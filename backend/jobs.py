"""Live pipeline runs: one subprocess `python -B -u -m demo.run_pipeline --size S --stage2 M` (cwd optimizer/) at a time.

Why a subprocess: the Stage-2 sampler already uses a multiprocessing spawn child (r3c_run.sampled_run), which is fragile
inside a Flask process on Windows; a subprocess also isolates crashes. Progress = the <KIND>_JSON stdout lines the
pipeline prints (demo.run_pipeline.parse_line). Job state lives in memory (lost on restart). On timeout the whole
process tree is killed (the spawn child too) and the job fails. A finished job is compared with the cached
results.json (matches_cached): same seeds from config, so it must reproduce it.
"""
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
import signal
import subprocess
import threading
import time
import uuid

from demo.run_pipeline import parse_line

METHOD_OF = {"csqa": "quantum", "csa": "control"}


class Busy(Exception):
    def __init__(self, job_id):
        super().__init__(job_id)
        self.job_id = job_id


def kill_tree(proc):
    """Kill the run and its spawn child (taskkill /T on Windows, the process group elsewhere)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()


def matches_cached(results_root, size, stage2, result):
    """True if a run's RESULT reproduces the cached method of results.json (assignment, chosen_ids where cached,
    gap to H_min i.e. V, joules to 1e-12 relative, and both validity flags); None without a cached file."""
    path = results_root / size / "results.json"
    if not path.is_file():
        return None
    cached = json.loads(path.read_text(encoding="utf-8"))
    method_id = METHOD_OF[stage2]
    m, out = cached["methods"][method_id], result["out"]
    ids = [s["id"] for s in cached["problem"]["servers"]]
    close = all(out.get(k) is not None and m[c] is not None and math.isclose(out[k], m[c], rel_tol=1e-12, abs_tol=1e-12)
                for k, c in (("gap_to_h_min_pct", "gap_to_h_min_pct"), ("linear_joules", "linear_joules")))
    return bool(close and result["assignment"] is not None
                and {v: ids[h] for v, h in result["assignment"].items()} == m["assignment"]
                and out["real_valid"] == m["valid_real"] and out["quantized_valid"] == m["valid_quantized"]
                and (method_id != "quantum" or out["chosen_ids"] == cached["stage1"]["chosen_ids"]))


class JobManager:
    def __init__(self, python, cwd, timeout_s, results_root):
        self.python, self.cwd, self.timeout_s, self.results_root = python, cwd, timeout_s, results_root
        self._lock = threading.Lock()
        self._jobs, self._procs, self._t0 = {}, {}, {}
        self._active = None

    def start(self, size, stage2):
        """Queue a run and return its id; raises Busy(running id) while another run is queued or running."""
        with self._lock:
            if self._active and self._jobs[self._active]["status"] in ("queued", "running"):
                raise Busy(self._active)
            job_id = uuid.uuid4().hex[:12]
            self._jobs[job_id] = {"job_id": job_id, "size": size, "stage2": stage2, "status": "queued",
                                  "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "elapsed_s": 0.0, "events": [], "result": None, "matches_cached": None,
                                  "error": None}
            self._t0[job_id], self._active = time.monotonic(), job_id
        threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
        return job_id

    def get(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            snapshot = {**job, "events": list(job["events"])}
            if job["status"] in ("queued", "running"):
                snapshot["elapsed_s"] = round(time.monotonic() - self._t0[job_id], 3)
            return snapshot

    def process(self, job_id):
        return self._procs.get(job_id)

    def _run(self, job_id):
        job, t0 = self._jobs[job_id], self._t0[job_id]
        cmd = [self.python, "-B", "-u", "-m", "demo.run_pipeline", "--size", job["size"], "--stage2", job["stage2"]]
        tail, timed_out = deque(maxlen=30), threading.Event()
        try:
            proc = subprocess.Popen(cmd, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace", start_new_session=os.name != "nt")
        except OSError as e:
            return self._finish(job_id, "failed", error=f"could not start the run: {e}")
        self._procs[job_id] = proc
        with self._lock:
            job["status"] = "running"
        timer = threading.Timer(self.timeout_s, lambda: (timed_out.set(), kill_tree(proc)))
        timer.start()
        result = None
        for line in proc.stdout:
            tail.append(line.rstrip())
            event = parse_line(line)
            if event:
                kind, data = event
                result = data if kind == "RESULT" else result
                with self._lock:
                    job["events"].append({"kind": kind, "t": round(time.monotonic() - t0, 3), "data": data})
        proc.wait()
        timer.cancel()
        if timed_out.is_set() and proc.returncode != 0:
            return self._finish(job_id, "failed", error=f"timed out after {self.timeout_s} s; process tree killed")
        if proc.returncode != 0 or result is None:
            return self._finish(job_id, "failed", error=f"exit code {proc.returncode}; last output:\n" + "\n".join(tail))
        self._finish(job_id, "done", result=result,
                     matches=matches_cached(self.results_root, job["size"], job["stage2"], result))

    def _finish(self, job_id, status, result=None, matches=None, error=None):
        with self._lock:
            self._jobs[job_id].update(status=status, result=result, matches_cached=matches, error=error,
                                      elapsed_s=round(time.monotonic() - self._t0[job_id], 3))
