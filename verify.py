"""GreenQuanta - verification runner.

Run it BEFORE you start a new step (to know the project is healthy) and AFTER you
finish it (to prove the new step works and nothing older broke).

    python verify.py              # all fast checks
    python verify.py --full       # also the slower solver checks (about 10-30 s)
    python verify.py --java       # also re-runs the CloudSim demo (needs Maven, a few minutes)
    python verify.py --phase 4    # only one area, e.g. 0, 1, 4

Statuses
    PASS     works
    FAIL     something that SHOULD work is broken -> fix before moving on
    SKIP     could not be checked here (reason is printed)
    TODO     a later phase that is not built yet (normal, not an error)
    REGRESSION  was PASS on the previous run and is FAIL now -> you broke something

Every time we build a phase, its checks are added to this file (see CHECKS and
PHASE_MARKERS). Intentional changes to a "golden" number must be edited here on
purpose, so a number can never change silently.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "optimizer"))
STATE = ROOT / "results" / "verify_last.json"


class Skip(Exception):
    pass


CHECKS = []  # (phase, name, func, kind)  kind: "fast" | "full" | "java"


def check(phase, name, kind="fast"):
    def deco(fn):
        CHECKS.append((phase, name, fn, kind))
        return fn
    return deco


def need(cond, msg):
    if not cond:
        raise AssertionError(msg)


def run_cmd(cmd, cwd=None, timeout=60):
    return subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=timeout)


# ----------------------------------------------------------------------------
# PHASE 0 - environment and repository
# ----------------------------------------------------------------------------
@check("0", "Python is 3.12")
def _():
    need(sys.version_info[:2] == (3, 12), f"Python is {sys.version.split()[0]}, expected 3.12.x. "
         "Activate the venv: .\\.venv\\Scripts\\Activate.ps1")


@check("0", "Virtual environment is active")
def _():
    need(sys.prefix != sys.base_prefix, "No (.venv) active. Run: .\\.venv\\Scripts\\Activate.ps1")


@check("0", "Required Python libraries import")
def _():
    missing = []
    for mod in ("numpy", "pandas", "dimod", "dwave.samplers"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    need(not missing, f"cannot import: {missing}. Run: python -m pip install -r optimizer/requirements.txt")


@check("0", "Project folders exist")
def _():
    gone = [d for d in ("simulation", "optimizer", "data", "results", "docs") if not (ROOT / d).is_dir()]
    need(not gone, f"missing folders: {gone}")


@check("0", "Git remote points to the GreenQuanta repository")
def _():
    if not shutil.which("git"):
        raise Skip("git not found")
    out = run_cmd("git remote -v", cwd=ROOT).stdout
    need("GreenQuanta" in out, "no GreenQuanta remote. Run: git remote add origin <repo url>")


@check("0", "Java 17 and Maven are available")
def _():
    if not shutil.which("mvn"):
        raise Skip("mvn not found on PATH")
    out = run_cmd("mvn -version").stdout
    need("Java version: 17" in out, "Maven is not using Java 17 (check JAVA_HOME)")


# ----------------------------------------------------------------------------
# PHASE 1 - things built from the original setup guide (must keep working)
# ----------------------------------------------------------------------------
@check("1", "D-Wave local annealer solves the 2-switch test (energy -1.0)")
def _():
    import dimod
    from dwave.samplers import SimulatedAnnealingSampler
    bqm = dimod.BinaryQuadraticModel({"x0": -1, "x1": -1}, {("x0", "x1"): 2}, 0.0, dimod.BINARY)
    s = SimulatedAnnealingSampler().sample(bqm, num_reads=10, seed=1)
    need(abs(s.first.energy + 1.0) < 1e-9, f"energy {s.first.energy}, expected -1.0")


@check("1", "data/hosts.csv and data/vms.csv are the 2-host / 4-VM demo")
def _():
    import pandas as pd
    h = pd.read_csv(ROOT / "data" / "hosts.csv")
    v = pd.read_csv(ROOT / "data" / "vms.csv")
    need(list(h.columns) == ["host_id", "pes", "ram_mb"], f"hosts.csv columns {list(h.columns)}")
    need(list(v.columns) == ["vm_id", "pes", "ram_mb"], f"vms.csv columns {list(v.columns)}")
    need(len(h) == 2 and len(v) == 4, f"expected 2 hosts / 4 VMs, found {len(h)} / {len(v)}")


@check("1", "QUBO placement result (results/assignment_qa.csv) is valid")
def _():
    import pandas as pd
    f = ROOT / "results" / "assignment_qa.csv"
    need(f.exists(), "missing. Run: python qa_placement.py (in optimizer/)")
    a = pd.read_csv(f)
    need(sorted(a["vm_id"]) == [0, 1, 2, 3], "every VM must appear exactly once")
    need(a["host_id"].nunique() == 1, "expected all 4 VMs on one host")


@check("1", "results/qa_summary.json says 1 active host, 250 W")
def _():
    f = ROOT / "results" / "qa_summary.json"
    need(f.exists(), "missing. Run: python qa_placement.py")
    d = json.loads(f.read_text())
    need(d["active_hosts"] == 1 and abs(d["power_watts"] - 250.0) < 1e-6, f"got {d}")


def _cloudsim_total(name):
    import pandas as pd
    f = ROOT / "results" / name
    need(f.exists(), f"missing {name}. Run the CloudSim GreenSimulation step (Step 11)")
    d = pd.read_csv(f)
    return float(d["energy_wh"].sum()), int(d["active"].sum())


@check("1", "CloudSim default mode: 2 hosts, 2.6534 Wh")
def _():
    e, n = _cloudsim_total("cloudsim_default.csv")
    need(n == 2 and abs(e - 2.6534) < 0.002, f"active={n}, energy={e:.4f}")


@check("1", "CloudSim QA mode: 1 host, 1.5362 Wh")
def _():
    e, n = _cloudsim_total("cloudsim_qa.csv")
    need(n == 1 and abs(e - 1.5362) < 0.002, f"active={n}, energy={e:.4f}")


@check("1", "CloudSim re-run reproduces both energy numbers", kind="java")
def _():
    sim = ROOT / "simulation"
    base = 'mvn -q compile exec:java "-Dexec.mainClass=greenquanta.GreenSimulation"'
    for args, want in (("", 2.6534), (' "-Dexec.args=qa"', 1.5362)):
        r = run_cmd(base + args, cwd=sim, timeout=600)
        need(r.returncode == 0, "Maven failed:\n" + (r.stdout + r.stderr)[-600:])
        m = re.search(r"Total energy:\s*([0-9.]+)", r.stdout)
        need(m and abs(float(m.group(1)) - want) < 0.002, f"energy {m.group(1) if m else None}, expected {want}")


# ----------------------------------------------------------------------------
# PHASE 4 - core optimizer pieces that already exist (model, baselines, QIEA)
# ----------------------------------------------------------------------------
def _inst(n):
    from gq_model import Params, make_synthetic
    p = Params()
    return make_synthetic(n, seed=n, params=p), p


@check("4", "Model: energy formula, cooling (PUE) and carbon match a hand calculation")
def _():
    import numpy as np
    from gq_model import Instance, Params, evaluate
    inst = Instance(np.array([10., 10.]), np.array([1e4, 1e4]), np.array([100., 100.]),
                    np.array([200., 200.]), np.array([0.5, 0.5]), np.array([5.]), np.array([1000.]))
    r = evaluate(inst, [0], Params(pue=1.4, hours=1.0))
    need(abs(r["server_w"] - 150.0) < 1e-9, f"server_w {r['server_w']} (idle host must add 0 W)")
    need(abs(r["cooling_w"] - 60.0) < 1e-9, f"cooling_w {r['cooling_w']}")
    need(abs(r["energy_kwh"] - 0.21) < 1e-9, f"energy_kwh {r['energy_kwh']}")
    need(abs(r["carbon_kg"] - 0.105) < 1e-9, f"carbon_kg {r['carbon_kg']}")
    need(r["active_hosts"] == 1, "active host count wrong")


@check("4", "Model: an overloaded assignment is flagged infeasible")
def _():
    import numpy as np
    from gq_model import evaluate
    inst, p = _inst(30)
    r = evaluate(inst, np.zeros(inst.n_vms, int), p)
    need(r["feasible"] is False, "30 VMs on host 0 should not be feasible")


@check("4", "Model: synthetic instances are reproducible")
def _():
    import numpy as np
    a, _p = _inst(30)
    b, _p = _inst(30)
    need(np.array_equal(a.vm_cpu, b.vm_cpu) and np.array_equal(a.cpu_cap, b.cpu_cap), "same seed gave different data")


@check("4", "Baselines: FFD/BFD feasible, within SLA headroom, known host counts")
def _():
    from baselines import best_fit_decreasing, first_fit_decreasing
    from gq_model import loads
    inst, p = _inst(30)
    for fn, hosts in ((first_fit_decreasing, 7), (best_fit_decreasing, 9)):
        r = fn(inst, p)
        need(r["feasible"], f"{r['method']} infeasible")
        cpu, _ram = loads(inst, r["assign"])
        need((cpu <= inst.cpu_cap * p.u_safe + 1e-9).all(), f"{r['method']} exceeded the headroom limit")
        need(r["active_hosts"] == hosts, f"{r['method']} used {r['active_hosts']} hosts, expected {hosts}")


@check("4", "QIEA repair step turns an overloaded assignment into a feasible one")
def _():
    import numpy as np
    from gq_model import is_feasible
    from qiea import repair
    inst, _p = _inst(30)
    fixed = repair(inst, np.zeros(inst.n_vms, int), np.random.default_rng(0))
    need(is_feasible(inst, fixed) and (fixed >= 0).all(), "repair did not give a complete feasible assignment")


@check("4", "QIEA: feasible, same seed gives same answer, best never gets worse")
def _():
    from qiea import qiea
    inst, p = _inst(10)
    a = qiea(inst, p, seed=0, generations=60)
    b = qiea(inst, p, seed=0, generations=60)
    need(a["feasible"], "QIEA returned an infeasible assignment")
    need(abs(a["objective"] - b["objective"]) < 1e-12, "same seed gave different results (not reproducible)")
    h = a["history"]
    need(all(h[i + 1] <= h[i] + 1e-12 for i in range(len(h) - 1)), "convergence curve got worse")
    need(a["iterations"] == 60 and a["seconds"] > 0, "iteration/time logging is wrong")


@check("4", "QIEA golden: 10 VMs reaches the known objective 0.40874", kind="full")
def _():
    from qiea import qiea
    inst, p = _inst(10)
    r = qiea(inst, p, seed=0)
    need(abs(r["objective"] - 0.40874) < 5e-4, f"objective {r['objective']:.5f}, expected 0.40874")


@check("4", "QIEA golden: 30 VMs beats FFD and uses 4 hosts", kind="full")
def _():
    from baselines import first_fit_decreasing
    from qiea import qiea
    inst, p = _inst(30)
    f = first_fit_decreasing(inst, p)
    r = qiea(inst, p, seed=0)
    need(r["feasible"], "infeasible")
    need(r["objective"] < f["objective"], f"QIEA {r['objective']:.4f} not better than FFD {f['objective']:.4f}")
    need(r["active_hosts"] <= 5, f"QIEA used {r['active_hosts']} hosts (expected about 4)")


@check("4", "results/benchmark.csv (if present) is complete and all runs feasible")
def _():
    import pandas as pd
    f = ROOT / "results" / "benchmark.csv"
    if not f.exists():
        raise Skip("run python run_benchmark.py 10 30 first")
    d = pd.read_csv(f)
    need({"vms", "method", "feasible", "active_hosts", "energy_kwh", "carbon_kg", "sla_rate",
          "objective", "seconds", "iterations"} <= set(d.columns), "benchmark.csv columns missing")
    need(d["feasible"].all(), "some benchmark rows are infeasible")
    need(set(d["method"]) >= {"FFD", "BFD", "QIEA"}, "a method is missing")


# ----------------------------------------------------------------------------
# PHASE 1 (decisions) - not code, but we still verify it
# ----------------------------------------------------------------------------
@check("D", "DECISIONS.md: every decision has been filled in")
def _():
    f = ROOT / "docs" / "DECISIONS.md"
    if not f.exists():
        raise Skip("DECISIONS.md is not in the project (decisions are tracked outside the repo)")
    blanks = re.findall(r"\*\*Our decision:\*\*\s*_{3,}", f.read_text(encoding="utf-8"))
    if blanks:
        raise Skip(f"{len(blanks)} decision(s) still blank (this is a TODO, not an error)")


# ----------------------------------------------------------------------------
# Later phases: shown as TODO until their marker file exists. When we build a
# phase we add real checks above and keep the marker here.
# ----------------------------------------------------------------------------
PHASE_MARKERS = {
    "2  Thin end-to-end slice": "optimizer/pipeline.py",
    "3  Data layer (time-varying workload, real trace)": "optimizer/workload.py",
    "4b Full QUBO + SA baseline + 2nd quantum-inspired solver": "optimizer/sa_baseline.py",
    "5  SLA validation module": "optimizer/sla.py",
    "6  Dynamic monitoring / re-optimization / migrations": "optimizer/dynamic_loop.py",
    "7  CloudSim cross-check (Python vs Java energy)": "optimizer/crosscheck.py",
    "8a Backend API": "backend/app.py",
    "8b Dashboard (React)": "dashboard/package.json",
    "9  Experiments on all sizes/seeds/trace": "results/experiments",
    "10 Automated tests": "tests",
    "11 Report chapters": "docs/report",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="include slower solver checks")
    ap.add_argument("--java", action="store_true", help="include the CloudSim re-run (needs Maven)")
    ap.add_argument("--phase", help="only this phase id, e.g. 0, 1, 4, D")
    a = ap.parse_args()

    prev = {}
    if STATE.exists():
        try:
            prev = json.loads(STATE.read_text())
        except Exception:
            prev = {}

    print("GreenQuanta verification\n" + "=" * 60)
    now, counts, regress = {}, {"PASS": 0, "FAIL": 0, "SKIP": 0}, []
    for phase, name, fn, kind in CHECKS:
        if a.phase and phase != a.phase:
            continue
        if kind == "full" and not (a.full or a.java):
            continue
        if kind == "java" and not a.java:
            continue
        t0 = time.perf_counter()
        try:
            fn()
            status, note = "PASS", ""
        except Skip as e:
            status, note = "SKIP", str(e)
        except AssertionError as e:
            status, note = "FAIL", str(e)
        except Exception as e:  # unexpected error is also a failure
            status, note = "FAIL", f"{type(e).__name__}: {e}"
        counts[status] += 1
        key = f"{phase}|{name}"
        now[key] = status
        tag = status
        if status == "FAIL" and prev.get(key) == "PASS":
            tag = "REGRESSION"
            regress.append(name)
        print(f"[{tag:^10}] P{phase}  {name}  ({time.perf_counter() - t0:.1f}s)")
        if note:
            print(f"             -> {note}")

    if not a.phase:
        print("\nLater phases (not built yet is normal):")
        for label, marker in PHASE_MARKERS.items():
            built = (ROOT / marker).exists()
            print(f"[{'FOUND' if built else 'TODO':^10}] {label}  ({marker})")

    print("=" * 60)
    print(f"PASS {counts['PASS']}   FAIL {counts['FAIL']}   SKIP {counts['SKIP']}")
    if regress:
        print("\nREGRESSION - these worked last time and are broken now:")
        for n in regress:
            print("  -", n)
    # remember results so the next run can spot regressions (only merge, never forget a PASS)
    merged = {**prev, **now}
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(merged, indent=2))
    if counts["FAIL"]:
        print("\nResult: NOT READY - fix the FAIL lines above before moving on.")
        sys.exit(1)
    print("\nResult: HEALTHY - safe to continue.")


if __name__ == "__main__":
    main()