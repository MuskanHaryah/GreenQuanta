"""Guide 2 step 2.4: live runs through the API (Flask test client; each run is a real subprocess of
demo.run_pipeline). A finished small_v2 csqa run appends one experiment-log row."""
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import create_app


def wait(client, job_id, cap_s):
    end = time.monotonic() + cap_s
    while time.monotonic() < end:
        job = client.get(f"/api/runs/{job_id}").get_json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.5)
    pytest.fail(f"job {job_id} not finished after {cap_s} s")


def test_live_run_small_done_exclusive_and_matches_cached():
    app = create_app({"TESTING": True})
    client = app.test_client()
    r = client.post("/api/runs", json={"size": "small_v2", "stage2": "csqa"})
    assert r.status_code == 202
    job_id = r.get_json()["job_id"]
    busy = client.post("/api/runs", json={"size": "medium_v2", "stage2": "csa"})
    assert busy.status_code == 409 and busy.get_json()["job_id"] == job_id
    assert busy.get_json()["error"]["code"] == "run_in_progress"
    job = wait(client, job_id, 900)
    assert job["status"] == "done", job["error"]
    assert job["matches_cached"] is True and job["error"] is None and job["elapsed_s"] > 0
    assert set(job) == {"job_id", "size", "stage2", "status", "started_at", "elapsed_s", "events", "result",
                        "matches_cached", "error"}
    kinds = [e["kind"] for e in job["events"]]
    order = [kinds.index(k) for k in ("PROPOSALS", "STAGE2", "HYBRID", "RESULT")]
    assert order == sorted(order) and kinds[-1] == "RESULT" and "LOGGED" in kinds
    assert [e["t"] for e in job["events"]] == sorted(e["t"] for e in job["events"])
    assert job["result"] == job["events"][-1]["data"] and job["result"]["out"]["chosen_ids"] == [2, 3]
    assert app.extensions["jobs"].process(job_id).poll() == 0


@pytest.mark.parametrize("body", [None, [], {"size": "small_v2"}, {"stage2": "csqa"},
                                  {"size": "huge_v2", "stage2": "csqa"}, {"size": "small_v2", "stage2": "exact"}])
def test_bad_body_400(body):
    client = create_app({"TESTING": True}).test_client()
    r = client.post("/api/runs", json=body) if body is not None else client.post("/api/runs", data="x")
    assert r.status_code == 400 and r.get_json()["error"]["code"] == "bad_request"


def test_unknown_job_404():
    r = create_app({"TESTING": True}).test_client().get("/api/runs/nope")
    assert r.status_code == 404 and r.get_json()["error"]["code"] == "unknown_job"


def test_timeout_kills_and_fails():
    app = create_app({"TESTING": True, "RUN_TIMEOUT_S": 1})
    client = app.test_client()
    job_id = client.post("/api/runs", json={"size": "medium_v2", "stage2": "csqa"}).get_json()["job_id"]
    job = wait(client, job_id, 60)
    assert job["status"] == "failed" and "timed out" in job["error"] and job["result"] is None
    assert app.extensions["jobs"].process(job_id).poll() is not None        # not left running
    again = client.post("/api/runs", json={"size": "medium_v2", "stage2": "csqa"})     # the lock is free again
    assert again.status_code == 202 and wait(client, again.get_json()["job_id"], 60)["status"] == "failed"
