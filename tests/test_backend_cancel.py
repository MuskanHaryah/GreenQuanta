"""POST /api/runs/<id>/cancel (the dashboard's Stop button): a running live run is killed with its process tree and
ends "cancelled"; nothing is logged for it; the run lock is free again afterwards."""
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import create_app

LOG = ROOT / "results" / "experiment_log.csv"


def wait(client, job_id, cap_s, statuses=("done", "failed", "cancelled")):
    end = time.monotonic() + cap_s
    while time.monotonic() < end:
        job = client.get(f"/api/runs/{job_id}").get_json()
        if job["status"] in statuses:
            return job
        time.sleep(0.25)
    pytest.fail(f"job {job_id} did not reach {statuses} within {cap_s} s")


def test_cancel_running_job_kills_it_and_frees_the_lock():
    app = create_app({"TESTING": True})
    client = app.test_client()
    log_rows = LOG.read_text(encoding="utf-8").count("\n")
    job_id = client.post("/api/runs", json={"size": "medium_v2", "stage2": "csqa"}).get_json()["job_id"]
    wait(client, job_id, 60, statuses=("running",))
    time.sleep(2)                                       # well inside the run (medium takes about a minute)
    r = client.post(f"/api/runs/{job_id}/cancel")
    assert r.status_code == 200 and r.get_json() == {"job_id": job_id, "status": "cancelled"}
    job = wait(client, job_id, 30)
    assert job["status"] == "cancelled" and job["error"] == "stopped by the user" and job["result"] is None
    assert app.extensions["jobs"].process(job_id).poll() is not None            # not left running
    assert LOG.read_text(encoding="utf-8").count("\n") == log_rows               # a stopped run logs nothing
    again = client.post(f"/api/runs/{job_id}/cancel")
    assert again.status_code == 409 and again.get_json()["error"]["code"] == "not_running"
    nxt = client.post("/api/runs", json={"size": "medium_v2", "stage2": "csqa"})  # the lock is free again
    assert nxt.status_code == 202
    assert client.post(f"/api/runs/{nxt.get_json()['job_id']}/cancel").status_code == 200
    assert wait(client, nxt.get_json()["job_id"], 30)["status"] == "cancelled"


def test_cancel_unknown_job_404():
    r = create_app({"TESTING": True}).test_client().post("/api/runs/nope/cancel")
    assert r.status_code == 404 and r.get_json()["error"]["code"] == "unknown_job"
