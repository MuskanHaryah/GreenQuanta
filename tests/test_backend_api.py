"""Guide 2 step 2.2: the cached-data endpoints on small_v2 (Flask test client), against results/mids/small_v2/results.json
and the VM sizes of diagnostics.common.build_problem."""
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import create_app
from diagnostics.common import build_problem

SIZE = "small_v2"
CACHED = json.loads((ROOT / "results" / "mids" / SIZE / "results.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client():
    return create_app({"TESTING": True}).test_client()


def get(client, url, status=200):
    r = client.get(url)
    assert r.status_code == status, (url, r.status_code, r.get_data(as_text=True)[:500])
    return r.get_json()


def test_health(client):
    body = get(client, "/api/health")
    assert body["status"] == "ok" and len(body["git_commit"]) == 40
    assert body["results_available"] == {"small_v2": True, "medium_v2": True, "large_v2": True}


def test_sizes(client):
    rows = {r["id"]: r for r in get(client, "/api/sizes")}
    assert list(rows) == ["small_v2", "medium_v2", "large_v2"]
    assert rows[SIZE] == {"id": SIZE, "label": "Small", "vms": CACHED["problem"]["vms"],
                          "servers": CACHED["problem"]["m"], "m": CACHED["problem"]["m"], "available": True}


def test_results_without_assignments(client):
    body = get(client, f"/api/results/{SIZE}")
    assert set(body) == set(CACHED) and set(body["methods"]) == set(CACHED["methods"])
    for method_id, r in body["methods"].items():
        assert "assignment" not in r
        assert r == {k: v for k, v in CACHED["methods"][method_id].items() if k != "assignment"}


@pytest.mark.parametrize("method_id", list(CACHED["methods"]))
def test_placement_loads_and_coverage(client, method_id):
    body = get(client, f"/api/results/{SIZE}/placement/{method_id}")
    vms, servers, _, _ = build_problem(SIZE)
    size_of = {v.vm_id: (v.cpu_size, v.mem_size) for v in vms}
    assert body["method_id"] == method_id and [h["host_id"] for h in body["hosts"]] == [s.server_id for s in servers]
    seen = [vm["vm_id"] for h in body["hosts"] for vm in h["vms"]]
    assert sorted(seen) == sorted(size_of)                       # every VM exactly once
    stored = CACHED["methods"][method_id]["assignment"]
    for h in body["hosts"]:
        assert all(stored[str(vm["vm_id"])] == h["host_id"] for vm in h["vms"])
        assert h["active"] == (h["host_id"] in CACHED["methods"][method_id]["active_servers"])
        assert math.isclose(h["cpu_load"], math.fsum(size_of[vm["vm_id"]][0] for vm in h["vms"]), abs_tol=1e-12)
        assert math.isclose(h["mem_load"], math.fsum(size_of[vm["vm_id"]][1] for vm in h["vms"]), abs_tol=1e-12)
        if CACHED["methods"][method_id]["valid_real"]:
            assert h["cpu_load"] <= h["cpu_limit"] + 1e-9 and h["mem_load"] <= h["mem_limit"] + 1e-9


def test_compare_equals_json(client):
    rows = get(client, f"/api/results/{SIZE}/compare")
    assert sorted(r["method_id"] for r in rows) == sorted(CACHED["methods"])
    assert [(r["diff_r1_pct"], r["method_id"]) for r in rows] == sorted((r["diff_r1_pct"], r["method_id"]) for r in rows)
    for row in rows:
        m = CACHED["methods"][row["method_id"]]
        assert row == {"method_id": row["method_id"], "label": m["label"], "kind": m["kind"],
                       "valid_real": m["valid_real"], "valid_quantized": m["valid_quantized"],
                       "active_servers": len(m["active_servers"]), "diff_r1_pct": m["diff_r1_pct"],
                       "diff_r2_pct": m["diff_r2_pct"], "gap_to_h_min_pct": m["gap_to_h_min_pct"],
                       "it_energy_kwh": m["energy"]["it_energy_kwh"], "co2_kg": m["energy"]["co2_kg"],
                       "cloudsim_ok": m["cloudsim"]["ok"]}


def test_claims(client):
    body = get(client, f"/api/claims?size={SIZE}")
    assert body["notes"] == CACHED["notes"]
    assert any("valid_quantized" in c for c in body["caveats"]) and any("one Stage-2 seed" in c for c in body["caveats"])
    assert {m["method_id"]: m["label"] for m in body["methods"]} == {k: r["label"] for k, r in CACHED["methods"].items()}
    assert get(client, "/api/claims")["size"] == "large_v2"


@pytest.mark.parametrize("url,code", [("/api/results/huge_v2", "unknown_size"),
                                      ("/api/results/huge_v2/compare", "unknown_size"),
                                      ("/api/results/small_v2/placement/nope", "unknown_method"),
                                      ("/api/claims?size=huge_v2", "unknown_size"),
                                      ("/api/nope", "not_found")])
def test_404(client, url, code):
    assert get(client, url, 404)["error"]["code"] == code


def test_missing_results_names_the_build_command(tmp_path):
    body = get(create_app({"TESTING": True, "RESULTS_ROOT": tmp_path}).test_client(), f"/api/results/{SIZE}", 404)
    assert body["error"]["code"] == "results_missing" and "build_results.py --size small_v2" in body["error"]["message"]
