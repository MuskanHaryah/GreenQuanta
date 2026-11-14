"""Guide 1 step 1.5: results/mids/<size>/results.json follows the Guide 2/3 contract, the quantum method matches the
golden snapshot, and every stored energy equals a fresh evaluate_allocation of the stored assignment."""
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimizer"))

from diagnostics.common import build_problem
from phase2.energy_model import evaluate_allocation

SIZES = ["small_v2", "medium_v2", "large_v2"]
METHODS = ["quantum", "control", "exact_r1", "cs_firstfit", "cs_bestfit", "cs_worstfit", "cs_roundrobin", "cs_ffd",
           "cs_bfd", "pabfd"]
TOP = {"size": str, "generated_at": str, "git_commit": str, "problem": dict, "references": dict, "stage1": dict,
       "methods": dict, "notes": list}
SERVER = ["id", "type", "cpu_capacity", "mem_capacity", "cpu_limit", "mem_limit", "idle_w", "max_w"]
STAGE1 = ["m", "states", "gas_budget", "gas_runs", "gas_oracle_calls", "gas_measurements", "exhaustive_evaluations",
          "y_star_ids", "chosen_ids"]
METHOD = {"label": str, "kind": str, "algorithm": str, "found": bool, "valid_real": bool, "valid_quantized": bool,
          "unplaced_vms": int, "active_servers": list, "assignment": dict, "resources": dict}
NULLABLE = ["reference", "linear_joules", "diff_r1_pct", "diff_r2_pct", "gap_to_h_min_pct", "energy", "cloudsim",
            "runtime_s"]
HOST = ["host_id", "type", "active", "vms", "mean_cpu_util_pct", "peak_cpu_util_pct", "mem_util_pct", "energy_kwh"]


def load(size):
    path = ROOT / "results" / "mids" / size / "results.json"
    if not path.exists():
        pytest.fail(f"missing {path}; run optimizer/demo/build_results.py --size {size}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("size", SIZES)
def test_schema(size):
    r = load(size)
    for key, kind in TOP.items():
        assert isinstance(r[key], kind), key
    assert r["size"] == size and r["problem"]["unit"] == 0.0025
    assert len(r["problem"]["servers"]) == r["problem"]["m"]
    assert all(set(SERVER) <= set(s) for s in r["problem"]["servers"])
    assert set(r["references"]) == {"h_min", "r1_joules", "r2_joules", "r2_curve_it_kwh"}
    assert set(STAGE1) <= set(r["stage1"]) and r["stage1"]["states"] == 2 ** r["problem"]["m"]
    assert list(r["methods"]) == METHODS
    for method_id, m in r["methods"].items():
        for key, kind in METHOD.items():
            assert isinstance(m[key], kind), (method_id, key)
        assert set(NULLABLE) <= set(m), method_id
        assert m["kind"] in ("quantum", "classical-control", "exact", "baseline")
        if m["found"]:
            assert len(m["assignment"]) == r["problem"]["vms"]
            assert set(m["energy"]) == {"it_energy_kwh", "facility_energy_kwh", "co2_kg", "per_host"}
            assert all(set(h) == set(HOST) for h in m["energy"]["per_host"])
            assert set(m["cloudsim"]) == {"it_energy_kwh", "co2_kg", "diff_it_pct_vs_python", "ok"}
    assert len(r["notes"]) >= 5


@pytest.mark.parametrize("size", SIZES)
def test_quantum_matches_golden(size):
    q = load(size)["methods"]["quantum"]
    g = json.loads((ROOT / "tests" / "data" / "mids_golden.json").read_text(encoding="utf-8"))[size]["csqa"]
    assert q["active_servers"] == g["chosen_ids"]
    assert (q["valid_real"], q["valid_quantized"]) == (g["real_valid"], g["quantized_valid"])
    assert math.isclose(q["linear_joules"], g["linear_joules"], rel_tol=1e-12)
    assert math.isclose(q["gap_to_h_min_pct"], g["gap_to_h_min_pct"], rel_tol=1e-12)


@pytest.mark.parametrize("size", SIZES)
def test_energy_is_fresh_evaluate_allocation(size):
    r = load(size)
    vms, servers, _, _ = build_problem(size)
    index = {s.server_id: h for h, s in enumerate(servers)}
    for method_id, m in r["methods"].items():
        if not m["found"]:
            continue
        fresh = evaluate_allocation(vms, {int(v): index[h] for v, h in m["assignment"].items()}, servers)
        for key in ("it_energy_kwh", "facility_energy_kwh", "co2_kg"):
            assert m["energy"][key] == fresh[key], (method_id, key)
        assert m["energy"]["per_host"] == [{k: h[k] for k in HOST} for h in fresh["per_host"]], method_id
