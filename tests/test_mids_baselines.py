"""Guide 1 step 1.4: CloudSim baselines on small_v2 (runs Maven; temp output folder)."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from demo.baselines import POLICIES, run_baseline
from diagnostics.common import build_problem


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    vms, servers, _, _ = build_problem("small_v2")
    out = tmp_path_factory.mktemp("mids")
    return vms, {p: run_baseline("small_v2", p, vms, servers, out) for p in POLICIES}


def test_every_assignment_covers_every_vm_once_or_is_flagged(runs):
    vms, results = runs
    ids = sorted(v.vm_id for v in vms)
    for policy, r in results.items():
        complete = sorted(r["assignment"]) == ids and r["unplaced_vms"] == 0
        assert complete or not (r["found"] and r["valid_real"]), policy
        assert complete == (r["energy"] is not None), policy


def test_sorted_submission_is_applied(runs):
    vms, results = runs
    size = {v.vm_id: (v.cpu_size, v.mem_size) for v in vms}
    expected = sorted(size, key=lambda i: (-size[i][0], -size[i][1], i))
    assert results["firstfit"]["submission_order"] == [v.vm_id for v in vms]
    for policy in ("ffd", "bfd", "pabfd"):
        assert results[policy]["submission_order"] == expected, policy
    assert results["ffd"]["submission_order"] != results["firstfit"]["submission_order"]
