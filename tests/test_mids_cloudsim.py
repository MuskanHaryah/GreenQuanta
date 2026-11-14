"""Guide 1 step 1.3: write_cloudsim_inputs writes exactly what write_outputs writes (temp folders only)."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.common import build_problem
from diagnostics.r3_run import saved_sample
from phase2 import config
from phase2.energy_model import evaluate_allocation
from phase2.pipeline import prepare, write_cloudsim_inputs, write_outputs


def test_write_cloudsim_inputs_matches_write_outputs(tmp_path):
    size = "small_v2"
    vms, servers, bqm, info = build_problem(size)
    _, percentile, tinfo, need = prepare(config.TRACE_DIR / config.SCENARIOS[size]["trace"], servers)
    _, assignment = saved_sample(config.RESULTS_DIR / size / "assignment_exact.csv", servers, bqm)   # read only
    write_outputs(tmp_path / "a", size, servers, vms, percentile, tinfo, need, info, {}, assignment,
                  evaluate_allocation(vms, assignment, servers), None, None, {})
    write_cloudsim_inputs(tmp_path / "b", servers, vms, assignment)
    names = sorted(p.name for p in (tmp_path / "a" / "cloudsim_input").iterdir())
    assert names == ["servers.csv", "settings.csv", "vm_cpu_demand.csv", "vms_assignment.csv"]
    assert names == sorted(p.name for p in (tmp_path / "b").iterdir())
    for name in names:
        assert (tmp_path / "a" / "cloudsim_input" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
