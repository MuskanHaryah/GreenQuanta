"""D3 helper checks: tiny synthetic reads and saved placements, no annealing."""
from pathlib import Path
import sys
from types import SimpleNamespace

import dimod
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.common import build_problem
from diagnostics.d2_scale import decompose, score_assignment
from diagnostics.d3_search import encode_assignment, flip_bound, read_landscape, write_csv
from phase1.qubo_builder import x_var, y_var
from phase2 import config


@pytest.mark.parametrize("variable, expected", [("a", 7.5), ("b", 7.25), ("c", 2.25)])
def test_flip_bound(variable, expected):
    bqm = dimod.BinaryQuadraticModel(
        {"a": -2.0, "b": 3.0, "c": -0.5},
        {("a", "b"): -4.0, ("a", "c"): 1.5, ("b", "c"): -0.25},
        100.0, dimod.BINARY,
    )
    assert flip_bound(bqm, variable) == expected


def test_read_landscape_threshold_and_multiplicity():
    vms = [SimpleNamespace(vm_id=1, cpu_size=0.25, mem_size=0.25)]
    servers = [SimpleNamespace(cpu_limit=1.0, mem_limit=1.0) for _ in range(2)]
    x0, x1, y0, y1 = x_var(1, 0), x_var(1, 1), y_var(0), y_var(1)
    sampleset = dimod.SampleSet.from_samples(
        [{x0: 1, x1: 0, y0: 1, y1: 0},
         {x0: 0, x1: 0, y0: 0, y1: 0},
         {x0: 0, x1: 1, y0: 0, y1: 1},
         {x0: 1, x1: 0, y0: 0, y1: 0}],
        vartype=dimod.BINARY, energy=[-1.0, 0.5, 2.0, 0.5000000005],
        num_occurrences=[2, 3, 1, 4],
    )
    metrics, leaks = read_landscape(sampleset, vms, servers, h_exact=0.5)
    assert metrics["reads_le_exact"] == 9
    assert metrics["reads_le_exact_pct"] == 90.0
    assert metrics["valid_reads_le_exact"] == 2
    assert metrics["valid_reads"] == 3
    assert metrics["valid_reads_pct"] == 30.0
    assert metrics["min_h_all"] == -1.0
    assert metrics["min_h_valid"] == -1.0
    assert metrics["valid_servers_on_histogram"] == {1: 3}
    assert metrics["leak_reads"] == 7
    assert metrics["leak_broken_rules"] == {"assign": 3, "link": 4}
    assert sum(leak["reads"] for leak in leaks) == 7


def test_csv_writer_refuses_overwrite(tmp_path):
    path = tmp_path / "d3" / "small_v2.csv"
    write_csv(path, [{"size": "small_v2", "setting": "S0"}])
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_csv(path, [{"size": "small_v2", "setting": "S1"}])
    assert path.read_bytes() == original


@pytest.fixture(scope="module")
def small_problem():
    vms, servers, bqm, _ = build_problem("small_v2")
    parts, _ = decompose(vms, servers, bqm)
    return vms, servers, bqm, parts


@pytest.mark.parametrize("filename, known_h", [
    ("assignment_exact.csv", 0.4366451311159),
    ("assignment_annealer_only.csv", 0.5012005266234),
])
def test_saved_small_encoding_matches_d2(small_problem, filename, known_h):
    vms, servers, bqm, parts = small_problem
    path = config.RESULTS_DIR / "small_v2" / filename
    sample = encode_assignment(path, servers, bqm)
    actual = float(bqm.energy(sample))
    d2_h = score_assignment(path, vms, servers, bqm, parts)["total_h"]
    assert actual == pytest.approx(d2_h, rel=0.0, abs=1e-9)
    assert actual == pytest.approx(known_h, rel=0.0, abs=1e-9)
