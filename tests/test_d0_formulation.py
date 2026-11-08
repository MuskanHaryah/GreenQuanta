"""D0 checks: real trace VMs and exhaustive enumeration, never annealing."""
from pathlib import Path
import math
import sys
from types import SimpleNamespace

import dimod
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.d0_formulation import (
    HAND_VALUES, SIZES, assignment_sample, enumerate_states, independent_terms,
    load_workload, placement_comparison, require_unbalanced, resource_statistics,
    state_validation, tiny_problem,
)
from diagnostics.d2_scale import decompose
from phase1.qubo_builder import x_var, y_var
from phase2 import config
from phase2.catalog import build_servers
from phase2.pipeline_v2 import auto_mix
from phase2.qubo_builder_v2 import add_capacity_rule
from phase2.validation import validate_allocation


@pytest.mark.parametrize("free_share,expected", HAND_VALUES)
def test_hand_capacity_values(free_share, expected):
    assert 0.05 * (free_share**2 - 2 * 0.02 * free_share) == pytest.approx(expected, abs=1e-12)
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    slack = add_capacity_rule(
        bqm, 0, "sc", {0: round((1 - free_share) * 100)}, 100, 0.05,
        [SimpleNamespace(vm_id=0)], "unbalanced", center=0.02,
    )
    assert slack == 0
    assert len(bqm.variables) == 2
    assert bqm.energy({x_var(0, 0): 1, y_var(0): 1}) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("encoding", ["slack", "slack_scaled", "unknown"])
def test_reject_non_unbalanced(encoding):
    with pytest.raises(ValueError, match="unbalanced only"):
        require_unbalanced(encoding)
    with pytest.raises(ValueError, match="unbalanced only"):
        tiny_problem(encoding)


@pytest.fixture(scope="module")
def tiny():
    vms, servers, bqm, info, scale = tiny_problem()
    states, maximum, coefficient_error = enumerate_states(vms, servers, bqm, info)
    return vms, servers, bqm, info, scale, states, maximum, coefficient_error


def test_all_32768_states_match_independent_formula(tiny):
    vms, servers, bqm, info, scale, states, maximum, coefficient_error = tiny
    assert len(vms) == 4
    assert [s.type_name for s in servers] == ["B", "D", "D"]
    assert len(bqm.variables) == 15
    assert info["slack_variables"] == 0
    assert len(states) == 2**15
    assert maximum < 1e-9
    assert coefficient_error <= 1e-12
    original = load_workload("small_v2")[:4]
    for vm, source in zip(vms, original):
        assert vm.vm_id == source.vm_id
        assert vm.cpu_size == source.cpu_size * scale
        assert vm.mem_size == source.mem_size * scale
        assert (vm.cpu_demand == source.cpu_demand).all()
        assert (vm.mem_demand == source.mem_demand).all()
    assert any(sum(getattr(v, f"{r}_size") for v in vms) > getattr(s, f"{r}_limit")
               for r in ("cpu", "mem") for s in servers)


def test_independent_terms_include_offsets_and_off_server_link(tiny):
    vms, servers, bqm, info, _, _, _, _ = tiny
    parts, _ = decompose(vms, servers, bqm)
    empty = dict.fromkeys(bqm.variables, 0)
    terms = independent_terms(empty, vms, servers, info, parts["energy"])
    assert terms["assign"] == pytest.approx(len(vms) * info["lambdas"]["assign"])
    assert terms["cpu"] == 0
    assert terms["ram"] == 0
    assert terms["link"] == 0
    assert math.fsum(terms.values()) == pytest.approx(bqm.energy(empty), abs=1e-9)
    empty[x_var(vms[0].vm_id, 0)] = 1
    terms = independent_terms(empty, vms, servers, info, parts["energy"])
    assert terms["link"] == info["lambdas"]["link"]
    assert math.fsum(terms.values()) == pytest.approx(bqm.energy(empty), abs=1e-9)


def test_all_81_assignments_use_real_size_validation_and_rank(tiny):
    vms, servers, bqm, _, _, states, _, _ = tiny
    result = placement_comparison(vms, servers, bqm, states)
    assert 0 < len(result["valid"]) < 3**4
    assert len({row["hosts"] for row in result["valid"]}) == len(result["valid"])
    for row in result["valid"]:
        assert validate_allocation(vms, row["assignment"], servers)["valid"]
        sample = assignment_sample(row["assignment"], bqm)
        assert state_validation(sample, vms, servers)["valid"]
        assert {h for h in range(3) if sample[y_var(h)]} == set(row["hosts"])
    for key, best in result["best"].items():
        assert best[key] == min(row[key] for row in result["valid"])
    assert 1 <= result["linear_h_rank_valid"] <= len(result["valid"])
    assert result["linear_h_rank_valid"] <= result["linear_h_rank_all"] <= len(states)


def test_validation_rejects_missing_duplicate_and_off_server(tiny):
    vms, servers, bqm, _, _, states, _, _ = tiny
    best = placement_comparison(vms, servers, bqm, states)["best"]["h"]
    sample = assignment_sample(best["assignment"], bqm)
    vm = vms[0]
    host = best["assignment"][vm.vm_id]
    sample[y_var(host)] = 0
    assert not state_validation(sample, vms, servers)["valid"]
    sample[y_var(host)] = 1
    sample[x_var(vm.vm_id, host)] = 0
    assert not state_validation(sample, vms, servers)["valid"]
    sample[x_var(vm.vm_id, host)] = 1
    sample[x_var(vm.vm_id, (host + 1) % 3)] = 1
    assert not state_validation(sample, vms, servers)["valid"]


@pytest.mark.parametrize("size", SIZES)
def test_resource_statistics_real_sizes_and_catalog(size):
    vms = load_workload(size)
    servers = build_servers(auto_mix(vms))
    stats = resource_statistics(vms, servers)
    for resource, unit in (("cpu", config.UNIT_NCU), ("mem", config.UNIT_NMU)):
        sizes = [getattr(v, f"{resource}_size") for v in vms]
        a = sorted(max(1, math.ceil(value / unit - 1e-9)) for value in sizes)
        row = stats[resource]
        assert row["min"] == a[0]
        assert row["max"] == a[-1]
        assert row["median"] == (a[(len(a) - 1) // 2] + a[len(a) // 2]) / 2
        assert row["sum_units"] == sum(a)
        assert row["sum_k_by_type"] == {
            name: sum(math.floor(getattr(s, f"{resource}_limit") / unit + 1e-9)
                      for s in servers if s.type_name == name)
            for name in {s.type_name for s in servers}
        }
        assert row["inflation"] == pytest.approx(sum(a) * unit / sum(sizes) - 1, abs=1e-12)
        assert row["pressure"] == row["sum_units"] / sum(row["sum_k_by_type"].values())
