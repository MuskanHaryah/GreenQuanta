"""Hand-sized real/quantized reference comparisons, without runners or sampling."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.exact_quantized import solve_exact_quantized
from phase2 import config
from phase2.exact_solver import linear_cost, solve_exact
from phase2.validation import validate_allocation


def hand_problem(resource, size_units, capacity_units):
    unit = config.UNIT_NCU if resource == "cpu" else config.UNIT_NMU
    sizes = {"cpu_size": config.UNIT_NCU, "mem_size": config.UNIT_NMU}
    sizes[f"{resource}_size"] = size_units * unit
    vms = [SimpleNamespace(vm_id=i, cpu_mean_served=sizes["cpu_size"], **sizes)
           for i in (10, 20)]
    limits = {"cpu_limit": 10 * config.UNIT_NCU, "mem_limit": 10 * config.UNIT_NMU}
    limits[f"{resource}_limit"] = capacity_units * unit
    servers = [SimpleNamespace(server_id=i, cpu_capacity=10 * config.UNIT_NCU,
                               fit_idle_w=10.0, fit_slope_w=2.0, **limits)
               for i in range(2)]
    return vms, servers


def assert_solution(vms, servers, assignment, info):
    assert assignment is not None
    assert info["status"] == "optimal"
    assert info["proven_optimal"]
    assert info["mip_gap"] == pytest.approx(0)
    assert info["objective_joule"] == pytest.approx(linear_cost(assignment, vms, servers))
    assert info["time_s"] >= 0
    assert validate_allocation(vms, assignment, servers)["valid"]


@pytest.mark.parametrize("resource", ["cpu", "mem"])
def test_rounding_requires_one_more_server(resource):
    # Real load is 0.98 units on a 1-unit host; rounded load is 2 units.
    vms, servers = hand_problem(resource, size_units=0.49, capacity_units=1)
    real, real_info = solve_exact(vms, servers)
    quantized, quantized_info = solve_exact_quantized(vms, servers)
    assert_solution(vms, servers, real, real_info)
    assert_solution(vms, servers, quantized, quantized_info)
    assert len(set(real.values())) == 1
    assert len(set(quantized.values())) == len(set(real.values())) + 1
    assert quantized_info["objective_joule"] > real_info["objective_joule"]
    assert not validate_allocation(vms, real, servers, quantized_unit=config.UNIT_NCU)["valid"]
    assert validate_allocation(vms, quantized, servers, quantized_unit=config.UNIT_NCU)["valid"]


@pytest.mark.parametrize("resource", ["cpu", "mem"])
def test_exact_unit_multiples_have_equal_optima(resource):
    vms, servers = hand_problem(resource, size_units=1, capacity_units=2)
    real, real_info = solve_exact(vms, servers)
    quantized, quantized_info = solve_exact_quantized(vms, servers)
    assert_solution(vms, servers, real, real_info)
    assert_solution(vms, servers, quantized, quantized_info)
    assert len(set(real.values())) == len(set(quantized.values())) == 1
    assert quantized_info["objective_joule"] == pytest.approx(real_info["objective_joule"])
    for assignment in (real, quantized):
        assert validate_allocation(vms, assignment, servers, quantized_unit=config.UNIT_NCU)["valid"]


def test_no_incumbent_has_complete_info():
    vms, servers = hand_problem("cpu", size_units=2, capacity_units=1)
    assignment, info = solve_exact_quantized(vms, servers)
    assert assignment is None
    assert info["status"]
    assert info["proven_optimal"] is False
    assert info["mip_gap"] is None
    assert info["objective_joule"] is None
    assert info["time_s"] >= 0
