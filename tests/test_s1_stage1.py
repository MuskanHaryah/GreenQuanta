"""Step 1.2: Stage-1 cost C(y) vs brute force on the tiny instance (4 VMs, 3 servers).

V(y) = exact fixed-y minimum of the slack QUBO at lambda = 1.1*U: all 2^12 x states per y, slack
eliminated by the Step 0 Lemma (d0_formulation.independent_terms, slack branch).
"""
from itertools import product
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.d0_formulation import independent_terms, tiny_problem
from diagnostics.r3c_build import build_qubo_custom
from diagnostics.s1_stage1 import aggregate_ok, all_costs, bits, stage1_cost, stage1_data
from phase1.qubo_builder import x_var, y_var
from phase2 import config
from phase2.validation import validate_allocation

FACTOR = 1.1
TOL = 1e-12


@pytest.fixture(scope="module")
def tiny():
    vms, servers, _, _, _ = tiny_problem()
    unit = config.UNIT_NCU
    data = stage1_data(vms, servers, unit, FACTOR)
    lam = data["lam"]
    _, info = build_qubo_custom(vms, servers, lam, lam, lam, lam, None, unit, "slack")
    energy, _ = build_qubo_custom(vms, servers, 0, 0, 0, 0, None, unit, "slack")
    n, m = len(vms), len(servers)
    base = dict.fromkeys(energy.variables, 0)
    values = {}
    for index in range(2 ** m):
        y = bits(index, m)
        best = math.inf
        for xs in product((0, 1), repeat=n * m):
            sample = dict(base)
            sample.update({y_var(h): int(y[h]) for h in range(m)})
            sample.update({x_var(v.vm_id, h): xs[i * m + h] for i, v in enumerate(vms) for h in range(m)})
            best = min(best, math.fsum(independent_terms(sample, vms, servers, info, energy).values()))
        values[index] = best
    return dict(vms=vms, servers=servers, unit=unit, data=data, V=values, m=m)


def test_cost_is_lower_bound_and_check_is_nonvacuous(tiny):
    data, m = tiny["data"], tiny["m"]
    passed = [aggregate_ok(bits(i, m), data) for i in range(2 ** m)]
    assert any(passed) and not all(passed)          # both branches of C are exercised
    for index, v in tiny["V"].items():
        assert stage1_cost(bits(index, m), data) <= v + TOL, (index, v)
    feasible_c = [stage1_cost(bits(i, m), data) for i in range(2 ** m) if passed[i]]
    assert max(feasible_c) <= data["U"] < data["M"] == data["lam"]


def test_aggregate_checks_are_necessary(tiny):
    vms, servers, m = tiny["vms"], tiny["servers"], tiny["m"]
    found = 0
    for hosts in product(range(m), repeat=len(vms)):
        assignment = {v.vm_id: h for v, h in zip(vms, hosts)}
        if not validate_allocation(vms, assignment, servers, quantized_unit=tiny["unit"])["valid"]:
            continue
        found += 1
        used = set(hosts)
        for index in range(2 ** m):
            y = bits(index, m)
            if all(y[h] for h in used):           # y switches on every used host
                assert aggregate_ok(y, tiny["data"]), (hosts, index)
    assert found > 0


def test_pruning_is_exact(tiny):
    data, m, values = tiny["data"], tiny["m"], tiny["V"]
    h_min = min(values.values())
    optimal = [i for i, v in values.items() if v <= h_min + TOL]
    for index in optimal:
        assert stage1_cost(bits(index, m), data) <= h_min + TOL
    assert data["M"] > h_min                         # failed-check states are always pruned


def test_all_costs_index_convention(tiny):
    data, m = tiny["data"], tiny["m"]
    costs = all_costs(data)
    assert costs.shape == (2 ** m,)
    for index in range(2 ** m):
        assert costs[index] == stage1_cost(bits(index, m), data)
    assert list(bits(0b101, 3)) == [True, False, True]


def test_factor_must_exceed_one(tiny):
    with pytest.raises(ValueError):
        stage1_data(tiny["vms"], tiny["servers"], tiny["unit"], 1.0)
