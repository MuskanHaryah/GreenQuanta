"""Local R3 contract checks; no sampling, runners, or result-file writes.

Run once diagnostics.r3_run exists:
    .venv-gq/Scripts/python.exe -B -m pytest -p no:cacheprovider tests/test_r3_run.py
"""
from dataclasses import replace
from itertools import product
import math
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.common import CENTER, LAMBDA_CONFIG
from diagnostics.d0_formulation import independent_terms, state_validation, tiny_problem
from diagnostics.r3_run import mechanical_label, solve_valid_qubo
from phase1.qubo_builder import x_var, y_var
from phase2.qubo_builder_v2 import build_qubo


TOLERANCE = 1e-9
SAFETY_FACTOR = 1 - 1e-6
RESULT_KEYS = {
    "sample", "assignment", "hosts", "bqm_H", "independent_H", "breakdown",
    "lower_bound", "proven_optimal", "status", "time_s", "iterations", "validation",
}


def within_margin(sample, vms, servers):
    return all(
        math.fsum(getattr(vm, f"{resource}_size") * sample[x_var(vm.vm_id, h)]
                  for vm in vms)
        <= getattr(server, f"{resource}_limit") * SAFETY_FACTOR
        for h, server in enumerate(servers)
        for resource in ("cpu", "mem")
    )


def enumerate_valid_states(vms, servers, bqm):
    """Enumerate singleton placements and every optional unused-host ON bit."""
    states = []
    for placement in product(range(len(servers)), repeat=len(vms)):
        used = set(placement)
        unused = sorted(set(range(len(servers))) - used)
        for on_bits in product((0, 1), repeat=len(unused)):
            sample = dict.fromkeys(bqm.variables, 0)
            for vm, host in zip(vms, placement):
                sample[x_var(vm.vm_id, host)] = 1
            for host in used:
                sample[y_var(host)] = 1
            for host, bit in zip(unused, on_bits):
                sample[y_var(host)] = bit
            if (state_validation(sample, vms, servers)["valid"]
                    and within_margin(sample, vms, servers)):
                states.append(sample)
    return states


def check_incumbent(result, vms, servers, bqm, info):
    assert RESULT_KEYS <= result.keys()
    sample = result["sample"]
    assert sample is not None
    assert set(sample) == set(bqm.variables)
    assert all(bit in (0, 1) for bit in sample.values())
    assignment = {}
    for vm in vms:
        hosts = [h for h in range(len(servers)) if sample[x_var(vm.vm_id, h)]]
        assert len(hosts) == 1
        assignment[vm.vm_id] = hosts[0]
    assert result["assignment"] == assignment
    assert set(result["hosts"]) == {
        h for h in range(len(servers)) if sample[y_var(h)]
    }
    assert all(sample[x_var(vm.vm_id, h)] <= sample[y_var(h)]
               for vm in vms for h in range(len(servers)))
    assert state_validation(sample, vms, servers)["valid"]
    assert within_margin(sample, vms, servers)
    assert result["validation"]["valid"]

    energy, _ = build_qubo(vms, servers, lambda_config="L0",
                           encoding="unbalanced", center=CENTER)
    true_h = math.fsum(independent_terms(sample, vms, servers, info, energy).values())
    for actual in (float(bqm.energy(sample)), result["bqm_H"],
                   result["independent_H"], math.fsum(result["breakdown"].values())):
        assert actual == pytest.approx(true_h, rel=0, abs=TOLERANCE)
    assert math.isfinite(result["lower_bound"])
    assert result["lower_bound"] <= true_h + TOLERANCE
    assert math.isfinite(result["time_s"]) and result["time_s"] >= 0
    return true_h


@pytest.fixture(scope="module")
def tiny():
    vms, servers, bqm, info, _ = tiny_problem()
    states = enumerate_valid_states(vms, servers, bqm)
    assert states, "tiny_problem must have a margin-safe complete placement"
    # An unused server may remain ON: excluding those states changes the oracle.
    assert any(sum(sample[y_var(h)] for h in range(len(servers)))
               > len({h for h in range(len(servers))
                      if any(sample[x_var(vm.vm_id, h)] for vm in vms)})
               for sample in states)
    return vms, servers, bqm, info, min(float(bqm.energy(s)) for s in states)


def test_constrained_tiny_minimum_is_proven_within_30_second_budget(tiny):
    vms, servers, bqm, info, minimum = tiny
    result = solve_valid_qubo(vms, servers, time_limit=30, max_iters=50)
    true_h = check_incumbent(result, vms, servers, bqm, info)
    assert result["proven_optimal"] is True
    assert result["status"] == "OPTIMAL"
    assert 1 <= result["iterations"] <= 50
    assert true_h == pytest.approx(minimum, rel=0, abs=TOLERANCE)
    assert result["lower_bound"] == pytest.approx(minimum, rel=0, abs=TOLERANCE)


@pytest.mark.parametrize("resource", ["cpu", "mem"])
def test_margin_excludes_near_limit_host_even_when_shared_validation_accepts(tiny, resource):
    source_vms, source_servers, _, _, _ = tiny
    first = source_servers[0]
    second = replace(first, server_id=1, cpu_capacity=2 * first.cpu_capacity,
                     mem_capacity=2 * first.mem_capacity)
    servers = [first, second]
    sizes = {f"{r}_size": getattr(first, f"{r}_limit") * 0.1
             for r in ("cpu", "mem")}
    sizes[f"{resource}_size"] = getattr(first, f"{resource}_limit") * (1 - 0.5e-6)
    vms = [replace(source_vms[0], **sizes)]
    bqm, info = build_qubo(vms, servers, lambda_config=LAMBDA_CONFIG,
                           encoding="unbalanced", center=CENTER)
    forbidden = dict.fromkeys(bqm.variables, 0)
    forbidden[x_var(vms[0].vm_id, 0)] = forbidden[y_var(0)] = 1
    assert state_validation(forbidden, vms, servers)["valid"]
    assert not within_margin(forbidden, vms, servers)

    result = solve_valid_qubo(vms, servers, time_limit=30, max_iters=50)
    true_h = check_incumbent(result, vms, servers, bqm, info)
    assert result["assignment"] == {vms[0].vm_id: 1}
    assert result["proven_optimal"] is True
    states = enumerate_valid_states(vms, servers, bqm)
    assert true_h == pytest.approx(min(float(bqm.energy(s)) for s in states),
                                 rel=0, abs=TOLERANCE)
    off_sample = dict(result["sample"])
    off_sample[y_var(1)] = 0
    check = state_validation(off_sample, vms, servers)
    assert not check["valid"]
    assert check["off_server_links"] == [(vms[0].vm_id, 1)]


@pytest.mark.parametrize("resource", ["cpu", "mem"])
def test_no_margin_safe_placement_does_not_return_invalid_fallback(tiny, resource):
    source_vms, source_servers, _, _, _ = tiny
    servers = [source_servers[0]]
    sizes = {f"{r}_size": getattr(servers[0], f"{r}_limit") * 0.1
             for r in ("cpu", "mem")}
    sizes[f"{resource}_size"] = getattr(servers[0], f"{resource}_limit")
    vms = [replace(source_vms[0], **sizes)]
    bqm, _ = build_qubo(vms, servers, lambda_config=LAMBDA_CONFIG,
                        encoding="unbalanced", center=CENTER)
    assert enumerate_valid_states(vms, servers, bqm) == []
    result = solve_valid_qubo(vms, servers, time_limit=30, max_iters=50)
    assert RESULT_KEYS <= result.keys()
    assert result["sample"] is None
    assert result["bqm_H"] is None
    label, _ = mechanical_label(result, False, 1000.0, 1000.0)
    assert label == ("INVALID" if result["proven_optimal"] else "UNPROVEN")


def test_one_iteration_preserves_honest_certificate_and_scoring(tiny):
    vms, servers, bqm, info, minimum = tiny
    result = solve_valid_qubo(vms, servers, time_limit=30, max_iters=1)
    assert RESULT_KEYS <= result.keys()
    assert 0 <= result["iterations"] <= 1
    if result["sample"] is None:
        assert result["bqm_H"] is None
        assert not result["proven_optimal"]
    else:
        true_h = check_incumbent(result, vms, servers, bqm, info)
        assert true_h >= minimum - TOLERANCE
        assert result["lower_bound"] <= minimum + TOLERANCE
        if result["proven_optimal"]:
            assert result["status"] == "OPTIMAL"
            assert true_h == pytest.approx(minimum, rel=0, abs=TOLERANCE)
            assert result["lower_bound"] == pytest.approx(minimum, rel=0, abs=TOLERANCE)
    if not result["proven_optimal"]:
        assert result["status"] == "UNPROVEN"
        assert mechanical_label(result, True, 1000.0, 1000.0)[0] == "UNPROVEN"


@pytest.mark.parametrize("real_valid", [False, True])
@pytest.mark.parametrize("cost", [990.0, 1000.0, 1010.0])
def test_unproven_label_takes_precedence_over_validity_and_cost(real_valid, cost):
    result = {"proven_optimal": False, "status": "UNPROVEN"}
    assert mechanical_label(result, real_valid, cost, 1000.0)[0] == "UNPROVEN"


@pytest.mark.parametrize("cost", [990.0, 1000.0, 1010.0])
def test_invalid_label_takes_precedence_over_cost_for_proven_result(cost):
    result = {"proven_optimal": True, "status": "OPTIMAL"}
    assert mechanical_label(result, False, cost, 1000.0)[0] == "INVALID"


@pytest.mark.parametrize("cost,expected_pct,expected_label", [
    (1000.0, 0.0, "MATCH"),
    (1000.5, 0.05, "MATCH"),
    (999.5, -0.05, "MATCH"),
    (1001.0, 0.1, "MATCH"),
    (999.0, -0.1, "MATCH"),
    (1001.001, 0.1001, "DIFFERENT"),
    (998.999, -0.1001, "DIFFERENT"),
    (1010.0, 1.0, "DIFFERENT"),
    (990.0, -1.0, "DIFFERENT"),
])
def test_proven_valid_label_uses_absolute_signed_percentage(cost, expected_pct, expected_label):
    result = {"proven_optimal": True, "status": "OPTIMAL"}
    label, pct = mechanical_label(result, True, cost, 1000.0)
    assert label == expected_label
    assert pct == pytest.approx(expected_pct, rel=0, abs=1e-12)
