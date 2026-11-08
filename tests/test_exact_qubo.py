"""Exact unbalanced-QUBO checks; no runners, annealing, or result writes.

Run with pytest -s to show exhaustive minima, mode results, and aggregate
fixed-state errors. The 1,224 fixed-state solves intentionally include invalid
allocations: this solver minimizes H, not a hard-constrained placement model.
"""
from itertools import product
import math
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.common import CENTER, ENCODING, LAMBDA_CONFIG
from diagnostics.d0_formulation import independent_terms, tiny_problem
from diagnostics.exact_qubo import solve_exact_qubo
from phase1.qubo_builder import x_var, y_var
from phase2 import config, qubo_builder_v2 as v2


TOLERANCE = 1e-9
RANDOM_STATES = 200
SETTINGS = (("L0.05", 0.02), ("L0.2", 0.07), ("L1.0", 0.0))
MODES = (False, True)
BREAKDOWN_KEYS = {"energy", "assign", "cpu", "ram", "link", "tie", "offset"}


def assert_close(actual, expected, context):
    error = abs(actual - expected)
    assert math.isfinite(error) and error <= TOLERANCE, (
        context, actual, expected, error,
    )
    return error


@pytest.fixture(scope="module")
def tiny():
    vms, servers, bqm, _, _ = tiny_problem()
    assert len(vms) == 4
    assert len(servers) == 3
    assert len(bqm.variables) == 15
    assert set(bqm.variables) == (
        {x_var(vm.vm_id, h) for vm in vms for h in range(len(servers))}
        | {y_var(h) for h in range(len(servers))}
    )
    return vms, servers


@pytest.fixture(scope="module", params=SETTINGS,
                ids=[f"{name}-c{center:g}" for name, center in SETTINGS])
def problem(request, tiny):
    vms, servers = tiny
    lambda_config, center = request.param
    bqm, info = v2.build_qubo(
        vms, servers, lambda_config=lambda_config, encoding=ENCODING, center=center,
    )
    # Only the energy part is taken from the builder. D0 independently computes
    # all penalties, including their constants, for the selected setting.
    energy, _ = v2.build_qubo(
        vms, servers, lambda_config="L0.0", encoding=ENCODING, center=center,
    )
    assert len(bqm.variables) == 15
    assert info["slack_variables"] == 0
    variables = tuple(bqm.variables)
    minimum = math.inf
    bqm_minimum = math.inf
    maximum_error = 0.0
    count = 0
    for bits in product((0, 1), repeat=len(variables)):
        sample = dict(zip(variables, bits))
        true_h = math.fsum(independent_terms(
            sample, vms, servers, info, energy,
        ).values())
        bqm_h = float(bqm.energy(sample))
        maximum_error = max(maximum_error, assert_close(
            bqm_h, true_h, (lambda_config, center, "exhaustive", sample),
        ))
        minimum = min(minimum, true_h)
        bqm_minimum = min(bqm_minimum, bqm_h)
        count += 1
    assert count == 2**15
    assert_close(bqm_minimum, minimum, "exhaustive minima")
    print(f"exact_qubo {lambda_config} center={center:g}: states={count} "
          f"exhaustive_min={minimum:.17g} "
          f"max_independent_bqm_error={maximum_error:.3e}")
    return {
        "vms": vms, "servers": servers, "bqm": bqm, "info": info,
        "energy": energy, "variables": variables, "minimum": minimum,
        "lambda_config": lambda_config, "center": center,
    }


def solve(problem, full_cuts, fixed_sample=None):
    kwargs = {"full_cuts": full_cuts, "fixed_sample": fixed_sample}
    setting = (problem["lambda_config"], problem["center"])
    if setting == SETTINGS[0]:
        assert setting == (LAMBDA_CONFIG, CENTER)
        # Exercise the documented default lambda, center, time limit and cap.
        return solve_exact_qubo(problem["vms"], problem["servers"], **kwargs)
    return solve_exact_qubo(
        problem["vms"], problem["servers"], lambda_config=setting[0],
        center=setting[1], **kwargs,
    )


def check_result(problem, result, full_cuts, context):
    vms, servers = problem["vms"], problem["servers"]
    sample = result["sample"]
    assert set(sample) == set(problem["variables"]), context
    assert all(value in (0, 1) for value in sample.values()), context
    placements = {
        vm.vm_id: tuple(h for h in range(len(servers))
                        if sample[x_var(vm.vm_id, h)])
        for vm in vms
    }
    assert result["placements"] == placements, context
    assert result["assignment"] == {
        vm_id: hosts[0] for vm_id, hosts in placements.items() if len(hosts) == 1
    }, context
    assert isinstance(result["hosts"], set), context
    assert result["hosts"] == {
        h for h in range(len(servers)) if sample[y_var(h)]
    }, context
    assert BREAKDOWN_KEYS <= result["breakdown"].keys(), context
    assert all(math.isfinite(result["breakdown"][key])
               for key in BREAKDOWN_KEYS), context

    true_h = math.fsum(independent_terms(
        sample, vms, servers, problem["info"], problem["energy"],
    ).values())
    bqm_h = float(problem["bqm"].energy(sample))
    errors = {
        "formula_bqm": assert_close(bqm_h, true_h, context),
        "reported_true": assert_close(result["incumbent_true_H"], true_h, context),
        "independent": assert_close(result["independent_H"], true_h, context),
        "bqm": assert_close(result["bqm_H"], bqm_h, context),
        "milp": assert_close(result["milp_objective"], true_h, context),
    }
    assert result["status"] == "OPTIMAL", context
    assert result["proven_optimal"], context
    assert result["converged"], context
    assert_close(result["incumbent_true_H"], result["milp_objective"], context)
    assert_close(result["lower_bound"], true_h, context)
    assert 0 <= result["mip_gap"] <= TOLERANCE, context
    for key in ("max_violation", "max_weighted_violation"):
        assert 0 <= result[key] <= TOLERANCE, (context, key, result[key])
    assert math.isfinite(result["time_s"]) and result["time_s"] >= 0, context
    assert 1 <= result["iterations"] <= 50, context
    assert len(result["history"]) == result["iterations"], context
    assert all(isinstance(record, dict) for record in result["history"]), context
    if full_cuts:
        assert result["iterations"] == 1, context
    return true_h, errors


def test_exhaustive_minimum_matches_full_and_cutting_plane(problem):
    results = {}
    for full_cuts in MODES:
        mode = "full" if full_cuts else "cutting"
        result = solve(problem, full_cuts)
        true_h, _ = check_result(problem, result, full_cuts, mode)
        # Compare energies, not placements: multiple minimizers are allowed.
        assert_close(true_h, problem["minimum"], mode)
        results[mode] = true_h
        print(f"exact_qubo {problem['lambda_config']} center={problem['center']:g} "
              f"mode={mode} H={true_h:.17g} "
              f"iterations={result['iterations']} status={result['status']} "
              f"violation={result['max_violation']:.3e} "
              f"weighted_violation={result['max_weighted_violation']:.3e}")
    assert_close(results["full"], results["cutting"], "mode minima")


@pytest.fixture(scope="module")
def random_bits():
    rng = np.random.default_rng(config.SEED)
    return rng.integers(0, 2, size=(RANDOM_STATES, 15))


def explicit_invalid_states(problem):
    vms, servers = problem["vms"], problem["servers"]
    empty = dict.fromkeys(problem["variables"], 0)
    duplicate = dict(empty)
    for h in (0, 1):
        duplicate[x_var(vms[0].vm_id, h)] = 1
        duplicate[y_var(h)] = 1
    # Select by actual demand instead of assuming which catalog host overloads.
    overloaded_host = next(
        h for h, server in enumerate(servers)
        if any(sum(getattr(vm, f"{resource}_size") for vm in vms)
               > getattr(server, f"{resource}_limit")
               for resource in ("cpu", "mem"))
    )
    overloaded = dict(empty)
    overloaded[y_var(overloaded_host)] = 1
    off_host = dict(empty)
    for vm in vms:
        overloaded[x_var(vm.vm_id, overloaded_host)] = 1
        off_host[x_var(vm.vm_id, 0)] = 1
    return [("empty", empty), ("duplicate", duplicate),
            ("overloaded", overloaded), ("off_host", off_host)]


@pytest.mark.parametrize("full_cuts", MODES, ids=["cutting", "full"])
def test_fixed_bounds_match_exact_h_for_random_and_invalid_states(
        problem, random_bits, full_cuts):
    states = explicit_invalid_states(problem)
    states.extend(
        (f"random-{index}", dict(zip(problem["variables"], map(int, bits))))
        for index, bits in enumerate(random_bits)
    )
    assert len(states) == RANDOM_STATES + 4
    maximum_errors = {}
    mode = "full" if full_cuts else "cutting"
    for label, fixed_sample in states:
        original = dict(fixed_sample)
        context = (problem["lambda_config"], problem["center"], mode, label, original)
        result = solve(problem, full_cuts, fixed_sample=fixed_sample)
        assert fixed_sample == original, (context, "fixed_sample mutated")
        assert result["sample"] == original, context
        _, errors = check_result(problem, result, full_cuts, context)
        for key, error in errors.items():
            maximum_errors[key] = max(maximum_errors.get(key, 0.0), error)
    summary = " ".join(f"max_{key}_error={value:.3e}"
                       for key, value in sorted(maximum_errors.items()))
    print(f"exact_qubo {problem['lambda_config']} center={problem['center']:g} "
          f"mode={mode} fixed_random={RANDOM_STATES} explicit_invalid=4 {summary}")


@pytest.fixture(scope="module")
def limit_problem(tiny):
    """Default tiny problem, without parametrizing limit paths over settings."""
    vms, servers = tiny
    bqm, info = v2.build_qubo(
        vms, servers, lambda_config=LAMBDA_CONFIG, encoding=ENCODING, center=CENTER,
    )
    energy, _ = v2.build_qubo(
        vms, servers, lambda_config="L0.0", encoding=ENCODING, center=CENTER,
    )
    return {
        "vms": vms, "servers": servers, "bqm": bqm, "info": info,
        "energy": energy, "variables": tuple(bqm.variables),
    }


def check_unproven_result(problem, result, context):
    """Check state scoring without treating the last MILP primal as a bound."""
    assert result["status"] == "UNPROVEN", context
    assert not result["proven_optimal"], context
    sample = result["sample"]
    assert set(sample) == set(problem["variables"]), context
    assert all(value in (0, 1) for value in sample.values()), context
    true_h = math.fsum(independent_terms(
        sample, problem["vms"], problem["servers"], problem["info"],
        problem["energy"],
    ).values())
    assert_close(problem["bqm"].energy(sample), true_h, context)
    for key in ("incumbent_true_H", "independent_H", "bqm_H"):
        assert_close(result[key], true_h, (context, key))
    assert BREAKDOWN_KEYS <= result["breakdown"].keys(), context
    assert_close(math.fsum(result["breakdown"].values()), true_h, context)
    assert math.isfinite(result["lower_bound"]), context
    assert result["lower_bound"] <= true_h + TOLERANCE, context
    assert len(result["history"]) == result["iterations"], context
    assert math.isfinite(result["time_s"]) and result["time_s"] >= 0, context
    print(f"exact_qubo limit={context} status={result['status']} "
          f"iterations={result['iterations']} true_H={true_h:.17g} "
          f"lower_bound={result['lower_bound']:.17g} "
          f"last_milp_objective={result['milp_objective']}")
    return true_h


def test_one_iteration_duplicate_state_is_unproven(limit_problem):
    fixed_sample = dict(explicit_invalid_states(limit_problem))["duplicate"]
    original = dict(fixed_sample)
    result = solve_exact_qubo(
        limit_problem["vms"], limit_problem["servers"],
        max_iters=1, full_cuts=False, fixed_sample=fixed_sample,
    )
    true_h = check_unproven_result(limit_problem, result, "max_iters_duplicate")
    assert fixed_sample == original
    assert result["sample"] == original
    assert result["iterations"] == 1
    assert not result["converged"]
    assert result["max_violation"] > TOLERANCE
    assert result["max_weighted_violation"] > TOLERANCE
    assert result["lower_bound"] < true_h
    assert result["milp_objective"] < true_h
    assert true_h - result["milp_objective"] > TOLERANCE


def test_time_exhausted_before_milp_has_finite_bound(limit_problem, monkeypatch):
    from types import SimpleNamespace

    import diagnostics.exact_qubo as solver

    ticks = iter((0.0, 1_000_000.0))
    monkeypatch.setattr(
        solver, "time",
        SimpleNamespace(perf_counter=lambda: next(ticks, 1_000_000.0)),
    )

    def forbidden_milp(*args, **kwargs):
        pytest.fail("MILP must not be called after the time budget is exhausted")

    monkeypatch.setattr(solver, "milp", forbidden_milp)
    result = solver.solve_exact_qubo(
        limit_problem["vms"], limit_problem["servers"], time_limit=0.01,
    )
    check_unproven_result(limit_problem, result, "time_before_milp")
    assert result["iterations"] == 0
    assert result["history"] == []
    assert result["milp_objective"] is None
    assert not result["converged"]
    assert result["sample"] == dict.fromkeys(limit_problem["variables"], 0)


def test_interrupted_exact_worse_candidate_retains_earlier_incumbent(
        limit_problem, monkeypatch):
    from scipy.optimize import Bounds, milp as real_milp

    import diagnostics.exact_qubo as solver

    nb = len(limit_problem["variables"])
    calls = []

    def interrupted_worse_milp(cost, *, bounds, **kwargs):
        # Restrict only this real MILP call, not the solver's original problem.
        # Its conditional optimum is a candidate, never a global certificate.
        lower = np.array(bounds.lb, copy=True)
        upper = np.array(bounds.ub, copy=True)
        assert np.all(lower[:nb] == 0)
        assert np.all(upper[:nb] == 1)
        lower[:nb] = upper[:nb] = 1
        result = real_milp(cost, bounds=Bounds(lower, upper), **kwargs)
        assert result.status == 0, result.message
        assert result.x is not None and np.all(result.x[:nb] == 1)
        calls.append(float(result.fun))
        result.status = 1
        result.message = "Time limit reached (simulated interrupted candidate)"
        # A dual bound from the all-one restriction is not an unrestricted bound.
        result.mip_dual_bound = None
        result.mip_gap = None
        return result

    monkeypatch.setattr(solver, "milp", interrupted_worse_milp)
    result = solver.solve_exact_qubo(
        limit_problem["vms"], limit_problem["servers"], full_cuts=True,
    )
    true_h = check_unproven_result(limit_problem, result, "worse_exact_candidate")
    assert len(calls) == 1
    assert result["iterations"] == 1
    assert result["milp_status"] == 1
    assert result["sample"] == dict.fromkeys(limit_problem["variables"], 0)
    assert_close(true_h, 0.2, "initial all-zero H")
    all_one = dict.fromkeys(limit_problem["variables"], 1)
    candidate_h = math.fsum(independent_terms(
        all_one, limit_problem["vms"], limit_problem["servers"],
        limit_problem["info"], limit_problem["energy"],
    ).values())
    assert_close(result["milp_objective"], candidate_h, "all-one candidate H")
    assert candidate_h > true_h + TOLERANCE
    assert 0 <= result["max_violation"] <= TOLERANCE
    assert 0 <= result["max_weighted_violation"] <= TOLERANCE
    record = result["history"][0]
    assert_close(record["candidate_true_H"], candidate_h, "candidate history")
    assert_close(record["incumbent_true_H"], true_h, "retained history")
    # Check the retained analytical bound against the unrestricted tiny optimum,
    # not against the artificial all-one conditional optimum.
    unrestricted_minimum = min(
        limit_problem["bqm"].energy(dict(zip(limit_problem["variables"], bits)))
        for bits in product((0, 1), repeat=nb)
    )
    assert result["lower_bound"] <= unrestricted_minimum + TOLERANCE


def test_milp_time_limit_without_candidate_keeps_true_scored_state(
        limit_problem, monkeypatch):
    from types import SimpleNamespace

    import diagnostics.exact_qubo as solver

    calls = []

    def no_candidate_milp(*args, **kwargs):
        calls.append(True)
        return SimpleNamespace(
            status=1, x=None, fun=None, mip_dual_bound=None, mip_gap=None,
            message="Time limit reached without an incumbent (simulated)",
        )

    monkeypatch.setattr(solver, "milp", no_candidate_milp)
    result = solver.solve_exact_qubo(
        limit_problem["vms"], limit_problem["servers"],
    )
    check_unproven_result(limit_problem, result, "milp_without_candidate")
    assert len(calls) == 1
    assert result["iterations"] == 1
    assert result["milp_status"] == 1
    assert result["milp_objective"] is None
    assert not result["converged"]
    assert result["sample"] == dict.fromkeys(limit_problem["variables"], 0)
