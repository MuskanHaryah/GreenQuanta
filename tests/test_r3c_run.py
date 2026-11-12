"""Fast R3c adapter, fixed-y and mechanical PASS checks; no sampling/writes."""
from itertools import product
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics import common, r3c_build
from diagnostics.d0_formulation import tiny_problem
from diagnostics.r3c_run import VARIANTS, metrics, pass_rule, solve_fixed, solver_context, violation_counts
from phase1.qubo_builder import x_var, y_var
from phase2.fingerprint import qubo_fingerprint


@pytest.fixture
def tiny():
    return tiny_problem()


def test_custom_matches_common_on_tiny(monkeypatch, tiny):
    vms, servers, _, _, _ = tiny
    monkeypatch.setattr(common.config, "SCENARIOS", {"tiny": {"trace": "unused.csv", "servers": "auto"}})
    monkeypatch.setattr(common, "load_trace", lambda _: None)
    monkeypatch.setattr(common, "build_vms", lambda *args: (vms, None))
    monkeypatch.setattr(common, "auto_mix", lambda _: {})
    monkeypatch.setattr(common, "build_servers", lambda _: servers)
    monkeypatch.setattr(common, "prepare", lambda *args: (vms, None, None, None))
    original = common.build_problem("tiny")[2]
    custom = r3c_build.build_problem_custom("tiny", 0.05, 0.05, 0.05, 0.05, 0.02, 0.0025)[2]
    assert dict(original.linear) == dict(custom.linear)
    assert dict(original.quadratic) == dict(custom.quadratic)
    assert original.offset == custom.offset
    assert qubo_fingerprint(original) == qubo_fingerprint(custom)


@pytest.mark.parametrize("term", range(4), ids=["assign", "cpu", "ram", "link"])
def test_per_term_lambda_changes_hash(tiny, term):
    vms, servers, _, _, _ = tiny
    weights = [0.05] * 4
    original, _ = r3c_build.build_qubo_custom(vms, servers, *weights, 0.02, 0.0025)
    weights[term] = 0.5
    changed, info = r3c_build.build_qubo_custom(vms, servers, *weights, 0.02, 0.0025)
    assert qubo_fingerprint(original)["sha256"] != qubo_fingerprint(changed)["sha256"]
    assert list(info["lambdas"].values()) == weights


@pytest.mark.parametrize("unit,weights,active", [
    (0.0025, [0.05] * 4, [0, 1]),
    (0.001, [0.5, 0.05, 0.05, 0.05], [0]),
])
def test_fixed_y_equals_bruteforce(tiny, unit, weights, active):
    vms, servers, _, _, _ = tiny
    variant = dict(id="test", unit=unit, lambdas=weights)
    bqm, _ = r3c_build.build_qubo_custom(vms, servers, *weights, 0.02, unit)
    fixed = {y_var(h): int(h in active) for h in range(len(servers))}
    free = [name for name in bqm.variables if name not in fixed]
    minimum = min(bqm.energy({**fixed, **dict(zip(free, bits))})
                  for bits in product((0, 1), repeat=len(free)))
    result = solve_fixed(vms, servers, variant, active, time_limit=20)
    assert result["status"] == "OPTIMAL"
    assert result["bqm_H"] == pytest.approx(minimum, abs=1e-9, rel=0)
    assert all(result["sample"][k] == v for k, v in fixed.items())
    assert result["independent_H"] == pytest.approx(bqm.energy(result["sample"]), abs=1e-9, rel=0)


@pytest.mark.parametrize("valid,d1,d2,unit,expected", [
    (True, 9., 0.1, 0.0025, True), (True, 0., 0.10001, 0.0025, False),
    (False, 0., 0., 0.0025, False), (True, -0.1, 8., 0.001, True),
    (True, 0.10001, 0., 0.0005, False), (True, None, None, 0.0025, False),
])
def test_pass_rule(valid, d1, d2, unit, expected):
    assert pass_rule(valid, d1, d2, unit) is expected


def test_linear_joules_use_actual_on_bits(tiny):
    vms, servers, _, _, _ = tiny
    energy, info = r3c_build.build_qubo_custom(vms, servers, 0, 0, 0, 0, 0.02, 0.0025)
    sample = dict.fromkeys(energy.variables, 0)
    sample[y_var(0)] = 1
    for vm in vms:
        sample[x_var(vm.vm_id, 1)] = 1
    actual = metrics(sample, vms, servers)
    assert not actual["GLOBAL_VALID"]
    assert actual["linear_joules"] == pytest.approx(energy.energy(sample) * info["e_ref_joule"], abs=1e-7, rel=0)


def test_violation_counts_include_duplicated_load(tiny):
    """Every VM on every host: the shared validator sees no single placement, hence no load (CHANGES #3);
    the breakdown must still flag the resources whose total exceeds the smallest limit (tiny_problem ensures one)."""
    vms, servers, bqm, _, _ = tiny
    sample = dict.fromkeys(bqm.variables, 0)
    for h in range(len(servers)):
        sample[y_var(h)] = 1
        for vm in vms:
            sample[x_var(vm.vm_id, h)] = 1
    check = metrics(sample, vms, servers)
    assert not any(v["type"].endswith("overload") for v in check["validation"]["allocation"]["violations"])
    expected = {r for r in ("cpu", "mem") if sum(getattr(vm, r + "_size") for vm in vms)
                > min(getattr(s, r + "_limit") for s in servers) + 1e-9}
    assert expected
    flags = violation_counts(check, check["GLOBAL_VALID"], True)
    assert {r for r, key in (("cpu", "cpu_over"), ("mem", "ram_over")) if flags[key]} == expected
    assert flags["duplicated"] and not flags["unplaced"] and not flags["off_host"]


def test_solver_bindings_do_not_mutate_shared_config(tiny):
    before = common.config.UNIT_NCU, common.config.UNIT_NMU
    solve, _ = solver_context(VARIANTS[-1])
    assert solve.__globals__["config"].UNIT_NCU == 0.0005
    assert (common.config.UNIT_NCU, common.config.UNIT_NMU) == before
