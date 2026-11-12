"""Step 2.2: reduced Stage-2 QUBO Q_y vs brute force on the tiny instance (4 VMs, servers B, D, D).

V(y) (full slack QUBO, lambda = 1.1*U, fixed y) comes from the Step 1.2 fixture `tiny` (reused, not copied).
W(y) = min of Q_y over all x on on-hosts, with the slack bits at their minimiser (exact_qubo.with_slack,
Step 0 Lemma); every Q_y energy is also checked against the independent term sum (d0_formulation).
"""
from itertools import product
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.d0_formulation import independent_terms
from diagnostics.exact_qubo import bqm_agrees, with_slack
from diagnostics.r3c_build import build_qubo_custom
from diagnostics.s0_exact import reference_u
from diagnostics.s1_stage1 import bits, stage1_cost
from diagnostics.s2_stage2 import coefficient_profile, load_factor, stage2_bqm
from phase1.qubo_builder import x_var, y_var
from phase2.validation import validate_allocation
from test_s1_stage1 import FACTOR, TOL, tiny  # noqa: F401  (module fixture reused)


def caps_loads(info, vms):
    """Rule order of exact_qubo.with_slack: cpu rules of hosts 0..m-1, then ram rules."""
    m = len(info["k_cpu"])
    cpu = [info["cpu_units"][v.vm_id] for v in vms]
    mem = [info["mem_units"][v.vm_id] for v in vms]
    return info["k_cpu"] + info["k_mem"], [cpu] * m + [mem] * m


@pytest.fixture(scope="module")
def reduced(tiny):
    """Per y: Q_y at lambda_y, W(y), its argmin placement, and the feasible-placement minimum E*(y)."""
    vms, servers, unit, m = tiny["vms"], tiny["servers"], tiny["unit"], tiny["m"]
    out = {}
    for index in range(1, 2 ** m):                       # y = 0 has no on host (stage2_bqm refuses it)
        y = bits(index, m)
        q, qinfo = stage2_bqm(vms, servers, y, unit, FACTOR)
        caps, loads = caps_loads(qinfo["info"], vms)
        on = qinfo["on"]
        best, best_hosts, feasible_e = math.inf, None, math.inf
        for choice in product((0, 1), repeat=len(vms) * len(on)):
            sample = {y_var(h): int(y[h]) for h in range(m)}
            sample.update({x_var(v.vm_id, h): 0 for v in vms for h in range(m)})
            sample.update({x_var(v.vm_id, h): choice[i * len(on) + j]
                           for i, v in enumerate(vms) for j, h in enumerate(on)})
            full = with_slack(sample, vms, caps, loads, m)
            assert all(full[k] == val for k, val in qinfo["fixed"].items())
            h_val = math.fsum(independent_terms(full, vms, servers, qinfo["info"], qinfo["energy"]).values())
            q_val = float(q.energy({k: full[k] for k in q.variables}))
            assert bqm_agrees(q, {k: full[k] for k in q.variables}, h_val, q_val), (index, h_val, q_val)
            if h_val < best:
                best = h_val
                best_hosts = {v.vm_id: [h for h in on if full[x_var(v.vm_id, h)]] for v in vms}
            hosts = {v.vm_id: [h for h in on if full[x_var(v.vm_id, h)]] for v in vms}
            if all(len(hs) == 1 for hs in hosts.values()):
                assignment = {k: hs[0] for k, hs in hosts.items()}
                if validate_allocation(vms, assignment, servers, quantized_unit=unit)["valid"]:
                    feasible_e = min(feasible_e, float(qinfo["energy"].energy(full)))
        out[index] = dict(q=q, qinfo=qinfo, W=best, argmin=best_hosts, E_star=feasible_e)
    return out


def test_q_y_has_no_off_host_or_y_variables(tiny, reduced):
    m = tiny["m"]
    for index, r in reduced.items():
        y = bits(index, m)
        names = set(r["q"].variables)
        assert not any(n.startswith("y_") for n in names)
        for h in range(m):
            if not y[h]:
                assert not any(n.endswith(f"_{h}") and n.startswith("x_") for n in names)
                assert not any(n.startswith((f"sc_{h}_", f"sm_{h}_")) for n in names)
        assert r["qinfo"]["num_variables"] == len(names) < len(r["qinfo"]["full"].variables)


def test_u_y_bounds(tiny, reduced):
    vms, servers, data = tiny["vms"], tiny["servers"], tiny["data"]
    energy = reduced[2 ** tiny["m"] - 1]["qinfo"]["energy"]
    assert reference_u(energy, vms, servers, on=range(tiny["m"])) == reference_u(energy, vms, servers) == data["U"]
    for r in reduced.values():
        assert r["qinfo"]["U_y"] <= data["U"] + TOL
        assert r["qinfo"]["lam"] == pytest.approx(FACTOR * r["qinfo"]["U_y"])


def test_theorem_2_2_minimiser_is_feasible_and_optimal(reduced):
    feasible = [r for r in reduced.values() if math.isfinite(r["E_star"])]
    assert feasible and len(feasible) < len(reduced)   # both cases occur on the tiny instance
    for r in feasible:
        assert all(len(hs) == 1 for hs in r["argmin"].values())
        assert abs(r["W"] - r["E_star"]) <= TOL, (r["W"], r["E_star"])


def test_restriction_equals_full_fixed_y_minimum(tiny, reduced):
    """W_{lambda_y}(y) = V_lambda(y) whenever a feasible placement exists (Prop. 1 + Theorem 2.2)."""
    for index, r in reduced.items():
        if math.isfinite(r["E_star"]):
            assert abs(r["W"] - tiny["V"][index]) <= TOL, (index, r["W"], tiny["V"][index])


def test_infeasible_y_costs_at_least_lambda_y(reduced):
    """Prop. 4: no feasible placement on ON(y) => W(y) >= lambda_y."""
    for r in reduced.values():
        if not math.isfinite(r["E_star"]):
            assert r["W"] >= r["qinfo"]["lam"] - TOL


def test_heuristic_pruning_bound(tiny, reduced):
    """Prop. 3: every quantized-feasible placement on ON(y) has E >= C(y)."""
    for index, r in reduced.items():
        if math.isfinite(r["E_star"]):
            assert stage1_cost(bits(index, tiny["m"]), tiny["data"]) <= r["E_star"] + TOL


def test_profile_and_load_factor(tiny, reduced):
    r = reduced[2 ** tiny["m"] - 1]
    qinfo = r["qinfo"]
    prof = coefficient_profile(qinfo["full"], qinfo["energy"], qinfo["fixed"])
    assert prof["variables"] == qinfo["num_variables"]
    assert prof["penalty_to_energy"] > 1 and prof["min_energy_coef"] > 0
    # Fixing fewer variables never lowers the variable count.
    assert coefficient_profile(qinfo["full"], qinfo["energy"], {})["variables"] >= prof["variables"]
    lf = load_factor(qinfo["info"], qinfo["on"])
    assert set(lf) == {"cpu", "ram"} and all(v > 0 for v in lf.values())


def test_rejects_bad_input(tiny):
    vms, servers, unit = tiny["vms"], tiny["servers"], tiny["unit"]
    with pytest.raises(ValueError):
        stage2_bqm(vms, servers, [0] * tiny["m"], unit)
    with pytest.raises(ValueError):
        stage2_bqm(vms, servers, [1] * tiny["m"], unit, factor=1.0)
    full, _ = build_qubo_custom(vms, servers, 1, 1, 1, 1, None, unit, "slack")
    assert len(full.variables) == len(stage2_bqm(vms, servers, [1] * tiny["m"], unit)[1]["full"].variables)
