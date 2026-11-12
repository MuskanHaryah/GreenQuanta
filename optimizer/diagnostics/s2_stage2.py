"""Step 2.2: reduced Stage-2 QUBO Q_y for a fixed server on/off vector y (proof: docs/reports/step2/2.2_proof.md).

Q_y = the Step 0 slack QUBO (build_qubo_custom, all four lambdas equal) with these variables fixed:
every y_h to its value, every x_vh with h off to 0, every slack bit of an off host to 0. dimod
fix_variables folds the fixed terms into the offset, so Q_y.energy(z) = H(z + fixed values).
Default weight lambda_y = factor * U_y, U_y = reference_u restricted to the on-set (Theorem 2.2).
No new builder logic: the full BQM comes from build_qubo_custom, U_y from reference_u.
"""
import math

from diagnostics.r3c_build import build_qubo_custom
from diagnostics.s0_exact import reference_u
from phase1.qubo_builder import slack_coefficients, x_var, y_var


def stage2_bqm(vms, servers, y, unit, factor=1.1, lam=None):
    """(Q_y, info). lam=None -> lambda_y = factor * U_y (factor > 1); else this lambda for all four terms."""
    m = len(servers)
    y = [bool(b) for b in y]
    if len(y) != m:
        raise ValueError("y must have one entry per server")
    on = [h for h in range(m) if y[h]]
    off = [h for h in range(m) if not y[h]]
    if not on:
        raise ValueError("y must switch on at least one server")
    energy, _ = build_qubo_custom(vms, servers, 0, 0, 0, 0, None, unit, "slack")
    u_y = reference_u(energy, vms, servers, on=on)
    if lam is None:
        if factor <= 1:
            raise ValueError("factor must be > 1 (Theorem 2.2)")
        lam = factor * u_y
    full, info = build_qubo_custom(vms, servers, lam, lam, lam, lam, None, unit, "slack")
    fixed = {y_var(h): int(y[h]) for h in range(m)}
    fixed.update({x_var(v.vm_id, h): 0 for v in vms for h in off})
    for tag, caps in (("sc", info["k_cpu"]), ("sm", info["k_mem"])):
        fixed.update({f"{tag}_{h}_{k}": 0 for h in off for k in range(len(slack_coefficients(caps[h])))})
    missing = set(fixed) - set(full.variables)
    if missing:
        raise AssertionError(("fixed labels absent from the builder BQM", sorted(missing)[:5]))
    reduced = full.copy()
    reduced.fix_variables(fixed)
    slack = sum(1 for v in reduced.variables if str(v).startswith(("sc_", "sm_")))
    return reduced, {"lam": lam, "U_y": u_y, "on": on, "fixed": fixed, "full": full, "energy": energy,
                     "info": info, "num_variables": len(reduced.variables), "slack_variables": slack,
                     "x_variables": len(reduced.variables) - slack}


def coefficient_profile(bqm, energy, fixed):
    """Coefficient magnitudes of bqm with `fixed` applied (Step 2.2, review finding R3).

    penalty part = bqm - energy (the lambda=0 BQM), both reduced by the same fixings; ratio =
    max |penalty coefficient| / min nonzero |energy coefficient| (offsets excluded)."""
    def reduce(model):
        model = model.copy()
        model.fix_variables({k: v for k, v in fixed.items() if k in model.variables})
        return model

    def magnitudes(model):
        return [abs(b) for b in list(model.linear.values()) + list(model.quadratic.values()) if b != 0]

    whole, penalty, base = reduce(bqm), reduce(bqm - energy), reduce(energy)
    pen, eng, all_ = magnitudes(penalty), magnitudes(base), magnitudes(whole)
    return {"variables": len(whole.variables), "max_abs_coef": max(all_), "min_abs_coef": min(all_),
            "max_penalty_coef": max(pen), "min_energy_coef": min(eng), "max_energy_coef": max(eng),
            "penalty_to_energy": max(pen) / min(eng)}


def load_factor(info, on):
    """Sum of quantized demands / sum of on-host quantized capacities, per resource (packing tightness)."""
    return {resource: math.fsum(info[units].values()) / math.fsum(info[caps][h] for h in on)
            for resource, units, caps in (("cpu", "cpu_units", "k_cpu"), ("ram", "mem_units", "k_mem"))}
