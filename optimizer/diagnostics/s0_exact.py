"""Step 0.5: check the lambda > F theorem on real sizes (exact slack-QUBO minimum). Stdout only.

Per size: F = E(R2 quantized-MILP assignment, y = used hosts); then the exact global minimum of
the slack QUBO at lambda = factor * F (all four equal) for each factor in RUN_CONFIG.
Step 1.1 (--reference U): lambda = factor * U with the solver-free bound U (reference_u); R2 is
still run, but only to compare against.
Reuses: exact_quantized.solve_exact_quantized (R2), r3c_run.solver_context / certificate /
metrics, phase2.validation.validate_allocation. No sampling, no result writes.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.common import build_problem
from diagnostics.d0_formulation import assignment_sample
from diagnostics.exact_quantized import solve_exact_quantized
from diagnostics.r3c_build import build_qubo_custom
from diagnostics.r3c_run import certificate, metrics, solver_context
from phase1.qubo_builder import x_var, y_var
from phase2 import config
from phase2.exact_solver import linear_cost
from phase2.validation import validate_allocation

RUN_CONFIG = {"sizes": ["small_v2", "medium_v2", "large_v2"], "factors": [1.1, 0.5],
              "u_factors": [1.1], "unit": 0.0025, "time_limit": 600, "max_iters": 50,
              "center": 0.02}


def emit(kind, **data):
    print(kind + "_JSON=" + json.dumps(data, sort_keys=True, allow_nan=False, default=str), flush=True)


def reference_f(vms, servers, unit):
    """R2 quantized MILP, then F = normalized E (tie-break included) of that feasible state."""
    if (config.UNIT_NCU, config.UNIT_NMU) != (unit, unit):
        raise ValueError("solve_exact_quantized uses config units; they must equal RUN_CONFIG unit")
    assignment, info = solve_exact_quantized(vms, servers, RUN_CONFIG["time_limit"])
    if assignment is None:
        return None, info, None
    quant = validate_allocation(vms, assignment, servers, quantized_unit=unit)
    if not quant["valid"]:
        raise AssertionError(("R2 assignment not quantized-feasible", quant))
    energy, _ = build_qubo_custom(vms, servers, 0, 0, 0, 0, None, unit, "slack")
    f = float(energy.energy(assignment_sample(assignment, energy)))
    return assignment, info, f


def reference_u(energy, vms, servers, on=None):
    """Step 1.1 solver-free bound U = sum_h c_y[h] + sum_v max_h c_x[v,h] (proof: 1.1_proof.md).

    energy is the lambda=0 slack BQM; U >= E(x, y) for every state with one host per VM.
    on (Step 2.2): restrict both sums to these host positions, giving U_y for the on-set of y
    (proof: docs/reports/step2/2.2_proof.md); None = all hosts (Step 1 behaviour)."""
    if energy.offset != 0 or min(energy.linear.values()) < 0:
        raise ValueError("U needs a zero offset and nonnegative energy coefficients (A4)")
    hosts = range(len(servers)) if on is None else sorted(on)
    if not hosts:
        raise ValueError("U needs at least one host")
    return math.fsum(energy.linear[y_var(h)] for h in hosts) + math.fsum(
        max(energy.linear[x_var(v.vm_id, h)] for h in hosts) for v in vms)


def coefficient_ratio(info):
    low, high = info["bias_range"]
    return high / low


def run(size, reference="F"):
    unit = RUN_CONFIG["unit"]
    vms, servers, _, _ = build_problem(size)
    emit("START", size=size, vms=len(vms), servers=len(servers), run_config=RUN_CONFIG,
         reference=reference)
    r2_assignment, r2_info, f = reference_f(vms, servers, unit)
    if f is None:
        emit("R2", size=size, info=r2_info, F=None)
        return
    r2_joules = linear_cost(r2_assignment, vms, servers)
    emit("R2", size=size, info=r2_info, F=f, r2_linear_joules=r2_joules,
         r2_hosts=sorted(servers[h].server_id for h in set(r2_assignment.values())))
    energy, _ = build_qubo_custom(vms, servers, 0, 0, 0, 0, None, unit, "slack")
    ref, factors = f, RUN_CONFIG["factors"]
    if reference == "U":
        ref, factors = reference_u(energy, vms, servers), RUN_CONFIG["u_factors"]
        _, info_f = build_qubo_custom(vms, servers, *[1.1 * f] * 4, None, unit, "slack")
        emit("U", size=size, U=ref, F=f, U_over_F=ref / f, U_ge_F=ref >= f,
             bias_range_1_1F=info_f["bias_range"], coef_ratio_1_1F=coefficient_ratio(info_f))
    for factor in factors:
        lam = factor * ref
        variant = {"id": f"S0-{factor:g}{reference}", "lambdas": [lam] * 4, "unit": unit,
                   "encoding": "slack"}
        bqm, info = build_qubo_custom(vms, servers, *variant["lambdas"], None, unit, "slack")
        solve, _ = solver_context(variant)
        result = solve(vms, servers, center=RUN_CONFIG["center"], time_limit=RUN_CONFIG["time_limit"],
                       max_iters=RUN_CONFIG["max_iters"])
        cert = certificate(result, vms, servers, bqm, info, energy)
        data = metrics(result["sample"], vms, servers)
        complete = len(result["assignment"]) == len(vms)
        quant_valid = complete and validate_allocation(vms, result["assignment"], servers,
                                                       quantized_unit=unit)["valid"]
        cost = data["linear_joules"]
        emit("S0", size=size, variant=variant["id"], factor=factor, lam=lam, F=f,
             reference=reference, bias_range=info["bias_range"],
             coef_ratio=coefficient_ratio(info),
             status=result["status"], proven_optimal=result["proven_optimal"],
             H_min=result["bqm_H"], H_min_minus_F=result["bqm_H"] - f,
             lower_bound=result["lower_bound"], mip_gap=result["mip_gap"],
             real_valid=data["GLOBAL_VALID"], quantized_valid=quant_valid,
             n_unplaced=data["n_unplaced"], n_duplicate=data["n_duplicate"],
             max_real_overload_pct=data["max_real_overload_pct"],
             active_hosts=sorted(servers[h].server_id for h in result["hosts"]),
             linear_joules=cost, r2_linear_joules=r2_joules,
             diff_r2_pct=None if cost is None else 100 * (cost - r2_joules) / r2_joules,
             certificate=cert, slack_variables=info["slack_variables"],
             num_variables=info["num_variables"], time_s=result["time_s"])
    emit("DONE", size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=RUN_CONFIG["sizes"], required=True)
    parser.add_argument("--reference", choices=["F", "U"], default="F")
    args = parser.parse_args()
    print("RUN_CONFIG=" + json.dumps(RUN_CONFIG, sort_keys=True), flush=True)
    run(args.size, args.reference)


if __name__ == "__main__":
    main()
