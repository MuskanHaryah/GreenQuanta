"""Step 0.6: annealer-relevant scale of the slack QUBO per unit (no sampling, no exact QUBO solve).

Per size: R1 real MILP once; per unit: R2 quantized MILP, F(u), lambda = 1.1*F(u), BQM size and
coefficient range. Appends rows to docs/reports/step0/0.6_scale.csv and prints them.
Reuses phase2.exact_solver.solve_exact (R1), exact_quantized.solve_exact_quantized (R2),
r3c_build.build_qubo_custom, phase2.validation.validate_allocation.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.common import build_problem
from diagnostics.d0_formulation import assignment_sample
from diagnostics.exact_quantized import solve_exact_quantized
from diagnostics.r3c_build import build_qubo_custom
from phase2 import config
from phase2.exact_solver import linear_cost, solve_exact
from phase2.validation import validate_allocation

RUN_CONFIG = {"sizes": ["small_v2", "medium_v2", "large_v2"], "units": [0.0025, 0.005, 0.01],
              "lambda_factor": 1.1, "time_limit": 600}
OUT = config.ROOT / "docs/reports/step0/0.6_scale.csv"
FIELDS = ["size", "unit", "vms", "servers", "x_vars", "y_vars", "slack_vars", "num_variables",
          "interactions", "min_abs_coef", "max_abs_coef", "coef_ratio", "min_energy_coef",
          "max_over_min_energy", "F", "lambda", "lambda_0025_rule_still_gt_F",
          "r1_joules", "r1_status", "r2_joules", "r2_status", "r2_mip_gap", "quantization_loss_pct",
          "r2_real_valid", "r2_quant_valid", "max_vms_on_a_host", "L3_condition_holds"]


def run(size):
    vms, servers, _, _ = build_problem(size)
    r1_assignment, r1_info = solve_exact(vms, servers, RUN_CONFIG["time_limit"])
    r1 = None if r1_assignment is None else linear_cost(r1_assignment, vms, servers)
    rows, lam_0025 = [], None
    for unit in RUN_CONFIG["units"]:
        energy, _ = build_qubo_custom(vms, servers, 0, 0, 0, 0, None, unit, "slack")
        assignment, info = solve_exact_quantized(vms, servers, RUN_CONFIG["time_limit"], unit=unit)
        row = {"size": size, "unit": unit, "vms": len(vms), "servers": len(servers),
               "r1_joules": r1, "r1_status": r1_info.get("status"),
               "r2_status": info["status"], "r2_mip_gap": info["mip_gap"]}
        if assignment is not None:
            f = float(energy.energy(assignment_sample(assignment, energy)))
            lam = RUN_CONFIG["lambda_factor"] * f
            if unit == 0.0025:
                lam_0025 = lam
            bqm, binfo = build_qubo_custom(vms, servers, *[lam] * 4, None, unit, "slack")
            low, high = binfo["bias_range"]
            min_e = min(abs(b) for b in energy.linear.values() if b != 0)
            r2 = linear_cost(assignment, vms, servers)
            per_host = [list(assignment.values()).count(h) for h in range(len(servers))]
            row.update(x_vars=binfo["assignment_variables"], y_vars=binfo["server_variables"],
                       slack_vars=binfo["slack_variables"], num_variables=binfo["num_variables"],
                       interactions=binfo["num_interactions"], min_abs_coef=low, max_abs_coef=high,
                       coef_ratio=high / low, min_energy_coef=min_e, max_over_min_energy=high / min_e,
                       F=f, **{"lambda": lam},
                       lambda_0025_rule_still_gt_F=None if lam_0025 is None else lam_0025 > f,
                       r2_joules=r2, quantization_loss_pct=None if r1 is None else 100 * (r2 - r1) / r1,
                       r2_real_valid=validate_allocation(vms, assignment, servers)["valid"],
                       r2_quant_valid=validate_allocation(vms, assignment, servers,
                                                          quantized_unit=unit)["valid"],
                       max_vms_on_a_host=max(per_host),
                       L3_condition_holds=(max(per_host) + 1) * unit <= 1)
        print("SCALE_JSON=" + json.dumps(row, sort_keys=True, default=str), flush=True)
        rows.append(row)
    new = not OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)
    print(f"APPENDED {len(rows)} rows to {OUT.relative_to(config.ROOT).as_posix()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=RUN_CONFIG["sizes"], required=True)
    args = parser.parse_args()
    print("RUN_CONFIG=" + json.dumps(RUN_CONFIG, sort_keys=True), flush=True)
    run(args.size)


if __name__ == "__main__":
    main()
