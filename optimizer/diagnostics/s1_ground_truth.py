"""Step 1.3: ground truth over y (classical enumeration; used only to score the quantum routines). Stdout only.

Per size: C(y) for all 2^m y (s1_stage1.all_costs); keep y with C(y) <= Step 0 H_min + margin (exact by
1.2 Prop. 3); exact fixed-y minimum V(y) for each survivor (r3c_run.solve_fixed: slack, lambda = 1.1*U,
y fixed through fixed_sample); y* = argmin V. --dry-run prints the survivor count without solving.
Later sub-steps read the GT_JSON line from docs/reports/step1/1.3_ground_truth.txt (ground_truth()).
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.common import build_problem
from diagnostics.r3c_run import metrics, solve_fixed
from diagnostics.s1_stage1 import aggregate_ok, all_costs, bits, stage1_data
from phase2 import config
from phase2.validation import validate_allocation

RAW = config.ROOT / "docs/reports/step1/1.3_ground_truth.txt"
RUN_CONFIG = {
    "sizes": ["small_v2", "medium_v2", "large_v2"], "unit": 0.0025, "factor": 1.1,
    "margin": 1e-9, "tie_tol": 1e-9, "time_limit": 600,
    # Step 0 global minimum (exact-sum H) and its linear joules: docs/reports/step1/1.1_exact.txt
    # (independent_H, linear_joules at lambda = 1.1*U), equal to Step 0 0.5_exact.txt.
    "step0_h_min": {"small_v2": 0.38505827990484337, "medium_v2": 0.40156535160258183,
                    "large_v2": 0.46360488610200884},
    "step0_joules": {"small_v2": 39951971.03582648, "medium_v2": 44211076.67748815,
                     "large_v2": 77970698.41683578},
}


def emit(kind, **data):
    print(kind + "_JSON=" + json.dumps(data, sort_keys=True, allow_nan=False), flush=True)


def ground_truth(size):
    """The GT_JSON record of `size` from the 1.3 raw output (last one wins)."""
    prefix = "GT_JSON="
    rows = [json.loads(line[len(prefix):]) for line in RAW.read_text(encoding="utf-8").splitlines()
            if line.startswith(prefix)]
    rows = [r for r in rows if r["size"] == size]
    if not rows:
        raise LookupError(f"no 1.3 ground truth for {size}")
    return rows[-1]


def candidates(size):
    vms, servers, _, _ = build_problem(size)
    data = stage1_data(vms, servers, RUN_CONFIG["unit"], RUN_CONFIG["factor"])
    costs = all_costs(data)
    threshold = RUN_CONFIG["step0_h_min"][size] + RUN_CONFIG["margin"]
    keep = sorted((int(i) for i in np.flatnonzero(costs <= threshold)), key=lambda i: (costs[i], i))
    return vms, servers, data, costs, keep, threshold


def run(size, dry_run):
    started = time.perf_counter()
    vms, servers, data, costs, keep, threshold = candidates(size)
    m = len(servers)
    n_ok = sum(aggregate_ok(bits(i, m), data) for i in range(2 ** m))
    emit("CANDIDATES", size=size, m=m, states=2 ** m, aggregate_ok=n_ok, survivors=len(keep),
         threshold=threshold, U=data["U"], lam=data["lam"], enumerate_s=time.perf_counter() - started)
    if dry_run:
        return
    unit = RUN_CONFIG["unit"]
    variant = {"id": "S1-1.1U", "lambdas": [data["lam"]] * 4, "unit": unit, "encoding": "slack"}
    rows = []
    for index in keep:
        active = [h for h in range(m) if bits(index, m)[h]]
        result = solve_fixed(vms, servers, variant, active, RUN_CONFIG["time_limit"])
        check = metrics(result["sample"], vms, servers)
        complete = len(result["assignment"]) == len(vms)
        row = {"index": index, "active_ids": [servers[h].server_id for h in active],
               "C": float(costs[index]), "V": result["independent_H"],
               "lower_bound": result["lower_bound"], "status": result["status"],
               "proven": result["proven_optimal"], "real_valid": check["GLOBAL_VALID"],
               "quantized_valid": complete and validate_allocation(
                   vms, result["assignment"], servers, quantized_unit=unit)["valid"],
               "linear_joules": check["linear_joules"], "time_s": result["time_s"]}
        emit("V", size=size, **row)
        rows.append(row)
    proven = [r for r in rows if r["proven"]]
    best = min(proven, key=lambda r: (r["V"], r["index"]))
    tol = RUN_CONFIG["tie_tol"]
    # An unproven survivor can only be excluded if its lower bound is above the best V.
    unresolved = [r["index"] for r in rows if not r["proven"] and r["lower_bound"] <= best["V"] + tol]
    optimal = [r["index"] for r in proven if r["V"] <= best["V"] + tol]
    argmin_c = int(np.argmin(costs))
    rank_c = 1 + int(np.sum(costs < costs[best["index"]] - tol))
    h0, j0 = RUN_CONFIG["step0_h_min"][size], RUN_CONFIG["step0_joules"][size]
    gate = (abs(best["V"] - h0) <= tol and best["real_valid"] and best["quantized_valid"]
            and best["linear_joules"] is not None and math.isclose(best["linear_joules"], j0, rel_tol=1e-9)
            and not unresolved)
    emit("GT", size=size, m=m, y_star=best["index"], y_star_ids=best["active_ids"], V_star=best["V"],
         C_star=best["C"], step0_h_min=h0, V_minus_step0=best["V"] - h0,
         joules=best["linear_joules"], step0_joules=j0, real_valid=best["real_valid"],
         quantized_valid=best["quantized_valid"], optimal_set=optimal, unresolved=unresolved,
         rank_of_y_star_under_C=rank_c, argmin_C=argmin_c,
         argmin_C_ids=[servers[h].server_id for h in range(m) if bits(argmin_c, m)[h]],
         argmin_C_is_y_star=argmin_c in optimal, survivors=len(keep), aggregate_ok=n_ok,
         solve_s=math.fsum(r["time_s"] for r in rows), wall_s=time.perf_counter() - started,
         GATE=bool(gate))
    print(f"TABLE | {size} | {m} | {2 ** m} | {n_ok} | {len(keep)} | {best['active_ids']} | "
          f"{best['V']:.12g} | {h0:.12g} | {rank_c} | {argmin_c in optimal} | "
          f"{best['linear_joules']:.10g} | {math.fsum(r['time_s'] for r in rows):.1f} | "
          f"{'PASS' if gate else 'FAIL'} |", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=RUN_CONFIG["sizes"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print("RUN_CONFIG=" + json.dumps(RUN_CONFIG, sort_keys=True), flush=True)
    run(args.size, args.dry_run)


if __name__ == "__main__":
    main()
