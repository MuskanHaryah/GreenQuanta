"""Step 1.7: end-to-end hybrid result, and the fixed-y SA probe for Step 2. Stdout + experiment-log rows.

--part hybrid: Stage 1 = quantum proposals of y, with no knowledge of y*:
  (a) the top-k sampled states of every CVaR-QAOA run at p = qaoa_p (read from the 1.5b raw output);
  (b) the best state of each GAS run, run to its full oracle budget (no stop at y*).
  The aggregate check (classical, O(m)) drops infeasible proposals. Stage 2 = exact fixed-y solve
  (s1_ground_truth.stage2): a STAND-IN for a perfect Stage 2, replaced by SQA in Step 2. Proposals are
  solved in increasing C order and skipped once C(y) > best proven V (exact by 1.2 Prop. 3).
--part probe: dwave SA (r3c_run.sampled_run: 200 reads x 2000 sweeps) on the slack QUBO, y = y* fixed.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.r3c_build import build_qubo_custom
from diagnostics.r3_run import historical_r2
from diagnostics.r3c_run import references, sampled_run
from diagnostics.s1_ground_truth import (RUN_CONFIG as GT_CONFIG, candidates, emit, ground_truth,
                                         log_run, records, stage2)
from diagnostics.s1_quantum import gas
from diagnostics.s1_stage1 import aggregate_ok, bits
from phase2 import config
from phase2.energy_model import evaluate_allocation

RUN_CONFIG = {"sizes": ["small_v2", "medium_v2", "large_v2"],
              "qaoa_raw": "docs/reports/step1/1.5b_qaoa_cvar.txt", "qaoa_objective": "cvar", "qaoa_p": 5,
              "gas_seeds": list(range(20)), "gas_growth": 1.2, "gas_budget_factor": 1.0,
              "probe_seeds": [0, 1, 2], "probe_cap": 900}


def proposals(size, costs):
    qaoa = [r for r in records("QAOA", size, config.ROOT / RUN_CONFIG["qaoa_raw"])
            if r["p"] == RUN_CONFIG["qaoa_p"] and r["objective"] == RUN_CONFIG["qaoa_objective"]]
    if not qaoa:
        raise LookupError(f"no {RUN_CONFIG['qaoa_objective']} QAOA rows at p={RUN_CONFIG['qaoa_p']} for {size}")
    budget = int(RUN_CONFIG["gas_budget_factor"] * len(costs))
    gas_best = [gas(costs, np.random.default_rng(s), budget, RUN_CONFIG["gas_growth"])["best"]
                for s in RUN_CONFIG["gas_seeds"]]
    return sorted({i for r in qaoa for i in r["top_k_sampled"]}), sorted(set(gas_best)), gas_best


def hybrid(size):
    started = time.perf_counter()
    vms, servers, data, costs, _, _ = candidates(size)
    gt = ground_truth(size)
    m = len(servers)
    from_qaoa, from_gas, gas_best = proposals(size, costs)
    pool = sorted(set(from_qaoa) | set(from_gas), key=lambda i: (costs[i], i))
    feasible = [i for i in pool if aggregate_ok(bits(i, m), data)]
    emit("PROPOSALS", size=size, qaoa_top_k=from_qaoa, gas_best=from_gas,
         gas_runs_ending_at_y_star=sum(i == gt["y_star"] for i in gas_best), gas_runs=len(gas_best),
         pool=pool, aggregate_ok=feasible, y_star_proposed=gt["y_star"] in pool)
    best, solved = None, []
    for index in feasible:
        if best is not None and costs[index] > best[0]["V"] + GT_CONFIG["margin"]:
            emit("SKIP", size=size, index=index, C=float(costs[index]), best_V=best[0]["V"],
                 reason="C(y) > best proven V (Prop. 3)")
            continue
        row, result = stage2(vms, servers, data, costs, index)
        emit("STAGE2", size=size, **row)
        solved.append(row)
        if row["proven"] and (best is None or row["V"] < best[0]["V"]):
            best = (row, result)
    row, result = best
    assignment = result["assignment"]
    lam = data["lam"]
    bqm, _ = build_qubo_custom(vms, servers, lam, lam, lam, lam, None, GT_CONFIG["unit"], "slack")
    r1, r2 = references(size, vms, servers, bqm)
    joules = row["linear_joules"]
    out = {"size": size, "chosen": row["index"], "chosen_ids": row["active_ids"],
           "is_y_star": row["index"] == gt["y_star"], "V": row["V"], "V_minus_step0": row["V"] - gt["step0_h_min"],
           "real_valid": row["real_valid"], "quantized_valid": row["quantized_valid"],
           "linear_joules": joules, "r1_joules": r1, "r2_joules": r2,
           "diff_r1_pct": 100 * (joules - r1) / r1, "diff_r2_pct": 100 * (joules - r2) / r2,
           "curve_it_energy_kwh": evaluate_allocation(vms, assignment, servers)["it_energy_kwh"],
           "r2_curve_it_energy_kwh": historical_r2(size)[2],
           "stage2_solves": len(solved), "stage2_s": math.fsum(r["time_s"] for r in solved),
           "wall_s": time.perf_counter() - started}
    emit("HYBRID", **out)
    print(f"TABLE | {size} | {len(pool)} | {len(feasible)} | {len(solved)} | {row['active_ids']} | "
          f"{out['is_y_star']} | {out['real_valid']}/{out['quantized_valid']} | {joules:.10g} | "
          f"{out['diff_r1_pct']:.4f} | {out['diff_r2_pct']:.4f} | {out['curve_it_energy_kwh']:.6g} |", flush=True)
    log_run("s1_hybrid", dict(RUN_CONFIG, size=size), RUN_CONFIG["gas_seeds"][0], size, vms,
            {"summary": out, "stage2": solved}, "PASS" if out["real_valid"] and out["quantized_valid"] else "FAIL")


def probe(size):
    vms, servers, data, _, _, _ = candidates(size)
    gt = ground_truth(size)
    m = len(servers)
    active = [h for h in range(m) if bits(gt["y_star"], m)[h]]
    variant = {"id": "S1-probe", "lambdas": [data["lam"]] * 4, "unit": GT_CONFIG["unit"], "encoding": "slack"}
    for seed in RUN_CONFIG["probe_seeds"]:
        result = sampled_run(size, variant, active, seed, gt["V_star"], cap=RUN_CONFIG["probe_cap"])
        row = {"size": size, "seed": seed, "y_star_ids": gt["y_star_ids"], **result}
        emit("PROBE", **row)
        print(f"TABLE | {size} | {seed} | {result['status']} | {result.get('valid_read_pct')} | "
              f"{result.get('gap_pct')} | {result.get('pure_and_optimal_count')} | {result['wall_s']:.1f} |", flush=True)
        log_run("s1_fixed_y_sa_probe", dict(RUN_CONFIG, size=size, seed=seed), seed, size, vms, row,
                "probe-recorded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["hybrid", "probe"], required=True)
    parser.add_argument("--size", choices=RUN_CONFIG["sizes"], required=True)
    args = parser.parse_args()
    print("RUN_CONFIG=" + json.dumps(RUN_CONFIG, sort_keys=True), flush=True)
    (hybrid if args.part == "hybrid" else probe)(args.size)


if __name__ == "__main__":
    main()
