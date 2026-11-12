"""Grover Adaptive Search (exact numpy statevector, s1_quantum.gas) on the Stage-1 cost C. Four modes:

first-hit (Step 1.6, default): per (size, seed), oracle calls and measurements until y* (1.3 ground truth,
  evaluation only) is first measured, within an oracle-call budget of budget_factor * 2^m.
budget (Step 2.1): GAS runs to a budget of c * sqrt(2^m) oracle calls with no stop, so it never uses y*.
  Per (size, c): P(returned best = y*), P(returned best is joule-optimal), total cost = oracle calls +
  measurements, and the computed r-restart success 1 - (1 - p)^r. y* is used only to score.
recompute-1.6 (Step 2.1): the 1.6 table with total cost = oracle calls + measurements, from the 1.6 raw file.
choose-c (Step 2.1): c_pipe = smallest c whose P(best = y*) >= c_pipe_min_success on every size, read from the
  budget-mode raw file. The rule is fixed in RUN_CONFIG before the runs; it is a calibration scored with y*.
Note: gas() checks the budget before each Grover round, so the last round may overshoot it by < sqrt(2^m);
the actual oracle calls are reported. No wall-clock speedup is claimed.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.s1_ground_truth import candidates, emit, ground_truth, log_run, records
from diagnostics.s1_quantum import first_hit, gas

RUN_CONFIG = {"sizes": ["small_v2", "medium_v2", "large_v2"], "seeds": list(range(20)),
              "growth": 1.2, "budget_factor": 1.0,          # first-hit: budget = budget_factor * 2^m
              # budget mode: budget = ceil(c * sqrt(2^m)) for each c; no stop at y*
              "sqrt_budget_factors": [1, 2, 4, 8, 16], "sqrt_budget_seeds": list(range(100)),
              "restarts": [1, 2, 4, 8], "c_pipe_min_success": 0.95}


def stats(values):
    return {"median": float(np.median(values)), "mean": float(np.mean(values)), "max": int(max(values))}


def run(size):
    vms, servers, _, costs, _, _ = candidates(size)
    gt = ground_truth(size)
    states = len(costs)
    y_star, argmin_c, joule_set = gt["y_star"], int(np.argmin(costs)), set(gt["joule_optimal"])
    budget = int(RUN_CONFIG["budget_factor"] * states)
    rows = []
    started = time.perf_counter()
    for seed in RUN_CONFIG["seeds"]:
        result = gas(costs, np.random.default_rng(seed), budget, RUN_CONFIG["growth"], stop={y_star})
        hit = {name: first_hit(result["history"], targets) for name, targets in
               (("y_star", {y_star}), ("argmin_C", {argmin_c}), ("joule_optimal", joule_set))}
        row = {"size": size, "seed": seed, "found_y_star": hit["y_star"] is not None,
               **{f"oracle_calls_to_{k}": None if v is None else v[0] for k, v in hit.items()},
               **{f"measurements_to_{k}": None if v is None else v[1] for k, v in hit.items()},
               "oracle_calls_total": result["oracle_calls"], "measurements_total": result["measurements"]}
        emit("GAS", **row)
        rows.append(row)
    calls = [r["oracle_calls_to_y_star"] for r in rows if r["found_y_star"]]
    meas = [r["measurements_to_y_star"] for r in rows if r["found_y_star"]]
    summary = {"size": size, "states": states, "budget": budget, "sqrt_states": math.sqrt(states),
               "seeds": len(rows), "success_rate": len(calls) / len(rows),
               "oracle_calls_median": float(np.median(calls)) if calls else None,
               "oracle_calls_mean": float(np.mean(calls)) if calls else None,
               "oracle_calls_max": max(calls) if calls else None,
               "measurements_median": float(np.median(meas)) if meas else None,
               "median_over_states": float(np.median(calls)) / states if calls else None,
               "median_over_sqrt_states": float(np.median(calls)) / math.sqrt(states) if calls else None,
               "wall_s": time.perf_counter() - started}
    emit("GAS_SUMMARY", **summary)
    print(f"TABLE | {size} | {states} | {math.sqrt(states):.0f} | {summary['success_rate']:.2f} | "
          f"{summary['oracle_calls_median']} | {summary['oracle_calls_mean']} | {summary['oracle_calls_max']} | "
          f"{summary['measurements_median']} |", flush=True)
    log_run("s1_gas", dict(RUN_CONFIG, size=size), RUN_CONFIG["seeds"][0], size, vms,
            {"summary": summary, "rows": rows}, "recorded")
    emit("DONE", size=size)


def run_budget(size):
    """Step 2.1: GAS to a fixed budget c*sqrt(2^m), no stop; the returned best is scored against y*."""
    vms, servers, _, costs, _, _ = candidates(size)
    gt = ground_truth(size)
    states = len(costs)
    y_star, joule_set = gt["y_star"], set(gt["joule_optimal"])
    print("TABLE | size | c | budget | seeds | P(best=y*) | P(best joule-opt) | "
          "calls med/max | meas med | total med/mean/max | " +
          " | ".join(f"r={r} (computed)" for r in RUN_CONFIG["restarts"]) + " |", flush=True)
    for c in RUN_CONFIG["sqrt_budget_factors"]:
        budget = math.ceil(c * math.sqrt(states))
        rows = []
        started = time.perf_counter()
        for seed in RUN_CONFIG["sqrt_budget_seeds"]:
            result = gas(costs, np.random.default_rng(seed), budget, RUN_CONFIG["growth"])
            rows.append({"seed": seed, "best": result["best"], "oracle_calls": result["oracle_calls"],
                         "measurements": result["measurements"],
                         "total_cost": result["oracle_calls"] + result["measurements"]})
        p_star = sum(r["best"] == y_star for r in rows) / len(rows)
        p_joule = sum(r["best"] in joule_set for r in rows) / len(rows)
        summary = {"size": size, "c": c, "states": states, "budget": budget, "seeds": len(rows),
                   "y_star": y_star, "p_best_is_y_star": p_star, "p_best_joule_optimal": p_joule,
                   "oracle_calls": stats([r["oracle_calls"] for r in rows]),
                   "measurements": stats([r["measurements"] for r in rows]),
                   "total_cost": stats([r["total_cost"] for r in rows]),
                   "restart_success_computed": {str(r): 1 - (1 - p_star) ** r for r in RUN_CONFIG["restarts"]},
                   "wall_s": time.perf_counter() - started}
        emit("GAS_BUDGET", size=size, c=c, rows=rows)
        emit("GAS_BUDGET_SUMMARY", **summary)
        t, o = summary["total_cost"], summary["oracle_calls"]
        print(f"TABLE | {size} | {c} | {budget} | {len(rows)} | {p_star:.2f} | {p_joule:.2f} | "
              f"{o['median']:.0f}/{o['max']} | {summary['measurements']['median']:.0f} | "
              f"{t['median']:.0f}/{t['mean']:.1f}/{t['max']} | " +
              " | ".join(f"{v:.4f}" for v in summary["restart_success_computed"].values()) + " |", flush=True)
        log_run("s2_gas_budget", dict(RUN_CONFIG, size=size, c=c), RUN_CONFIG["sqrt_budget_seeds"][0], size,
                vms, {"summary": summary, "rows": rows}, "recorded")
    emit("DONE", size=size)


def recompute_16(raw):
    """Step 2.1: the 1.6 first-hit table with total cost = oracle calls + measurements (no rerun)."""
    print("TABLE | size | seeds | found | oracle med/mean/max | total med/mean/max |", flush=True)
    for size in RUN_CONFIG["sizes"]:
        rows = records("GAS", size, raw)
        hit = [r for r in rows if r["found_y_star"]]
        oracle = stats([r["oracle_calls_to_y_star"] for r in hit])
        total = stats([r["oracle_calls_to_y_star"] + r["measurements_to_y_star"] for r in hit])
        emit("GAS_16_TOTAL", size=size, seeds=len(rows), found=len(hit), oracle_calls=oracle, total_cost=total)
        print(f"TABLE | {size} | {len(rows)} | {len(hit)} | {oracle['median']}/{oracle['mean']}/{oracle['max']} | "
              f"{total['median']}/{total['mean']}/{total['max']} |", flush=True)


def choose_c(raw):
    """Step 2.1: apply the c_pipe rule fixed in RUN_CONFIG to the budget-mode summaries."""
    need = RUN_CONFIG["c_pipe_min_success"]
    by_size = {size: {s["c"]: s["p_best_is_y_star"] for s in records("GAS_BUDGET_SUMMARY", size, raw)}
               for size in RUN_CONFIG["sizes"]}
    missing = [size for size, table in by_size.items() if not table]
    if missing:
        raise LookupError(f"no budget-mode summaries for {missing} in {raw}")
    ok = [c for c in RUN_CONFIG["sqrt_budget_factors"]
          if all(c in table and table[c] >= need for table in by_size.values())]
    emit("C_PIPE", rule=f"smallest c with P(best = y*) >= {need} on every size (scored with y*)",
         c_pipe=min(ok) if ok else None, table=by_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["first-hit", "budget", "recompute-1.6", "choose-c"],
                        default="first-hit")
    parser.add_argument("--size", choices=RUN_CONFIG["sizes"])
    parser.add_argument("--raw", help="raw output file read by recompute-1.6 / choose-c")
    args = parser.parse_args()
    if args.mode in ("first-hit", "budget") and not args.size:
        parser.error("--size is required for this mode")
    if args.mode in ("recompute-1.6", "choose-c") and not args.raw:
        parser.error("--raw is required for this mode")
    print("RUN_CONFIG=" + json.dumps(RUN_CONFIG, sort_keys=True), flush=True)
    {"first-hit": lambda: run(args.size), "budget": lambda: run_budget(args.size),
     "recompute-1.6": lambda: recompute_16(args.raw), "choose-c": lambda: choose_c(args.raw)}[args.mode]()


if __name__ == "__main__":
    main()
