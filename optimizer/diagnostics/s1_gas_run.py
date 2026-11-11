"""Step 1.6: Grover Adaptive Search (exact numpy statevector, s1_quantum.gas) on the Stage-1 cost C.
Per (size, seed): oracle calls and measurements until y* (1.3 ground truth, evaluation only) is first
measured, within an oracle-call budget of 2^m. Compared with exhaustive search (2^m evaluations) and a
sqrt(2^m) reference line. Stdout + one experiment-log row per size. No wall-clock speedup is claimed.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.s1_ground_truth import candidates, emit, ground_truth, log_run
from diagnostics.s1_quantum import first_hit, gas

RUN_CONFIG = {"sizes": ["small_v2", "medium_v2", "large_v2"], "seeds": list(range(20)),
              "growth": 1.2, "budget_factor": 1.0}          # oracle-call budget = budget_factor * 2^m


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=RUN_CONFIG["sizes"], required=True)
    args = parser.parse_args()
    print("RUN_CONFIG=" + json.dumps(RUN_CONFIG, sort_keys=True), flush=True)
    run(args.size)


if __name__ == "__main__":
    main()
