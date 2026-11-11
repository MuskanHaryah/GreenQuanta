"""Step 1.5: QAOA (exact numpy statevector, s1_quantum) on the Stage-1 cost C/M; scored against the
1.3 ground truth y* (classical enumeration, evaluation only). Stdout + one experiment-log row per
(size, run). No wall-clock speedup is claimed; counts are circuit evaluations and shots.
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
from diagnostics.s1_quantum import optimise_qaoa, probabilities, qaoa_state, walsh_coefficients

RUN_CONFIG = {"sizes": ["small_v2", "medium_v2", "large_v2"], "p_max": 5,
              "runs": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],      # one optimiser run = these restart seeds
              "maxiter": 300, "gamma_max": 200.0, "grid": [2000, 16], "rhobeg": 0.1,
              "shots": 1000, "top_k": 5, "success": 0.99, "walsh_rel_tol": 1e-12,
              "cvar_alpha": 0.05}                              # used only with --objective cvar (1.5b)


def shots_for(probability):
    """Shots for RUN_CONFIG['success'] chance of seeing the target at least once."""
    if probability <= 0:
        return None
    if probability >= 1:
        return 1
    return math.ceil(math.log(1 - RUN_CONFIG["success"]) / math.log(1 - probability))


def walsh_summary(costs):
    w = walsh_coefficients(costs)
    nonzero = np.flatnonzero(np.abs(w) > RUN_CONFIG["walsh_rel_tol"] * np.abs(w).max())
    orders = [bin(int(s)).count("1") for s in nonzero]
    return {"walsh_terms": len(nonzero), "walsh_max_order": max(orders),
            "walsh_terms_by_order": {k: orders.count(k) for k in sorted(set(orders))}}


def run(size, objective="mean"):
    alpha = RUN_CONFIG["cvar_alpha"] if objective == "cvar" else None
    vms, servers, data, costs, _, _ = candidates(size)
    gt = ground_truth(size)
    m, states = len(servers), 2 ** len(servers)
    y_star, argmin_c, joule_set = gt["y_star"], int(np.argmin(costs)), gt["joule_optimal"]
    norm = costs / data["M"]                                  # C/M in [0, 1]; M -> 1
    emit("BASELINE", size=size, m=m, states=states, y_star=y_star, argmin_C=argmin_c,
         joule_optimal=joule_set, uniform_P=1 / states, uniform_shots_99=shots_for(1 / states),
         exhaustive_evaluations=states, **walsh_summary(costs))
    for run_index, seeds in enumerate(RUN_CONFIG["runs"]):
        started = time.perf_counter()
        rows = optimise_qaoa(norm, RUN_CONFIG["p_max"], seeds, RUN_CONFIG["maxiter"],
                             RUN_CONFIG["gamma_max"], tuple(RUN_CONFIG["grid"]), RUN_CONFIG["rhobeg"],
                             alpha)
        wall = time.perf_counter() - started
        rng = np.random.default_rng(seeds[0])
        results = []
        for row in rows:
            probs = probabilities(qaoa_state(norm, row["gammas"], row["betas"]))
            counts = np.bincount(rng.choice(states, size=RUN_CONFIG["shots"], p=probs), minlength=states)
            top = [int(i) for i in np.argsort(-counts, kind="stable")[:RUN_CONFIG["top_k"]] if counts[i]]
            p_star, p_joule = float(probs[y_star]), float(probs[joule_set].sum())
            result = {"size": size, "run": run_index, "seeds": seeds, "p": row["p"],
                      "objective": objective, "alpha": alpha, "objective_value": row["expected"],
                      "P_y_star": p_star, "P_argmin_C": float(probs[argmin_c]),
                      "P_joule_optimal": p_joule, "P_y_star_over_uniform": p_star * states,
                      "rank_y_star_by_P": 1 + int(np.sum(probs > p_star)),
                      "expected_C_over_M": float(probs @ norm), "expected_C": float(probs @ norm) * data["M"],
                      "nfev_this_p": row["nfev"], "nfev_cumulative": row["nfev_cumulative"],
                      "shots_99_y_star": shots_for(p_star), "shots_99_joule_optimal": shots_for(p_joule),
                      "y_star_in_top_k": y_star in top, "top_k_sampled": top,
                      "gammas": row["gammas"], "betas": row["betas"]}
            emit("QAOA", **result)
            results.append(result)
            print(f"TABLE | {size} | {run_index} | {row['p']} | {p_star:.4f} | {p_joule:.4f} | "
                  f"{result['rank_y_star_by_P']} | {row['expected']:.6f} | {row['nfev_cumulative']} | "
                  f"{result['shots_99_y_star']} | {result['y_star_in_top_k']} |", flush=True)
        method = "s1_qaoa" if objective == "mean" else "s1_qaoa_cvar"
        log_run(method, dict(RUN_CONFIG, size=size, run=run_index, seeds=seeds, objective=objective),
                seeds[0], size, vms,
                {"rows": results, "wall_s": wall}, "recorded")
    emit("DONE", size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=RUN_CONFIG["sizes"], required=True)
    parser.add_argument("--objective", choices=["mean", "cvar"], default="mean")
    args = parser.parse_args()
    print("RUN_CONFIG=" + json.dumps(RUN_CONFIG, sort_keys=True), flush=True)
    run(args.size, args.objective)


if __name__ == "__main__":
    main()
