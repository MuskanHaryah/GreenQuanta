"""Step 2.5: SA (dwave-samplers) vs SQA (OpenJij, s2_sqa) on Q_y at y = y*, lambda_y, at equal spin-update budget.

Modes (run in this order; outputs under docs/reports/step2/):
  --dry-run       one SA and one SQA run on small (calibration seed), time per spin update, estimated wall time of
                  the calibration and of the main matrix per size. -> 2.5_dry_run.txt
  --calibrate     SQA grid beta x gamma x trotter on small only, calibration seed (not a main seed), calib_reads
                  reads at budget level 1. Choice rule: max valid %, then optimal reads, then smallest best valid
                  gap, then lowest mean raw H (the only signal if no setting gives a valid read), then grid order.
                  Every run and its spin updates are counted (calibration cost). -> 2.5_calibration.txt
  --size S        main runs: SA and SQA (calibrated settings, unchanged on every size), budget levels B and 4B,
                  main seeds, RUN_CONFIG reads. -> 2.5_sa_vs_sqa.txt
Equal budget: per read, SA sweeps = base_sa_sweeps x level and SQA sweeps x trotter = the same number, so
reads x sweeps x trotter x variables match (SA trotter = 1). B at level 1 equals the 2.3 SA budget.
beta and gamma are in units of max|coef| (s2_sqa rescales). Grid range: on Q_y(lambda_y) the smallest coefficient is
~3e-5 of the largest and the energy coefficients ~1e-10..1e-6 of it (2.2_table.md), so beta spans 1e1..1e6.
Scoring, validation, violation breakdown: shared r3c_run.sample_worker path. Wall times are indicative only.
"""
import argparse
from itertools import product
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics import r3c_run
from diagnostics.s1_ground_truth import emit, log_run, records
from diagnostics.s2_sa_reduced import RUN_CONFIG as S23, setup, summarize, variant
from phase2 import config

CALIB = config.ROOT / "docs/reports/step2/2.5_calibration.txt"
RUN_CONFIG = {"sizes": S23["sizes"], "seeds": S23["seeds"], "reads": S23["reads"], "base_sa_sweeps": S23["sweeps"],
              "budget_levels": [1, 4], "calib_size": "small_v2", "calib_seed": 100, "calib_reads": 50,
              "grid": {"beta": [1e1, 1e2, 1e3, 1e4, 1e5, 1e6], "gamma": [0.3, 1.0, 3.0], "trotter": [4, 8]},
              "dry_sqa": {"beta": 1e3, "gamma": 1.0, "trotter": 4}, "cap_s": 1800}


def spec(method, reads, level, params=None):
    sweeps = RUN_CONFIG["base_sa_sweeps"] * level
    if method == "sa":
        return {"method": "sa", "reads": reads, "sweeps": sweeps}
    if sweeps % params["trotter"]:
        raise ValueError("SA sweeps must be a multiple of trotter for an equal budget")
    return {"method": "sqa", "reads": reads, "sweeps": sweeps // params["trotter"], **params}


def sample(size, ctx, seed, s):
    _, _, _, active, exact_h, qinfo = ctx
    return r3c_run.sampled_run(size, variant("lambda_y", qinfo["lam"]), active, seed, exact_h,
                               cap=RUN_CONFIG["cap_s"], reduce=True, sampler=s)


def dry_run():
    size = RUN_CONFIG["calib_size"]
    ctx = setup(size)
    rate = {}
    for method in ("sa", "sqa"):
        result = sample(size, ctx, RUN_CONFIG["calib_seed"],
                        spec(method, RUN_CONFIG["calib_reads"], 1, RUN_CONFIG["dry_sqa"]))
        emit("DRY", size=size, **result)
        if result["status"] != "DONE":
            raise RuntimeError(("dry run failed", method, result))
        rate[method] = result["sampler_wall_s"] / result["spin_updates"]
    n = {s: ctx[5]["num_variables"] if s == size else setup(s)[5]["num_variables"] for s in RUN_CONFIG["sizes"]}
    grid = list(product(*RUN_CONFIG["grid"].values()))
    calib_updates = len(grid) * RUN_CONFIG["calib_reads"] * RUN_CONFIG["base_sa_sweeps"] * n[size]
    main = {s: sum(RUN_CONFIG["budget_levels"]) * RUN_CONFIG["reads"] * RUN_CONFIG["base_sa_sweeps"] * n[s]
            * len(RUN_CONFIG["seeds"]) for s in RUN_CONFIG["sizes"]}
    emit("ESTIMATE", s_per_spin_update=rate, q_y_variables=n, calibration_runs=len(grid),
         calibration_spin_updates=calib_updates, calibration_est_s=calib_updates * rate["sqa"],
         main_spin_updates_per_method=main,
         main_est_s={s: u * (rate["sa"] + rate["sqa"]) for s, u in main.items()},
         note="per-update rates measured on small; larger Q_y may run at a different rate")


def calibrate():
    size = RUN_CONFIG["calib_size"]
    ctx = setup(size)
    vms, exact_h = ctx[0], ctx[4]
    rows = []
    for order, (beta, gamma, trotter) in enumerate(product(*RUN_CONFIG["grid"].values())):
        params = {"beta": beta, "gamma": gamma, "trotter": trotter}
        result = sample(size, ctx, RUN_CONFIG["calib_seed"], spec("sqa", RUN_CONFIG["calib_reads"], 1, params))
        row = {"size": size, "order": order, **params, **result}
        emit("CALIB", **row)
        rows.append(row)
    done = [r for r in rows if r["status"] == "DONE"]
    if not done:
        raise RuntimeError("no calibration run finished")
    best = max(done, key=lambda r: (r["valid_read_pct"], r["pure_and_optimal_count"],
                                    -(r["gap_pct"] if r["gap_pct"] is not None else math.inf),
                                    -r["mean_raw_H"], -r["order"]))
    choice = {"size": size, "beta": best["beta"], "gamma": best["gamma"], "trotter": best["trotter"],
              "chosen_order": best["order"], "chosen_valid_pct": best["valid_read_pct"],
              "chosen_mean_raw_H": best["mean_raw_H"], "any_valid": any(r["valid_read_pct"] > 0 for r in done),
              "runs": len(rows), "done": len(done), "statuses": sorted({r["status"] for r in rows}),
              "spin_updates_total": sum(r.get("spin_updates", 0) for r in done),
              "sampler_wall_s_total": math.fsum(r["sampler_wall_s"] for r in done), "exact_h": exact_h}
    emit("CALIB_CHOICE", **choice)
    for r in sorted(done, key=lambda r: r["order"]):
        print(f"CTABLE | {r['beta']:g} | {r['gamma']:g} | {r['trotter']} | {r['valid_read_pct']} | "
              f"{r['pure_and_optimal_count']} | {r['mean_raw_H']:.6g} | {r['min_raw_H']:.6g} | "
              f"unplaced {r['violation_reads']['unplaced']}, duplicated {r['violation_reads']['duplicated']}, "
              f"cpu {r['violation_reads']['cpu_over']}, ram {r['violation_reads']['ram_over']} |", flush=True)
    log_run("s2_sqa_calibration", RUN_CONFIG, RUN_CONFIG["calib_seed"], size, vms,
            {"choice": choice, "rows": rows}, "recorded")


def main_runs(size):
    choice = records("CALIB_CHOICE", RUN_CONFIG["calib_size"], CALIB)[-1]
    params = {k: choice[k] for k in ("beta", "gamma", "trotter")}
    ctx = setup(size)
    vms, index, exact_h = ctx[0], ctx[2], ctx[4]
    for level in RUN_CONFIG["budget_levels"]:
        for method in ("sa", "sqa"):
            s = spec(method, RUN_CONFIG["reads"], level, params)
            rows = []
            for seed in RUN_CONFIG["seeds"]:
                row = {"size": size, "method": method, "level": level, "seed": seed, "y_star": index,
                       **sample(size, ctx, seed, s)}
                emit("RUN", **row)
                rows.append(row)
            summary = {"size": size, "method": method, "level": level, "sampler": s, **summarize(rows, exact_h)}
            emit("SUMMARY", **summary)
            v = summary["violation_pct"]
            print(f"TABLE | {size} | {method} | {level}B | {summary['spin_updates_per_seed']} | "
                  f"{summary['valid_pct_per_seed']} | {summary['p_optimal_read']} | {summary['best_valid_gap_pct']} | "
                  f"{summary['tts99_spin_updates'] or 'not reached'} | "
                  + ", ".join(f"{k} {val:.1f}%" for k, val in v.items()) +
                  f" | mean raw H {summary['mean_raw_H']} | {summary['statuses']} | wall {summary['sampler_wall_s']} |",
                  flush=True)
            log_run("s2_sa_vs_sqa", dict(RUN_CONFIG, size=size, method=method, level=level, sampler=s),
                    RUN_CONFIG["seeds"][0], size, vms, {"summary": summary, "rows": rows}, "recorded")
    emit("DONE", size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument("--size", choices=RUN_CONFIG["sizes"])
    args = parser.parse_args()
    print("RUN_CONFIG=" + json.dumps(RUN_CONFIG, sort_keys=True), flush=True)
    if args.dry_run:
        dry_run()
    elif args.calibrate:
        calibrate()
    else:
        main_runs(args.size)


if __name__ == "__main__":
    main()
