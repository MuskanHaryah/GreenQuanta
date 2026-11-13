"""Step 2.5: SA (dwave-samplers) vs SQA (OpenJij, s2_sqa) on Q_y at y = y*, lambda_y, at equal spin-update budget.

Modes (run in this order; outputs under docs/reports/step2/):
  --dry-run       one SA and one SQA run on small (calibration seed), time per spin update, estimated wall time of
                  the calibration and of the main matrix per size. -> 2.5_dry_run.txt
  --calibrate     SQA grid beta x gamma x trotter on small only, calibration seed (not a main seed), calib_reads
                  reads at budget level 1. Choice rule: max valid %, then optimal reads, then smallest best valid
                  gap, then lowest mean raw H (the only signal if no setting gives a valid read), then grid order.
                  Every run and its spin updates are counted (calibration cost). -> 2.5_calibration.txt
  --size S        main runs: SA and SQA (calibrated settings, unchanged on every size), budget levels B and 4B,
                  main seeds, RUN_CONFIG reads. -> 2.5_sa_vs_sqa.txt (RUN_TABLE lines)
  --lever-l4 S    2.5b lever L4: assignment weight l4_ratios x lambda_y (exact: s2_stage2.stage2_bqm docstring),
                  SA and SQA at budget B, main seeds. -> 2.5b_levers.txt (L4_TABLE lines). Mean raw H includes the
                  larger assignment penalty, so compare valid % / P(optimal) across ratios, not raw H.
Step 2C (constraint-preserving track, s2_constrained; methods csa / csqa; budget unit = move attempts):
  --c-dry-run     one csa and one csqa run (c_dry settings, calibration seed, calib_reads) on every size, timed;
                  estimated wall time of the 2C calibration and of the 2C.4 matrix per size. -> 2C.3_dry_run.txt
  --c-calibrate   small only, calibration seed: c_grid["csa"] and c_grid["csqa"] have the same number of runs at the
                  same budget (equal calibration budget). Same choice rule as --calibrate. -> 2C.3_calibration.txt
  --c-size S      2C.4: csa vs csqa (calibrated settings, unchanged on every size), levels B and 4B, main seeds;
                  asserts no unplaced/duplicated VM. -> 2C.4_compare.txt (C_TABLE lines)
Step 2D (protocol pre-registered in docs/reports/step2/2D.0_protocol.md; csqa with world-line moves):
  --d-freeze      2D.2: csqa at the 2C.4 settings, without and with world-line moves, d_freeze size/seed/reads;
                  per-sweep counters in d_freeze windows; C1 verdict by the 2D.0 rule. -> 2D.2_freeze.txt
  --d-calibrate S 2D.3: d_grid for csa and csqa (16 each), d_calib_seeds pooled, calib_reads, budget B; selection
                  rule select(); saturation flag. Medium first; large only if medium saturates. -> 2D.3_calibration.txt
  --d-size S      2D.4: csa and csqa at the 2D.3 choice + csqa with world-line at the 2C.4 settings, levels B and 4B,
                  main seeds; decide() per level; C1/C2 effects vs the 2C.4 rows. -> 2D.4_eval.txt
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics import r3c_run
from diagnostics.s1_ground_truth import emit, log_run, records
from diagnostics.s2_sa_reduced import RUN_CONFIG as S23, setup, summarize, variant
from phase2 import config

CALIB = config.ROOT / "docs/reports/step2/2.5_calibration.txt"
CALIB_C = config.ROOT / "docs/reports/step2/2C.3_calibration.txt"
RUN_CONFIG = {"sizes": S23["sizes"], "seeds": S23["seeds"], "reads": S23["reads"], "base_sa_sweeps": S23["sweeps"],
              "budget_levels": [1, 4], "calib_size": "small_v2", "calib_seed": 100, "calib_reads": 50,
              "grid": {"beta": [1e1, 1e2, 1e3, 1e4, 1e5, 1e6], "gamma": [0.3, 1.0, 3.0], "trotter": [4, 8]},
              "dry_sqa": {"beta": 1e3, "gamma": 1.0, "trotter": 4}, "cap_s": 1800,
              "l4_ratios": [10.0, 100.0, 1000.0, 10000.0],
              # Step 2C. beta, gamma dimensionless (G_y / lambda_y, s2_constrained docstring). 12 runs per method.
              "c_grid": {"csa": {"beta_min": [0.01, 0.1, 1.0], "beta_max": [1e2, 1e3, 1e4, 1e5]},
                         "csqa": {"beta": [1e2, 1e3, 1e4], "gamma": [1.0, 4.0], "trotter": [4, 8]}},
              "c_dry": {"csa": {"beta_min": 0.1, "beta_max": 1e3}, "csqa": {"beta": 1e3, "gamma": 1.0, "trotter": 4}},
              # Step 2D (2D.0_protocol.md). Grid order = product order of these lists. Ties: |difference| <= d_tie.
              "d_grid": {"csa": {"beta_min": [0.01, 0.1, 1.0, 10.0], "beta_max": [1e3, 1e4, 1e5, 1e6]},
                         "csqa": {"beta": [1e3, 1e4], "gamma": [0.3, 1.0, 3.0, 10.0], "trotter": [8, 16],
                                  "worldline": [True]}},
              "d_calib_seeds": [100, 101], "d_calib_sizes": ["medium_v2", "large_v2"],
              "d_eval_sizes": ["medium_v2", "large_v2"], "d_tie": 1e-9,
              "d_freeze": {"size": "large_v2", "seed": 100, "reads": 50, "windows": 10}}
CALIB_D = config.ROOT / "docs/reports/step2/2D.3_calibration.txt"
COMPARE_C = config.ROOT / "docs/reports/step2/2C.4_compare.txt"


def spec(method, reads, level, params=None):
    """Sampler spec at budget level x B. P = 1 methods (sa, csa) get base sweeps x level; Trotter methods
    (sqa, csqa) get that / trotter, so reads x sweeps x P x units is equal within a track. csqa with world-line
    moves costs 2P per VM per sweep (s2_constrained.move_attempts): sweeps = floor(base x level / 2P) (2D.0)."""
    sweeps = RUN_CONFIG["base_sa_sweeps"] * level
    params = params or {}
    if method in ("sa", "csa"):           # QUBO SA uses dwave defaults: SQA params passed to it are ignored (2.5)
        return {"method": method, "reads": reads, "sweeps": sweeps, **(params if method == "csa" else {})}
    if params.get("worldline"):
        if sweeps < 2 * params["trotter"]:
            raise ValueError("budget below one world-line sweep")
        return {"method": method, "reads": reads, "sweeps": sweeps // (2 * params["trotter"]), **params}
    if sweeps % params["trotter"]:
        raise ValueError("SA sweeps must be a multiple of trotter for an equal budget")
    return {"method": method, "reads": reads, "sweeps": sweeps // params["trotter"], **params}


def sample(size, ctx, seed, s, ratio=1.0):
    """One sampled run on Q_y at lambda_y; ratio > 1: assignment weight ratio * lambda_y (lever L4)."""
    _, _, _, active, exact_h, qinfo = ctx
    lam = qinfo["lam"]
    return r3c_run.sampled_run(size, variant("lambda_y", lam, None if ratio == 1 else ratio * lam), active, seed,
                               exact_h, cap=RUN_CONFIG["cap_s"], reduce=True, sampler=s)


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


def calibrate(method, grid, log_method):
    """Grid of `method` on small at budget level 1, calibration seed. Choice rule: max valid %, then optimal reads,
    then smallest best valid gap, then lowest mean raw H, then grid order. Every run is counted (cost)."""
    size = RUN_CONFIG["calib_size"]
    ctx = setup(size)
    vms, exact_h = ctx[0], ctx[4]
    rows = []
    for order, combo in enumerate(product(*grid.values())):
        params = dict(zip(grid, combo))
        result = sample(size, ctx, RUN_CONFIG["calib_seed"], spec(method, RUN_CONFIG["calib_reads"], 1, params))
        row = {"size": size, "method": method, "order": order, "params": params, **result}
        emit("CALIB", **row)
        rows.append(row)
    done = [r for r in rows if r["status"] == "DONE"]
    if not done:
        raise RuntimeError("no calibration run finished")
    best = max(done, key=lambda r: (r["valid_read_pct"], r["pure_and_optimal_count"],
                                    -(r["gap_pct"] if r["gap_pct"] is not None else math.inf),
                                    -r["mean_raw_H"], -r["order"]))
    choice = {"size": size, "method": method, "params": best["params"], **best["params"],
              "chosen_order": best["order"], "chosen_valid_pct": best["valid_read_pct"],
              "chosen_mean_raw_H": best["mean_raw_H"], "any_valid": any(r["valid_read_pct"] > 0 for r in done),
              "runs": len(rows), "done": len(done), "statuses": sorted({r["status"] for r in rows}),
              "budget_total": sum(r.get("spin_updates", 0) for r in done),
              "budget_unit": done[0].get("budget_unit", "spin_updates"),
              "sampler_wall_s_total": math.fsum(r["sampler_wall_s"] for r in done), "exact_h": exact_h}
    emit("CALIB_CHOICE", **choice)
    for r in sorted(done, key=lambda r: r["order"]):
        v = r["violation_reads"]
        print(f"CTABLE | {method} | {json.dumps(r['params'], sort_keys=True)} | {r['valid_read_pct']} | "
              f"{r['pure_and_optimal_count']} | {r['gap_pct']} | {r['mean_raw_H']:.6g} | {r['min_raw_H']:.6g} | "
              f"unplaced {v['unplaced']}, duplicated {v['duplicated']}, cpu {v['cpu_over']}, ram {v['ram_over']} | "
              f"{r['sampler_wall_s']:.1f} s |", flush=True)
    log_run(log_method, dict(RUN_CONFIG, method=method), RUN_CONFIG["calib_seed"], size, vms,
            {"choice": choice, "rows": rows}, "recorded")


def calibrated(method):
    """Chosen params of `method` (last CALIB_CHOICE). 2.5 rows (sqa) predate the 'method'/'params' keys."""
    path = CALIB if method == "sqa" else CALIB_C
    rows = [r for r in records("CALIB_CHOICE", RUN_CONFIG["calib_size"], path) if r.get("method", "sqa") == method]
    if not rows:
        raise LookupError(f"no calibration choice for {method} in {path.name}")
    return rows[-1].get("params") or {k: rows[-1][k] for k in ("beta", "gamma", "trotter")}


def cell(size, ctx, method, level, params, ratio, kind, log_method):
    """One (method, budget level, assignment ratio) cell over the main seeds: rows, summary, table line, log row."""
    vms, index, exact_h = ctx[0], ctx[2], ctx[4]
    s = spec(method, RUN_CONFIG["reads"], level, params)
    labels = {"size": size, "method": method, "level": level, "assign_ratio": ratio}
    rows = []
    for seed in RUN_CONFIG["seeds"]:
        row = {**labels, "seed": seed, "y_star": index, **sample(size, ctx, seed, s, ratio)}
        emit(kind, **row)
        rows.append(row)
    summary = {**labels, "sampler": s, "budget_unit": rows[0].get("budget_unit", "spin_updates"),
               **summarize(rows, exact_h)}
    emit(kind + "_SUMMARY", **summary)
    v = summary["violation_pct"]
    print(f"{kind}_TABLE | {size} | {method} | {level}B | r {ratio:g} | {summary['spin_updates_per_seed']} "
          f"{summary['budget_unit']} | "
          f"{summary['valid_pct_per_seed']} | {summary['p_optimal_read']} | {summary['best_valid_gap_pct']} | "
          f"{summary['tts99_spin_updates'] or 'not reached'} | "
          + ", ".join(f"{k} {val:.1f}%" for k, val in v.items()) +
          f" | mean raw H {summary['mean_raw_H']} | {summary['statuses']} | wall {summary['sampler_wall_s']} |",
          flush=True)
    log_run(log_method, dict(RUN_CONFIG, **labels, sampler=s), RUN_CONFIG["seeds"][0], size, vms,
            {"summary": summary, "rows": rows}, "recorded")
    return summary, rows


def main_runs(size):
    params, ctx = calibrated("sqa"), setup(size)
    for level in RUN_CONFIG["budget_levels"]:
        for method in ("sa", "sqa"):
            cell(size, ctx, method, level, params, 1.0, "RUN", "s2_sa_vs_sqa")
    emit("DONE", size=size)


def lever_l4(size):
    """2.5b lever L4 (user-approved, CHANGES #4): assignment weight ratio * lambda_y, other weights lambda_y,
    SA and SQA (2.5 calibrated settings, unchanged) at budget level 1, main seeds."""
    params, ctx = calibrated("sqa"), setup(size)
    for ratio in RUN_CONFIG["l4_ratios"]:
        for method in ("sa", "sqa"):
            cell(size, ctx, method, 1, params, ratio, "L4", "s2_lever_l4")
    emit("DONE", size=size, lever="L4")


def c_dry_run():
    """2C.3 dry run: one csa and one csqa run per size, timed; estimates for the 2C calibration and 2C.4 matrix."""
    rate, n_vms = {}, {}
    for size in RUN_CONFIG["sizes"]:
        ctx = setup(size)
        n_vms[size] = len(ctx[0])
        for method in ("csa", "csqa"):
            result = sample(size, ctx, RUN_CONFIG["calib_seed"],
                            spec(method, RUN_CONFIG["calib_reads"], 1, RUN_CONFIG["c_dry"][method]))
            emit("C_DRY", size=size, method=method, **result)
            if result["status"] != "DONE":
                raise RuntimeError(("2C dry run failed", size, method, result.get("error")))
            rate[f"{size}/{method}"] = result["sampler_wall_s"] / result["spin_updates"]
    per_read = {s: RUN_CONFIG["base_sa_sweeps"] * n for s, n in n_vms.items()}     # move attempts per read at B
    small = RUN_CONFIG["calib_size"]
    calib_s = sum(len(list(product(*RUN_CONFIG["c_grid"][m].values()))) * RUN_CONFIG["calib_reads"] * per_read[small]
                  * rate[f"{small}/{m}"] for m in ("csa", "csqa"))
    main = {s: sum(RUN_CONFIG["budget_levels"]) * RUN_CONFIG["reads"] * per_read[s] * len(RUN_CONFIG["seeds"])
            * (rate[f"{s}/csa"] + rate[f"{s}/csqa"]) for s in RUN_CONFIG["sizes"]}
    emit("C_ESTIMATE", s_per_move_attempt=rate, n_vms=n_vms, calibration_est_s=calib_s, main_est_s=main,
         note="indicative; Python per-step overhead dominates, so the per-attempt rate depends on reads and P")


def c_main(size):
    """2C.4: constraint-preserving SA vs SQA at equal move-attempt budget; one-hot must never break."""
    ctx = setup(size)
    for level in RUN_CONFIG["budget_levels"]:
        for method in ("csa", "csqa"):
            summary, _ = cell(size, ctx, method, level, calibrated(method), 1.0, "C", "s2c_compare")
            assert summary["violation_reads"]["unplaced"] == summary["violation_reads"]["duplicated"] == 0, summary
    emit("DONE", size=size, track="2C")


def select(rows, tie):
    """2D.0 selection rule: highest p_opt, then highest valid_pct, then lowest mean_g, then grid order; values within
    `tie` of the best tie. Returns (chosen row, criterion that decided, saturated: >= 2 rows with p_opt = 1)."""
    cand, decided = sorted(rows, key=lambda r: r["order"]), "grid order"
    for name, best_of in (("p_opt", max), ("valid_pct", max), ("mean_g", min)):
        best = best_of(r[name] for r in cand)
        narrowed = [r for r in cand if abs(r[name] - best) <= tie]
        if len(narrowed) == 1 < len(cand):
            decided = name
        cand = narrowed
        if len(cand) == 1:
            break
    return cand[0], decided, sum(r["p_opt"] >= 1 - tie for r in rows) >= 2


def seed_metrics(row):
    """(P(opt), best valid gap %) of one DONE run; no valid read -> gap +inf."""
    if row["status"] != "DONE":
        raise RuntimeError(("run not DONE", row.get("status"), row.get("error")))
    return row["pure_and_optimal_count"] / row["sampler"]["reads"], math.inf if row["gap_pct"] is None else row["gap_pct"]


def decide(rows_a, rows_b, tie, names=("SQA", "SA")):
    """2D.0 decision rule over seed-matched runs: per seed, higher P(opt) wins, then lower best gap, else tie.
    '<a> better' iff a wins every seed, '<b> better' iff b does; otherwise no significant difference."""
    winners = []
    for ra, rb in zip(rows_a, rows_b, strict=True):
        if ra["seed"] != rb["seed"]:
            raise ValueError(("seed mismatch", ra["seed"], rb["seed"]))
        (pa, ga), (pb, gb) = seed_metrics(ra), seed_metrics(rb)
        if abs(pa - pb) > tie:
            winners.append(names[0] if pa > pb else names[1])
        elif ga == gb or abs(ga - gb) <= tie:
            winners.append("tie")
        else:
            winners.append(names[0] if ga < gb else names[1])
    for name in names:
        if all(w == name for w in winners):
            return f"{name} better", winners
    return "no significant difference at this budget", winners


def d_freeze():
    """2D.2: csqa at the 2C.4 settings, without and with world-line moves; per-sweep counters in windows; C1 rule."""
    from diagnostics.s2_constrained import anneal, build_model, move_attempts, schedule
    f = RUN_CONFIG["d_freeze"]
    ctx = setup(f["size"])
    vms, qinfo = ctx[0], ctx[5]
    model = build_model(qinfo, vms)
    q = len(model["on"])
    final, last = {}, {}
    for worldline in (False, True):
        s = spec("csqa", f["reads"], 1, {**calibrated("csqa"), "worldline": worldline})
        beta_g, ell, p = schedule(s, float(model["lam"][0]), q)
        stats = []
        anneal(model, beta_g, ell, p, f["reads"], np.random.default_rng(f["seed"]), worldline, stats)
        for k, idx in enumerate(np.array_split(np.arange(s["sweeps"]), f["windows"])):
            tot = {key: sum(stats[t][key] for t in idx) for key in stats[0] if key.startswith(("local", "wl"))}
            end = stats[idx[-1]]
            row = {"size": f["size"], "worldline": worldline, "window": k, "sweeps": [int(idx[0]), int(idx[-1])],
                   "s": [(idx[0] + 0.5) / s["sweeps"], (idx[-1] + 0.5) / s["sweeps"]],
                   "ell": [float(ell[idx[0]]), float(ell[idx[-1]])],
                   "local_acc_rate": tot["local_accepted"] / tot["local_proposed"],
                   "blocked_share": tot["local_blocked"] / tot["local_energy_ok"] if tot["local_energy_ok"] else None,
                   "wl_acc_rate": tot["wl_accepted"] / tot["wl_proposed"] if tot["wl_proposed"] else None,
                   "mismatch_vm_frac": end["mismatch_vm_frac"], "mean_g0": end["mean_g0"],
                   "overloaded_reads0": end["overloaded_reads0"], **tot}
            emit("FREEZE", **row)
            print(f"FTABLE | wl {worldline} | {k} | s {row['s'][0]:.3f}-{row['s'][1]:.3f} | "
                  f"ell {row['ell'][0]:.3g}..{row['ell'][1]:.3g} | local acc {row['local_acc_rate']:.3g} | "
                  f"blocked {row['blocked_share']} | wl acc {row['wl_acc_rate']} | "
                  f"mismatch {row['mismatch_vm_frac']:.3g} | mean G0 {row['mean_g0']:.10g} | "
                  f"overloaded {row['overloaded_reads0']:.3g} |", flush=True)
            last[worldline] = row
        final[worldline] = {"spec": s, "mean_g0": stats[-1]["mean_g0"],
                            "overloaded_reads0": stats[-1]["overloaded_reads0"],
                            "move_attempts": move_attempts(s, p, len(vms))}
        emit("FREEZE_FINAL", size=f["size"], worldline=worldline, **final[worldline])
    off = last[False]
    checks = {"1_local_acc_below_1e-3": off["local_acc_rate"] < 1e-3,
              "2_blocked_share_at_least_0.5": off["blocked_share"] is not None and off["blocked_share"] >= 0.5,
              "3_worldline_lowers_final_mean_g0": final[False]["mean_g0"] - final[True]["mean_g0"]
                  > 1e-9 * abs(final[False]["mean_g0"])}
    verdict = {"size": f["size"], "checks": checks, "C1": "confirmed" if all(checks.values()) else "refuted"}
    emit("C1_VERDICT", **verdict)
    log_run("s2d_freeze", dict(RUN_CONFIG, stage="2D.2"), f["seed"], f["size"], vms,
            {"final": final, "last_window": last, "verdict": verdict}, verdict["C1"])


def d_calibrate(size):
    """2D.3: both methods, equal grid size and budget, calibration seeds pooled; select() chooses."""
    ctx = setup(size)
    vms = ctx[0]
    for method in ("csa", "csqa"):
        grid, rows = RUN_CONFIG["d_grid"][method], []
        for order, combo in enumerate(product(*grid.values())):
            params = dict(zip(grid, combo))
            s = spec(method, RUN_CONFIG["calib_reads"], 1, params)
            runs = []
            for seed in RUN_CONFIG["d_calib_seeds"]:
                result = sample(size, ctx, seed, s)
                emit("D_CALIB", size=size, method=method, order=order, params=params, seed=seed, **result)
                if result["status"] != "DONE":
                    raise RuntimeError(("calibration run failed", size, method, params, seed, result.get("error")))
                runs.append(result)
            reads = sum(r["sampler"]["reads"] for r in runs)
            gaps = [r["gap_pct"] for r in runs if r["gap_pct"] is not None]
            row = {"order": order, "params": params, "p_opt": sum(r["pure_and_optimal_count"] for r in runs) / reads,
                   "valid_pct": math.fsum(r["valid_read_pct"] for r in runs) / len(runs),
                   "mean_g": math.fsum(r["mean_raw_H"] for r in runs) / len(runs),
                   "best_gap_pct": min(gaps) if gaps else None,
                   "move_attempts": sum(r["spin_updates"] for r in runs),
                   "sampler_wall_s": math.fsum(r["sampler_wall_s"] for r in runs)}
            rows.append(row)
            print(f"DTABLE | {size} | {method} | {order} | {json.dumps(params, sort_keys=True)} | "
                  f"P(opt) {row['p_opt']} | valid {row['valid_pct']} | mean G {row['mean_g']:.12g} | "
                  f"best gap {row['best_gap_pct']} | moves {row['move_attempts']} | {row['sampler_wall_s']:.1f} s |",
                  flush=True)
        best, decided, saturated = select(rows, RUN_CONFIG["d_tie"])
        choice = {"size": size, "method": method, "params": best["params"], "chosen_order": best["order"],
                  "decided_by": decided, "saturated": saturated, "chosen": best,
                  "calibration_move_attempts": sum(r["move_attempts"] for r in rows),
                  "sampler_wall_s_total": math.fsum(r["sampler_wall_s"] for r in rows)}
        emit("D_CHOICE", **choice)
        log_run("s2d_calibration", dict(RUN_CONFIG, method=method, size=size), RUN_CONFIG["d_calib_seeds"][0], size,
                vms, {"choice": choice, "rows": rows}, "recorded")
    emit("DONE", size=size, track="2D.3")


def d_calibrated(method):
    """2D.3 choice of `method` by the 2D.0 rule: medium, unless either method saturated there, then large."""
    medium, large = RUN_CONFIG["d_calib_sizes"]

    def last(size, m):
        rows = [r for r in records("D_CHOICE", size, CALIB_D) if r["method"] == m]
        return rows[-1] if rows else None

    on_medium = {m: last(medium, m) for m in ("csa", "csqa")}
    if None in on_medium.values():
        raise LookupError(f"2D.3 medium calibration missing in {CALIB_D.name}")
    if any(r["saturated"] for r in on_medium.values()):
        chosen = last(large, method)
        if chosen is None:
            raise LookupError("medium saturated: run --d-calibrate large_v2 first (2D.0)")
        return chosen["params"]
    return on_medium[method]["params"]


def d_main(size):
    """2D.4: held-out seeds; csa / csqa at the 2D.3 choice, csqa with world-line at the 2C.4 settings; decide()."""
    ctx = setup(size)
    tie = RUN_CONFIG["d_tie"]
    settings = {"csa_2d": ("csa", d_calibrated("csa")), "csqa_2d": ("csqa", d_calibrated("csqa")),
                "csqa_wl_2c4": ("csqa", {**calibrated("csqa"), "worldline": True})}
    old = records("C", size, COMPARE_C)
    for level in RUN_CONFIG["budget_levels"]:
        rows = {}
        for label, (method, params) in settings.items():
            emit("D_CELL", size=size, level=level, label=label, method=method, params=params)
            summary, rows[label] = cell(size, ctx, method, level, params, 1.0, "D", "s2d_compare")
            assert summary["violation_reads"]["unplaced"] == summary["violation_reads"]["duplicated"] == 0, summary
        verdict, winners = decide(rows["csqa_2d"], rows["csa_2d"], tie)
        emit("D_DECISION", size=size, level=level, verdict=verdict, seed_winners=winners,
             per_seed={k: [seed_metrics(r) for r in v] for k, v in rows.items()})
        print(f"DECISION | {size} | {level}B | {verdict} | {winners} |", flush=True)
        c2c4 = sorted((r for r in old if r["method"] == "csqa" and r["level"] == level), key=lambda r: r["seed"])
        c1 = decide(rows["csqa_wl_2c4"], c2c4, tie, ("world-line", "no world-line"))
        c2 = decide(rows["csqa_2d"], rows["csqa_wl_2c4"], tie, ("2D.3 settings", "2C.4 settings"))
        emit("D_EFFECT", size=size, level=level, note="diagnostic only (2D.0), not a claim",
             C1_worldline_at_2C4=c1, C2_settings_with_worldline=c2)
    emit("DONE", size=size, track="2D.4")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument("--size", choices=RUN_CONFIG["sizes"])
    mode.add_argument("--lever-l4", choices=RUN_CONFIG["sizes"], metavar="SIZE")
    mode.add_argument("--c-dry-run", action="store_true")
    mode.add_argument("--c-calibrate", action="store_true")
    mode.add_argument("--c-size", choices=RUN_CONFIG["sizes"])
    mode.add_argument("--d-freeze", action="store_true")
    mode.add_argument("--d-calibrate", choices=RUN_CONFIG["d_calib_sizes"])
    mode.add_argument("--d-size", choices=RUN_CONFIG["d_eval_sizes"])
    args = parser.parse_args()
    print("RUN_CONFIG=" + json.dumps(RUN_CONFIG, sort_keys=True), flush=True)
    if args.dry_run:
        dry_run()
    elif args.calibrate:
        calibrate("sqa", RUN_CONFIG["grid"], "s2_sqa_calibration")
    elif args.lever_l4:
        lever_l4(args.lever_l4)
    elif args.c_dry_run:
        c_dry_run()
    elif args.c_calibrate:
        for method in ("csa", "csqa"):
            calibrate(method, RUN_CONFIG["c_grid"][method], "s2c_calibration")
    elif args.c_size:
        c_main(args.c_size)
    elif args.d_freeze:
        d_freeze()
    elif args.d_calibrate:
        d_calibrate(args.d_calibrate)
    elif args.d_size:
        d_main(args.d_size)
    else:
        main_runs(args.size)


if __name__ == "__main__":
    main()
