"""Step 2.3: classical SA baseline (dwave-samplers) on the reduced Stage-2 QUBO Q_y at y = y*. Stdout + log rows.

Two weights per size: (a) lambda = 1.1*U (Step 1), (b) lambda_y = 1.1*U_y (Theorem 2.2). This separates the
effect of the reduction (a vs the Step 1 probe, which fixed only y: docs/reports/step1/1.7_end_to_end.txt) from
the effect of the smaller weight (b vs a). Sampling: r3c_run.sampled_run(..., reduce=True) (reads/sweeps from
r3c_run.RUN_CONFIG); every read is scored on the full BQM with the fixed values restored, validated by the
shared validator, and classified by violation (r3c_run.violation_counts). exact_h = V(y*) from 1.3.
Spin updates = reads x sweeps x Q_y variables. Wall time is indicative only.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics import r3c_run
from diagnostics.s1_ground_truth import RUN_CONFIG as S1, candidates, emit, ground_truth, log_run, records
from diagnostics.s1_stage1 import bits
from diagnostics.s2_stage2 import stage2_bqm

RUN_CONFIG = {"sizes": S1["sizes"], "unit": S1["unit"], "factor": S1["factor"], "seeds": [0, 1, 2],
              "reads": r3c_run.RUN_CONFIG["reads"], "sweeps": r3c_run.RUN_CONFIG["sweeps"], "cap_s": 900}


def run(size):
    vms, servers, data, _, _, _ = candidates(size)
    m, unit = len(servers), RUN_CONFIG["unit"]
    gt = ground_truth(size)
    index = gt["y_star"]
    exact_h = [r for r in records("V", size) if r["index"] == index][-1]["V"]
    y = bits(index, m)
    active = [h for h in range(m) if y[h]]
    _, qinfo = stage2_bqm(vms, servers, y, unit, RUN_CONFIG["factor"])
    cases = {"lambda_1.1U": data["lam"], "lambda_y": qinfo["lam"]}
    for case, lam in cases.items():
        variant = {"id": f"S2-{case}", "lambdas": [lam] * 4, "unit": unit, "encoding": "slack"}
        rows = []
        for seed in RUN_CONFIG["seeds"]:
            result = r3c_run.sampled_run(size, variant, active, seed, exact_h, cap=RUN_CONFIG["cap_s"], reduce=True)
            row = {"size": size, "case": case, "lam": lam, "seed": seed, "y_star": index, **result}
            emit("SA_QY", **row)
            rows.append(row)
        done = [r for r in rows if r["status"] == "DONE"]
        reads = RUN_CONFIG["reads"] * len(done)
        breakdown = {k: sum(r["violation_reads"][k] for r in done) for k in done[0]["violation_reads"]} if done else {}
        gaps = [r["gap_pct"] for r in done if r["gap_pct"] is not None]
        summary = {"size": size, "case": case, "lam": lam, "seeds": len(rows), "done": len(done),
                   "valid_pct_per_seed": [r["valid_read_pct"] for r in done],
                   "valid_pct_pooled": math.fsum(r["valid_read_pct"] for r in done) / len(done) if done else None,
                   "optimal_reads": sum(r["pure_and_optimal_count"] for r in done),
                   "p_optimal_read": sum(r["pure_and_optimal_count"] for r in done) / reads if reads else None,
                   "best_valid_gap_pct": min(gaps) if gaps else None, "violation_reads": breakdown,
                   "violation_pct": {k: 100 * v / reads for k, v in breakdown.items()} if reads else {},
                   "sampled_variables": done[0]["sampled_variables"] if done else None,
                   "spin_updates_per_seed": done[0]["spin_updates"] if done else None,
                   "exact_h": exact_h, "statuses": [r["status"] for r in rows],
                   "sampler_wall_s": [r.get("sampler_wall_s") for r in rows]}
        emit("SA_QY_SUMMARY", **summary)
        v = summary["violation_pct"]
        print(f"TABLE | {size} | {case} | {lam:.6g} | {summary['sampled_variables']} | "
              f"{summary['spin_updates_per_seed']} | {summary['valid_pct_per_seed']} | {summary['p_optimal_read']} | "
              f"{summary['best_valid_gap_pct']} | " + ", ".join(f"{k} {val:.1f}%" for k, val in v.items()) +
              f" | {summary['statuses']} |", flush=True)
        log_run("s2_sa_qy", dict(RUN_CONFIG, size=size, case=case, lam=lam), RUN_CONFIG["seeds"][0], size, vms,
                {"summary": summary, "rows": rows}, "recorded")
    emit("DONE", size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", choices=RUN_CONFIG["sizes"], required=True)
    args = parser.parse_args()
    print("RUN_CONFIG=" + json.dumps(RUN_CONFIG, sort_keys=True), flush=True)
    run(args.size)


if __name__ == "__main__":
    main()
