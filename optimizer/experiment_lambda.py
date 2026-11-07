"""
GreenQuanta Phase 1 - penalty weight (lambda) experiment.

The main run keeps every lambda = 2. This script tries other weights on the SAME
workload and servers and records how much of the placement the annealer finds by itself.

    lambda_assign  : weight of "every VM on exactly one server" (also used for the link rule)
    lambda_cap     : weight of the CPU and memory capacity rules

Usage (defaults shown):
    python experiment_lambda.py --assign 2 5 10 20 --cap 0.5 2 --seeds 3 --reads 100

Outputs in results/phase1/:
    lambda_experiment.csv   one row per (lambda_assign, lambda_cap, seed)
    lambda_experiment.png   annealer share and valid reads vs lambda
"""
import argparse
import itertools
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from phase1 import config
from phase1.energy_model import evaluate_allocation
from phase1.inputs import default_inputs
from phase1.pipeline import optimize, prepare


def parse_args():
    p = argparse.ArgumentParser(description="Lambda experiment for the Phase 1 QUBO")
    p.add_argument("--assign", type=float, nargs="+", default=[2, 5, 10, 20])
    p.add_argument("--cap", type=float, nargs="+", default=[0.5, 2])
    p.add_argument("--seeds", type=int, default=3, help="repeat each setting with this many seeds")
    p.add_argument("--reads", type=int, default=100)
    p.add_argument("--sweeps", type=int, default=config.NUM_SWEEPS)
    p.add_argument("--servers", type=int, default=config.NUM_SERVERS)
    p.add_argument("--trace", default=str(config.TRACE_CSV))
    p.add_argument("--no-reduction", action="store_true")
    return p.parse_args()


def run_grid(args):
    from dataclasses import replace
    d = default_inputs()
    inputs = replace(d, trace_csv=args.trace, server=replace(d.server, count=args.servers))
    if args.no_reduction:
        config.CANDIDATE_REDUCTION = False
    vms, _, tinfo, lb, binding = prepare(inputs)
    print(f"{len(vms)} VMs, lower bound {lb:.2f} servers ({binding}), {args.servers} user servers")

    rows = []
    grid = list(itertools.product(args.assign, args.cap, range(1, args.seeds + 1)))
    for i, (la, lc, seed) in enumerate(grid, 1):
        config.LAMBDA_ASSIGN = config.LAMBDA_LINK = la
        config.LAMBDA_CPU = config.LAMBDA_RAM = lc
        start = time.perf_counter()
        assignment, stats, qinfo = optimize(vms, inputs.server, lb, args.reads, args.sweeps, seed)
        result = evaluate_allocation(vms, assignment, inputs.server)
        rows.append({
            "lambda_assign": la,
            "lambda_cap": lc,
            "seed": seed,
            "candidate_servers": qinfo["candidate_servers"],
            "qubo_variables": qinfo["num_variables"],
            "valid_reads_pct": stats["valid_reads_pct"],
            "vms_by_annealer": stats["vms_placed_by_annealer"],
            "vms_by_annealer_pct": 100.0 * stats["vms_placed_by_annealer"] / len(vms),
            "pure_annealer_result": not stats["repaired"],
            "active_hosts": result["active_hosts"],
            "it_energy_kwh": result["it_energy_kwh"],
            "anneal_time_s": stats["anneal_time_s"],
            "wall_time_s": time.perf_counter() - start,
        })
        r = rows[-1]
        print(f"[{i:>2}/{len(grid)}] assign={la:<5g} cap={lc:<5g} seed={seed}  "
              f"valid {r['valid_reads_pct']:5.1f}%  annealer {r['vms_by_annealer']:>3}/{len(vms)}  "
              f"servers {r['active_hosts']}")

    df = pd.DataFrame(rows)
    df["lower_bound_servers"] = round(lb, 3)
    return df, len(vms)


def summarize_and_plot(df, n_vms):
    out = config.RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "lambda_experiment.csv", index=False)

    table = (df.groupby(["lambda_assign", "lambda_cap"])
               .agg(valid_reads_pct=("valid_reads_pct", "mean"),
                    vms_by_annealer_pct=("vms_by_annealer_pct", "mean"),
                    pure_results=("pure_annealer_result", "sum"),
                    active_hosts=("active_hosts", "mean"),
                    anneal_time_s=("anneal_time_s", "mean"))
               .round(2).reset_index())
    print("\nAverage over seeds:")
    print(table.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for lc, g in table.groupby("lambda_cap"):
        axes[0].plot(g["lambda_assign"], g["vms_by_annealer_pct"], marker="o", label=f"capacity lambda = {lc:g}")
        axes[1].plot(g["lambda_assign"], g["valid_reads_pct"], marker="o", label=f"capacity lambda = {lc:g}")
    axes[0].set_ylabel(f"VMs placed by annealer (% of {n_vms})")
    axes[1].set_ylabel("Reads valid without repair (%)")
    for ax in axes:
        ax.set_xlabel("assign / link lambda")
        ax.set_xscale("log")
        ax.set_xticks(sorted(table["lambda_assign"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Phase 1: effect of penalty weights on the simulated annealer")
    fig.tight_layout()
    fig.savefig(out / "lambda_experiment.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out / 'lambda_experiment.csv'}\n       {out / 'lambda_experiment.png'}")


if __name__ == "__main__":
    a = parse_args()
    frame, n = run_grid(a)
    summarize_and_plot(frame, n)
