"""
GreenQuanta Phase 2 - lambda configuration experiment.

Runs lambda configurations A-E (see phase2/qubo_builder.py) with several seeds on one size,
and compares every QA answer with the exact optimum (SciPy HiGHS).

    python experiment_phase2.py --size small
    python experiment_phase2.py --size medium --configs A C D E --seeds 5 --reads 200

Outputs in results/phase2/<size>/:
    lambda_experiment.csv   one row per (config, seed)
    lambda_experiment.png   valid reads, annealer share, pure-and-optimal, energy gap per config
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from phase2 import config
from phase2.catalog import build_servers, parse_mix
from phase2.energy_model import evaluate_allocation
from phase2.pipeline import exact_reference, gap_metrics, optimize, prepare
from phase2.qubo_builder import LAMBDA_CONFIGS


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 lambda experiment")
    p.add_argument("--size", choices=sorted(config.SCENARIOS), default=config.DEFAULT_SIZE)
    p.add_argument("--trace")
    p.add_argument("--servers")
    p.add_argument("--configs", nargs="+", default=sorted(LAMBDA_CONFIGS))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--reads", type=int, default=config.NUM_READS)
    p.add_argument("--sweeps", type=int, default=config.NUM_SWEEPS)
    p.add_argument("--candidate", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.candidate:
        config.CANDIDATE_REDUCTION = True
    scenario = config.SCENARIOS[a.size]
    trace = Path(a.trace) if a.trace else config.TRACE_DIR / scenario["trace"]
    servers = build_servers(parse_mix(a.servers) if a.servers else scenario["servers"])
    vms, _, _, _ = prepare(trace, servers)

    exact_assignment, exact_info = exact_reference(vms, servers)
    ex = exact_info.get("energy", {})
    print(f"{len(vms)} VMs on {len(servers)} servers. Exact optimum: {ex.get('active_hosts')} servers, "
          f"{ex.get('it_energy_kwh', float('nan')):.4f} kWh ({exact_info.get('status')})")

    rows = []
    runs = [(c, s) for c in a.configs for s in range(1, a.seeds + 1)]
    for i, (cfg, seed) in enumerate(runs, 1):
        start = time.perf_counter()
        assignment, stats, qinfo = optimize(vms, servers, cfg, a.reads, a.sweeps, seed)
        energy = evaluate_allocation(vms, assignment, servers)
        gap = gap_metrics(assignment, energy, exact_assignment, exact_info, vms, servers, stats)
        lam = qinfo["lambdas"]
        rows.append({
            "config": cfg, "seed": seed,
            "lambda_assign": lam["assign"], "lambda_cpu": lam["cpu"],
            "lambda_ram": lam["ram"], "lambda_link": lam["link"],
            "qubo_variables": qinfo["num_variables"],
            "valid_reads_pct": stats["valid_reads_pct"],
            "vms_by_annealer_pct": 100.0 * stats["vms_placed_by_annealer"] / len(vms),
            "pure_result": not stats["repaired"],
            "pure_and_optimal": gap.get("pure_and_optimal", False),
            "matches_optimum": gap.get("qa_matches_optimum", False),
            "active_hosts": energy["active_hosts"],
            "it_energy_kwh": energy["it_energy_kwh"],
            "gap_energy_pct": gap.get("gap_energy_pct"),
            "anneal_time_s": stats["anneal_time_s"],
            "wall_time_s": time.perf_counter() - start,
        })
        r = rows[-1]
        print(f"[{i:>2}/{len(runs)}] config {cfg} seed {seed}: valid {r['valid_reads_pct']:5.1f}%  "
              f"annealer {r['vms_by_annealer_pct']:5.1f}%  servers {r['active_hosts']}  "
              f"gap {r['gap_energy_pct']:+6.2f}%  pure&optimal {r['pure_and_optimal']}")

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / a.size
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "lambda_experiment.csv", index=False)

    table = (df.groupby("config")
               .agg(valid_reads_pct=("valid_reads_pct", "mean"),
                    vms_by_annealer_pct=("vms_by_annealer_pct", "mean"),
                    pure_results=("pure_result", "sum"),
                    pure_and_optimal=("pure_and_optimal", "sum"),
                    active_hosts=("active_hosts", "mean"),
                    gap_energy_pct=("gap_energy_pct", "mean"),
                    anneal_time_s=("anneal_time_s", "mean"))
               .round(2).reset_index())
    print(f"\nAverage over {a.seeds} seeds (pure counts are out of {a.seeds}); "
          f"exact optimum = {ex.get('active_hosts')} servers:")
    print(table.to_string(index=False))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [("valid_reads_pct", "Reads valid without repair (%)"),
              ("vms_by_annealer_pct", "VMs placed by annealer (%)"),
              ("pure_and_optimal", f"Pure and optimal runs (of {a.seeds})"),
              ("gap_energy_pct", "Energy gap vs exact optimum (%)")]
    for ax, (col, title) in zip(axes, panels):
        ax.bar(table["config"], table[col], color="#4C72B0")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("lambda configuration")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Phase 2 ({a.size}): lambda configurations A-E")
    fig.tight_layout()
    fig.savefig(out / "lambda_experiment.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out / 'lambda_experiment.csv'}\n       {out / 'lambda_experiment.png'}")


if __name__ == "__main__":
    main()
