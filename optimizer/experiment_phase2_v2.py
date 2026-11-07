"""
GreenQuanta Phase 2 v2 experiment: encoding x lambda x center x anneal schedule x unit x seeds.

    python experiment_phase2_v2.py --size small_v2
    python experiment_phase2_v2.py --size medium_v2 --encodings unbalanced --configs C \
           --lambdas 0.05 0.1 0.25 0.5 1 --centers 0 0.02 0.05 --schedules 200x2000 50x10000 --seeds 3

--configs   named lambda configs (A-E)
--lambdas   extra scan values: every value v becomes config "L<v>" (all four lambdas = v)
--centers   unbalanced penalty centers (only used with the unbalanced encoding)
--schedules annealer budgets as READSxSWEEPS
Outputs in results/phase2/<size>/: experiment_v2.csv and experiment_v2.png
"""
import argparse
import itertools
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase2 import config
from phase2.energy_model import evaluate_allocation
from phase2.pipeline import exact_reference, prepare
from phase2.pipeline_v2 import gap_metrics_v2, optimize
from phase2.qubo_builder import LAMBDA_CONFIGS
from phase2.qubo_builder_v2 import ENCODINGS
import run_phase2_v2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=sorted(config.SCENARIOS), default="small_v2")
    p.add_argument("--trace")
    p.add_argument("--servers")
    p.add_argument("--auto-factor", type=float, default=config.AUTO_SERVER_FACTOR)
    p.add_argument("--encodings", nargs="+", default=["slack", "slack_scaled", "unbalanced"])
    p.add_argument("--configs", nargs="*", default=["A", "C", "E"])
    p.add_argument("--lambdas", type=float, nargs="*", default=[])
    p.add_argument("--centers", type=float, nargs="+", default=[config.UNBALANCED_CENTER])
    p.add_argument("--schedules", nargs="+", default=[f"{config.NUM_READS}x{config.NUM_SWEEPS}"])
    p.add_argument("--units", type=float, nargs="+", default=[config.UNIT_NCU])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--descent", action="store_true")
    p.add_argument("--candidate", action="store_true")
    a = p.parse_args()

    configs = list(a.configs) + [f"L{v:g}" for v in a.lambdas]
    for e in a.encodings:
        assert e in ENCODINGS, f"unknown encoding {e}"
    for c in a.configs:
        assert c in LAMBDA_CONFIGS, f"unknown lambda config {c}"
    schedules = [tuple(int(x) for x in s.lower().split("x")) for s in a.schedules]

    ns = run_phase2_v2.build_args(["--size", a.size, "--auto-factor", str(a.auto_factor)]
                                  + (["--trace", a.trace] if a.trace else [])
                                  + (["--servers", a.servers] if a.servers else [])
                                  + (["--candidate"] if a.candidate else []))
    trace, servers, mix = run_phase2_v2.setup(ns)
    vms, _, _, _ = prepare(trace, servers)
    ex_a, ex_i = exact_reference(vms, servers)
    ex = ex_i.get("energy", {})
    print(f"{len(vms)} VMs on {len(servers)} servers {mix}. Exact optimum: {ex.get('active_hosts')} servers, "
          f"{ex.get('it_energy_kwh', float('nan')):.4f} kWh ({ex_i.get('status')})")

    grid = []
    for unit, enc, cfg, (reads, sweeps) in itertools.product(a.units, a.encodings, configs, schedules):
        for center in (a.centers if enc == "unbalanced" else [None]):
            for seed in range(1, a.seeds + 1):
                grid.append((unit, enc, cfg, center, reads, sweeps, seed))

    rows = []
    for i, (unit, enc, cfg, center, reads, sweeps, seed) in enumerate(grid, 1):
        config.UNIT_NCU = config.UNIT_NMU = unit
        t = time.perf_counter()
        final, pure, stats, qinfo = optimize(vms, servers, cfg, enc, reads, sweeps, seed, a.descent, center)
        gap = gap_metrics_v2(final, pure, vms, servers, ex_a, ex_i, stats)
        pure_servers = gap.get("annealer_only_servers", np.nan)
        r = {
            "unit": unit, "encoding": enc, "config": cfg, "lambda_assign": qinfo["lambdas"]["assign"],
            "center": center if center is not None else np.nan, "schedule": f"{reads}x{sweeps}", "seed": seed,
            "qubo_variables": qinfo["num_variables"],
            "valid_raw_pct": stats["valid_reads_raw_pct"],
            "valid_pct": stats["valid_reads_pct"],
            "annealer_only_found": gap.get("annealer_only_found", False),
            "annealer_only_servers": pure_servers,
            "annealer_only_gap_pct": gap.get("annealer_only_gap_energy_pct", np.nan),
            "pure_and_optimal": gap.get("pure_and_optimal", False),
            "final_servers": evaluate_allocation(vms, final, servers)["active_hosts"],
            "final_gap_pct": gap.get("final_gap_energy_pct", np.nan),
            "vms_by_annealer_pct": 100.0 * stats["vms_placed_by_annealer"] / len(vms),
            "anneal_time_s": stats["anneal_time_s"],
            "wall_time_s": time.perf_counter() - t,
        }
        rows.append(r)
        cen = f"c={center:g}" if center is not None else "     "
        print(f"[{i:>3}/{len(grid)}] {enc:<12} {cfg:<6} {cen:<6} {reads}x{sweeps:<6} seed {seed}: "
              f"valid {r['valid_pct']:5.1f}%  annealer-only servers {pure_servers}  "
              f"gap {r['annealer_only_gap_pct']:+7.2f}%  final gap {r['final_gap_pct']:+6.2f}%  "
              f"pure&optimal {r['pure_and_optimal']}")

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / a.size
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "experiment_v2.csv", index=False)
    keys = ["unit", "encoding", "config", "center", "schedule"]
    table = (df.groupby(keys, dropna=False)
               .agg(variables=("qubo_variables", "mean"),
                    valid_pct=("valid_pct", "mean"),
                    annealer_only_runs=("annealer_only_found", "sum"),
                    annealer_only_servers=("annealer_only_servers", "mean"),
                    annealer_only_gap_pct=("annealer_only_gap_pct", "mean"),
                    best_gap_pct=("annealer_only_gap_pct", "min"),
                    pure_and_optimal=("pure_and_optimal", "sum"),
                    final_gap_pct=("final_gap_pct", "mean"),
                    anneal_time_s=("anneal_time_s", "mean"))
               .round(2).reset_index())
    pd.set_option("display.width", 250)
    print(f"\nAverage over {a.seeds} seeds (run counts are out of {a.seeds}); exact optimum "
          f"{ex.get('active_hosts')} servers:")
    print(table.to_string(index=False))
    best = table.dropna(subset=["annealer_only_gap_pct"]).sort_values("annealer_only_gap_pct").head(3)
    if len(best):
        print("\nBest settings (lowest average annealer-only gap):")
        print(best[keys + ["valid_pct", "annealer_only_servers", "annealer_only_gap_pct", "pure_and_optimal"]]
              .to_string(index=False))

    labels = []
    for r in table.itertuples():
        lab = f"{r.encoding[:5]} {r.config}"
        if not np.isnan(r.center):
            lab += f"\nc={r.center:g}"
        if len(schedules) > 1:
            lab += f"\n{r.schedule}"
        if len(a.units) > 1:
            lab += f"\nu={r.unit:g}"
        labels.append(lab)
    fig, axes = plt.subplots(1, 4, figsize=(max(18, 0.55 * len(table) * 4), 4.8))
    for ax, (col, title) in zip(axes, [("valid_pct", "Valid reads, no repair (%)"),
                                       ("annealer_only_servers", "Servers on (annealer only)"),
                                       ("pure_and_optimal", f"Pure and optimal runs (of {a.seeds})"),
                                       ("annealer_only_gap_pct", "Annealer-only energy gap (%)")]):
        ax.bar(range(len(table)), table[col].fillna(0), color="#4C72B0")
        if col == "annealer_only_servers" and ex.get("active_hosts"):
            ax.axhline(ex["active_hosts"], color="black", linestyle="--", linewidth=1, label="optimum")
            ax.legend(fontsize=7)
        ax.set_xticks(range(len(table)), labels, fontsize=6)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Phase 2 v2 ({a.size}): encodings, lambdas, centers, schedules")
    fig.tight_layout()
    fig.savefig(out / "experiment_v2.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out / 'experiment_v2.csv'}\n       {out / 'experiment_v2.png'}")


if __name__ == "__main__":
    main()
