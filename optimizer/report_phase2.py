"""
GreenQuanta Phase 2 - report. Run AFTER run_phase2.py and Phase2Simulation.java:
    python report_phase2.py --size small

Cross-checks Python vs CloudSim and saves to results/phase2/<size>/:
    host_utilization_24h.png   CPU utilization of every active server over the day
    host_energy.png            energy per active server, Python vs CloudSim, by type
"""
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase1 import config as p1config
from phase2 import config


def compare(r, cs):
    """Python result vs CloudSim summary: ([(metric, python, cloudsim, |diff| % of python)], worst diff %)."""
    rows = [("Active servers", r["active_hosts"], cs["active_hosts"]),
            ("IT energy (kWh)", r["it_energy_kwh"], cs["it_energy_kwh"]),
            ("Facility energy (kWh)", r["facility_energy_kwh"], cs["facility_energy_kwh"]),
            ("Carbon (kg CO2)", r["co2_kg"], cs["co2_kg"])]
    out = [(name, x, y, 0.0 if x == 0 else 100 * abs(x - y) / abs(x)) for name, x, y in rows]
    return out, max(d for *_, d in out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=sorted(config.SCENARIOS), default=config.DEFAULT_SIZE)
    a = p.parse_args()
    out = config.RESULTS_DIR / a.size
    need = [out / "summary.json", out / "energy_hosts_python.csv",
            out / "cloudsim_summary.csv", out / "cloudsim_energy_hosts.csv"]
    missing = [str(x) for x in need if not x.exists()]
    if missing:
        sys.exit("Missing (run run_phase2.py, then Phase2Simulation.java):\n  " + "\n  ".join(missing))

    s = json.load(open(out / "summary.json"))
    py = pd.read_csv(out / "energy_hosts_python.csv")
    cs_hosts = pd.read_csv(out / "cloudsim_energy_hosts.csv")
    cs = pd.read_csv(out / "cloudsim_summary.csv").set_index("metric")["value"]
    r, st, q = s["result"], s["solver"], s["qubo"]

    line = "=" * 66
    print(line)
    print(f" GreenQuanta Phase 2 report ({a.size})")
    print(line)
    print(f" Workload : {s['workload']['vms']} VMs from {s['workload']['collections']} Google jobs")
    print(f" Servers  : " + ", ".join(f"{t} x{c}" for t, c in s["servers"].items()))
    print(f" Lambdas  : config {q['lambda_config']} {q['lambdas']}")
    print(f" Annealer : {st['anneal_time_s']} s, valid {st['valid_reads_pct']}%, "
          f"{st['vms_placed_by_annealer']} VMs by annealer + {st['vms_placed_by_repair']} by repair")
    print(line)
    print(f" {'Metric':<22}{'Python':>14}{'CloudSim':>14}{'Diff %':>10}")
    rows, worst = compare(r, cs)
    for name, x, y, d in rows:
        print(f" {name:<22}{x:>14.4f}{y:>14.4f}{d:>10.3f}")
    print(line)
    print(f" SLA violation {r['sla_violation_pct']:.2f}%, unmet demand {r['unmet_cpu_demand_pct']:.2f}%, "
          f"overloads {r['host_overload_windows']}")
    gap = s.get("optimality", {})
    if gap.get("exact_available"):
        ex = s["exact_reference"]["energy"]
        print(f" Exact optimum: {ex['active_hosts']} servers, {ex['it_energy_kwh']:.4f} kWh; "
              f"QA gap {gap.get('gap_energy_pct', gap.get('final_gap_energy_pct', float('nan'))):+.2f}%, "
              f"pure and optimal: {gap.get('pure_and_optimal')}")
    print(" Cross-check: " + ("OK (< 1%)" if worst < 1 else "WARNING, Python and CloudSim differ by more than 1%"))

    # ---- graphs
    cs_in = out / "cloudsim_input"
    servers = pd.read_csv(cs_in / "servers.csv").set_index("host_id")
    vms = pd.read_csv(cs_in / "vms_assignment.csv")
    demand = pd.read_csv(cs_in / "vm_cpu_demand.csv")
    hours = demand["window_idx"] * p1config.WINDOW_SECONDS / 3600
    fig, ax = plt.subplots(figsize=(10, 5))
    for h, g in vms.groupby("host_id"):
        used = sum(np.minimum(demand[f"vm_{int(v.vm_id)}"], v.cpu_size_ncu) for v in g.itertuples())
        ax.plot(hours, 100 * used / servers.loc[h, "cpu_ncu"], linewidth=1.1,
                label=f"Server {h} ({servers.loc[h, 'type']})")
    ax.axhline(85, color="black", linestyle="--", linewidth=1, label="Safe limit 85%")
    ax.set(xlabel="Hour of day (Google trace 2019, cell a, day 2)", ylabel="CPU utilization (%)",
           title=f"Phase 2 ({a.size}): server CPU utilization over 24 h", xlim=(0, 24), ylim=(0, 100))
    ax.legend(ncol=4, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "host_utilization_24h.png", dpi=150)
    plt.close(fig)

    act = py[py["active"]]
    c2 = cs_hosts.set_index("host_id").loc[act["host_id"]]
    x = np.arange(len(act))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, act["energy_kwh"], 0.4, label="Python (SPEC curve)")
    ax.bar(x + 0.2, c2["energy_kwh"], 0.4, label="CloudSim Plus")
    ax.set_xticks(x, [f"S{h} ({t})" for h, t in zip(act["host_id"], act["type"])])
    ax.set(ylabel="IT energy over 24 h (kWh)", title=f"Phase 2 ({a.size}): energy per active server")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "host_energy.png", dpi=150)
    plt.close(fig)
    print(f" Graphs: {out / 'host_utilization_24h.png'}\n         {out / 'host_energy.png'}")


if __name__ == "__main__":
    main()
