"""
GreenQuanta Phase 1 - final report.

Run AFTER run_phase1.py and Phase1Simulation.java:
    python report_phase1.py

1. Cross-checks the Python energy model against CloudSim (they should agree closely).
2. Saves two graphs to results/phase1/:
     host_utilization_24h.png  CPU utilization of every active server over the day
     host_energy.png           energy per server, Python vs CloudSim
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase1 import config

OUT = config.RESULTS_DIR
CS_IN = config.CLOUDSIM_INPUT_DIR


def load():
    needed = [OUT / "summary.json", OUT / "energy_hosts_python.csv",
              OUT / "cloudsim_summary.csv", OUT / "cloudsim_energy_hosts.csv"]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        sys.exit("Missing files (run run_phase1.py, then Phase1Simulation.java):\n  " + "\n  ".join(missing))
    with open(OUT / "summary.json") as f:
        summary = json.load(f)
    py_hosts = pd.read_csv(OUT / "energy_hosts_python.csv")
    cs_hosts = pd.read_csv(OUT / "cloudsim_energy_hosts.csv")
    cs = pd.read_csv(OUT / "cloudsim_summary.csv").set_index("metric")["value"]
    return summary, py_hosts, cs_hosts, cs


def pct_diff(a, b):
    return 0.0 if a == 0 else 100.0 * abs(a - b) / abs(a)


def plot_utilization(summary):
    settings = pd.read_csv(CS_IN / "settings.csv").set_index("key")["value"]
    vms = pd.read_csv(CS_IN / "vms_assignment.csv")
    demand = pd.read_csv(CS_IN / "vm_cpu_demand.csv")
    hours = demand["window_idx"] * config.WINDOW_SECONDS / 3600
    cap = settings["cpu_capacity_ncu"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for host_id, group in vms.groupby("host_id"):
        served = sum(np.minimum(demand[f"vm_{int(r.vm_id)}"], r.cpu_size_ncu) for r in group.itertuples())
        ax.plot(hours, 100 * served / cap, label=f"Server {host_id}", linewidth=1.2)
    safe = summary["inputs"]["server"]["safe_utilization"] * 100
    ax.axhline(safe, color="black", linestyle="--", linewidth=1, label=f"Safe limit {safe:.0f}%")
    ax.set_xlabel("Hour of day (Google trace 2019, cell a, day 2)")
    ax.set_ylabel("CPU utilization (%)")
    ax.set_title("Phase 1: server CPU utilization over 24 h (QA placement)")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 100)
    ax.legend(ncol=4, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "host_utilization_24h.png", dpi=150)
    plt.close(fig)


def plot_energy(py_hosts, cs_hosts):
    active = py_hosts[py_hosts["active"]]
    cs = cs_hosts.set_index("host_id").loc[active["host_id"]]
    x = np.arange(len(active))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, active["energy_kwh"], 0.4, label="Python energy model")
    ax.bar(x + 0.2, cs["energy_kwh"], 0.4, label="CloudSim Plus")
    ax.set_xticks(x, [f"Server {h}" for h in active["host_id"]])
    ax.set_ylabel("IT energy over 24 h (kWh)")
    ax.set_title("Phase 1: energy per active server")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "host_energy.png", dpi=150)
    plt.close(fig)


def main():
    summary, py_hosts, cs_hosts, cs = load()
    r, s = summary["result"], summary["solver"]
    line = "=" * 62
    print(line)
    print(" GreenQuanta Phase 1 report")
    print(line)
    print(f" Workload : {summary['workload']['vms']} VMs from {summary['workload']['collections']} Google jobs")
    print(f" QUBO     : {summary['qubo']['num_variables']} variables, lambda = 2 for all penalties")
    print(f" Annealer : {s['anneal_time_s']} s, valid without repair {s['valid_reads_pct']}%, "
          f"best answer: {s['vms_placed_by_annealer']} VMs by annealer + {s['vms_placed_by_repair']} by repair")
    print(line)
    print(f" {'Metric':<22}{'Python':>14}{'CloudSim':>14}{'Diff %':>10}")
    rows = [
        ("Active servers", r["active_hosts"], cs["active_hosts"]),
        ("IT energy (kWh)", r["it_energy_kwh"], cs["it_energy_kwh"]),
        ("Facility energy (kWh)", r["facility_energy_kwh"], cs["facility_energy_kwh"]),
        ("Carbon (kg CO2)", r["co2_kg"], cs["co2_kg"]),
    ]
    for name, a, b in rows:
        print(f" {name:<22}{a:>14.4f}{b:>14.4f}{pct_diff(a, b):>10.3f}")
    print(line)
    print(f" SLA violation : {r['sla_violation_pct']:.2f}% of VM time "
          f"(limit {summary['inputs']['sla_max_violation_pct']}%)")
    print(f" Unmet demand  : {r['unmet_cpu_demand_pct']:.2f}%   Host overloads: {r['host_overload_windows']}")
    print(f" DCiE {r['dcie_pct']:.2f}%   CUE {r['cue_kg_per_kwh_it']:.4f} kg CO2 per kWh IT")

    worst = max(pct_diff(a, b) for _, a, b in rows)
    print(line)
    print(" Cross-check: " + ("OK, Python and CloudSim agree (< 1%)" if worst < 1
                              else "WARNING, Python and CloudSim differ by more than 1%"))

    plot_utilization(summary)
    plot_energy(py_hosts, cs_hosts)
    print(f" Graphs saved: {OUT / 'host_utilization_24h.png'}")
    print(f"               {OUT / 'host_energy.png'}")


if __name__ == "__main__":
    main()
