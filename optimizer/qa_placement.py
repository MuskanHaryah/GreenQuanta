"""GreenQuanta - VM placement as QUBO, solved with local simulated annealing."""
import json
import time
from pathlib import Path

import dimod
import pandas as pd
from dwave.samplers import SimulatedAnnealingSampler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Simple linear power model (watts), used only as an estimate
P_IDLE = 100.0
P_MAX = 250.0

# Penalty for breaking a constraint
PENALTY = 5.0


def build_bqm(hosts, vms):
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)

    # Objective: each active host costs 1 (fewer hosts = less energy)
    for h in hosts.itertuples():
        bqm.add_variable(f"y_{h.host_id}", 1.0)

    # Constraint 1: each VM is placed on exactly one host
    for v in vms.itertuples():
        terms = [(f"x_{v.vm_id}_{h.host_id}", 1) for h in hosts.itertuples()]
        bqm.add_linear_equality_constraint(
            terms, lagrange_multiplier=PENALTY, constant=-1
        )

    # Constraint 2: host CPU capacity, and VMs only on active hosts
    for h in hosts.itertuples():
        terms = [(f"x_{v.vm_id}_{h.host_id}", int(v.pes)) for v in vms.itertuples()]
        terms.append((f"y_{h.host_id}", -int(h.pes)))
        bqm.add_linear_inequality_constraint(
            terms,
            lagrange_multiplier=PENALTY,
            label=f"cpu_{h.host_id}",
            lb=-int(h.pes),
            ub=0,
        )
    return bqm


def decode(sample, hosts, vms):
    """Turn the switches into an assignment. Returns None if a constraint is broken."""
    assignment = {}
    for v in vms.itertuples():
        chosen = [
            h.host_id for h in hosts.itertuples()
            if sample[f"x_{v.vm_id}_{h.host_id}"] == 1
        ]
        if len(chosen) != 1:
            return None
        assignment[v.vm_id] = chosen[0]

    for h in hosts.itertuples():
        used = sum(v.pes for v in vms.itertuples() if assignment[v.vm_id] == h.host_id)
        if used > h.pes:
            return None
    return assignment


def estimate_power(hosts, vms, assignment):
    total = 0.0
    active = 0
    for h in hosts.itertuples():
        used = sum(v.pes for v in vms.itertuples() if assignment[v.vm_id] == h.host_id)
        if used == 0:
            continue
        active += 1
        total += P_IDLE + (P_MAX - P_IDLE) * used / h.pes
    return active, total


def main():
    hosts = pd.read_csv(DATA / "hosts.csv")
    vms = pd.read_csv(DATA / "vms.csv")
    RESULTS.mkdir(exist_ok=True)

    bqm = build_bqm(hosts, vms)
    print(f"QUBO variables (switches): {len(bqm.variables)}")

    start = time.perf_counter()
    sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=200, seed=42)
    runtime = time.perf_counter() - start

    assignment = None
    for datum in sampleset.data(fields=["sample", "energy"], sorted_by="energy"):
        assignment = decode(datum.sample, hosts, vms)
        if assignment is not None:
            break

    if assignment is None:
        print("No valid solution found. Try increasing PENALTY or num_reads.")
        return

    active, power = estimate_power(hosts, vms, assignment)

    print("\n=== QUANTUM ANNEALING (simulated) ===")
    for vm_id in sorted(assignment):
        print(f"  VM {vm_id} -> Host {assignment[vm_id]}")
    print(f"  Active hosts: {active}")
    print(f"  Estimated power: {power:.1f} W")
    print(f"  Annealing time: {runtime:.3f} s")

    pd.DataFrame(
        {"vm_id": list(assignment.keys()), "host_id": list(assignment.values())}
    ).sort_values("vm_id").to_csv(RESULTS / "assignment_qa.csv", index=False)

    with open(RESULTS / "qa_summary.json", "w") as f:
        json.dump(
            {"active_hosts": active, "power_watts": power, "runtime_s": round(runtime, 4)},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()