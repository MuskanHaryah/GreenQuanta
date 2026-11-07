"""
Phase 2 solving: simulated annealer -> decode -> validate -> (repair) -> pick the best read.

Repair (only when a read breaks a rule):
  keep every VM the annealer placed validly, empty overloaded servers (largest VM first),
  then first-fit the rest: servers already ON first, then the most energy-efficient OFF server
  (lowest watts per NCU when filled to its safe limit). Always reported.
"""
import time

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler

from phase1.qubo_builder import x_var, y_var
from . import config
from .energy_model import evaluate_allocation


def efficiency_key(server):
    return float(server.power(server.safe_utilization)) / server.cpu_limit


def capacity_ok(vm_list, server):
    return (sum(v.cpu_size for v in vm_list) <= server.cpu_limit + 1e-12
            and sum(v.mem_size for v in vm_list) <= server.mem_limit + 1e-12)


def decode(sample, vms, servers):
    broken, assignment = set(), {}
    for v in vms:
        chosen = [h for h in range(len(servers)) if sample[x_var(v.vm_id, h)] == 1]
        if len(chosen) != 1:
            broken.add("assign")
            continue
        assignment[v.vm_id] = chosen[0]
        if sample[y_var(chosen[0])] == 0:
            broken.add("link")
    if len(assignment) == len(vms):
        for h, s in enumerate(servers):
            on_h = [v for v in vms if assignment[v.vm_id] == h]
            if sum(v.cpu_size for v in on_h) > s.cpu_limit + 1e-12:
                broken.add("cpu_capacity")
            if sum(v.mem_size for v in on_h) > s.mem_limit + 1e-12:
                broken.add("mem_capacity")
    return (assignment if not broken else None), broken


def repair(sample, vms, servers):
    by_id = {v.vm_id: v for v in vms}
    assignment = {}
    for v in vms:
        chosen = [h for h in range(len(servers)) if sample[x_var(v.vm_id, h)] == 1]
        if len(chosen) == 1 and sample[y_var(chosen[0])] == 1:
            assignment[v.vm_id] = chosen[0]
    for h, s in enumerate(servers):
        on_h = sorted([i for i, x in assignment.items() if x == h], key=lambda i: by_id[i].cpu_size, reverse=True)
        while on_h and not capacity_ok([by_id[i] for i in on_h], s):
            del assignment[on_h.pop(0)]
    kept = len(assignment)

    order = sorted(range(len(servers)), key=lambda h: efficiency_key(servers[h]))
    for v in sorted([v for v in vms if v.vm_id not in assignment], key=lambda v: v.cpu_size, reverse=True):
        active = set(assignment.values())
        for h in [h for h in order if h in active] + [h for h in order if h not in active]:
            if capacity_ok([by_id[i] for i, x in assignment.items() if x == h] + [v], servers[h]):
                assignment[v.vm_id] = h
                break
        else:
            return None, kept, 0
    return assignment, kept, len(vms) - kept


def solve(bqm, vms, servers, num_reads=None, num_sweeps=None, seed=None):
    num_reads = num_reads or config.NUM_READS
    num_sweeps = num_sweeps or config.NUM_SWEEPS
    seed = config.SEED if seed is None else seed

    start = time.perf_counter()
    sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    anneal_s = time.perf_counter() - start

    best, best_score, best_info = None, None, None
    valid, repaired_ok, rule_counts = 0, 0, {}
    for rec in sampleset.data(fields=["sample", "energy"], sorted_by="energy"):
        assignment, broken = decode(rec.sample, vms, servers)
        repaired, kept, by_repair = False, len(vms), 0
        if assignment is None:
            for r in broken:
                rule_counts[r] = rule_counts.get(r, 0) + 1
            assignment, kept, by_repair = repair(rec.sample, vms, servers)
            if assignment is None:
                continue
            repaired = True
            repaired_ok += 1
        else:
            valid += 1
        energy = evaluate_allocation(vms, assignment, servers)["it_energy_kwh"]
        score = (repaired, energy)
        if best_score is None or score < best_score:
            best, best_score = assignment, score
            best_info = {"repaired": repaired, "vms_placed_by_annealer": kept,
                         "vms_placed_by_repair": by_repair, "qubo_energy_of_best": float(rec.energy)}
    if best is None:
        raise RuntimeError("No usable placement: the servers are too small for this workload.")

    energies = sampleset.record.energy
    stats = {
        "solver": "dwave.samplers.SimulatedAnnealingSampler",
        "num_reads": num_reads, "num_sweeps": num_sweeps, "seed": seed,
        "anneal_time_s": round(anneal_s, 4),
        "total_solve_time_s": round(time.perf_counter() - start, 4),
        "valid_reads": valid,
        "valid_reads_pct": round(100.0 * valid / num_reads, 2),
        "repaired_reads": repaired_ok,
        "reads_breaking_rule": rule_counts,
        "qubo_energy_min": float(np.min(energies)),
        "qubo_energy_mean": float(np.mean(energies)),
    }
    stats.update(best_info)
    return best, stats
