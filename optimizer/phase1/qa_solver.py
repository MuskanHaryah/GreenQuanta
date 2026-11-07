"""
Solve the QUBO with D-Wave's local simulated annealer (dwave-samplers).
The same BQM could be sent to a real D-Wave QPU by swapping the sampler.

Returns the best placement plus computational-overhead numbers:
runtime, reads, sweeps, % of reads that were valid without repair, energy spread.
"""
import time

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler

from . import config
from .energy_model import evaluate_allocation
from .validator import decode, repair


def _rule_name(message):
    if "switched-off" in message:
        return "link"
    if "CPU" in message:
        return "cpu_capacity"
    if "memory" in message:
        return "mem_capacity"
    return "assign"


def solve(bqm, vms, server, num_reads=None, num_sweeps=None, seed=None):
    num_reads = num_reads or config.NUM_READS
    num_sweeps = num_sweeps or config.NUM_SWEEPS
    seed = config.SEED if seed is None else seed

    start = time.perf_counter()
    sampleset = SimulatedAnnealingSampler().sample(
        bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    anneal_seconds = time.perf_counter() - start

    best, best_score, best_info = None, None, None
    valid_reads, repaired_ok, broken_examples = 0, 0, {}
    for record in sampleset.data(fields=["sample", "energy"], sorted_by="energy"):
        assignment, broken = decode(record.sample, vms, server)
        repaired = False
        kept, placed_by_repair = len(vms), 0
        if assignment is None:
            for kind in {_rule_name(b) for b in broken}:
                broken_examples[kind] = broken_examples.get(kind, 0) + 1
            assignment, kept, placed_by_repair = repair(record.sample, vms, server)
            if assignment is None:
                continue
            repaired = True
            repaired_ok += 1
        else:
            valid_reads += 1

        result = evaluate_allocation(vms, assignment, server)
        # pure annealer answers are preferred over repaired ones; then lower energy wins
        score = (repaired, result["active_hosts"], result["it_energy_kwh"])
        if best_score is None or score < best_score:
            best, best_score = assignment, score
            best_info = {"repaired": repaired,
                         "vms_placed_by_annealer": kept,
                         "vms_placed_by_repair": placed_by_repair,
                         "qubo_energy_of_best": float(record.energy)}

    total_seconds = time.perf_counter() - start
    energies = sampleset.record.energy
    stats = {
        "solver": "dwave.samplers.SimulatedAnnealingSampler",
        "num_reads": num_reads,
        "num_sweeps": num_sweeps,
        "seed": seed,
        "anneal_time_s": round(anneal_seconds, 4),
        "total_solve_time_s": round(total_seconds, 4),
        "valid_reads": valid_reads,
        "valid_reads_pct": round(100.0 * valid_reads / num_reads, 2),
        "repaired_reads": repaired_ok,
        "qubo_energy_min": float(np.min(energies)),
        "qubo_energy_mean": float(np.mean(energies)),
        "qubo_energy_max": float(np.max(energies)),
        "reads_breaking_rule": broken_examples,
    }
    if best is None:
        raise RuntimeError("No usable placement found: not enough servers for this workload.")
    stats.update(best_info)
    return best, stats
