"""
Phase 2 solver, version 2.

Three stages, each reported separately:
  1. raw      : D-Wave simulated annealer reads
  2. descent  : (optional) D-Wave SteepestDescentSolver started from every raw read. It only
                moves downhill on the SAME QUBO, so it is QUBO-native post-processing, not a
                packing heuristic.
  3. repair   : first-fit repair (from solver.py), ONLY for reads still breaking a rule.

Two answers come out:
  * annealer_only : best read that is valid after stage 1 or 2 (no repair), or None
  * final         : best answer overall (repair allowed), always a valid placement
"""
import time

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSolver

from phase1.qubo_builder import x_var
from . import config
from .energy_model import evaluate_allocation
from .solver import decode, repair


def _placed_count(sample, vms, servers):
    return sum(1 for v in vms if sum(sample[x_var(v.vm_id, h)] for h in range(len(servers))) == 1)


def solve(bqm, vms, servers, num_reads=None, num_sweeps=None, seed=None, descent=True):
    num_reads = num_reads or config.NUM_READS
    num_sweeps = num_sweeps or config.NUM_SWEEPS
    seed = config.SEED if seed is None else seed

    t0 = time.perf_counter()
    raw = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    anneal_s = time.perf_counter() - t0
    raw_valid = sum(1 for r in raw.data(["sample"]) if decode(r.sample, vms, servers)[0] is not None)

    t1 = time.perf_counter()
    final_set = SteepestDescentSolver().sample(bqm, initial_states=raw) if descent else raw
    descent_s = time.perf_counter() - t1 if descent else 0.0

    best_any, best_any_score, best_any_info = None, None, None
    best_pure, best_pure_energy = None, None
    valid_after, rule_counts, placed = 0, {}, []
    for rec in final_set.data(fields=["sample", "energy"]):
        placed.append(_placed_count(rec.sample, vms, servers))
        assignment, broken = decode(rec.sample, vms, servers)
        repaired, kept, by_repair = False, len(vms), 0
        if assignment is None:
            for r in broken:
                rule_counts[r] = rule_counts.get(r, 0) + 1
            assignment, kept, by_repair = repair(rec.sample, vms, servers)
            if assignment is None:
                continue
            repaired = True
        else:
            valid_after += 1
        e = evaluate_allocation(vms, assignment, servers)["it_energy_kwh"]
        if not repaired and (best_pure_energy is None or e < best_pure_energy):
            best_pure, best_pure_energy = assignment, e
        score = (repaired, e)
        if best_any_score is None or score < best_any_score:
            best_any, best_any_score = assignment, score
            best_any_info = {"repaired": repaired, "vms_placed_by_annealer": kept,
                             "vms_placed_by_repair": by_repair, "qubo_energy_of_best": float(rec.energy)}
    if best_any is None:
        raise RuntimeError("No usable placement: the servers are too small for this workload.")

    stats = {
        "solver": "SimulatedAnnealingSampler" + (" + SteepestDescentSolver" if descent else ""),
        "num_reads": num_reads, "num_sweeps": num_sweeps, "seed": seed, "descent": descent,
        "anneal_time_s": round(anneal_s, 4),
        "descent_time_s": round(descent_s, 4),
        "total_solve_time_s": round(time.perf_counter() - t0, 4),
        "valid_reads_raw_pct": round(100.0 * raw_valid / num_reads, 2),
        "valid_reads_pct": round(100.0 * valid_after / num_reads, 2),
        "mean_vms_placed_pct": round(100.0 * float(np.mean(placed)) / len(vms), 2) if placed else 0.0,
        "reads_breaking_rule": rule_counts,
        "annealer_only_found": best_pure is not None,
        "annealer_only_energy_kwh": best_pure_energy,
    }
    stats.update(best_any_info)
    return best_any, best_pure, stats
