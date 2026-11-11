"""Step 0.7: classical SA sanity probe on the corrected (slack, lambda = 1.1*F) QUBO.

Baseline for Step 2's SQA comparison. Full BQM, no fixed variables, no repair. Per size: exact
global minimum (as in 0.5), then dwave SimulatedAnnealingSampler for each seed in RUN_CONFIG.
Every read is scored on the genuine BQM (raw H, slack bits as sampled) and with slack eliminated
(r3c_run.independent_terms = Lemma). One experiment-log row per (size, seed); no results/runs folder.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dwave.samplers import SimulatedAnnealingSampler

from diagnostics.common import build_problem
from diagnostics.r3c_build import build_qubo_custom
from diagnostics.r3c_run import independent_terms, metrics, solver_context
from diagnostics.s0_exact import reference_f
from phase1.qubo_builder import x_var, y_var
from phase2 import config, manifest
from phase2.validation import validate_allocation

RUN_CONFIG = {"sizes": ["small_v2", "medium_v2"], "seeds": [0, 1, 2], "reads": 200, "sweeps": 2000,
              "unit": 0.0025, "lambda_factor": 1.1, "center": 0.02, "time_limit": 600, "max_iters": 50}
LOG = config.ROOT / "results/experiment_log.csv"
TOL = 1e-9


def emit(kind, **data):
    print(kind + "_JSON=" + json.dumps(data, sort_keys=True, allow_nan=False, default=str), flush=True)


def run(size):
    unit = RUN_CONFIG["unit"]
    vms, servers, _, _ = build_problem(size)
    _, _, f = reference_f(vms, servers, unit)
    lam = RUN_CONFIG["lambda_factor"] * f
    variant = {"id": f"S0-{RUN_CONFIG['lambda_factor']:g}F", "lambdas": [lam] * 4, "unit": unit,
               "encoding": "slack"}
    bqm, info = build_qubo_custom(vms, servers, *variant["lambdas"], None, unit, "slack")
    energy, _ = build_qubo_custom(vms, servers, 0, 0, 0, 0, None, unit, "slack")
    solve, _ = solver_context(variant)
    exact = solve(vms, servers, center=RUN_CONFIG["center"], time_limit=RUN_CONFIG["time_limit"],
                  max_iters=RUN_CONFIG["max_iters"])
    exact_h = exact["bqm_H"] if exact["proven_optimal"] else None
    emit("EXACT", size=size, F=f, lam=lam, status=exact["status"], exact_H=exact["bqm_H"],
         num_variables=info["num_variables"], interactions=info["num_interactions"])
    trace = config.TRACE_DIR / config.SCENARIOS[size]["trace"]
    workload = [(v.vm_id, v.cpu_size, v.mem_size) for v in vms]
    for seed in RUN_CONFIG["seeds"]:
        started = time.perf_counter()
        samples = SimulatedAnnealingSampler().sample(bqm, num_reads=RUN_CONFIG["reads"],
                                                     num_sweeps=RUN_CONFIG["sweeps"], seed=seed)
        wall = time.perf_counter() - started
        total = valid = quant_valid = hit_raw = hit_elim = 0
        best = None
        for row in samples.data(fields=["sample", "energy", "num_occurrences"]):
            sample = {k: int(v) for k, v in row.sample.items()}
            raw = float(row.energy)
            assert abs(raw - float(bqm.energy(sample))) <= TOL
            count = int(row.num_occurrences)
            total += count
            data = metrics(sample, vms, servers)
            if not data["GLOBAL_VALID"]:
                continue
            valid += count
            assignment = {vm.vm_id: next(h for h in range(len(servers)) if sample[x_var(vm.vm_id, h)])
                          for vm in vms}
            quant_valid += count * validate_allocation(vms, assignment, servers, quantized_unit=unit)["valid"]
            elim = math.fsum(independent_terms(sample, vms, servers, info, energy).values())
            if exact_h is not None:
                hit_raw += count * (abs(raw - exact_h) <= TOL)
                hit_elim += count * (abs(elim - exact_h) <= TOL)
            if best is None or elim < best["elim_H"]:
                best = {"raw_H": raw, "elim_H": elim, "linear_joules": data["linear_joules"],
                        "active_hosts": sorted(servers[h].server_id for h in range(len(servers))
                                               if sample[y_var(h)])}
        assert total == RUN_CONFIG["reads"]
        gap = None if best is None or exact_h is None else 100 * (best["elim_H"] - exact_h) / exact_h
        result = {"size": size, "unit": unit, "lambda": lam, "seed": seed,
                  "reads": total, "sweeps": RUN_CONFIG["sweeps"],
                  "valid_read_pct": 100 * valid / total, "quant_valid_read_pct": 100 * quant_valid / total,
                  "exact_H": exact_h, "best_valid": best, "best_valid_gap_pct": gap,
                  "reads_hitting_exact_raw": hit_raw, "reads_hitting_exact_slack_eliminated": hit_elim,
                  "sampler_wall_s": wall, "exact_status": exact["status"]}
        emit("SA", **result)
        run_config = dict(RUN_CONFIG, size=size, seed=seed, lambda_value=lam, encoding="slack")
        m = manifest.build_manifest("s0_sa_probe", run_config, seed, 0, trace, workload, config.ROOT,
                                    ("numpy", "scipy", "dimod", "dwave-samplers"))
        manifest.append_experiment_log(LOG, {
            "run_id": m["run_id"], "git_commit": m["code_revision"], "config_hash": m["config_hash"],
            "sampler_seed": seed, "workload_seed": m["workload_seed"], "method": m["method"],
            "metrics_json": json.dumps(result, default=str),
            "gate_result": "probe-recorded", "timestamp": m["timestamp"]})
        emit("LOGGED", run_id=m["run_id"], dirty_worktree=m["dirty_worktree"])
    emit("DONE", size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=RUN_CONFIG["sizes"], required=True)
    args = parser.parse_args()
    print("RUN_CONFIG=" + json.dumps(RUN_CONFIG, sort_keys=True), flush=True)
    run(args.size)


if __name__ == "__main__":
    main()
