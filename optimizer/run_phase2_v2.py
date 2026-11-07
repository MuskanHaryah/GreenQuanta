"""
GreenQuanta Phase 2 update (v2) - static allocation, heterogeneous servers.

New in v2: capacity encodings (slack / slack_scaled / unbalanced), steepest-descent post-processing,
"annealer-only" result next to the final (repaired) result, auto server mix, memory-bound check.

    python run_phase2_v2.py --size small_v2
    python run_phase2_v2.py --size medium_v2 --encoding slack --lambda-config C
    python run_phase2_v2.py --size small --encoding unbalanced     # first-run data, new encoding
    python run_phase2_v2.py --size large_v2 --unit 0.005 --reads 50 --sweeps 5000
    python run_phase2_v2.py --size medium_v2 --lambda-config L0.25 --center 0.02 --reads 50 --sweeps 10000
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from phase1.trace_loader import load_trace
from phase1.vm_builder import build_vms
from phase2 import config
from phase2.catalog import build_servers, catalog_table, parse_mix
from phase2.energy_model import evaluate_allocation
from phase2.pipeline import exact_reference, prepare, write_outputs
from phase2.pipeline_v2 import auto_mix, gap_metrics_v2, optimize
from phase2.qubo_builder import LAMBDA_CONFIGS
from phase2.qubo_builder_v2 import ENCODINGS


def build_args(argv=None):
    p = argparse.ArgumentParser(description="GreenQuanta Phase 2 v2")
    p.add_argument("--size", choices=sorted(config.SCENARIOS), default="small_v2")
    p.add_argument("--trace")
    p.add_argument("--servers", help="A=2,B=1,... or 'auto'")
    p.add_argument("--auto-factor", type=float, default=config.AUTO_SERVER_FACTOR)
    p.add_argument("--encoding", choices=sorted(ENCODINGS), default=config.DEFAULT_ENCODING)
    p.add_argument("--lambda-config", default=config.DEFAULT_LAMBDA_CONFIG,
                   help="A, B, C, D, E or L<value> (all lambdas = value), e.g. L0.25")
    p.add_argument("--center", type=float, help="unbalanced penalty center (free share), default 0.05")
    p.add_argument("--unit", type=float, help="capacity unit for CPU and memory (default 0.0025)")
    p.add_argument("--reads", type=int, default=config.NUM_READS)
    p.add_argument("--sweeps", type=int, default=config.NUM_SWEEPS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--descent", action="store_true", help="steepest-descent post-processing (off by default)")
    p.add_argument("--candidate", action="store_true")
    p.add_argument("--no-exact", action="store_true")
    return p.parse_args(argv)


def setup(a):
    """Apply options; return trace path, servers and the server mix."""
    if a.candidate:
        config.CANDIDATE_REDUCTION = True
    cfg = a.lambda_config
    if not (cfg in LAMBDA_CONFIGS or (cfg.upper().startswith("L") and cfg[1:].replace(".", "", 1).isdigit())):
        raise SystemExit(f"--lambda-config must be one of {sorted(LAMBDA_CONFIGS)} or L<value>, got '{cfg}'")
    if a.unit:
        config.UNIT_NCU = config.UNIT_NMU = a.unit
    config.AUTO_SERVER_FACTOR = a.auto_factor
    scenario = config.SCENARIOS[a.size]
    trace = Path(a.trace) if a.trace else config.TRACE_DIR / scenario["trace"]
    mix = a.servers or scenario["servers"]
    if mix == "auto":
        vms, _ = build_vms(load_trace(trace), config.INSTANCES_PER_VM, config.SLA_MAX_VIOLATION_PCT)
        mix = auto_mix(vms)
    elif isinstance(mix, str):
        mix = parse_mix(mix)
    return trace, build_servers(mix), mix


def main():
    a = build_args()
    trace, servers, mix = setup(a)
    line = "=" * 72
    print(line)
    print(f" GreenQuanta Phase 2 v2 ({a.size}) - encoding '{a.encoding}', lambda config {a.lambda_config}")
    print(line)
    vms, percentile, tinfo, need = prepare(trace, servers)
    print(f"Trace     : {tinfo['collections']} jobs, {tinfo['instances']} instances -> {len(vms)} VMs")
    print(f"Demand    : {need['cpu_ncu']:.4f} NCU, {need['mem_nmu']:.4f} NMU")
    print(f"Servers   : " + ", ".join(f"{t} x{c}" for t, c in mix.items()) + f"  ({len(servers)} total)")

    final, pure, stats, qinfo = optimize(vms, servers, a.lambda_config, a.encoding,
                                         a.reads, a.sweeps, a.seed, a.descent, a.center)
    lam = qinfo["lambdas"]
    print(f"Lambdas   : assign {lam['assign']:.4g}, cpu {lam['cpu']:.4g}, ram {lam['ram']:.4g}, link {lam['link']:.4g}")
    print(f"QUBO      : {qinfo['num_variables']} variables ({qinfo['slack_variables']} slack), "
          f"bias range {qinfo['bias_range'][0]:.2e} .. {qinfo['bias_range'][1]:.2e}")
    print(f"Annealer  : {stats['anneal_time_s']:.2f} s (+ descent {stats['descent_time_s']:.2f} s); valid reads: "
          f"raw {stats['valid_reads_raw_pct']:.1f}%, after descent {stats['valid_reads_pct']:.1f}%; "
          f"VMs placed per read {stats['mean_vms_placed_pct']:.1f}%")

    energy = evaluate_allocation(vms, final, servers)
    ex_a, ex_i = (None, {"skipped": True}) if a.no_exact else exact_reference(vms, servers)
    gap = gap_metrics_v2(final, pure, vms, servers, ex_a, ex_i, stats)

    out = config.RESULTS_DIR / a.size
    summary = write_outputs(out, a.size, servers, vms, percentile, tinfo, need, qinfo, stats, final,
                            energy, ex_a, ex_i, gap)
    summary["version"] = "v2"
    if pure:
        pd.DataFrame({"vm_id": sorted(pure), "host_id": [pure[v] for v in sorted(pure)]}) \
            .to_csv(out / "assignment_annealer_only.csv", index=False)
        summary["annealer_only_result"] = {k: v for k, v in evaluate_allocation(vms, pure, servers).items()
                                           if not k.startswith("_") and k != "per_host"}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(line)
    print(" Final answer (repair allowed)")
    print(f"  Servers on       : {energy['active_hosts']}  " +
          ", ".join(f"{t}: {d['servers_on']}/{d['servers_total']}" for t, d in sorted(energy['by_type'].items())))
    print(f"  VMs by annealer  : {stats['vms_placed_by_annealer']} of {len(vms)} (repair {stats['vms_placed_by_repair']})")
    print(f"  IT / facility    : {energy['it_energy_kwh']:.4f} / {energy['facility_energy_kwh']:.4f} kWh, "
          f"CO2 {energy['co2_kg']:.4f} kg")
    print(f"  SLA / memory     : SLA {energy['sla_violation_pct']:.2f}%, memory-bound servers "
          f"{energy['memory_bound_servers']}")
    if gap.get("exact_available"):
        ex = ex_i["energy"]
        print(f"  Exact optimum    : {ex['active_hosts']} servers, {ex['it_energy_kwh']:.4f} kWh ({ex_i['status']})")
        print(f"  Final gap        : {gap['final_gap_energy_pct']:+.2f}%")
        if gap["annealer_only_found"]:
            print(f"  Annealer only    : {gap['annealer_only_servers']} servers, gap "
                  f"{gap['annealer_only_gap_energy_pct']:+.2f}%, pure and optimal: {gap['pure_and_optimal']}")
        else:
            print("  Annealer only    : no valid read without repair")
    print(line)
    print(f" Saved: {out}")
    print(f" Next : mvn -q compile exec:java \"-Dexec.args={a.size}\"   then   python report_phase2.py --size {a.size}")


if __name__ == "__main__":
    main()
