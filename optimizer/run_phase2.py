"""
GreenQuanta Phase 2 - static VM allocation on heterogeneous servers.

    python run_phase2.py                          # small size, lambda config A
    python run_phase2.py --size medium --lambda-config E
    python run_phase2.py --servers A=3,B=1,C=2,D=2 --trace ..\\data\\trace\\phase2\\my.csv
    python run_phase2.py --candidate              # optional candidate rule
"""
import argparse
from pathlib import Path

from phase2 import config
from phase2.catalog import build_servers, catalog_table, parse_mix
from phase2.energy_model import evaluate_allocation
from phase2.pipeline import exact_reference, gap_metrics, optimize, prepare, write_outputs
from phase2.qubo_builder import LAMBDA_CONFIGS


def parse_args():
    p = argparse.ArgumentParser(description="GreenQuanta Phase 2 (static, heterogeneous servers)")
    p.add_argument("--size", choices=sorted(config.SCENARIOS), default=config.DEFAULT_SIZE)
    p.add_argument("--trace", help="override the trace CSV of the chosen size")
    p.add_argument("--servers", help="override the server mix, e.g. A=2,B=1,C=2,D=1")
    p.add_argument("--lambda-config", choices=sorted(LAMBDA_CONFIGS), default=config.DEFAULT_LAMBDA_CONFIG)
    p.add_argument("--reads", type=int, default=config.NUM_READS)
    p.add_argument("--sweeps", type=int, default=config.NUM_SWEEPS)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--candidate", action="store_true", help="use the optional candidate-server rule")
    p.add_argument("--no-exact", action="store_true", help="skip the exact reference solver")
    return p.parse_args()


def main():
    a = parse_args()
    if a.candidate:
        config.CANDIDATE_REDUCTION = True
    scenario = config.SCENARIOS[a.size]
    trace = Path(a.trace) if a.trace else config.TRACE_DIR / scenario["trace"]
    servers = build_servers(parse_mix(a.servers) if a.servers else scenario["servers"])
    line = "=" * 70

    print(line)
    print(f" GreenQuanta Phase 2 - static allocation, heterogeneous servers ({a.size})")
    print(line)
    vms, percentile, tinfo, need = prepare(trace, servers)
    print(f"Trace     : {tinfo['collections']} jobs, {tinfo['instances']} instances -> {len(vms)} VMs "
          f"(CPU size = P{percentile:g})")
    print(f"Demand    : {need['cpu_ncu']:.4f} NCU CPU, {need['mem_nmu']:.4f} NMU memory")
    print(" Servers  : type  count  CPU(NCU)  MEM(NMU)  idle W  peak W  SPEC server")
    for t in catalog_table(servers):
        print(f"            {t['type']:<5}{t['count']:>5}  {t['cpu_ncu']:>8.4f}  {t['mem_nmu']:>8.4f}  "
              f"{t['idle_w']:>6.1f}  {t['peak_w']:>6.1f}  {t['spec_server']}")

    assignment, stats, qinfo = optimize(vms, servers, a.lambda_config, a.reads, a.sweeps, a.seed)
    lam = qinfo["lambdas"]
    print(f"Lambdas   : config {a.lambda_config} -> assign {lam['assign']:.4g}, cpu {lam['cpu']:.4g}, "
          f"ram {lam['ram']:.4g}, link {lam['link']:.4g}")
    print(f"QUBO      : {qinfo['num_variables']} variables ({qinfo['assignment_variables']} placement, "
          f"{qinfo['server_variables']} server, {qinfo['slack_variables']} slack), "
          f"{qinfo['candidate_servers']} of {qinfo['user_servers']} servers")
    print(f"Annealer  : {stats['anneal_time_s']:.2f} s, valid without repair {stats['valid_reads_pct']:.1f}%, "
          f"best: {stats['vms_placed_by_annealer']} VMs by annealer + {stats['vms_placed_by_repair']} by repair")

    energy = evaluate_allocation(vms, assignment, servers)
    exact_assignment, exact_info = (None, {"skipped": True}) if a.no_exact else exact_reference(vms, servers)
    gap = gap_metrics(assignment, energy, exact_assignment, exact_info, vms, servers, stats)

    out = config.RESULTS_DIR / a.size
    write_outputs(out, a.size, servers, vms, percentile, tinfo, need, qinfo, stats, assignment,
                  energy, exact_assignment, exact_info, gap)

    print(line)
    print(" Placement (server type: VMs)")
    for ph in energy["per_host"]:
        if ph["active"]:
            vm_list = [v for v in sorted(assignment) if assignment[v] == ph["host_id"]]
            print(f"  Server {ph['host_id']:>2} ({ph['type']}): {len(vm_list)} VMs, "
                  f"mean CPU {ph['mean_cpu_util_pct']:.1f}%, peak {ph['peak_cpu_util_pct']:.1f}%")
    print(line)
    print(" 24-hour result (SPEC power curves)")
    print(f"  Active servers   : {energy['active_hosts']} of {len(servers)}  "
          + ", ".join(f"{t}: {d['servers_on']}/{d['servers_total']}" for t, d in sorted(energy['by_type'].items())))
    print(f"  IT energy        : {energy['it_energy_kwh']:.4f} kWh")
    print(f"  Facility energy  : {energy['facility_energy_kwh']:.4f} kWh (PUE {config.PUE})")
    print(f"  Carbon           : {energy['co2_kg']:.4f} kg CO2 (EF {config.EMISSION_FACTOR})")
    print(f"  SLA violation    : {energy['sla_violation_pct']:.2f}% (limit {config.SLA_MAX_VIOLATION_PCT}%)")
    if gap.get("exact_available"):
        ex = exact_info["energy"]
        print(f"  Exact reference  : {ex['active_hosts']} servers, {ex['it_energy_kwh']:.4f} kWh "
              f"({exact_info['status']}, {exact_info['time_s']} s)")
        print(f"  Optimality gap   : {gap['gap_energy_pct']:+.2f}% energy, "
              f"{'matches' if gap['qa_matches_optimum'] else 'does not match'} the optimum, "
              f"pure and optimal: {gap['pure_and_optimal']}")
    print(line)
    print(f" Saved: {out}")
    print(f" Next : mvn -q compile exec:java \"-Dexec.args={a.size}\"   (in the simulation folder)")


if __name__ == "__main__":
    main()
