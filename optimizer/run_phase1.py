"""
GreenQuanta Phase 1 - static VM allocation on identical servers.

Flow:
  Google 2019 trace -> VMs -> QUBO -> simulated annealer -> validate/repair
  -> 24 h energy, carbon, SLA (Python) -> files for CloudSim (Phase1Simulation.java)

Run with the hardcoded development inputs:
    python run_phase1.py
Override any user input from the command line, e.g.:
    python run_phase1.py --servers 12 --sla 2 --instances-per-vm 3
"""
import argparse
from dataclasses import replace
from pathlib import Path

from phase1 import config
from phase1.energy_model import evaluate_allocation
from phase1.inputs import default_inputs
from phase1.outputs import write_cloudsim_inputs, write_results
from phase1.pipeline import optimize, prepare


def parse_args():
    d = default_inputs()
    p = argparse.ArgumentParser(description="GreenQuanta Phase 1 (static, homogeneous servers)")
    p.add_argument("--trace", default=d.trace_csv, help="CSV exported from BigQuery")
    p.add_argument("--servers", type=int, default=d.server.count, help="number of identical servers")
    p.add_argument("--server-cpu", type=float, default=d.server.cpu_capacity, help="CPU per server (NCU)")
    p.add_argument("--server-mem", type=float, default=d.server.mem_capacity, help="memory per server (NMU)")
    p.add_argument("--idle-power", type=float, default=d.server.idle_power_w, help="watts when idle")
    p.add_argument("--peak-power", type=float, default=d.server.peak_power_w, help="watts at 100%% CPU")
    p.add_argument("--safe-util", type=float, default=d.server.safe_utilization, help="0..1, default 0.85")
    p.add_argument("--sla", type=float, default=d.sla_max_violation_pct,
                   help="max %% of VM time that may be throttled")
    p.add_argument("--instances-per-vm", type=int, default=d.instances_per_vm)
    p.add_argument("--reads", type=int, default=config.NUM_READS, help="annealer reads")
    p.add_argument("--sweeps", type=int, default=config.NUM_SWEEPS, help="annealer sweeps per read")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--no-reduction", action="store_true",
                   help="give the QUBO all servers instead of only the candidate servers")
    a = p.parse_args()

    inputs = replace(
        d,
        trace_csv=a.trace,
        sla_max_violation_pct=a.sla,
        instances_per_vm=a.instances_per_vm,
        server=replace(d.server, count=a.servers, cpu_capacity=a.server_cpu, mem_capacity=a.server_mem,
                       idle_power_w=a.idle_power, peak_power_w=a.peak_power, safe_utilization=a.safe_util),
    )
    inputs.validate()
    if a.no_reduction:
        config.CANDIDATE_REDUCTION = False
    return inputs, a


def main():
    inputs, args = parse_args()
    server = inputs.server
    line = "=" * 62

    print(line)
    print(" GreenQuanta Phase 1 - static allocation, identical servers")
    print(line)

    vms, percentile, tinfo, lower_bound, binding = prepare(inputs)
    print(f"Trace        : {tinfo['collections']} jobs, {tinfo['instances']} instances, {tinfo['rows']} rows")
    print(f"VMs          : {len(vms)}  ({inputs.instances_per_vm} sibling instances per VM, "
          f"CPU size = P{percentile:g} of 24 h demand)")
    print(f"Servers      : {server.count} x (CPU {server.cpu_capacity} NCU, memory {server.mem_capacity} NMU), "
          f"safe limit {server.safe_utilization:.0%}")
    print(f"Lower bound  : at least {lower_bound:.2f} servers are needed ({binding} fills up first)")

    assignment, stats, qinfo = optimize(vms, server, lower_bound, args.reads, args.sweeps, args.seed)
    print(f"Candidates   : QUBO uses {qinfo['candidate_servers']} of {server.count} servers "
          f"(others stay OFF)")
    print(f"QUBO         : {qinfo['num_variables']} variables "
          f"({qinfo['assignment_variables']} placement, {qinfo['server_variables']} server, "
          f"{qinfo['slack_variables']} slack), {qinfo['num_interactions']} interactions")

    print(f"Annealer     : {stats['num_reads']} reads x {stats['num_sweeps']} sweeps in "
          f"{stats['anneal_time_s']:.2f} s, valid without repair: {stats['valid_reads_pct']:.1f}%")
    print(f"Best answer  : {stats['vms_placed_by_annealer']} VMs placed by the annealer, "
          f"{stats['vms_placed_by_repair']} by repair"
          + ("  (repair was needed)" if stats["repaired"] else "  (pure annealer result)"))

    energy = evaluate_allocation(vms, assignment, server)
    out = Path(config.RESULTS_DIR)
    write_results(out, inputs, vms, percentile, tinfo, qinfo, stats, assignment, energy)
    write_cloudsim_inputs(Path(config.CLOUDSIM_INPUT_DIR), inputs, vms, assignment)

    print(line)
    print(" Placement (server: VMs)")
    for h in range(server.count):
        on_h = [v for v in sorted(assignment) if assignment[v] == h]
        if on_h:
            ph = energy["per_host"][h]
            print(f"  Server {h:>2}: VMs {on_h}  mean CPU {ph['mean_cpu_util_pct']:.1f}%, "
                  f"peak {ph['peak_cpu_util_pct']:.1f}%")
    print(line)
    print(" 24-hour result (Python energy model)")
    print(f"  Active servers     : {energy['active_hosts']} of {server.count}")
    print(f"  IT energy          : {energy['it_energy_kwh']:.4f} kWh")
    print(f"  Facility energy    : {energy['facility_energy_kwh']:.4f} kWh  (PUE {config.PUE})")
    print(f"  Overhead energy    : {energy['overhead_energy_kwh']:.4f} kWh")
    print(f"  Carbon             : {energy['co2_kg']:.4f} kg CO2  (EF {config.EMISSION_FACTOR})")
    sla_ok = energy["sla_violation_pct"] <= inputs.sla_max_violation_pct
    print(f"  SLA violation      : {energy['sla_violation_pct']:.2f}% of VM time "
          f"(limit {inputs.sla_max_violation_pct}%) -> {'MET' if sla_ok else 'NOT MET'}")
    print(f"  Unmet CPU demand   : {energy['unmet_cpu_demand_pct']:.2f}%")
    print(f"  Host overloads     : {energy['host_overload_windows']} windows")
    print(line)
    print(f" Saved results to      {out}")
    print(f" Saved CloudSim input to {config.CLOUDSIM_INPUT_DIR}")
    print(" Next: run Phase1Simulation.java to replay the same 24 h in CloudSim.")


if __name__ == "__main__":
    main()
