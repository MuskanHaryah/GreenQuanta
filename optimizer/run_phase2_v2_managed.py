"""Run the Phase 2 v2 pipeline in a new, provenance-managed directory."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from run_phase2_v2 import (
    build_args,
    evaluate_allocation,
    exact_reference,
    gap_metrics_v2,
    optimize,
    prepare,
    setup,
    write_outputs,
)
from phase2 import manifest, validation


def build_run_config(args) -> dict:
    """Return result-affecting CLI options, including the sampler seed."""
    names = (
        "size", "trace", "encoding", "lambda_config", "center", "unit",
        "reads", "sweeps", "seed", "descent", "candidate", "no_exact",
        "auto_factor", "servers",
    )
    return {name: getattr(args, name) for name in names}


def managed_args(argv=None):
    """Consume managed output options and reuse the original CLI parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runs-root", default="results/runs",
                        help="parent directory for new managed runs")
    parser.add_argument("--log", default="results/experiment_log.csv",
                        help="append-only experiment CSV")
    extra, remaining = parser.parse_known_args(argv)
    if "--help" in remaining or "-h" in remaining:
        print("Managed output options:")
        parser.print_help()
    args = build_args(remaining)
    args.runs_root = extra.runs_root
    args.log = extra.log
    return args


def main(argv=None):
    args = managed_args(argv)
    repo_dir = Path(__file__).resolve().parents[1]
    protected = (repo_dir / "results" / "phase2").resolve()
    for path in (args.runs_root, args.log):
        if Path(path).resolve().is_relative_to(protected):
            raise SystemExit("Managed outputs must not write under results/phase2/.")

    trace, servers, mix = setup(args)
    line = "=" * 72
    print(line)
    print(f" GreenQuanta Phase 2 v2 ({args.size}) - encoding '{args.encoding}', "
          f"lambda config {args.lambda_config}")
    print(line)
    vms, percentile, tinfo, need = prepare(trace, servers)
    print(f"Trace     : {tinfo['collections']} jobs, {tinfo['instances']} instances -> {len(vms)} VMs")
    print(f"Demand    : {need['cpu_ncu']:.4f} NCU, {need['mem_nmu']:.4f} NMU")
    print("Servers   : " + ", ".join(f"{t} x{c}" for t, c in mix.items()) +
          f"  ({len(servers)} total)")

    workload_obj = [
        {"vm_id": int(vm.vm_id), "cpu_size": float(vm.cpu_size),
         "mem_size": float(vm.mem_size),
         "cpu_demand": [float(value) for value in vm.cpu_demand],
         "mem_demand": [float(value) for value in vm.mem_demand]}
        for vm in vms
    ]
    # This pipeline has no workload seed: trace-derived VM construction is deterministic.
    run_manifest = manifest.build_manifest(
        method="phase2_v2", config=build_run_config(args), sampler_seed=args.seed,
        workload_seed=0, dataset_path=trace, workload_obj=workload_obj,
        repo_dir=repo_dir,
        package_names=("pandas", "numpy", "scipy", "dimod", "dwave-samplers", "pytest"),
    )
    try:
        out = manifest.write_manifest(args.runs_root, run_manifest).parent
    except FileExistsError as exc:
        raise SystemExit(f"Run already exists; refusing to overwrite: {exc}") from exc

    final, pure, stats, qinfo = optimize(
        vms, servers, args.lambda_config, args.encoding, args.reads, args.sweeps,
        args.seed, args.descent, args.center,
    )
    lam = qinfo["lambdas"]
    print(f"Lambdas   : assign {lam['assign']:.4g}, cpu {lam['cpu']:.4g}, "
          f"ram {lam['ram']:.4g}, link {lam['link']:.4g}")
    print(f"QUBO      : {qinfo['num_variables']} variables ({qinfo['slack_variables']} slack), "
          f"bias range {qinfo['bias_range'][0]:.2e} .. {qinfo['bias_range'][1]:.2e}")
    print(f"Annealer  : {stats['anneal_time_s']:.2f} s (+ descent {stats['descent_time_s']:.2f} s); "
          f"valid reads: raw {stats['valid_reads_raw_pct']:.1f}%, "
          f"after descent {stats['valid_reads_pct']:.1f}%; "
          f"VMs placed per read {stats['mean_vms_placed_pct']:.1f}%")

    energy = evaluate_allocation(vms, final, servers)
    ex_a, ex_i = (None, {"skipped": True}) if args.no_exact else exact_reference(vms, servers)
    gap = gap_metrics_v2(final, pure, vms, servers, ex_a, ex_i, stats)
    checks = {
        "final": validation.validate_allocation(vms, final, servers),
        "pure": validation.validate_allocation(vms, pure, servers) if pure is not None else None,
        "exact": validation.validate_allocation(vms, ex_a, servers) if ex_a is not None else None,
    }
    status = "ok" if checks["final"]["valid"] else "failed"
    summary = write_outputs(
        out, args.size, servers, vms, percentile, tinfo, need, qinfo, stats, final,
        energy, ex_a, ex_i, gap,
    )
    summary["version"] = "v2"
    summary["validation"] = checks
    summary["status"] = status
    if pure:
        pd.DataFrame({"vm_id": sorted(pure), "host_id": [pure[v] for v in sorted(pure)]}).to_csv(
            out / "assignment_annealer_only.csv", index=False,
        )
        summary["annealer_only_result"] = {
            k: v for k, v in evaluate_allocation(vms, pure, servers).items()
            if not k.startswith("_") and k != "per_host"
        }
    with (out / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    metrics = {
        "servers_on": energy["active_hosts"],
        "it_energy_kwh": energy["it_energy_kwh"],
        "sla_violation_pct": energy["sla_violation_pct"],
        "final_gap_energy_pct": gap.get("final_gap_energy_pct"),
        "annealer_only_found": gap["annealer_only_found"],
        "anneal_time_s": stats["anneal_time_s"],
        "status": status,
    }
    manifest.append_experiment_log(args.log, {
        "run_id": run_manifest["run_id"], "git_commit": run_manifest["code_revision"],
        "config_hash": run_manifest["config_hash"], "sampler_seed": args.seed,
        "workload_seed": run_manifest["workload_seed"], "method": run_manifest["method"],
        "metrics_json": json.dumps(metrics),
        "gate_result": "pass" if status == "ok" else "fail",
        "timestamp": run_manifest["timestamp"],
    })

    print(line)
    print(" Final answer (repair allowed)" if status == "ok" else
          " FAILED: final allocation did not pass validation")
    print(f"  Servers on       : {energy['active_hosts']}  " +
          ", ".join(f"{t}: {d['servers_on']}/{d['servers_total']}"
                    for t, d in sorted(energy['by_type'].items())))
    print(f"  VMs by annealer  : {stats['vms_placed_by_annealer']} of {len(vms)} "
          f"(repair {stats['vms_placed_by_repair']})")
    print(f"  IT / facility    : {energy['it_energy_kwh']:.4f} / "
          f"{energy['facility_energy_kwh']:.4f} kWh, CO2 {energy['co2_kg']:.4f} kg")
    print(f"  SLA / memory     : SLA {energy['sla_violation_pct']:.2f}%, "
          f"memory-bound servers {energy['memory_bound_servers']}")
    if gap.get("exact_available"):
        ex = ex_i["energy"]
        print(f"  Exact optimum    : {ex['active_hosts']} servers, "
              f"{ex['it_energy_kwh']:.4f} kWh ({ex_i['status']})")
        print(f"  Final gap        : {gap['final_gap_energy_pct']:+.2f}%")
        if gap["annealer_only_found"]:
            print(f"  Annealer only    : {gap['annealer_only_servers']} servers, gap "
                  f"{gap['annealer_only_gap_energy_pct']:+.2f}%, "
                  f"pure and optimal: {gap['pure_and_optimal']}")
        else:
            print("  Annealer only    : no valid read without repair")
    print(f"  Status           : {status}")
    print(line)
    print(f" Run directory: {out}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
