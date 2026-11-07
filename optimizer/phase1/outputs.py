"""Write Phase 1 results (for us) and CloudSim input files (for Phase1Simulation.java)."""
import json

import pandas as pd

from . import config
from .vm_builder import vms_table


def write_results(out_dir, inputs, vms, percentile, trace_info, qubo_info, solver_stats, assignment, energy):
    out_dir.mkdir(parents=True, exist_ok=True)

    vms_table(vms).to_csv(out_dir / "vms.csv", index=False)
    pd.DataFrame({"vm_id": sorted(assignment), "host_id": [assignment[v] for v in sorted(assignment)]}) \
        .to_csv(out_dir / "assignment_qa.csv", index=False)
    pd.DataFrame(energy["per_host"]).to_csv(out_dir / "energy_hosts_python.csv", index=False)

    summary = {
        "phase": "1 - static allocation, homogeneous servers",
        "inputs": inputs.to_dict(),
        "constants": {
            "pue": config.PUE,
            "emission_factor_kg_per_kwh": config.EMISSION_FACTOR,
            "simulation_hours": config.SIMULATION_SECONDS / 3600,
            "window_seconds": config.WINDOW_SECONDS,
            "lambda": {"assign": config.LAMBDA_ASSIGN, "cpu": config.LAMBDA_CPU,
                       "ram": config.LAMBDA_RAM, "link": config.LAMBDA_LINK},
            "resource_resolution": config.RESOURCE_RESOLUTION,
            "tie_break": config.TIE_BREAK,
        },
        "workload": {**trace_info, "vms": len(vms), "vm_cpu_size_percentile": percentile},
        "qubo": {k: v for k, v in qubo_info.items() if k not in ("cpu_units", "mem_units")},
        "solver": solver_stats,
        "result": {k: v for k, v in energy.items() if not k.startswith("_") and k != "per_host"},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def write_cloudsim_inputs(cs_dir, inputs, vms, assignment):
    """Plain CSV files so the Java side needs no JSON library."""
    cs_dir.mkdir(parents=True, exist_ok=True)
    s = inputs.server
    settings = {
        "num_servers": s.count,
        "cpu_capacity_ncu": s.cpu_capacity,
        "mem_capacity_nmu": s.mem_capacity,
        "idle_power_w": s.idle_power_w,
        "peak_power_w": s.peak_power_w,
        "window_seconds": config.WINDOW_SECONDS,
        "windows": config.WINDOWS_PER_DAY,
        "pue": config.PUE,
        "emission_factor": config.EMISSION_FACTOR,
    }
    pd.DataFrame({"key": list(settings), "value": list(settings.values())}) \
        .to_csv(cs_dir / "settings.csv", index=False)

    pd.DataFrame({
        "vm_id": [v.vm_id for v in vms],
        "cpu_size_ncu": [v.cpu_size for v in vms],
        "mem_size_nmu": [v.mem_size for v in vms],
        "host_id": [assignment[v.vm_id] for v in vms],
    }).to_csv(cs_dir / "vms_assignment.csv", index=False)

    demand = pd.DataFrame({f"vm_{v.vm_id}": v.cpu_demand for v in vms})
    demand.insert(0, "window_idx", range(config.WINDOWS_PER_DAY))
    demand.to_csv(cs_dir / "vm_cpu_demand.csv", index=False, float_format="%.8f")
