"""Phase 2 pipeline shared by run_phase2.py and experiment_phase2.py."""
import json

import pandas as pd

from phase1 import config as p1config
from phase1.trace_loader import load_trace, trace_summary
from phase1.vm_builder import build_vms, vms_table
from . import config
from .catalog import catalog_table
from .energy_model import evaluate_allocation
from .exact_solver import linear_cost, solve_exact
from .qubo_builder import build_qubo
from .solver import efficiency_key, solve


def prepare(trace_csv, servers, sla_pct=None, instances_per_vm=None):
    sla_pct = config.SLA_MAX_VIOLATION_PCT if sla_pct is None else sla_pct
    instances_per_vm = instances_per_vm or config.INSTANCES_PER_VM
    trace = load_trace(trace_csv)
    vms, percentile = build_vms(trace, instances_per_vm, sla_pct)
    cpu_need = sum(v.cpu_size for v in vms)
    mem_need = sum(v.mem_size for v in vms)
    if cpu_need > sum(s.cpu_limit for s in servers) or mem_need > sum(s.mem_limit for s in servers):
        raise SystemExit(f"Workload needs {cpu_need:.3f} NCU / {mem_need:.3f} NMU but the servers' safe "
                         f"capacity is {sum(s.cpu_limit for s in servers):.3f} / "
                         f"{sum(s.mem_limit for s in servers):.3f}. Add servers.")
    return vms, percentile, trace_summary(trace), {"cpu_ncu": cpu_need, "mem_nmu": mem_need}


def candidate_servers(vms, servers):
    """Optional rule: most efficient servers until capacity >= (1 + margin) x demand, plus one."""
    if not config.CANDIDATE_REDUCTION:
        return list(range(len(servers)))
    need_cpu = sum(v.cpu_size for v in vms) * (1 + config.CANDIDATE_MARGIN)
    need_mem = sum(v.mem_size for v in vms) * (1 + config.CANDIDATE_MARGIN)
    chosen, cpu, mem = [], 0.0, 0.0
    for h in sorted(range(len(servers)), key=lambda h: efficiency_key(servers[h])):
        if cpu >= need_cpu and mem >= need_mem:
            break
        chosen.append(h)
        cpu += servers[h].cpu_limit
        mem += servers[h].mem_limit
    order = sorted(range(len(servers)), key=lambda h: efficiency_key(servers[h]))
    extra = [h for h in order if h not in chosen][:1]
    return sorted(chosen + extra)


def optimize(vms, servers, lambda_config, reads=None, sweeps=None, seed=None):
    idx = candidate_servers(vms, servers)
    sub = [servers[h] for h in idx]
    bqm, qinfo = build_qubo(vms, sub, lambda_config)
    assignment_sub, stats = solve(bqm, vms, sub, reads, sweeps, seed)
    assignment = {v: idx[h] for v, h in assignment_sub.items()}
    qinfo["candidate_servers"] = len(idx)
    qinfo["user_servers"] = len(servers)
    return assignment, stats, qinfo


def exact_reference(vms, servers, time_limit=None):
    assignment, info = solve_exact(vms, servers, time_limit)
    if assignment is not None:
        info["energy"] = {k: v for k, v in evaluate_allocation(vms, assignment, servers).items()
                          if not k.startswith("_") and k != "per_host"}
    return assignment, info


def gap_metrics(qa_assignment, qa_energy, exact_assignment, exact_info, vms, servers, stats):
    """Optimality gap of the QA answer against the exact reference."""
    if exact_assignment is None:
        return {"exact_available": False}
    qa_lin = linear_cost(qa_assignment, vms, servers)
    ex_lin = exact_info["objective_joule"]
    ex_kwh = exact_info["energy"]["it_energy_kwh"]
    optimal = qa_lin <= ex_lin * (1 + 1e-3)
    return {
        "exact_available": True,
        "exact_proven_optimal": exact_info.get("proven_optimal", False),
        "gap_linear_pct": 100.0 * (qa_lin - ex_lin) / ex_lin,
        "gap_energy_pct": 100.0 * (qa_energy["it_energy_kwh"] - ex_kwh) / ex_kwh,
        "qa_matches_optimum": bool(optimal),
        "pure_and_optimal": bool(optimal and not stats["repaired"]),
    }


def write_outputs(out_dir, size, servers, vms, percentile, tinfo, need, qinfo, stats, assignment,
                  energy, exact_assignment, exact_info, gap):
    out_dir.mkdir(parents=True, exist_ok=True)
    cs_dir = out_dir / "cloudsim_input"
    cs_dir.mkdir(exist_ok=True)

    vms_table(vms).to_csv(out_dir / "vms.csv", index=False)
    pd.DataFrame(catalog_table(servers)).to_csv(out_dir / "server_types.csv", index=False)
    pd.DataFrame({"vm_id": sorted(assignment), "host_id": [assignment[v] for v in sorted(assignment)],
                  "host_type": [servers[assignment[v]].type_name for v in sorted(assignment)]}) \
        .to_csv(out_dir / "assignment_qa.csv", index=False)
    if exact_assignment:
        pd.DataFrame({"vm_id": sorted(exact_assignment),
                      "host_id": [exact_assignment[v] for v in sorted(exact_assignment)]}) \
            .to_csv(out_dir / "assignment_exact.csv", index=False)
    pd.DataFrame(energy["per_host"]).to_csv(out_dir / "energy_hosts_python.csv", index=False)

    summary = {
        "phase": "2 - static allocation, heterogeneous servers",
        "size": size,
        "servers": {t["type"]: t["count"] for t in catalog_table(servers)},
        "server_types": catalog_table(servers),
        "constants": {"pue": config.PUE, "emission_factor": config.EMISSION_FACTOR,
                      "sla_max_violation_pct": config.SLA_MAX_VIOLATION_PCT,
                      "instances_per_vm": config.INSTANCES_PER_VM, "unit_ncu": config.UNIT_NCU,
                      "unit_nmu": config.UNIT_NMU, "candidate_reduction": config.CANDIDATE_REDUCTION},
        "workload": {**tinfo, "vms": len(vms), "vm_cpu_size_percentile": percentile,
                     "total_cpu_ncu": round(need["cpu_ncu"], 6), "total_mem_nmu": round(need["mem_nmu"], 6)},
        "qubo": {k: v for k, v in qinfo.items() if not k.startswith("capacity_units")},
        "solver": stats,
        "result": {k: v for k, v in energy.items() if not k.startswith("_") and k != "per_host"},
        "exact_reference": exact_info,
        "optimality": gap,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- CloudSim inputs (plain CSV)
    settings = {"window_seconds": p1config.WINDOW_SECONDS, "windows": p1config.WINDOWS_PER_DAY,
                "pue": config.PUE, "emission_factor": config.EMISSION_FACTOR}
    pd.DataFrame({"key": list(settings), "value": list(settings.values())}).to_csv(cs_dir / "settings.csv", index=False)
    rows = []
    for s in servers:
        row = {"host_id": s.server_id, "type": s.type_name, "cpu_ncu": s.cpu_capacity, "mem_nmu": s.mem_capacity}
        row.update({f"p{10 * i}": p for i, p in enumerate(s.power_w)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(cs_dir / "servers.csv", index=False)
    pd.DataFrame({"vm_id": [v.vm_id for v in vms], "cpu_size_ncu": [v.cpu_size for v in vms],
                  "mem_size_nmu": [v.mem_size for v in vms],
                  "host_id": [assignment[v.vm_id] for v in vms]}).to_csv(cs_dir / "vms_assignment.csv", index=False)
    demand = pd.DataFrame({f"vm_{v.vm_id}": v.cpu_demand for v in vms})
    demand.insert(0, "window_idx", range(p1config.WINDOWS_PER_DAY))
    demand.to_csv(cs_dir / "vm_cpu_demand.csv", index=False, float_format="%.8f")
    return summary
