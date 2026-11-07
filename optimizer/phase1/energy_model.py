"""
Energy, carbon and SLA model (Formulation Research document, Phase 1 version).

    host utilization  u_h(t) = (CPU served to VMs on h) / C_cpu
    host power        P_h(t) = 0                                   if h is OFF
                             = P_idle + (P_peak - P_idle) * u_h(t)  if h is ON
    IT energy (kWh)   E_IT   = sum_h sum_t P_h(t) * dt / 3,600,000
    facility energy          = E_IT * PUE
    overhead energy          = E_IT * (PUE - 1)
    carbon (kg CO2)          = facility energy * EF
A host is ON when at least one VM is placed on it; an empty host is switched off (0 W).
"""
import numpy as np

from . import config


def host_power(util, server):
    """Linear power model, util between 0 and 1 (array or number)."""
    return server.idle_power_w + (server.peak_power_w - server.idle_power_w) * util


def evaluate_allocation(vms, assignment, server):
    """
    Replay the real 24 h trace for a placement and return energy, carbon and SLA numbers.
    assignment: dict vm_id -> host index
    """
    dt = config.WINDOW_SECONDS
    steps = config.WINDOWS_PER_DAY
    served = np.zeros((server.count, steps))
    for vm in vms:
        served[assignment[vm.vm_id]] += vm.cpu_served

    active = np.array([any(assignment[v.vm_id] == h for v in vms) for h in range(server.count)])
    util = np.clip(served / server.cpu_capacity, 0.0, 1.0)
    power = np.where(active[:, None], host_power(util, server), 0.0)

    host_energy_kwh = power.sum(axis=1) * dt / 3_600_000
    e_it = float(host_energy_kwh.sum())
    e_facility = e_it * config.PUE

    # SLA: a VM window is "violated" when real demand is above the VM's reserved size
    demand = np.array([v.cpu_demand for v in vms])
    size = np.array([v.cpu_size for v in vms])[:, None]
    throttled_windows = int((demand > size + 1e-12).sum())
    total_demand = float(demand.sum())
    unmet = float(np.clip(demand - size, 0, None).sum())

    # host overload check (should be 0: sizes are packed below the safe limit)
    overloaded = int(((served > server.cpu_capacity + 1e-12) & active[:, None]).sum())

    return {
        "active_hosts": int(active.sum()),
        "it_energy_kwh": e_it,
        "facility_energy_kwh": e_facility,
        "overhead_energy_kwh": e_it * (config.PUE - 1),
        "co2_kg": e_facility * config.EMISSION_FACTOR,
        "dcie_pct": 100.0 / config.PUE,
        "cue_kg_per_kwh_it": config.PUE * config.EMISSION_FACTOR,
        "sla_violation_pct": 100.0 * throttled_windows / demand.size,
        "unmet_cpu_demand_pct": 100.0 * unmet / total_demand if total_demand > 0 else 0.0,
        "host_overload_windows": overloaded,
        "per_host": [{
            "host_id": h,
            "active": bool(active[h]),
            "vms": int(sum(assignment[v.vm_id] == h for v in vms)),
            "mean_cpu_util_pct": float(util[h].mean() * 100) if active[h] else 0.0,
            "peak_cpu_util_pct": float(util[h].max() * 100) if active[h] else 0.0,
            "mean_power_w": float(power[h].mean()),
            "energy_kwh": float(host_energy_kwh[h]),
        } for h in range(server.count)],
        "_util_matrix": util,        # used for plots, not written to JSON
        "_power_matrix": power,
    }


def qubo_energy_terms(vms, server):
    """
    Energy pieces the QUBO objective needs, for the whole 24 h:
      idle_cost      = energy of keeping one server ON at idle
      dynamic_cost[v]= extra energy VM v causes on any server (same on every server,
                       because servers are identical and the model is linear)
      e_ref          = energy with ALL servers ON (no consolidation) -> used to normalize
    """
    t = config.SIMULATION_SECONDS
    idle_cost = server.idle_power_w * t
    slope = (server.peak_power_w - server.idle_power_w) / server.cpu_capacity
    dynamic_cost = {v.vm_id: slope * v.cpu_mean_served * t for v in vms}
    e_ref = server.count * idle_cost + sum(dynamic_cost.values())
    return idle_cost, dynamic_cost, e_ref
