"""
Phase 2 energy, carbon and SLA model (heterogeneous servers).

    served_v(t)   = min(demand_v(t), size_v)
    u_h(t)        = sum of served CPU on h / C_cpu(h)
    P_h(t)        = 0 if h has no VM, else SPEC curve of h's type at u_h(t) (interpolated)
    E_IT (kWh)    = sum_h sum_t P_h(t) * 300 s / 3,600,000
    E_facility    = E_IT * PUE,   CO2 = E_facility * EF
"""
import numpy as np

from phase1 import config as p1config
from . import config


def evaluate_allocation(vms, assignment, servers):
    dt = p1config.WINDOW_SECONDS
    steps = p1config.WINDOWS_PER_DAY
    n = len(servers)
    served = np.zeros((n, steps))
    for vm in vms:
        served[assignment[vm.vm_id]] += vm.cpu_served

    active = np.zeros(n, dtype=bool)
    for h in assignment.values():
        active[h] = True
    util = np.array([np.clip(served[h] / s.cpu_capacity, 0, 1) for h, s in enumerate(servers)])
    power = np.array([s.power(util[h]) if active[h] else np.zeros(steps) for h, s in enumerate(servers)])

    host_kwh = power.sum(axis=1) * dt / 3_600_000
    e_it = float(host_kwh.sum())
    e_fac = e_it * config.PUE

    demand = np.array([v.cpu_demand for v in vms])
    size = np.array([v.cpu_size for v in vms])[:, None]
    total_demand = float(demand.sum())

    mem_util = np.zeros(n)
    for vm in vms:
        mem_util[assignment[vm.vm_id]] += vm.mem_size
    mem_util = np.array([mem_util[h] / s.mem_capacity for h, s in enumerate(servers)])
    memory_bound = [h for h in range(n) if active[h] and mem_util[h] > util[h].mean()]

    by_type = {}
    for h, s in enumerate(servers):
        t = by_type.setdefault(s.type_name, {"servers_on": 0, "servers_total": 0, "energy_kwh": 0.0})
        t["servers_total"] += 1
        t["servers_on"] += int(active[h])
        t["energy_kwh"] += float(host_kwh[h])

    return {
        "active_hosts": int(active.sum()),
        "it_energy_kwh": e_it,
        "facility_energy_kwh": e_fac,
        "overhead_energy_kwh": e_it * (config.PUE - 1),
        "co2_kg": e_fac * config.EMISSION_FACTOR,
        "sla_violation_pct": 100.0 * int((demand > size + 1e-12).sum()) / demand.size,
        "unmet_cpu_demand_pct": 100.0 * float(np.clip(demand - size, 0, None).sum()) / total_demand
        if total_demand > 0 else 0.0,
        "host_overload_windows": int(((served > np.array([s.cpu_capacity for s in servers])[:, None] + 1e-12)
                                      & active[:, None]).sum()),
        "memory_bound_servers": len(memory_bound),
        "by_type": by_type,
        "per_host": [{
            "host_id": h,
            "type": s.type_name,
            "active": bool(active[h]),
            "vms": int(sum(1 for v in assignment.values() if v == h)),
            "mean_cpu_util_pct": float(util[h].mean() * 100) if active[h] else 0.0,
            "peak_cpu_util_pct": float(util[h].max() * 100) if active[h] else 0.0,
            "mem_util_pct": float(mem_util[h] * 100),
            "memory_bound": h in memory_bound,
            "mean_power_w": float(power[h].mean()),
            "energy_kwh": float(host_kwh[h]),
        } for h, s in enumerate(servers)],
        "_util_matrix": util,
    }


def linear_energy_terms(vms, servers):
    """
    Energy pieces for the QUBO and the exact solver (linear fit of each SPEC curve), whole day:
      idle_cost[h]      = fit_idle_w(h) * T
      dyn_cost[v][h]    = fit_slope_w(h) * mean served CPU of v / C_cpu(h) * T
      e_ref             = energy with all servers ON (used to normalize the objective to ~0..1)
    """
    t = p1config.SIMULATION_SECONDS
    idle_cost = [s.fit_idle_w * t for s in servers]
    dyn_cost = {v.vm_id: [s.fit_slope_w * v.cpu_mean_served / s.cpu_capacity * t for s in servers]
                for v in vms}
    e_ref = sum(idle_cost) + sum(float(np.mean(c)) for c in dyn_cost.values())
    return idle_cost, dyn_cost, e_ref
