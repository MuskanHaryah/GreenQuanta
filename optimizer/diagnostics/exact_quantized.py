"""Read-only comparison of real and QUBO-integer capacity MILPs; no sampling."""
import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.common import build_problem
from phase2 import config
from phase2.energy_model import evaluate_allocation, linear_energy_terms
from phase2.exact_solver import linear_cost, solve_exact
from phase2.qubo_builder import units
from phase2.validation import validate_allocation


SIZES = ("small_v2", "medium_v2", "large_v2")


def solve_exact_quantized(vms, servers, time_limit=None):
    """Use solve_exact's unmodified joule objective and links, with integer capacities."""
    n, m = len(vms), len(servers)
    idle_cost, dyn_cost, _ = linear_energy_terms(vms, servers)
    nx = n * m

    def xi(i, h):
        return i * m + h

    c = np.zeros(nx + m)
    for i, v in enumerate(vms):
        for h in range(m):
            c[xi(i, h)] = dyn_cost[v.vm_id][h]
    c[nx:] = idle_cost

    rows = n + 2 * m + nx
    a = lil_matrix((rows, nx + m))
    lb, ub = np.full(rows, -np.inf), np.zeros(rows)
    r = 0
    for i in range(n):
        for h in range(m):
            a[r, xi(i, h)] = 1
        lb[r] = ub[r] = 1
        r += 1
    for h, s in enumerate(servers):
        for i, v in enumerate(vms):
            a[r, xi(i, h)] = units(v.cpu_size, config.UNIT_NCU)
            a[r + 1, xi(i, h)] = units(v.mem_size, config.UNIT_NMU)
        a[r, nx + h] = -math.floor(s.cpu_limit / config.UNIT_NCU + 1e-9)
        a[r + 1, nx + h] = -math.floor(s.mem_limit / config.UNIT_NMU + 1e-9)
        r += 2
    for i in range(n):
        for h in range(m):
            a[r, xi(i, h)] = 1
            a[r, nx + h] = -1
            r += 1

    start = time.perf_counter()
    res = milp(c, constraints=LinearConstraint(a.tocsr(), lb, ub),
               integrality=np.ones(nx + m), bounds=Bounds(0, 1),
               options={"time_limit": time_limit or config.EXACT_TIME_LIMIT_S, "disp": False})
    seconds = time.perf_counter() - start
    if res.x is None:
        return None, {
            "status": res.message,
            "proven_optimal": False,
            "mip_gap": None,
            "objective_joule": None,
            "time_s": round(seconds, 3),
        }

    x = np.round(res.x).astype(int)
    assignment = {v.vm_id: int(np.argmax(x[i * m:(i + 1) * m])) for i, v in enumerate(vms)}
    return assignment, {
        "status": "optimal" if res.status == 0 else f"stopped ({res.message})",
        "proven_optimal": res.status == 0,
        "mip_gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
        "objective_joule": float(res.fun),
        "time_s": round(seconds, 3),
    }


def _print_solution(label, assignment, info, vms, servers):
    print(f"{label}: status={info['status']}")
    print(f"  proven_optimal={info.get('proven_optimal', False)} "
          f"mip_gap={info.get('mip_gap')} time_s={info['time_s']}")
    if assignment is None:
        print("  assignment unavailable; metrics and validators unavailable")
        return None
    active = sorted(set(assignment.values()))
    cost = linear_cost(assignment, vms, servers)
    energy = evaluate_allocation(vms, assignment, servers)["it_energy_kwh"]
    print(f"  servers_on={len(active)} host_ids={[servers[h].server_id for h in active]}")
    print(f"  linear_cost_J={cost:.9f} curve_it_energy_kWh={energy:.12f}")
    print(f"  real_size_validator={validate_allocation(vms, assignment, servers)}")
    print("  quantized_validator="
          f"{validate_allocation(vms, assignment, servers, quantized_unit=config.UNIT_NCU)}")
    return len(active), cost, energy


def _print_active_capacity(assignment, info, vms, servers):
    if assignment is None:
        print("Real-MILP active-set quantization: unavailable (no assignment)")
        return
    active = sorted(set(assignment.values()))
    print("Real-MILP active-set quantization (positive deficit means excess units):")
    for resource, unit in (("cpu", config.UNIT_NCU), ("mem", config.UNIT_NMU)):
        vm_units = {v.vm_id: units(getattr(v, f"{resource}_size"), unit) for v in vms}
        capacities = {h: math.floor(getattr(servers[h], f"{resource}_limit") / unit + 1e-9)
                      for h in active}
        loads = {h: sum(vm_units[v.vm_id] for v in vms if assignment[v.vm_id] == h)
                 for h in active}
        total, capacity = sum(vm_units.values()), sum(capacities.values())
        print(f"  {resource.upper()}: total_units={total} active_K={capacity} "
              f"total_minus_K={total - capacity:+d} "
              f"aggregate_deficit_units={max(0, total - capacity)} "
              f"placement_excess_units={sum(max(0, loads[h] - capacities[h]) for h in active)}")
        for h in active:
            print(f"    host_id={servers[h].server_id} load_units={loads[h]} K={capacities[h]} "
                  f"load_minus_K={loads[h] - capacities[h]:+d}")
    valid = validate_allocation(vms, assignment, servers, quantized_unit=config.UNIT_NCU)["valid"]
    print(f"  real-MILP assignment quantized-feasible: {'yes' if valid else 'no'} "
          f"(proven_optimal={info.get('proven_optimal', False)})")


def _percent_difference(value, reference):
    if reference == 0:
        return "0.000000000" if value == 0 else "undefined (zero real reference)"
    return f"{100.0 * (value - reference) / reference:+.9f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--time-limit", type=float, default=config.EXACT_TIME_LIMIT_S)
    args = parser.parse_args()
    if args.time_limit <= 0 or not math.isfinite(args.time_limit):
        parser.error("--time-limit must be finite and positive")
    if config.UNIT_NCU != config.UNIT_NMU:
        parser.error("the shared quantized validator requires equal CPU and RAM units")
    print(f"units: CPU={config.UNIT_NCU} RAM={config.UNIT_NMU}; "
          f"time_limit_s={args.time_limit}; objective=linear joules, no tie-break")
    for size in args.sizes:
        print(f"\n=== {size} ===", flush=True)
        vms, servers, _, _ = build_problem(size)
        print(f"vms={len(vms)} servers={len(servers)}", flush=True)
        real_assignment, real_info = solve_exact(vms, servers, args.time_limit)
        real = _print_solution("real MILP", real_assignment, real_info, vms, servers)
        sys.stdout.flush()
        quantized_assignment, quantized_info = solve_exact_quantized(vms, servers, args.time_limit)
        quantized = _print_solution("quantized MILP", quantized_assignment, quantized_info, vms, servers)
        if real is None or quantized is None:
            print("quantized minus real: unavailable (missing assignment)")
        else:
            print(f"quantized minus real: servers={quantized[0] - real[0]:+d} "
                  f"linear_cost_pct={_percent_difference(quantized[1], real[1])} "
                  f"curve_energy_pct={_percent_difference(quantized[2], real[2])}")
        _print_active_capacity(real_assignment, real_info, vms, servers)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
