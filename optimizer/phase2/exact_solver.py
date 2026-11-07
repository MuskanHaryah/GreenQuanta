"""
Exact reference solver: the same placement problem as a Mixed-Integer Linear Program,
solved with SciPy's HiGHS (scipy.optimize.milp). Used ONLY as a yardstick (optimality gap),
not as a competing method.

    minimize   sum_h y_h * idle_cost_h + sum_{v,h} x_vh * dyn_cost_vh     (same linear energy as the QUBO)
    subject to sum_h x_vh = 1                       every VM on one server
               sum_v cpu_v x_vh <= safe * C_h * y_h  CPU (real, un-rounded sizes)
               sum_v mem_v x_vh <= safe * M_h * y_h  memory
               x_vh <= y_h                           no VM on a switched-off server
"""
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from . import config
from .energy_model import linear_energy_terms


def linear_cost(assignment, vms, servers):
    """Linear (fitted) energy of a placement, in joules. Same objective as the MILP and the QUBO."""
    idle_cost, dyn_cost, _ = linear_energy_terms(vms, servers)
    on = set(assignment.values())
    return sum(idle_cost[h] for h in on) + sum(dyn_cost[v.vm_id][assignment[v.vm_id]] for v in vms)


def solve_exact(vms, servers, time_limit=None):
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
            a[r, xi(i, h)] = v.cpu_size
            a[r + 1, xi(i, h)] = v.mem_size
        a[r, nx + h] = -s.cpu_limit
        a[r + 1, nx + h] = -s.mem_limit
        r += 2
    for i in range(n):
        for h in range(m):
            a[r, xi(i, h)] = 1
            a[r, nx + h] = -1
            r += 1

    start = time.perf_counter()
    res = milp(c, constraints=LinearConstraint(a.tocsr(), lb, ub), integrality=np.ones(nx + m),
               bounds=Bounds(0, 1),
               options={"time_limit": time_limit or config.EXACT_TIME_LIMIT_S, "disp": False})
    seconds = time.perf_counter() - start
    if res.x is None:
        return None, {"status": res.message, "time_s": round(seconds, 3)}

    x = np.round(res.x).astype(int)
    assignment = {v.vm_id: int(np.argmax(x[i * m:(i + 1) * m])) for i, v in enumerate(vms)}
    return assignment, {
        "status": "optimal" if res.status == 0 else f"stopped ({res.message})",
        "proven_optimal": res.status == 0,
        "mip_gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
        "objective_joule": float(res.fun),
        "time_s": round(seconds, 3),
    }
