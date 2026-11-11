"""Step 1.2: Stage-1 cost C(y) over the server on/off bits y (proof: docs/reports/step1/1.2_proof.md).

C(y) = sum_{h on} c_y[h] + sum_v min_{h on} c_x[v,h]  if y passes the aggregate checks, else M = lambda.
c_y, c_x are the linear coefficients of the lambda=0 slack BQM (build_qubo_custom), so tie-break and
normalisation are Step 0's. lambda = factor * U (Step 1.1, reference_u). Index convention for
all 2^m states: y_h = bit h of the integer index (h = server position in `servers`).
"""
import math

import numpy as np

from diagnostics.r3c_build import build_qubo_custom
from diagnostics.s0_exact import reference_u
from phase1.qubo_builder import x_var, y_var


def stage1_data(vms, servers, unit, factor):
    """Coefficients, quantized demands (n,2) and capacities (m,2), U and lambda = M = factor * U."""
    if factor <= 1:
        raise ValueError("factor must be > 1 (Theorem 1.1)")
    energy, info = build_qubo_custom(vms, servers, 0, 0, 0, 0, None, unit, "slack")
    hosts = range(len(servers))
    u = reference_u(energy, vms, servers)
    return {"c_y": np.array([energy.linear[y_var(h)] for h in hosts]),
            "c_x": np.array([[energy.linear[x_var(v.vm_id, h)] for h in hosts] for v in vms]),
            "demand": np.array([[info["cpu_units"][v.vm_id], info["mem_units"][v.vm_id]] for v in vms]),
            "capacity": np.array([info["k_cpu"], info["k_mem"]]).T,
            "U": u, "lam": factor * u, "M": factor * u, "unit": unit, "factor": factor}


def bits(index, m):
    return np.array([(index >> h) & 1 for h in range(m)], dtype=bool)


def aggregate_ok(y, data):
    """Necessary for a quantized-feasible placement (Prop. 1): total capacity and largest VM, per resource."""
    y = np.asarray(y, dtype=bool)
    if not y.any():
        return False
    capacity = data["capacity"][y]
    demand = data["demand"]
    return bool(np.all(capacity.sum(axis=0) >= demand.sum(axis=0))
                and np.all(demand.max(axis=0) <= capacity.max(axis=0)))


def stage1_cost(y, data):
    y = np.asarray(y, dtype=bool)
    if not aggregate_ok(y, data):
        return float(data["M"])
    return math.fsum(data["c_y"][y]) + math.fsum(data["c_x"][:, y].min(axis=1))


def all_costs(data):
    """C over all 2^m states; entry i is C(bits(i, m))."""
    m = len(data["c_y"])
    return np.array([stage1_cost(bits(i, m), data) for i in range(2 ** m)])
