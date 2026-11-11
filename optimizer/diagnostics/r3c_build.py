"""Local explicit-parameter adapter; no shared module/config mutation."""
import math

import dimod

from diagnostics.common import build_problem
from phase1.qubo_builder import add_squared_penalty, x_var, y_var
from phase2 import config
from phase2.energy_model import linear_energy_terms
from phase2.qubo_builder import units
from phase2.qubo_builder_v2 import ENCODINGS, add_capacity_rule


def build_qubo_custom(vms, servers, lambda_assign, lambda_cpu, lambda_ram,
                      lambda_link, center, unit, encoding="unbalanced"):
    """build_qubo_v2 assembly with explicit weights/units; encoding 'unbalanced' or 'slack'.

    center is used only by 'unbalanced' (pass None for 'slack')."""
    lam = dict(assign=lambda_assign, cpu=lambda_cpu, ram=lambda_ram, link=lambda_link)
    if encoding not in ("unbalanced", "slack"):
        raise ValueError("encoding must be 'unbalanced' or 'slack'")
    if not math.isfinite(unit) or unit <= 0:
        raise ValueError("unit must be finite and positive")
    if encoding == "unbalanced" and (center is None or not math.isfinite(center)):
        raise ValueError("finite center required for unbalanced")
    if any(not math.isfinite(v) or v < 0 for v in lam.values()):
        raise ValueError("finite nonnegative lambdas required")
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    hosts = range(len(servers))
    cpu_units = {v.vm_id: units(v.cpu_size, unit) for v in vms}
    mem_units = {v.vm_id: units(v.mem_size, unit) for v in vms}
    k_cpu = [int(math.floor(s.cpu_limit / unit + 1e-9)) for s in servers]
    k_mem = [int(math.floor(s.mem_limit / unit + 1e-9)) for s in servers]
    if min(k_cpu + k_mem) <= 0:
        raise ValueError("positive quantized capacities required")
    idle_cost, dyn_cost, e_ref = linear_energy_terms(vms, servers)
    type_index, seen = [], {}
    for s in servers:
        type_index.append(seen.get(s.type_name, 0))
        seen[s.type_name] = type_index[-1] + 1
    for h, s in enumerate(servers):
        tie = config.TIE_BREAK * type_index[h] / max(1, seen[s.type_name] - 1)
        bqm.add_linear(y_var(h), idle_cost[h] * (1 + tie) / e_ref)
    for v in vms:
        for h in hosts:
            bqm.add_linear(x_var(v.vm_id, h), dyn_cost[v.vm_id][h] / e_ref)
    for v in vms:
        add_squared_penalty(bqm, [(x_var(v.vm_id, h), 1) for h in hosts], -1, lam["assign"])
    slack_vars = 0
    for h in hosts:
        slack_vars += add_capacity_rule(bqm, h, "sc", cpu_units, k_cpu[h], lam["cpu"], vms, encoding, center)
        slack_vars += add_capacity_rule(bqm, h, "sm", mem_units, k_mem[h], lam["ram"], vms, encoding, center)
    for v in vms:
        for h in hosts:
            bqm.add_linear(x_var(v.vm_id, h), lam["link"])
            bqm.add_quadratic(x_var(v.vm_id, h), y_var(h), -lam["link"])
    biases = [abs(b) for b in bqm.linear.values()] + [abs(b) for b in bqm.quadratic.values()]
    biases = [b for b in biases if b > 0]
    return bqm, {
        "encoding": encoding, "encoding_description": ENCODINGS[encoding],
        "lambda_config": "custom", "lambda_description": repr(lam),
        "unbalanced_center": center if encoding == "unbalanced" else None, "lambdas": lam,
        "num_variables": len(bqm.variables), "num_interactions": len(bqm.quadratic),
        "assignment_variables": len(vms) * len(servers), "server_variables": len(servers),
        "slack_variables": slack_vars, "unit_ncu": unit, "unit_nmu": unit,
        "bias_range": [min(biases), max(biases)] if biases else [0, 0],
        "e_ref_joule": e_ref,
    }


def build_problem_custom(size, lambda_assign, lambda_cpu, lambda_ram, lambda_link,
                         center, unit, encoding="unbalanced"):
    """Return the same (vms, servers, bqm, info) tuple as common.build_problem."""
    vms, servers, _, _ = build_problem(size)
    bqm, info = build_qubo_custom(vms, servers, lambda_assign, lambda_cpu,
                                  lambda_ram, lambda_link, center, unit, encoding)
    return vms, servers, bqm, info
