"""
Phase 2 QUBO (heterogeneous servers):

    H = E_hat + lA*H_assign + lCPU*H_CPU + lRAM*H_RAM + lL*H_link

    E_hat    = [ sum_h y_h * idle_cost_h * (1 + tie_h) + sum_{v,h} x_vh * dyn_cost_vh ] / e_ref
    H_assign = sum_v ( sum_h x_vh - 1 )^2
    H_CPU    = sum_h ( sum_v a_v x_vh + slack_h - K_h * y_h )^2
    H_RAM    = sum_h ( sum_v b_v x_vh + slack_h - M_h * y_h )^2
    H_link   = sum_{v,h} x_vh * (1 - y_h)

Changes from Phase 1:
  * idle_cost_h and dyn_cost_vh depend on the server type (linear fit of its SPEC curve)
  * ONE fixed unit for every server (UNIT_NCU / UNIT_NMU): a_v is the same on every server,
    K_h = floor(safe_util_h * C_h / unit) differs per server
  * the four lambdas are separate and come from a named configuration (A-E)
"""
import math

import dimod

from phase1.qubo_builder import add_squared_penalty, slack_coefficients, x_var, y_var
from . import config
from .energy_model import linear_energy_terms

LAMBDA_CONFIGS = {
    "A": "Baseline: all lambdas = 2 (Phase 1)",
    "B": "Glover et al. 2019: lambda = 1 (75-150% of the normalized objective range)",
    "C": "Verma & Lewis (VLM): all lambdas = largest objective change of a single flip",
    "D": "MOMC-style, per rule: largest objective change that breaking that rule can save / 1",
    "E": "Phase 1 experiment best: assign = link = 20, capacity = 0.5",
}


def units(value, unit):
    return max(1, math.ceil(value / unit - 1e-9))


def resolve_lambdas(name, y_coeffs, x_coeffs):
    """Return {'assign','cpu','ram','link'} for configuration name, given the objective coefficients."""
    max_y = max(y_coeffs)
    max_x = max(x_coeffs)
    max_any = max(max_y, max_x)
    if name == "A":
        lam = dict(assign=2.0, cpu=2.0, ram=2.0, link=2.0)
    elif name == "B":
        lam = dict(assign=1.0, cpu=1.0, ram=1.0, link=1.0)
    elif name == "C":
        lam = dict(assign=max_any, cpu=max_any, ram=max_any, link=max_any)
    elif name == "D":
        # dropping a VM saves at most max_x; switching a server off (link or capacity rule) saves at most max_y;
        # the smallest violation of every rule is 1
        lam = dict(assign=max_x, cpu=max_y, ram=max_y, link=max_y)
    elif name == "E":
        lam = dict(assign=20.0, cpu=0.5, ram=0.5, link=20.0)
    else:
        raise ValueError(f"Unknown lambda configuration '{name}'. Use one of {sorted(LAMBDA_CONFIGS)}.")
    return {k: float(v) for k, v in lam.items()}


def build_qubo(vms, servers, lambda_config=None):
    lambda_config = lambda_config or config.DEFAULT_LAMBDA_CONFIG
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    hosts = range(len(servers))

    cpu_units = {v.vm_id: units(v.cpu_size, config.UNIT_NCU) for v in vms}
    mem_units = {v.vm_id: units(v.mem_size, config.UNIT_NMU) for v in vms}
    k_cpu = [int(math.floor(s.cpu_limit / config.UNIT_NCU + 1e-9)) for s in servers]
    k_mem = [int(math.floor(s.mem_limit / config.UNIT_NMU + 1e-9)) for s in servers]
    if not any(cpu_units[v.vm_id] <= kc and mem_units[v.vm_id] <= km
               for v in vms for kc, km in zip(k_cpu, k_mem)):
        raise ValueError("Some VM fits on no server. Use bigger servers or fewer instances per VM.")

    # ---- objective: normalized IT energy (linear fit of each server's SPEC curve)
    idle_cost, dyn_cost, e_ref = linear_energy_terms(vms, servers)
    type_index, seen = [], {}
    for s in servers:
        type_index.append(seen.get(s.type_name, 0))
        seen[s.type_name] = type_index[-1] + 1
    y_coeffs = []
    for h, s in enumerate(servers):
        n_same = seen[s.type_name]
        tie = config.TIE_BREAK * type_index[h] / max(1, n_same - 1)
        y_coeffs.append(idle_cost[h] * (1 + tie) / e_ref)
        bqm.add_linear(y_var(h), y_coeffs[-1])
    x_coeffs = []
    for v in vms:
        for h in hosts:
            c = dyn_cost[v.vm_id][h] / e_ref
            x_coeffs.append(c)
            bqm.add_linear(x_var(v.vm_id, h), c)

    lam = resolve_lambdas(lambda_config, y_coeffs, x_coeffs)

    # ---- H_assign
    for v in vms:
        add_squared_penalty(bqm, [(x_var(v.vm_id, h), 1) for h in hosts], -1, lam["assign"])

    # ---- H_CPU and H_RAM, each server with its own capacity in the common unit
    slack_vars = 0
    for h in hosts:
        for tag, unit_map, cap, weight in (("sc", cpu_units, k_cpu[h], lam["cpu"]),
                                           ("sm", mem_units, k_mem[h], lam["ram"])):
            coeffs = slack_coefficients(cap)
            slack_vars += len(coeffs)
            terms = [(x_var(v.vm_id, h), unit_map[v.vm_id]) for v in vms]
            terms += [(f"{tag}_{h}_{k}", c) for k, c in enumerate(coeffs)]
            terms.append((y_var(h), -cap))
            add_squared_penalty(bqm, terms, 0, weight)

    # ---- H_link
    for v in vms:
        for h in hosts:
            bqm.add_linear(x_var(v.vm_id, h), lam["link"])
            bqm.add_quadratic(x_var(v.vm_id, h), y_var(h), -lam["link"])

    info = {
        "lambda_config": lambda_config,
        "lambda_description": LAMBDA_CONFIGS[lambda_config],
        "lambdas": lam,
        "num_variables": len(bqm.variables),
        "num_interactions": len(bqm.quadratic),
        "assignment_variables": len(vms) * len(servers),
        "server_variables": len(servers),
        "slack_variables": slack_vars,
        "unit_ncu": config.UNIT_NCU,
        "unit_nmu": config.UNIT_NMU,
        "capacity_units_cpu": k_cpu,
        "capacity_units_mem": k_mem,
        "e_ref_joule": e_ref,
    }
    return bqm, info
