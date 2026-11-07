"""
Build the Phase 1 QUBO (Formulation Research document, final version):

    H = E_hat(x, y) + lA*H_assign + lCPU*H_CPU + lRAM*H_RAM + lL*H_link

Binary variables
    x_{v}_{h}   1 if VM v is placed on server h
    y_{h}       1 if server h is ON
    sc_{h}_{k}  CPU slack bits of server h (unused CPU units)
    sm_{h}_{k}  memory slack bits of server h

Terms
    E_hat   = [ sum_h y_h * idle_cost * (1 + tie_h) + sum_{v,h} x_vh * dynamic_cost_v ] / e_ref
    H_assign= sum_v ( sum_h x_vh - 1 )^2
    H_CPU   = sum_h ( sum_v a_v x_vh + slack_h - K_cpu * y_h )^2
    H_RAM   = sum_h ( sum_v b_v x_vh + slack_h - K_mem * y_h )^2
    H_link  = sum_{v,h} x_vh * (1 - y_h)

a_v, b_v, K are integers: each server's capacity is split into RESOLUTION units,
VM sizes are rounded UP (so a QUBO-feasible answer is always really feasible),
and K = floor(SAFE_UTILIZATION * RESOLUTION).
"""
import math

import dimod

from . import config
from .energy_model import qubo_energy_terms


def x_var(v, h):
    return f"x_{v}_{h}"


def y_var(h):
    return f"y_{h}"


def slack_coefficients(max_value):
    """Bounded binary encoding: the bits can add up to any integer 0..max_value, never more."""
    if max_value <= 0:
        return []
    n = int(math.floor(math.log2(max_value))) + 1
    coeffs = [2 ** k for k in range(n - 1)]
    coeffs.append(max_value - (2 ** (n - 1) - 1))
    return coeffs


def add_squared_penalty(bqm, terms, constant, weight):
    """Add weight * (sum_i c_i z_i + constant)^2 for binary z_i (z_i^2 = z_i)."""
    for i, (vi, ci) in enumerate(terms):
        bqm.add_linear(vi, weight * (ci * ci + 2 * constant * ci))
        for vj, cj in terms[i + 1:]:
            bqm.add_quadratic(vi, vj, weight * 2 * ci * cj)
    bqm.offset += weight * constant * constant


def to_units(value, capacity):
    """Round a VM size UP to integer resource units of the server."""
    return max(1, math.ceil(value / capacity * config.RESOURCE_RESOLUTION - 1e-9))


def build_qubo(vms, server):
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    hosts = range(server.count)
    k_cap = int(math.floor(server.safe_utilization * config.RESOURCE_RESOLUTION))

    cpu_units = {v.vm_id: to_units(v.cpu_size, server.cpu_capacity) for v in vms}
    mem_units = {v.vm_id: to_units(v.mem_size, server.mem_capacity) for v in vms}
    too_big = [v.vm_id for v in vms if cpu_units[v.vm_id] > k_cap or mem_units[v.vm_id] > k_cap]
    if too_big:
        raise ValueError(f"VMs {too_big} are larger than one server's safe capacity. "
                         "Lower INSTANCES_PER_VM or use bigger servers.")

    # ---- objective: normalized IT energy
    idle_cost, dynamic_cost, e_ref = qubo_energy_terms(vms, server)
    for h in hosts:
        tie = config.TIE_BREAK * h / max(1, server.count - 1)
        bqm.add_linear(y_var(h), idle_cost * (1 + tie) / e_ref)
    for v in vms:
        for h in hosts:
            bqm.add_linear(x_var(v.vm_id, h), dynamic_cost[v.vm_id] / e_ref)

    # ---- H_assign: every VM on exactly one server
    for v in vms:
        add_squared_penalty(bqm, [(x_var(v.vm_id, h), 1) for h in hosts], -1, config.LAMBDA_ASSIGN)

    # ---- H_CPU and H_RAM: capacity (only on ON servers), with slack bits
    slack = slack_coefficients(k_cap)
    for h in hosts:
        for resource, units, weight, tag in (
            ("cpu", cpu_units, config.LAMBDA_CPU, "sc"),
            ("mem", mem_units, config.LAMBDA_RAM, "sm"),
        ):
            terms = [(x_var(v.vm_id, h), units[v.vm_id]) for v in vms]
            terms += [(f"{tag}_{h}_{k}", c) for k, c in enumerate(slack)]
            terms.append((y_var(h), -k_cap))
            add_squared_penalty(bqm, terms, 0, weight)

    # ---- H_link: no VM on a switched-off server   x*(1-y) = x - x*y
    for v in vms:
        for h in hosts:
            bqm.add_linear(x_var(v.vm_id, h), config.LAMBDA_LINK)
            bqm.add_quadratic(x_var(v.vm_id, h), y_var(h), -config.LAMBDA_LINK)

    info = {
        "num_variables": len(bqm.variables),
        "num_interactions": len(bqm.quadratic),
        "assignment_variables": len(vms) * server.count,
        "server_variables": server.count,
        "slack_variables": 2 * server.count * len(slack),
        "capacity_units_per_server": k_cap,
        "resource_resolution": config.RESOURCE_RESOLUTION,
        "e_ref_joule": e_ref,
        "cpu_units": cpu_units,
        "mem_units": mem_units,
    }
    return bqm, info
