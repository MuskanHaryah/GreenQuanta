"""
Phase 2 QUBO, version 2: same objective and rules, three ways to write the capacity rule.

    H = E_hat + lA*H_assign + lCPU*H_CPU + lRAM*H_RAM + lL*H_link

Capacity encodings (for server h, load L_h = sum_v a_v x_vh in units, capacity K_h units):

  "slack"         team formulation (Phase 1 / Phase 2 first run)
                  ( L_h + slack_h - K_h * y_h )^2
  "slack_scaled"  same, divided by K_h, so big and small servers weigh the same
                  ( L_h + slack_h - K_h * y_h )^2 / K_h
  "unbalanced"    unbalanced penalization (Montanez-Barrera et al., Quantum Sci. Technol. 2024):
                  no slack bits; g_h = (K_h * y_h - L_h) / K_h is the free share of the server,
                  penalty = g_h^2 - 2 * c * g_h  (minimum at c = UNBALANCED_CENTER free space,
                  grows fast when g_h < 0, i.e. when the server is overloaded)

Everything else (objective, assign, link, lambda configs A-E) is the same as qubo_builder.py.

Lambda configuration names: A, B, C, D, E (see qubo_builder.py) or "L<value>", e.g. "L0.25",
which sets all four lambdas to that value (used for the small-lambda scan).
"""
import math

import dimod

from phase1.qubo_builder import add_squared_penalty, slack_coefficients, x_var, y_var
from . import config
from .energy_model import linear_energy_terms
from .qubo_builder import LAMBDA_CONFIGS, resolve_lambdas, units

ENCODINGS = {
    "slack": "Slack bits, (L + s - K*y)^2 (team formulation)",
    "slack_scaled": "Slack bits, (L + s - K*y)^2 / K (scaled per server)",
    "unbalanced": "No slack, g^2 - 2*c*g with g = (K*y - L)/K (unbalanced penalization)",
}


def lambda_setup(name, y_coeffs, x_coeffs):
    """Returns (lambdas dict, description) for 'A'..'E' or 'L<value>'."""
    if name.upper().startswith("L"):
        value = float(name[1:])
        return {k: value for k in ("assign", "cpu", "ram", "link")}, f"All lambdas = {value:g} (scan)"
    return resolve_lambdas(name, y_coeffs, x_coeffs), LAMBDA_CONFIGS[name]


def add_capacity_rule(bqm, h, tag, unit_map, cap, weight, vms, encoding, center=None):
    """Adds one capacity rule of server h; returns the number of slack bits used."""
    if encoding in ("slack", "slack_scaled"):
        coeffs = slack_coefficients(cap)
        terms = [(x_var(v.vm_id, h), unit_map[v.vm_id]) for v in vms]
        terms += [(f"{tag}_{h}_{k}", c) for k, c in enumerate(coeffs)]
        terms.append((y_var(h), -cap))
        w = weight / cap if encoding == "slack_scaled" else weight
        add_squared_penalty(bqm, terms, 0, w)
        return len(coeffs)
    if encoding == "unbalanced":
        # g = y - sum_v (a_v / K) x_v   ;  penalty = weight * (g^2 - 2 c g)
        c = config.UNBALANCED_CENTER if center is None else center
        terms = [(y_var(h), 1.0)] + [(x_var(v.vm_id, h), -unit_map[v.vm_id] / cap) for v in vms]
        add_squared_penalty(bqm, terms, 0, weight)              # weight * g^2
        for name, coef in terms:                                # weight * (-2 c g)
            bqm.add_linear(name, -2 * c * coef * weight)
        return 0
    raise ValueError(f"Unknown encoding '{encoding}'. Use one of {sorted(ENCODINGS)}.")


def build_qubo(vms, servers, lambda_config=None, encoding="slack", center=None):
    lambda_config = lambda_config or config.DEFAULT_LAMBDA_CONFIG
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    hosts = range(len(servers))

    cpu_units = {v.vm_id: units(v.cpu_size, config.UNIT_NCU) for v in vms}
    mem_units = {v.vm_id: units(v.mem_size, config.UNIT_NMU) for v in vms}
    k_cpu = [int(math.floor(s.cpu_limit / config.UNIT_NCU + 1e-9)) for s in servers]
    k_mem = [int(math.floor(s.mem_limit / config.UNIT_NMU + 1e-9)) for s in servers]

    idle_cost, dyn_cost, e_ref = linear_energy_terms(vms, servers)
    type_index, seen = [], {}
    for s in servers:
        type_index.append(seen.get(s.type_name, 0))
        seen[s.type_name] = type_index[-1] + 1
    y_coeffs, x_coeffs = [], []
    for h, s in enumerate(servers):
        tie = config.TIE_BREAK * type_index[h] / max(1, seen[s.type_name] - 1)
        y_coeffs.append(idle_cost[h] * (1 + tie) / e_ref)
        bqm.add_linear(y_var(h), y_coeffs[-1])
    for v in vms:
        for h in hosts:
            x_coeffs.append(dyn_cost[v.vm_id][h] / e_ref)
            bqm.add_linear(x_var(v.vm_id, h), x_coeffs[-1])

    lam, lam_desc = lambda_setup(lambda_config, y_coeffs, x_coeffs)

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
    info = {
        "encoding": encoding,
        "encoding_description": ENCODINGS[encoding],
        "lambda_config": lambda_config,
        "lambda_description": lam_desc,
        "unbalanced_center": (config.UNBALANCED_CENTER if center is None else center)
        if encoding == "unbalanced" else None,
        "lambdas": lam,
        "num_variables": len(bqm.variables),
        "num_interactions": len(bqm.quadratic),
        "assignment_variables": len(vms) * len(servers),
        "server_variables": len(servers),
        "slack_variables": slack_vars,
        "unit_ncu": config.UNIT_NCU,
        "unit_nmu": config.UNIT_NMU,
        "bias_range": [min(biases), max(biases)] if biases else [0, 0],
        "e_ref_joule": e_ref,
    }
    return bqm, info
