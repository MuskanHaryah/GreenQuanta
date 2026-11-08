"""D0: deterministic, exhaustive checks of the unbalanced formulation only.

Independently recompute H_assign = sum_v (sum_h x_vh - 1)^2,
H_resource = sum_h (g_h^2 - 2*c*g_h), g_h = y_h - sum_v a_v*x_vh/K_h,
and H_link = sum_vh x_vh*(1-y_h). Use only the energy part from D2.
No runners, stochastic sampling, result writes, or slack encodings are used.
"""
import argparse
from dataclasses import replace
from itertools import product
import math
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import dimod


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.common import CENTER, ENCODING, LAMBDA_CONFIG
from diagnostics.d2_scale import decompose
from phase1.qubo_builder import x_var, y_var
from phase1.trace_loader import load_trace
from phase1.vm_builder import build_vms
from phase2 import config
from phase2.catalog import build_servers
from phase2.energy_model import evaluate_allocation
from phase2.exact_solver import linear_cost
from phase2.pipeline_v2 import auto_mix
from phase2.qubo_builder import units
from phase2.qubo_builder_v2 import add_capacity_rule, build_qubo
from phase2.validation import validate_allocation


SIZES = ("small_v2", "medium_v2", "large_v2")
HAND_VALUES = ((1.0, 0.048), (0.02, -0.00002), (0.0, 0.0), (-0.5, 0.0135))
ENERGY_TOLERANCE = 1e-9
RANK_TOLERANCE = 1e-12


def require_unbalanced(encoding):
    if encoding != "unbalanced":
        raise ValueError("D0 supports unbalanced only; slack and slack_scaled are rejected")


def load_workload(size):
    """Read the same trace and VM configuration as diagnostics.common, without a QUBO."""
    trace = config.TRACE_DIR / config.SCENARIOS[size]["trace"]
    vms, _ = build_vms(load_trace(trace), config.INSTANCES_PER_VM,
                       config.SLA_MAX_VIOLATION_PCT)
    return vms


def hand_values():
    """One ON server, one resource, K=100; binary x=1 realizes each load."""
    rows = []
    for free_share, expected in HAND_VALUES:
        bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
        vm = SimpleNamespace(vm_id=0)
        load = round((1 - free_share) * 100)
        slack = add_capacity_rule(bqm, 0, "sc", {0: load}, 100, 0.05,
                                  [vm], "unbalanced", center=0.02)
        actual = float(bqm.energy({x_var(0, 0): 1, y_var(0): 1}))
        assert slack == 0
        assert abs(actual - expected) < 1e-12, (free_share, actual, expected)
        rows.append((free_share, expected, actual))
    return rows


def tiny_problem(encoding=ENCODING):
    require_unbalanced(encoding)
    original = load_workload("small_v2")[:4]
    if len(original) != 4:
        raise ValueError("small_v2 must supply at least four real VMs")
    servers = build_servers({"B": 1, "D": 2})
    # Make at least one all-on-one-server placement exceed a real safe limit.
    pressure = max(sum(getattr(vm, f"{r}_size") for vm in original)
                   / min(getattr(s, f"{r}_limit") for s in servers)
                   for r in ("cpu", "mem"))
    scale = 1.1 / pressure if pressure <= 1.0 else 1.0
    vms = [replace(vm, cpu_size=vm.cpu_size * scale, mem_size=vm.mem_size * scale)
           for vm in original]
    bqm, info = build_qubo(vms, servers, lambda_config=LAMBDA_CONFIG,
                           encoding=encoding, center=CENTER)
    assert len(bqm.variables) == 15 and info["slack_variables"] == 0
    return vms, servers, bqm, info, scale


def independent_terms(sample, vms, servers, info, energy):
    """Use scalar docstring formulas, not builder penalty/decomposition helpers."""
    # ExactSolver stores int8 bits; capacity arithmetic must not stay in int8.
    sample = {variable: int(value) for variable, value in sample.items()}
    assign = math.fsum((sum(sample[x_var(vm.vm_id, h)] for h in range(len(servers))) - 1)**2
                       for vm in vms)
    capacity = {}
    for resource, unit, term in (("cpu", info["unit_ncu"], "cpu"),
                                 ("mem", info["unit_nmu"], "ram")):
        # Repeat the documented quantization independently of qubo_builder.units.
        a = {vm.vm_id: max(1, math.ceil(getattr(vm, f"{resource}_size") / unit - 1e-9))
             for vm in vms}
        values = []
        for h, server in enumerate(servers):
            k = math.floor(getattr(server, f"{resource}_limit") / unit + 1e-9)
            load = sum(a[vm.vm_id] * sample[x_var(vm.vm_id, h)] for vm in vms)
            g = (k * sample[y_var(h)] - load) / k
            values.append(g*g - 2 * info["unbalanced_center"] * g)
        capacity[term] = math.fsum(values)
    link = sum(sample[x_var(vm.vm_id, h)] * (1 - sample[y_var(h)])
               for vm in vms for h in range(len(servers)))
    raw = {"assign": assign, **capacity, "link": link}
    return {"energy": float(energy.energy(sample)),
            **{term: info["lambdas"][term] * value for term, value in raw.items()}}


def enumerate_states(vms, servers, bqm, info):
    parts, coefficient_error = decompose(vms, servers, bqm)
    # ExactSolver.sample is dimod's exhaustive enumerator, not stochastic sampling.
    states = dimod.ExactSolver().sample(bqm)
    assert len(states) == 2**15
    maximum = 0.0
    for row in states.data(fields=["sample", "energy"], sorted_by=None):
        terms = independent_terms(row.sample, vms, servers, info, parts["energy"])
        error = abs(math.fsum(terms.values()) - float(row.energy))
        assert math.isfinite(error) and error < ENERGY_TOLERANCE, (row.sample, error)
        maximum = max(maximum, error)
    return states, maximum, coefficient_error


def state_validation(sample, vms, servers):
    """Use shared real-size validation, additionally checking explicit x/y state."""
    assignments = {vm.vm_id: [h for h in range(len(servers))
                              if sample[x_var(vm.vm_id, h)]] for vm in vms}
    assignment = {vm_id: hosts[0] for vm_id, hosts in assignments.items() if len(hosts) == 1}
    check = validate_allocation(vms, assignment, servers)
    multiplicities = {vm_id: len(hosts) for vm_id, hosts in assignments.items() if len(hosts) != 1}
    off_links = [(vm_id, h) for vm_id, hosts in assignments.items() for h in hosts
                 if not sample[y_var(h)]]
    return {"valid": check["valid"] and not multiplicities and not off_links,
            "allocation": check, "assignment_multiplicities": multiplicities,
            "off_server_links": off_links}


def assignment_sample(assignment, bqm):
    sample = dict.fromkeys(bqm.variables, 0)
    for vm_id, host in assignment.items():
        sample[x_var(vm_id, host)] = 1
        sample[y_var(host)] = 1
    return sample


def placement_comparison(vms, servers, bqm, states):
    valid = []
    for hosts in product(range(len(servers)), repeat=len(vms)):
        assignment = {vm.vm_id: h for vm, h in zip(vms, hosts)}
        if not validate_allocation(vms, assignment, servers)["valid"]:
            continue
        sample = assignment_sample(assignment, bqm)
        valid.append({"hosts": hosts, "assignment": assignment,
                      "h": float(bqm.energy(sample)),
                      "linear_joule": float(linear_cost(assignment, vms, servers)),
                      "curve_kwh": evaluate_allocation(vms, assignment, servers)["it_energy_kwh"]})
    if not valid:
        raise ValueError("Tiny instance has no valid real-size placement")
    best = {key: min(valid, key=lambda row: (row[key], row["hosts"]))
            for key in ("h", "linear_joule", "curve_kwh")}
    linear_h = best["linear_joule"]["h"]
    return {"valid": valid, "best": best,
            "equal": best["h"]["hosts"] == best["linear_joule"]["hosts"] == best["curve_kwh"]["hosts"],
            "linear_h_rank_valid": 1 + sum(row["h"] < linear_h - RANK_TOLERANCE for row in valid),
            "linear_h_rank_all": 1 + int((states.record.energy < linear_h - RANK_TOLERANCE).sum())}


def resource_statistics(vms, servers):
    stats = {}
    for resource, unit in (("cpu", config.UNIT_NCU), ("mem", config.UNIT_NMU)):
        sizes = [getattr(vm, f"{resource}_size") for vm in vms]
        a = [units(size, unit) for size in sizes]
        by_type = {}
        for server in servers:
            by_type[server.type_name] = by_type.get(server.type_name, 0) + math.floor(
                getattr(server, f"{resource}_limit") / unit + 1e-9)
        stats[resource] = {"min": min(a), "median": statistics.median(a), "max": max(a),
                           "sum_units": sum(a), "sum_k_by_type": by_type,
                           "inflation": sum(value * unit for value in a) / sum(sizes) - 1,
                           "pressure": sum(a) / sum(by_type.values())}
    return stats


def print_size_statistics(size):
    vms = load_workload(size)
    scenario = config.SCENARIOS[size]
    mix = auto_mix(vms) if scenario["servers"] == "auto" else scenario["servers"]
    servers = build_servers(mix)
    stats = resource_statistics(vms, servers)
    print(f"T4 {size}: VMs={len(vms)} servers={len(servers)} mix={mix}")
    for resource, label in (("cpu", "CPU"), ("mem", "RAM")):
        row = stats[resource]
        print(f"  {label} units/VM min/median/max={row['min']}/{row['median']}/{row['max']} "
              f"sum_units={row['sum_units']} sum_K_by_type={row['sum_k_by_type']} "
              f"rounding_inflation={row['inflation']:.17g} pressure={row['pressure']:.17g}")
    binding = max(stats, key=lambda r: stats[r]["pressure"])
    print(f"  binding_resource={'CPU' if binding == 'cpu' else 'RAM'} "
          "(largest sum_units / sum_K; tie selects CPU)")
    for name in sorted(mix):
        binding = max(stats, key=lambda r: stats[r]["sum_units"] / stats[r]["sum_k_by_type"][name])
        print(f"  type={name} binding_resource={'CPU' if binding == 'cpu' else 'RAM'} "
              "(largest sum_units / type sum_K)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding", default=ENCODING)
    args = parser.parse_args(argv)
    require_unbalanced(args.encoding)
    print(f"D0 encoding={args.encoding} lambda_config={LAMBDA_CONFIG} center={CENTER}")
    print("Slack and slack_scaled encodings are rejected. No stochastic sampling; exhaustive enumeration only.")
    print(f"VM construction: instances_per_vm={config.INSTANCES_PER_VM} "
          f"sla_max_violation_pct={config.SLA_MAX_VIOLATION_PCT}")
    for g, expected, actual in hand_values():
        print(f"T1 g={g:g} lambda=0.05 c=0.02 expected={expected:.12g} "
              f"add_capacity_rule={actual:.12g} PASS")
    vms, servers, bqm, info, scale = tiny_problem(args.encoding)
    print(f"T2 tiny: VMs={len(vms)} servers={[s.type_name for s in servers]} "
          f"variables={len(bqm.variables)} reservation_scale={scale:.17g}")
    print("  Original trace demand arrays are unchanged; CPU/RAM reservations are multiplied by scale.")
    for vm in vms:
        print(f"  VM {vm.vm_id}: cpu_size={vm.cpu_size:.17g} mem_size={vm.mem_size:.17g}")
    for h, server in enumerate(servers):
        print(f"  server {h} {server.type_name}: cpu_limit={server.cpu_limit:.17g} "
              f"mem_limit={server.mem_limit:.17g}")
    states, maximum, coefficient_error = enumerate_states(vms, servers, bqm, info)
    print(f"T2 states={len(states)} expected={2**15} max_abs_H_error={maximum:.17g} "
          f"required=<{ENERGY_TOLERANCE:g} PASS; D2_coefficient_error={coefficient_error:.17g}")
    first = states.first
    print(f"T3 (i) argmin_H_all: H={first.energy:.17g} state={dict(sorted(first.sample.items()))}")
    print(f"  validation={state_validation(first.sample, vms, servers)}")
    comparison = placement_comparison(vms, servers, bqm, states)
    print(f"T3 assignments_enumerated={len(servers)**len(vms)} valid_real_size={len(comparison['valid'])} "
          "y=servers_used; minima ties select lexicographic host tuple")
    for label, key in (("(ii) argmin_H_valid", "h"), ("(iii) argmin_linear_cost_valid", "linear_joule"),
                       ("(iv) argmin_curve_energy_valid", "curve_kwh")):
        row = comparison["best"][key]
        print(f"T3 {label}: assignment={row['assignment']} H={row['h']:.17g} "
              f"linear_cost_joule={row['linear_joule']:.17g} curve_energy_kwh={row['curve_kwh']:.17g}")
    print(f"T3 (ii)==(iii)==(iv): {comparison['equal']}")
    print(f"T3 H-rank_of_(iii)_among_valid={comparison['linear_h_rank_valid']} "
          f"among_all_states={comparison['linear_h_rank_all']} "
          f"(1 + count H < H(iii) - {RANK_TOLERANCE:g})")
    for size in SIZES:
        print_size_statistics(size)


if __name__ == "__main__":
    main()
