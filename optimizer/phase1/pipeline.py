"""
Shared Phase 1 pipeline, used by run_phase1.py and experiment_lambda.py.

prepare()  : trace -> VMs -> lower bound on servers
optimize() : candidate-server reduction -> QUBO -> annealer -> validate/repair
"""
import math
from dataclasses import replace

from . import config
from .qa_solver import solve
from .qubo_builder import build_qubo
from .trace_loader import load_trace, trace_summary
from .vm_builder import build_vms


def lower_bound_servers(vms, server):
    """Fewest servers the total CPU or memory could ever fit in (ignores packing gaps)."""
    cpu = sum(v.cpu_size for v in vms) / (server.safe_utilization * server.cpu_capacity)
    mem = sum(v.mem_size for v in vms) / (server.safe_utilization * server.mem_capacity)
    return max(cpu, mem), ("CPU" if cpu >= mem else "memory")


def prepare(inputs):
    trace = load_trace(inputs.trace_csv)
    vms, percentile = build_vms(trace, inputs.instances_per_vm, inputs.sla_max_violation_pct)
    lb, binding = lower_bound_servers(vms, inputs.server)
    if lb > inputs.server.count:
        raise SystemExit(f"Workload needs at least {lb:.1f} servers but only {inputs.server.count} given. "
                         "Increase --servers.")
    return vms, percentile, trace_summary(trace), lb, binding


def candidate_count(lower_bound, user_servers):
    if not config.CANDIDATE_REDUCTION:
        return user_servers
    k = math.ceil(lower_bound * (1 + config.CANDIDATE_MARGIN)) + config.CANDIDATE_EXTRA
    return max(1, min(user_servers, k))


def optimize(vms, server, lower_bound, reads=None, sweeps=None, seed=None):
    """
    Build and solve the QUBO on the candidate servers only (servers 0..k-1).
    Servers k..count-1 stay OFF. If the workload cannot be fitted on k servers,
    k grows by one until it fits or reaches the user's server count.
    """
    k = candidate_count(lower_bound, server.count)
    while True:
        qubo_server = replace(server, count=k)
        bqm, qinfo = build_qubo(vms, qubo_server)
        try:
            assignment, stats = solve(bqm, vms, qubo_server, reads, sweeps, seed)
            break
        except RuntimeError:
            if k >= server.count:
                raise
            k += 1
    qinfo["lower_bound_servers"] = round(lower_bound, 3)
    qinfo["candidate_servers"] = k
    qinfo["user_servers"] = server.count
    return assignment, stats, qinfo
