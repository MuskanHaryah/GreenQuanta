"""Phase 2 update (v2) pipeline: auto server mix, capacity encodings, descent, annealer-only result."""
import math

from . import config
from .catalog import build_servers, load_catalog
from .energy_model import evaluate_allocation
from .exact_solver import linear_cost
from .pipeline import candidate_servers
from .qubo_builder_v2 import build_qubo
from .solver_v2 import solve


def auto_mix(vms, factor=None, catalog=None):
    """Same number n of every server type, so that n x (one of each type) >= factor x demand."""
    factor = factor or config.AUTO_SERVER_FACTOR
    catalog = catalog or load_catalog()
    one_each = build_servers({t: 1 for t in catalog}, catalog=catalog)
    cpu = sum(s.cpu_limit for s in one_each)
    mem = sum(s.mem_limit for s in one_each)
    need = max(sum(v.cpu_size for v in vms) / cpu, sum(v.mem_size for v in vms) / mem)
    n = max(1, math.ceil(factor * need))
    return {t: n for t in sorted(catalog)}


def optimize(vms, servers, lambda_config, encoding, reads=None, sweeps=None, seed=None, descent=None,
             center=None):
    descent = config.DESCENT if descent is None else descent
    idx = candidate_servers(vms, servers)
    sub = [servers[h] for h in idx]
    bqm, qinfo = build_qubo(vms, sub, lambda_config, encoding, center)
    best_sub, pure_sub, stats = solve(bqm, vms, sub, reads, sweeps, seed, descent)
    best = {v: idx[h] for v, h in best_sub.items()}
    pure = {v: idx[h] for v, h in pure_sub.items()} if pure_sub else None
    qinfo["candidate_servers"] = len(idx)
    qinfo["user_servers"] = len(servers)
    return best, pure, stats, qinfo


def gap_metrics_v2(final, pure, vms, servers, exact_assignment, exact_info, stats):
    if exact_assignment is None:
        return {"exact_available": False}
    ex_lin = exact_info["objective_joule"]
    ex_kwh = exact_info["energy"]["it_energy_kwh"]
    fin_kwh = evaluate_allocation(vms, final, servers)["it_energy_kwh"]
    out = {
        "exact_available": True,
        "exact_proven_optimal": exact_info.get("proven_optimal", False),
        "final_gap_energy_pct": 100.0 * (fin_kwh - ex_kwh) / ex_kwh,
        "final_matches_optimum": linear_cost(final, vms, servers) <= ex_lin * (1 + 1e-3),
        # linear-vs-curve check: the exact optimum is optimal for the LINEAR model;
        # if a QA answer is lower on the real SPEC curve, the linear model missed something
        "final_beats_linear_optimum_on_curve": fin_kwh < ex_kwh * (1 - 1e-4),
        "annealer_only_found": pure is not None,
    }
    if pure is not None:
        pure_kwh = evaluate_allocation(vms, pure, servers)["it_energy_kwh"]
        out["annealer_only_gap_energy_pct"] = 100.0 * (pure_kwh - ex_kwh) / ex_kwh
        out["annealer_only_servers"] = len(set(pure.values()))
        out["pure_and_optimal"] = linear_cost(pure, vms, servers) <= ex_lin * (1 + 1e-3)
    else:
        out["pure_and_optimal"] = False
    return out
