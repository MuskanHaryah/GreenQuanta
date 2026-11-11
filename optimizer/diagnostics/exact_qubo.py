"""Exact QUBO minimization with local HiGHS, without placement rules.

Integer assignment/capacity residuals have square epigraphs. Capacity cuts use
q = g/K and t = (g/K)^2, avoiding coefficients proportional to 1/K^2 on an
integer column. No physical feasibility constraints, repair, or sampling apply.

encoding="slack" (Step 0): slack bits are eliminated exactly (Lemma in
docs/reports/step0/0.2_proof.md). The capacity penalty of a rule becomes
lambda * o^2 with integer overload o = max(0, L - K*y), so q is the integer
overload o >= -g, t >= o^2, and cuts are unscaled. Returned samples carry the
optimal slack bits, so the genuine slack BQM scores them.
"""
import math
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from diagnostics.common import CENTER, LAMBDA_CONFIG
from diagnostics.d0_formulation import independent_terms, state_validation
from phase1.qubo_builder import slack_coefficients, x_var, y_var
from phase2 import config
from phase2.energy_model import linear_energy_terms
from phase2.qubo_builder import units
from phase2.qubo_builder_v2 import build_qubo


TOLERANCE = 1e-9
BQM_RELATIVE = 1e-12


def bqm_agrees(bqm, sample, exact_h, bqm_h):
    """|exact-sum H - dimod bqm.energy(sample)| <= TOLERANCE, else <= 1e-12 x the summed |active terms|.

    dimod sums in plain float; at large lambda its rounding exceeds 1e-9 (Step 1 CHANGES #1).
    The term sum is computed only when the absolute check fails."""
    difference = abs(exact_h - bqm_h)
    if difference <= TOLERANCE:
        return True
    active = math.fsum(abs(b) for v, b in bqm.linear.items() if sample[v])
    active += math.fsum(abs(b) for (u, v), b in bqm.quadratic.items() if sample[u] and sample[v])
    return difference <= BQM_RELATIVE * (active + abs(bqm.offset))


def with_slack(sample, vms, caps, loads, m):
    """Add the optimal slack bits S = clip(K*y - L, 0, K) of every capacity rule.

    Rule r = resource_index * m + host; tags 'sc' (cpu) / 'sm' (ram) as in
    qubo_builder_v2. S is written greedily with slack_coefficients(K)."""
    full = dict(sample)
    for r, (cap, demand) in enumerate(zip(caps, loads)):
        h = r % m
        load = sum(a * int(sample[x_var(vm.vm_id, h)]) for vm, a in zip(vms, demand))
        value = min(cap, max(0, cap * int(sample[y_var(h)]) - load))
        coeffs = slack_coefficients(cap)
        bits = [0] * len(coeffs)
        for k in [len(coeffs) - 1] + list(range(len(coeffs) - 2, -1, -1)):
            if value >= coeffs[k]:
                bits[k], value = 1, value - coeffs[k]
        assert value == 0, "slack value not representable"
        tag = "sc" if r < m else "sm"
        full.update({f"{tag}_{h}_{k}": b for k, b in enumerate(bits)})
    return full


def solve_exact_qubo(vms, servers, lambda_config=LAMBDA_CONFIG, center=CENTER,
                     time_limit=None, *, max_iters=50, full_cuts=False,
                     fixed_sample=None, encoding="unbalanced"):
    """Return an independently scored x/y incumbent and its QUBO certificate.

    ``fixed_sample`` optionally fixes any binary variable by its BQM name; it
    does not impose placement feasibility. With fixed bounds the optimality
    certificate is conditional on those bounds, not the unrestricted QUBO. ``assignment`` contains singleton
    placements only; ``placements`` and ``sample`` preserve missing/duplicate
    placements and explicit off hosts. The shared D0 validator is diagnostic.

    ``lower_bound`` is the best HiGHS dual bound or an initial analytic bound,
    NOT an interrupted solve's primal objective. ``milp_objective`` is the final relaxation incumbent's
    objective; it is a lower bound only when that relaxation is solved optimally.
    ``mip_gap`` is the true-QUBO incumbent/dual-bound relative gap; the last
    relaxation's gap is also returned as ``milp_gap``. Limits yield UNPROVEN.

    The assign breakdown includes its constant, lambda_assign * len(vms),
    already present in (sum(x)-1)^2. Thus ``offset`` is only the remaining
    constant (zero for this builder), not a second copy of ``bqm_offset``.
    """
    start = time.perf_counter()
    time_limit = config.EXACT_TIME_LIMIT_S if time_limit is None else time_limit
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("time_limit must be finite and positive")
    if not isinstance(max_iters, int) or isinstance(max_iters, bool) or max_iters < 1:
        raise ValueError("max_iters must be a positive integer")
    if not math.isfinite(center):
        raise ValueError("center must be finite")
    if encoding not in ("unbalanced", "slack"):
        raise ValueError("encoding must be 'unbalanced' or 'slack'")
    slack = encoding == "slack"
    n, m = len(vms), len(servers)
    if not n or not m or len({vm.vm_id for vm in vms}) != n:
        raise ValueError("require nonempty VMs/servers and unique VM IDs")
    caps = []
    loads = []
    weights = []
    for resource, unit, term in (("cpu", config.UNIT_NCU, "cpu"),
                                 ("mem", config.UNIT_NMU, "ram")):
        demand = [units(getattr(vm, f"{resource}_size"), unit) for vm in vms]
        for server in servers:
            cap = math.floor(getattr(server, f"{resource}_limit") / unit + 1e-9)
            if cap <= 0:
                raise ValueError("unbalanced encoding requires positive integer capacities")
            caps.append(cap)
            loads.append(demand)
            weights.append(term)

    bqm, info = build_qubo(vms, servers, lambda_config=lambda_config,
                           encoding=encoding, center=center)
    lam = info["lambdas"]
    if any(not math.isfinite(value) or value < 0 for value in lam.values()):
        raise ValueError("square epigraphs require finite, nonnegative lambdas")
    if slack and min(lam.values()) <= 0:
        raise ValueError("slack mode requires positive lambdas (overload cost must grow)")
    # This is the builder's energy objective only; D0 independently evaluates
    # every penalty rather than using the builder's quadratic coefficients.
    energy, _ = build_qubo(vms, servers, lambda_config="L0",
                           encoding=encoding, center=center)
    idle, _, e_ref = linear_energy_terms(vms, servers)
    tie_coeff = [energy.linear[y_var(h)] - idle[h] / e_ref for h in range(m)]
    residual_offset = float(bqm.offset - n * lam["assign"] - energy.offset)
    assert abs(residual_offset) <= TOLERANCE, "unexpected builder offset"

    binary_names = [x_var(vm.vm_id, h) for vm in vms for h in range(m)]
    binary_names += [y_var(h) for h in range(m)]
    fixed_sample = {} if fixed_sample is None else dict(fixed_sample)
    if set(fixed_sample) - set(binary_names):
        raise ValueError("fixed_sample contains unknown x/y variables")
    if any(value not in (0, 1) for value in fixed_sample.values()):
        raise ValueError("fixed_sample values must be binary")

    nx, nb, nr = n * m, n * m + m, 2 * m
    d0 = nb
    g0 = d0 + n
    s0 = g0 + nr
    t0 = s0 + n
    q0 = t0 + nr
    z0 = q0 + nr
    size = z0 + nx
    cost = np.zeros(size)
    lower, upper = np.zeros(size), np.full(size, np.inf)
    integer = np.zeros(size, dtype=np.uint8)
    integer[:g0 + nr] = 1
    if slack:
        integer[q0:q0 + nr] = 1          # q = integer overload o
    scale = [1 if slack else cap for cap in caps]
    upper[:nb] = 1
    lower[d0:g0], upper[d0:g0] = -1, m - 1
    upper[s0:t0] = max(1, m - 1) ** 2
    upper[z0:] = 1
    for j, name in enumerate(binary_names):
        cost[j] = energy.linear[name]
        if name in fixed_sample:
            lower[j] = upper[j] = int(fixed_sample[name])
    cost[s0:t0] = lam["assign"]
    cost[z0:] = lam["link"]
    for r, (cap, demand, term) in enumerate(zip(caps, loads, weights)):
        total = sum(demand)
        lower[g0 + r], upper[g0 + r] = -total, cap
        if slack:
            lower[q0 + r], upper[q0 + r] = 0, total
            upper[t0 + r] = max(1, total) ** 2
        else:
            lower[q0 + r], upper[q0 + r] = -total / cap, 1
            upper[t0 + r] = max(1, total / cap) ** 2
            cost[q0 + r] = -2 * center * lam[term]
        cost[t0 + r] = lam[term]
    objective_offset = float(energy.offset)

    rows, row_lower, row_upper = [], [], []

    def row(coefficients, lo, hi=np.inf):
        rows.append(coefficients)
        row_lower.append(lo)
        row_upper.append(hi)

    for i in range(n):
        row({d0 + i: 1, **{i * m + h: -1 for h in range(m)}}, -1, -1)
        for k in range(-1, m):
            row({s0 + i: 1, d0 + i: -(2 * k + 1)}, -k * (k + 1))
    for r, (cap, demand) in enumerate(zip(caps, loads)):
        h = r % m
        row({g0 + r: 1, nx + h: -cap,
             **{i * m + h: a for i, a in enumerate(demand)}}, 0, 0)
        if slack:
            row({g0 + r: 1, q0 + r: 1}, 0)       # o >= -g = L - K*y (and o >= 0)
        else:
            row({g0 + r: 1 / cap, q0 + r: -1}, 0, 0)
    for i in range(n):
        for h in range(m):
            j = i * m + h
            row({z0 + j: 1, j: -1, nx + h: 1}, 0)

    cuts = [set() for _ in range(nr)]

    def add_cut(r, k):
        if k in cuts[r]:
            return False
        cap = scale[r]
        row({t0 + r: 1, q0 + r: -(2 * k + 1) / cap},
            -(k / cap) * ((k + 1) / cap))
        cuts[r].add(k)
        return True

    for r, (cap, demand) in enumerate(zip(caps, loads)):
        if slack:
            # Secants through integer overloads; valid because o is integer.
            hi = sum(demand)
            grid = range(0, hi + 1) if full_cuts else (
                set(range(0, min(hi, 32) + 1)) | {2 ** j for j in range(hi.bit_length())}
                | {hi // 2, hi})
        else:
            lo, hi = -sum(demand), cap
            grid = range(lo, hi + 1) if full_cuts else {
                lo, hi, -1, 0, (lo + hi) // 2, lo // 2, hi // 2,
            }
        for k in sorted(grid):
            add_cut(r, k)

    def score(sample):
        if slack:
            sample = with_slack(sample, vms, caps, loads, m)
        terms = independent_terms(sample, vms, servers, info, energy)
        independent_h = math.fsum(terms.values())
        tie = math.fsum(tie_coeff[h] * sample[y_var(h)] for h in range(m))
        breakdown = {**terms, "energy": terms["energy"] - tie,
                     "tie": tie, "offset": 0.0}
        true_h = math.fsum(breakdown.values())
        bqm_h = float(bqm.energy(sample))
        assert (abs(true_h - independent_h) <= TOLERANCE
                and bqm_agrees(bqm, sample, true_h, bqm_h)
                and bqm_agrees(bqm, sample, independent_h, bqm_h)), (true_h, independent_h, bqm_h)
        return {"sample": sample, "breakdown": breakdown,
                "incumbent_true_H": true_h, "independent_H": independent_h,
                "bqm_H": bqm_h}

    # Every binary state is QUBO-feasible, even if the first MILP times out.
    incumbent = score({name: int(fixed_sample.get(name, 0)) for name in binary_names})
    history = []
    # Assignment/link penalties are nonnegative. Each capacity parabola is
    # bounded below by its continuous minimum on the residual's full range.
    # This certificate remains available if preprocessing consumes the budget.
    bound_terms = [objective_offset]
    bound_terms.extend(min(cost[j] * lower[j], cost[j] * upper[j]) for j in range(nb))
    for r, term in enumerate(weights):
        if slack:
            continue                      # lambda * o^2 >= 0
        q = min(upper[q0 + r], max(lower[q0 + r], center))
        bound_terms.append(lam[term] * (q * q - 2 * center * q))
    lower_bound = math.fsum(bound_terms)
    last = None
    converged = False
    proven = False
    reason = "max_iters"
    max_violation = max_weighted = None
    objective = None
    for iteration in range(1, max_iters + 1):
        remaining = time_limit - (time.perf_counter() - start)
        if remaining <= 0:
            reason = "time_limit"
            break
        rr, cc, vv = [], [], []
        for index, coefficients in enumerate(rows):
            for column, value in coefficients.items():
                rr.append(index)
                cc.append(column)
                vv.append(value)
        matrix = coo_matrix((vv, (rr, cc)), shape=(len(rows), size)).tocsc()
        remaining = time_limit - (time.perf_counter() - start)
        if remaining <= 0:
            reason = "time_limit"
            break
        # SciPy forwards these HiGHS options verbatim, issuing a warning solely
        # because they are not in SciPy's small public milp option allow-list.
        options = {"time_limit": remaining, "mip_rel_gap": 1e-9,
                   "mip_abs_gap": 1e-10, "mip_feasibility_tolerance": 1e-10,
                   "primal_feasibility_tolerance": 1e-10,
                   "dual_feasibility_tolerance": 1e-10,
                   "random_seed": config.SEED, "disp": False}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*",
                                    category=RuntimeWarning)
            last = milp(cost, integrality=integer, bounds=Bounds(lower, upper),
                        constraints=LinearConstraint(matrix, row_lower, row_upper),
                        options=options)
        bound = getattr(last, "mip_dual_bound", None)
        if bound is not None and math.isfinite(bound):
            bound = float(bound) + objective_offset
            lower_bound = max(lower_bound, bound)
        objective = None if last.fun is None else float(last.fun) + objective_offset
        record = {"iteration": iteration, "milp_objective": objective,
                  "lower_bound": lower_bound, "milp_status": int(last.status),
                  "max_violation": None, "max_weighted_violation": None,
                  "incumbent_true_H": incumbent["incumbent_true_H"]}
        if last.x is None:
            history.append(record)
            print(f"iteration={iteration} max_violation=None milp_objective={objective} "
                  f"lower_bound={lower_bound} incumbent_true_H={incumbent['incumbent_true_H']:.17g}",
                  flush=True)
            reason = str(last.message)
            break
        bits = np.rint(last.x[:nb]).astype(int)
        assert np.max(np.abs(last.x[:nb] - bits)) <= 1e-7
        candidate = score(dict(zip(binary_names, map(int, bits))))
        if candidate["incumbent_true_H"] < incumbent["incumbent_true_H"]:
            incumbent = candidate
        residuals, violations = [], []
        for r, (cap, demand) in enumerate(zip(caps, loads)):
            h = r % m
            g = cap * int(bits[nx + h]) - sum(
                a * int(bits[i * m + h]) for i, a in enumerate(demand))
            if slack:
                g = max(0, -g)            # cut index = true integer overload
                violations.append(max(0.0, g ** 2 - float(last.x[t0 + r])))
            else:
                violations.append(max(0.0, (g / cap) ** 2 - float(last.x[t0 + r])))
            residuals.append(g)
        max_violation = max(violations, default=0.0)
        max_weighted = max((lam[term] * violation
                            for term, violation in zip(weights, violations)), default=0.0)
        error = abs(candidate["incumbent_true_H"] - objective)
        record.update(max_violation=max_violation, max_weighted_violation=max_weighted,
                      candidate_true_H=candidate["incumbent_true_H"],
                      incumbent_true_H=incumbent["incumbent_true_H"], objective_error=error)
        history.append(record)
        print(f"iteration={iteration} max_violation={max_violation:.17g} "
              f"max_weighted_violation={max_weighted:.17g} "
              f"milp_objective={objective:.17g} lower_bound={lower_bound} "
              f"incumbent_true_H={incumbent['incumbent_true_H']:.17g}", flush=True)
        converged = (max_violation <= TOLERANCE and max_weighted <= TOLERANCE
                     and error <= TOLERANCE)
        if converged:
            assert abs(candidate["incumbent_true_H"] - objective) <= TOLERANCE
            proven = last.status == 0 and (
                abs(incumbent["incumbent_true_H"] - lower_bound) <= TOLERANCE
                and abs(incumbent["incumbent_true_H"] - objective) <= TOLERANCE)
            if proven:
                assert abs(incumbent["incumbent_true_H"] - objective) <= TOLERANCE
            if proven:
                reason = ("proven fixed-bounds QUBO minimum" if fixed_sample
                          else "proven global QUBO minimum")
            else:
                reason = str(last.message)
            break
        if last.status != 0:
            reason = str(last.message)
            break
        added = 0
        for r, (g, violation) in enumerate(zip(residuals, violations)):
            if violation > 0:
                added += add_cut(r, g)
                added += add_cut(r, g - 1)
        if not added:
            reason = "numerical objective/cut mismatch"
            break

    # Recompute all reported energies from the returned state, including on
    # time/iteration limits; never relabel a relaxed objective as true H.
    incumbent = score(incumbent["sample"])
    sample = incumbent["sample"]
    placements = {vm.vm_id: tuple(h for h in range(m) if sample[x_var(vm.vm_id, h)])
                  for vm in vms}
    gap = None if lower_bound is None else max(
        0.0, incumbent["incumbent_true_H"] - lower_bound) / max(
            abs(incumbent["incumbent_true_H"]), 1e-10)
    return {**incumbent, "status": "OPTIMAL" if proven else "UNPROVEN",
            "reason": reason, "proven_optimal": bool(proven), "converged": converged,
            "lower_bound": lower_bound, "milp_objective": objective, "mip_gap": gap,
            "milp_gap": getattr(last, "mip_gap", None),
            "milp_status": getattr(last, "status", None),
            "iterations": len(history), "history": history,
            "max_violation": max_violation, "max_weighted_violation": max_weighted,
            "time_s": time.perf_counter() - start, "bqm_offset": float(bqm.offset),
            "assignment": {vm_id: hosts[0] for vm_id, hosts in placements.items()
                           if len(hosts) == 1}, "placements": placements,
            "hosts": {h for h in range(m) if sample[y_var(h)]},
            "validation": state_validation(sample, vms, servers),
            "full_cuts": full_cuts, "lambdas": dict(lam), "center": center,
            "encoding": encoding}
