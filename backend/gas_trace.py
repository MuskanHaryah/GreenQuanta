"""Grover visualisation data: one pipeline GAS run (s1_quantum.gas with the pipeline's budget and growth), replayed
step by step with s1_quantum.grover_state / probabilities.

Replay of the history [(cumulative oracle calls, measured index, cost)]: step 0 is the first uniform sample (no
threshold, 0 rotations); at step i >= 1 the rotation count is the difference in cumulative oracle calls and the marked
set is costs < threshold, with threshold = the best cost measured before step i.
"""
from functools import lru_cache

import numpy as np

from diagnostics.s1_end_to_end import RUN_CONFIG, proposals
from diagnostics.s1_ground_truth import candidates, ground_truth
from diagnostics.s1_quantum import gas, grover_state, probabilities
from diagnostics.s1_stage1 import bits

TOP = 32


def run(size, seed):
    """(servers, costs, budget, gas result) of the pipeline's GAS run with `seed`."""
    _, servers, _, costs, _, _ = candidates(size)
    budget = proposals(size, costs, "gas")[3]["gas_budget"]
    return servers, costs, budget, gas(costs, np.random.default_rng(seed), budget, RUN_CONFIG["gas_growth"])


def replay(costs, history):
    """Yields (threshold or None, rotations, marked mask, probabilities) for each measurement of a GAS history."""
    calls_before, threshold = 0, None
    for calls, _, cost in history:
        marked = np.zeros(len(costs), bool) if threshold is None else costs < threshold
        rotations = calls - calls_before
        yield threshold, rotations, marked, probabilities(grover_state(marked, rotations))
        calls_before = calls
        threshold = cost if threshold is None else min(threshold, cost)


@lru_cache(maxsize=None)
def trace(size, seed, top=TOP):
    servers, costs, budget, result = run(size, seed)
    m, gt = len(servers), ground_truth(size)
    y_star = gt["y_star"]

    def ids(index):
        return [servers[h].server_id for h, on in enumerate(bits(index, m)) if on]

    steps = []
    for step, ((calls, index, cost), (threshold, rotations, marked, p)) in enumerate(
            zip(result["history"], replay(costs, result["history"]))):
        order = np.argsort(-p, kind="stable")[:top]
        steps.append({"step": step, "oracle_calls": calls, "rotations": rotations, "threshold": threshold,
                      "marked_count": int(marked.sum()), "measured_index": index, "measured_cost": cost,
                      "improved": threshold is None or cost < threshold,
                      "top": [{"index": int(i), "ids": ids(int(i)), "p": float(p[i]), "cost": float(costs[i]),
                               "marked": bool(marked[i])} for i in order],
                      "p_y_star": float(p[y_star]), "p_marked_total": float(p[marked].sum())})
    return {"size": size, "seed": seed, "m": m, "states": len(costs), "budget": budget, "y_star_index": y_star,
            "y_star_ids": gt["y_star_ids"], "steps": steps,
            "totals": {"oracle_calls": result["oracle_calls"], "measurements": result["measurements"],
                       "exhaustive": len(costs)}}
