"""Step 1.4: exact numpy statevector simulation of QAOA and Grover Adaptive Search on a diagonal cost.

Index convention (as s1_stage1): basis state i has y_h = bit h of i; qubit h <-> server h.
QAOA: |+>^n, then p layers of exp(-i*gamma*C) and the X-mixer exp(-i*beta*sum_h X_h).
GAS (Durr-Hoyer / Gilliam et al.): oracle marks C(y) < threshold; random Grover rotation count
r in {0..ceil(k)-1}, k grows by `growth` on failure (BBHT); every oracle call is counted.
No wall-clock claims: this is a classical simulation (docs/plans/STEP1_PLAN.md, honest-claims rules).
"""
import math

import numpy as np
from scipy.optimize import minimize


def n_qubits(costs):
    n = int(round(math.log2(len(costs))))
    if 2 ** n != len(costs):
        raise ValueError("cost vector length must be a power of two")
    return n


def apply_mixer(psi, beta):
    """exp(-i*beta*X) on every qubit; X commute, so the order does not matter."""
    n = n_qubits(psi)
    c, s = math.cos(beta), -1j * math.sin(beta)
    for q in range(n):
        view = psi.reshape(2 ** (n - 1 - q), 2, 2 ** q)   # axis 1 = bit q of the index
        zero, one = view[:, 0, :].copy(), view[:, 1, :].copy()
        view[:, 0, :] = c * zero + s * one
        view[:, 1, :] = s * zero + c * one
    return psi


def qaoa_state(costs, gammas, betas):
    costs = np.asarray(costs, dtype=float)
    psi = np.full(len(costs), 1 / math.sqrt(len(costs)), dtype=complex)
    for gamma, beta in zip(gammas, betas, strict=True):
        psi *= np.exp(-1j * gamma * costs)
        apply_mixer(psi, beta)
    return psi


def probabilities(psi):
    p = np.abs(psi) ** 2
    return p / p.sum()


def expectation(costs, gammas, betas):
    return float(probabilities(qaoa_state(costs, gammas, betas)) @ np.asarray(costs, dtype=float))


def optimise_qaoa(costs, p_max, seeds, maxiter):
    """COBYLA per depth; one restart per seed (random p=1 start); layer-wise init for p>1
    (previous optimum of the same restart + a zero layer, so a restart never gets worse with p).
    Returns one row per p with the best restart; nfev counts every circuit evaluation of all restarts."""
    chains = {}
    rows = []
    total = 0
    for p in range(1, p_max + 1):
        best, nfev = None, 0
        for seed in seeds:
            if p == 1:
                rng = np.random.default_rng(seed)
                start = np.array([rng.uniform(0, math.pi), rng.uniform(0, math.pi / 2)])
            else:
                g, b = chains[seed]
                start = np.concatenate([g, [0.0], b, [0.0]])
            result = minimize(lambda x: expectation(costs, x[:p], x[p:]), start, method="COBYLA",
                              options={"maxiter": maxiter, "rhobeg": 0.5})
            nfev += int(result.nfev)
            chains[seed] = (result.x[:p], result.x[p:])
            if best is None or result.fun < best[0]:
                best = (float(result.fun), seed, result.x.copy())
        total += nfev
        rows.append({"p": p, "expected": best[0], "seed": best[1], "gammas": best[2][:p].tolist(),
                     "betas": best[2][p:].tolist(), "nfev": nfev, "nfev_cumulative": total})
    return rows


def grover_state(marked, rotations):
    """Uniform state after `rotations` Grover iterations (oracle = sign flip on marked, then diffusion)."""
    marked = np.asarray(marked, dtype=bool)
    psi = np.full(len(marked), 1 / math.sqrt(len(marked)), dtype=complex)
    for _ in range(rotations):
        psi[marked] *= -1
        psi = 2 * psi.mean() - psi
    return psi


def gas(costs, rng, max_calls, growth=1.2, stop=None):
    """Grover Adaptive Search for the minimum of `costs`. `stop` (evaluation only) ends the run as soon
    as one of those indices is measured; GAS itself has no stopping certificate.
    Returns the measurement history [(oracle_calls_so_far, index, cost)] and the counters."""
    costs = np.asarray(costs, dtype=float)
    size = len(costs)
    stop = set() if stop is None else set(stop)
    index = int(rng.integers(size))                 # first threshold: one uniform sample, 0 oracle calls
    threshold, calls, k = costs[index], 0, 1.0
    history = [(calls, index, float(costs[index]))]
    while index not in stop and calls < max_calls:
        rotations = int(rng.integers(0, math.ceil(k)))
        psi = grover_state(costs < threshold, rotations)
        calls += rotations
        index = int(rng.choice(size, p=probabilities(psi)))
        history.append((calls, index, float(costs[index])))
        if costs[index] < threshold:
            threshold, k = costs[index], 1.0
        else:
            k = min(growth * k, math.sqrt(size))
    return {"history": history, "oracle_calls": calls, "measurements": len(history),
            "best": min(history, key=lambda h: (h[2], h[0]))[1]}


def first_hit(history, targets):
    """(oracle calls, measurements) when any target was first measured, or None."""
    for count, (calls, index, _) in enumerate(history, start=1):
        if index in targets:
            return calls, count
    return None


def walsh_coefficients(costs):
    """C(y) = sum_S w_S prod_{h in S} z_h with z_h = 1 - 2*y_h; fast Walsh-Hadamard, O(N log N)."""
    w = np.array(costs, dtype=float)
    h = 1
    while h < len(w):
        w = w.reshape(-1, 2, h)
        w = np.stack([w[:, 0] + w[:, 1], w[:, 0] - w[:, 1]], axis=1).reshape(-1)
        h *= 2
    return w / len(w)
