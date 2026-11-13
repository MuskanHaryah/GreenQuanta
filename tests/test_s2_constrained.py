"""Step 2C.2: constraint-preserving annealer (s2_constrained) — equivalence and minimum on the 1.2 tiny instance,
PIMC correctness against exact matrices (CHANGES #8c/#8d), conversion to Q_y samples, reproducibility."""
from itertools import product
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.d0_formulation import independent_terms
from diagnostics.exact_qubo import bqm_agrees, with_slack
from diagnostics.s1_stage1 import bits
from diagnostics.s2_constrained import (anneal, build_model, constrained_sample, energy_of, log_coupling, loads,
                                        schedule, trotter_weights)
from diagnostics.s2_stage2 import stage2_bqm
from phase1.qubo_builder import x_var, y_var
from phase2.validation import validate_allocation
from test_s1_stage1 import FACTOR, TOL, tiny  # noqa: F401  (module fixture reused)

TV_TOL = 0.03          # total-variation tolerance, fixed before the first run (CHANGES #8c)
NOISE_FRACTION = 0.2   # sample count chosen so the expected sampling TV is <= TV_TOL * NOISE_FRACTION


def all_y(tiny):
    m = tiny["m"]
    for index in range(1, 2 ** m):
        y = bits(index, m)
        q, qinfo = stage2_bqm(tiny["vms"], tiny["servers"], y, tiny["unit"], FACTOR)
        yield index, y, q, qinfo


def full_sample(tiny, qinfo, assignment):
    """Q_y-domain sample of a host-index assignment, slack at its minimiser (as constrained_sample builds it)."""
    vms, m, info, on = tiny["vms"], tiny["m"], qinfo["info"], qinfo["on"]
    base = {y_var(h): int(h in on) for h in range(m)}
    base.update({x_var(v.vm_id, h): 0 for v in vms for h in range(m)})
    base.update({x_var(v.vm_id, on[j]): 1 for v, j in zip(vms, assignment)})
    demand = [[info["cpu_units"][v.vm_id] for v in vms]] * m + [[info["mem_units"][v.vm_id] for v in vms]] * m
    return with_slack(base, vms, info["k_cpu"] + info["k_mem"], demand, m)


def test_g_equals_slack_eliminated_q_y_on_every_one_hot_state(tiny):
    """2C.1 Prop. 1 on the tiny instance: every y, every assignment."""
    checked = 0
    for _, _, q, qinfo in all_y(tiny):
        model = build_model(qinfo, tiny["vms"])
        states = np.array(list(product(range(len(qinfo["on"])), repeat=len(tiny["vms"]))))
        g = energy_of(model, states)
        for assignment, g_val in zip(states, g):
            full = full_sample(tiny, qinfo, assignment)
            h = math.fsum(independent_terms(full, tiny["vms"], tiny["servers"], qinfo["info"],
                                            qinfo["energy"]).values())
            assert abs(g_val - h) <= 1e-9 * max(1.0, abs(h)), (assignment, g_val, h)
            z = {k: full[k] for k in q.variables}
            assert bqm_agrees(q, z, h, float(q.energy(z)))
            checked += 1
    assert checked > 0


def test_min_g_equals_fixed_y_minimum(tiny):
    """2C.1 Prop. 2: min G_y = V(y) when a quantized-feasible placement exists; otherwise min G_y >= lambda_y."""
    vms, servers, unit = tiny["vms"], tiny["servers"], tiny["unit"]
    feasible_seen = infeasible_seen = 0
    for index, _, _, qinfo in all_y(tiny):
        model, on = build_model(qinfo, vms), qinfo["on"]
        states = np.array(list(product(range(len(on)), repeat=len(vms))))
        g = energy_of(model, states)
        feasible = [validate_allocation(vms, {v.vm_id: on[j] for v, j in zip(vms, s)}, servers,
                                        quantized_unit=unit)["valid"] for s in states]
        if any(feasible):
            assert abs(g.min() - tiny["V"][index]) <= 1e-9, (index, g.min(), tiny["V"][index])
            assert feasible[int(np.argmin(g))]
            feasible_seen += 1
        else:
            assert g.min() >= qinfo["lam"] - TOL
            infeasible_seen += 1
    assert feasible_seen and infeasible_seen


@pytest.mark.parametrize("a", [0.05, 0.5, 2.0])
def test_trotter_weights(a):
    """2C.1 Prop. 4: q = 2 gives w_diff / w_same = tanh(a), i.e. the standard coupling -(1/2) ln tanh(a);
    for q = 3, 5 the weights equal the entries of e^{a(J - I)} from numpy eigh."""
    w_same, w_diff = trotter_weights(a, 2)
    assert w_diff / w_same == pytest.approx(math.tanh(a), rel=1e-12)
    assert -0.5 * float(log_coupling(a, 2)) == pytest.approx(-0.5 * math.log(math.tanh(a)), rel=1e-12)
    for q in (3, 5):
        vals, vecs = np.linalg.eigh(np.ones((q, q)) - np.eye(q))
        expm = (vecs * np.exp(a * vals)) @ vecs.T
        w_same, w_diff = trotter_weights(a, q)
        assert np.allclose(np.diag(expm), w_same, rtol=1e-12) and w_diff > 0
        assert np.allclose(expm[~np.eye(q, dtype=bool)], w_diff, rtol=1e-12)
        assert float(log_coupling(a, q)) == pytest.approx(math.log(w_diff / w_same), rel=1e-12)


def synthetic():
    """Tiny-shaped model (4 VMs, 3 hosts) with small integer units, so the Gibbs weights are spread (CHANGES #8d)."""
    rng = np.random.default_rng(0)
    return {"cost": rng.uniform(0, 1, (4, 3)), "const": 0.0,
            "units": np.array([[1, 2, 1, 3], [2, 1, 2, 1]], dtype=np.int64),
            "caps": np.array([[3, 4, 3], [4, 3, 3]], dtype=np.int64), "lam": np.array([0.5, 0.5]),
            "on": [0, 1, 2]}


def exact_trotter(model, beta, gamma, p):
    """Slice-0 distribution of the discrete-Trotter path integral: diag((D T)^P), D = e^{-(beta/P) G},
    T = tensor product over VMs of e^{a(J - I)}, a = beta * gamma / P."""
    n, q = model["cost"].shape
    states = np.array(list(product(range(q), repeat=n)))
    d = np.exp(-(beta / p) * energy_of(model, states))
    if gamma > 0:
        w_same, w_diff = trotter_weights(beta * gamma / p, q)
        t = np.prod(np.where(states[:, None, :] == states[None, :, :], w_same, w_diff), axis=2)
    else:
        t = np.eye(len(states))
    prob = np.diag(np.linalg.matrix_power(d[:, None] * t, p))
    return states, prob / prob.sum()


def quantum_gibbs(model, beta, gamma):
    """diag(e^{-beta (G + gamma H_D)}) / Z by numpy eigh, H_D = -sum_v (J - I)_v."""
    n, q = model["cost"].shape
    states = np.array(list(product(range(q), repeat=n)))
    hops = (states[:, None, :] != states[None, :, :]).sum(axis=2) == 1
    ham = np.diag(energy_of(model, states)) - gamma * hops
    vals, vecs = np.linalg.eigh(ham)
    rho = (vecs * np.exp(-beta * (vals - vals.min()))) @ vecs.T
    return np.diag(rho) / np.trace(rho)


@pytest.mark.parametrize("p,gamma", [(1, 0.0), (2, 1.0), (4, 1.0)])
def test_pimc_matches_exact_trotter_distribution(p, gamma):
    """CHANGES #8c (i): sampler slice-0 histogram vs the exact discrete-Trotter distribution (P = 1: Boltzmann)."""
    model, beta = synthetic(), 2.0
    states, prob = exact_trotter(model, beta, gamma, p)
    noise_per_sqrt_n = 0.5 * np.sum(np.sqrt(2 * prob * (1 - prob) / math.pi))
    reads = int(math.ceil((noise_per_sqrt_n / (TV_TOL * NOISE_FRACTION)) ** 2))
    sweeps = 60
    ell = np.full(sweeps, float(log_coupling(beta * gamma / p, 3)) if p > 1 else 0.0)
    hosts, _ = anneal(model, np.full(sweeps, beta / p), ell, p, reads, np.random.default_rng(1))
    codes = hosts[:, 0, :] @ (3 ** np.arange(3, -1, -1))
    hist = np.bincount(codes, minlength=len(states)) / reads
    assert np.array_equal(states @ (3 ** np.arange(3, -1, -1)), np.arange(len(states)))
    tv = 0.5 * np.abs(hist - prob).sum()
    assert tv <= TV_TOL, (tv, reads)


def test_exact_trotter_converges_to_quantum_gibbs():
    """CHANGES #8c (ii): the discrete-Trotter distribution approaches diag(e^{-beta H}) / Z as P grows."""
    model, beta, gamma = synthetic(), 2.0, 1.0
    target = quantum_gibbs(model, beta, gamma)
    tv = [0.5 * np.abs(exact_trotter(model, beta, gamma, p)[1] - target).sum() for p in (8, 32, 128)]
    assert tv[0] > tv[1] > tv[2], tv
    assert tv[2] <= tv[1] / 4 and tv[2] <= 0.02, tv


def test_anneal_loads_reproducible_and_odd_trotter_rejected(tiny):
    *_, qinfo = list(all_y(tiny))[-1]                     # all hosts on: q = m
    model = build_model(qinfo, tiny["vms"])
    q = len(qinfo["on"])
    beta_g, ell = np.full(5, 1.0), np.full(5, float(log_coupling(0.5, q)))
    first = anneal(model, beta_g, ell, 4, 16, np.random.default_rng(7))   # asserts incremental loads internally
    again = anneal(model, beta_g, ell, 4, 16, np.random.default_rng(7))
    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])
    assert first[0].shape == (16, 4, len(tiny["vms"])) and first[0].min() >= 0 and first[0].max() < q
    assert np.allclose(first[1], energy_of(model, first[0]), rtol=1e-12)
    assert loads(model, first[0]).sum(axis=-1).tolist() == [[model["units"].sum(axis=1).tolist()] * 4] * 16
    with pytest.raises(ValueError):
        anneal(model, beta_g, ell, 3, 4, np.random.default_rng(0))


@pytest.mark.parametrize("spec", [
    {"method": "csa", "reads": 8, "sweeps": 20, "beta_min": 0.1, "beta_max": 100.0},
    {"method": "csqa", "reads": 8, "sweeps": 10, "trotter": 2, "beta": 50.0, "gamma": 1.0},
])
def test_constrained_sample_gives_one_hot_q_y_samples(tiny, spec):
    *_, (index, y, q, qinfo) = list(all_y(tiny))
    samples, info = constrained_sample(qinfo, tiny["vms"], tiny["servers"], spec, seed=3)
    assert set(samples.variables) == set(q.variables)
    assert info["move_attempts"] == spec["reads"] * spec["sweeps"] * info["trotter"] * len(tiny["vms"])
    for row in samples.data(fields=["sample", "energy"]):
        sample = dict(row.sample)
        for v in tiny["vms"]:
            assert sum(sample[x_var(v.vm_id, h)] for h in qinfo["on"]) == 1
        raw = float(q.energy(sample))
        assert abs(raw - float(row.energy)) <= 1e-6 * max(1.0, abs(raw))


def test_schedule_shapes():
    beta_g, ell, p = schedule({"method": "csqa", "sweeps": 10, "trotter": 4, "beta": 8.0, "gamma": 2.0}, 0.5, 3)
    assert p == 4 and beta_g.shape == ell.shape == (10,)
    assert np.all(np.diff(beta_g) > 0) and np.all(np.isfinite(ell)) and np.all(ell < 0)
    assert np.all(np.diff(ell) < 0)                           # bonds stiffen as the field goes to 0
    beta_g, ell, p = schedule({"method": "csa", "sweeps": 5, "beta_min": 1.0, "beta_max": 16.0}, 2.0, 3)
    assert p == 1 and beta_g[0] == pytest.approx(0.5) and beta_g[-1] == pytest.approx(8.0)
    with pytest.raises(ValueError):
        schedule({"method": "csqa", "sweeps": 5, "trotter": 2, "beta": 1.0, "gamma": 0.0}, 1.0, 2)
