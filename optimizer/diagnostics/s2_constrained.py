"""Step 2C.2: constraint-preserving annealing for Stage 2 (proofs: docs/reports/step2/2C.1_proof.md).

State: one host index per VM, h: V -> ON(y) (q = |ON(y)| hosts), so a VM is always placed exactly once.
Problem energy (diagonal), from the same coefficients, units and lambda as Q_y (s2_stage2.stage2_bqm):
    G_y(h) = const + sum_v c[v, h(v)] + sum_{r in cpu,ram} lambda_r * sum_j max(0, L_rj(h) - K_rj)^2
which equals Q_y with the slack bits at their minimiser (2C.1 Prop. 1).
Driver: H_D = -Gamma * sum_v sum_{a != b in ON} |b><a|_v (relocate one VM; stoquastic; q = 2 -> -Gamma sigma^x).

One sampler, anneal(), serves both methods (2C plan: no second sampler):
  * SQA = path-integral Monte Carlo of H(s) = (1 - s) * Gamma * H_D + s * G_y at fixed beta, P Trotter slices
    (P even: slices are updated in two checkerboard halves). Per VM and slice bond, the imaginary-time weight is
    e^{a(J - I)} with a = beta * Gamma * (1 - s) / P: w_same = (e^{a(q-1)} + (q-1) e^{-a}) / q,
    w_diff = (e^{a(q-1)} - e^{-a}) / q (2C.1 Prop. 4). s_t = (t + 1/2) / sweeps, so s < 1 and a > 0.
  * Classical SA = the same relocate moves with P = 1 and no driver, beta geometric from beta_min to beta_max.
Units: beta and Gamma are dimensionless; G_y is divided by lambda_y (the cpu weight), so beta = 1 means one unit of
overload costs one temperature unit. beta * Gamma is unchanged by that scaling.
Move: pick VM v (fixed order), propose a uniformly random other host (symmetric), Metropolis on
-(beta_t / P) * dG + ell_t * d(mismatched bonds), ell = ln(w_diff / w_same).
Budget unit: move attempts = reads x sweeps x P x n_VMs (both methods).
Output: final state of slice 0 (CHANGES #8f), converted to a Q_y sample with the slack bits at their minimiser
(exact_qubo.with_slack), scored by the shared path in r3c_run.sample_worker (methods "csa", "csqa").
Tracked G is checked against an exact recomputation after every sweep (relative 1e-6, then resynced).
"""
import math

import dimod
import numpy as np

from diagnostics.exact_qubo import with_slack
from phase1.qubo_builder import x_var, y_var


def build_model(qinfo, vms):
    """Arrays of G_y from stage2_bqm's info: cost [n, q], const, units [2, n], caps [2, q], lam [2], on."""
    energy, info, on = qinfo["energy"], qinfo["info"], qinfo["on"]
    if any(b != 0 for b in energy.quadratic.values()):
        raise ValueError("G_y needs an energy part linear in x and y (2C.1 Prop. 1)")
    return {"cost": np.array([[float(energy.linear[x_var(v.vm_id, h)]) for h in on] for v in vms]),
            "const": float(energy.offset) + math.fsum(float(energy.linear[y_var(h)]) for h in on),
            "units": np.array([[info["cpu_units"][v.vm_id] for v in vms],
                               [info["mem_units"][v.vm_id] for v in vms]], dtype=np.int64),
            "caps": np.array([[info["k_cpu"][h] for h in on], [info["k_mem"][h] for h in on]], dtype=np.int64),
            "lam": np.array([info["lambdas"]["cpu"], info["lambdas"]["ram"]], dtype=float), "on": list(on)}


def loads(model, hosts):
    """Integer loads [..., 2, q] of assignments hosts [..., n]."""
    q = model["caps"].shape[1]
    onehot = (hosts[..., None] == np.arange(q)).astype(np.int64)
    return np.einsum("...vq,rv->...rq", onehot, model["units"])


def energy_of(model, hosts):
    """Exact G_y of assignments hosts [..., n]."""
    n = hosts.shape[-1]
    over = np.maximum(0, loads(model, hosts) - model["caps"]).astype(float)
    return model["const"] + model["cost"][np.arange(n), hosts].sum(-1) + (over ** 2).sum(-1) @ model["lam"]


def trotter_weights(a, q):
    """(w_same, w_diff) of e^{a(J - I)} on q hosts (2C.1 Prop. 4)."""
    return (math.exp(a * (q - 1)) + (q - 1) * math.exp(-a)) / q, (math.exp(a * (q - 1)) - math.exp(-a)) / q


def log_coupling(a, q):
    """ln(w_diff / w_same) = ln(1 - e^{-aq}) - ln(1 + (q - 1) e^{-aq}), stable for a > 0 (array or scalar)."""
    e = np.exp(-np.asarray(a, dtype=float) * q)
    return np.log1p(-e) - np.log1p((q - 1) * e)


def anneal(model, beta_g, ell, trotter, reads, rng):
    """Run len(beta_g) sweeps. beta_g[t]: weight of dG per slice; ell[t]: log bond ratio (ignored if P = 1).
    Returns (hosts [reads, P, n], tracked G [reads, P])."""
    n, q = model["cost"].shape
    units, caps, lam, cost = model["units"], model["caps"], model["lam"], model["cost"]
    if trotter != 1 and trotter % 2:
        raise ValueError("trotter must be 1 or even (checkerboard slice update)")
    hosts = rng.integers(0, q, size=(reads, trotter, n))
    load = loads(model, hosts)
    g_track = energy_of(model, hosts)
    groups = [np.arange(trotter)] if trotter == 1 else [np.arange(0, trotter, 2), np.arange(1, trotter, 2)]

    def pen(x):
        return np.maximum(0, x).astype(float) ** 2

    for t in range(len(beta_g)):
        for g in groups if q > 1 else []:
            left, right = (g - 1) % trotter, (g + 1) % trotter
            for v in range(n):
                old = hosts[:, g, v]
                new = (old + rng.integers(1, q, size=old.shape)) % q
                sub = load[:, g]
                shape = old.shape + (2, 1)
                lo = np.take_along_axis(sub, np.broadcast_to(old[..., None, None], shape), axis=3)[..., 0]
                ln = np.take_along_axis(sub, np.broadcast_to(new[..., None, None], shape), axis=3)[..., 0]
                ko, kn, av = caps.T[old], caps.T[new], units[:, v]
                dpen = pen(ln + av - kn) - pen(ln - kn) + pen(lo - av - ko) - pen(lo - ko)
                d_g = cost[v, new] - cost[v, old] + dpen @ lam
                log_acc = -beta_g[t] * d_g
                if trotter > 1:
                    hl, hr = hosts[:, left, v], hosts[:, right, v]
                    mismatch = ((new != hl).astype(int) + (new != hr)) - ((old != hl).astype(int) + (old != hr))
                    log_acc = log_acc + ell[t] * mismatch
                accept = np.log1p(-rng.random(old.shape)) < log_acc
                ri, ci = np.nonzero(accept)
                if ri.size:
                    sl, o, w = g[ci], old[ri, ci], new[ri, ci]
                    hosts[ri, sl, v] = w
                    for r in range(2):
                        load[ri, sl, r, o] -= units[r, v]
                        load[ri, sl, r, w] += units[r, v]
                    g_track[ri, sl] += d_g[ri, ci]
        exact = energy_of(model, hosts)
        if not np.all(np.abs(g_track - exact) <= 1e-6 * np.maximum(1.0, np.abs(exact))):
            raise AssertionError(("tracked G drifted", float(np.max(np.abs(g_track - exact)))))
        g_track = exact
    assert np.array_equal(load, loads(model, hosts)), "incremental loads diverged"
    return hosts, g_track


def schedule(spec, scale, q):
    """(beta_g, ell, P) per sweep. csa: geometric beta, P = 1. csqa: s_t = (t + 1/2)/T, fixed beta, P slices."""
    sweeps = spec["sweeps"]
    if spec["method"] == "csa":
        return np.geomspace(spec["beta_min"], spec["beta_max"], sweeps) / scale, np.zeros(sweeps), 1
    if spec["method"] != "csqa" or not spec["gamma"] > 0:
        raise ValueError("csqa needs gamma > 0")
    p, s = spec["trotter"], (np.arange(sweeps) + 0.5) / sweeps
    return spec["beta"] * s / (p * scale), log_coupling(spec["beta"] * spec["gamma"] * (1 - s) / p, q), p


def constrained_sample(qinfo, vms, servers, spec, seed):
    """Reads of method spec["method"] ("csa" | "csqa") on G_y. Returns (SampleSet over Q_y's variables with
    slack at its minimiser, energies = tracked G of slice 0; info)."""
    model = build_model(qinfo, vms)
    q, m = len(model["on"]), len(servers)
    beta_g, ell, trotter = schedule(spec, float(model["lam"][0]), q)
    hosts, g = anneal(model, beta_g, ell, trotter, spec["reads"], np.random.default_rng(seed))
    info = qinfo["info"]
    caps = info["k_cpu"] + info["k_mem"]
    demand = [[info["cpu_units"][v.vm_id] for v in vms]] * m + [[info["mem_units"][v.vm_id] for v in vms]] * m
    keep = set(qinfo["full"].variables) - set(qinfo["fixed"])
    states = []
    for read in hosts[:, 0, :]:
        base = {y_var(h): int(h in model["on"]) for h in range(m)}
        base.update({x_var(v.vm_id, h): 0 for v in vms for h in range(m)})
        base.update({x_var(v.vm_id, model["on"][j]): 1 for v, j in zip(vms, read)})
        full = with_slack(base, vms, caps, demand, m)
        states.append({k: full[k] for k in keep})
    samples = dimod.SampleSet.from_samples(states, dimod.BINARY, [float(e) for e in g[:, 0]])
    return samples, {"trotter": trotter, "move_attempts": spec["reads"] * spec["sweeps"] * trotter * len(vms)}
