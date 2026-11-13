"""Step 2D.1: world-line move on the constraint-preserving sampler (s2_constrained.anneal(worldline=True)),
the one budget formula, and the 2D.0 selection / decision rules. Old tests are unchanged (plan rule 5)."""
import math
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimizer"))

from diagnostics.s2_constrained import (anneal, build_model, constrained_sample, energy_of, log_coupling, loads,
                                        move_attempts)
from phase1.qubo_builder import x_var
from test_s1_stage1 import tiny  # noqa: F401  (module fixture reused)
from test_s2_constrained import NOISE_FRACTION, TV_TOL, all_y, exact_trotter, synthetic

COMMIT_2C = "9155960a"     # last Step 2 commit: the sampler that produced 2C.4 and 2.6


def tiny_model(tiny):
    *_, qinfo = list(all_y(tiny))[-1]                     # all hosts on: q = m
    return build_model(qinfo, tiny["vms"])


@pytest.mark.parametrize("p", [2, 4])
def test_worldline_matches_exact_trotter_distribution(p):
    """2D.1_proof: with world-line moves the stationary law is still diag((D T)^P); same tolerance and sample-count
    rule as test_pimc_matches_exact_trotter_distribution."""
    model, beta, gamma = synthetic(), 2.0, 1.0
    states, prob = exact_trotter(model, beta, gamma, p)
    noise_per_sqrt_n = 0.5 * np.sum(np.sqrt(2 * prob * (1 - prob) / math.pi))
    reads = int(math.ceil((noise_per_sqrt_n / (TV_TOL * NOISE_FRACTION)) ** 2))
    sweeps = 60
    ell = np.full(sweeps, float(log_coupling(beta * gamma / p, 3)))
    stats = []
    hosts, _ = anneal(model, np.full(sweeps, beta / p), ell, p, reads, np.random.default_rng(1), True, stats)
    assert sum(s["wl_accepted"] for s in stats) > 0
    codes = hosts[:, 0, :] @ (3 ** np.arange(3, -1, -1))
    hist = np.bincount(codes, minlength=len(states)) / reads
    tv = 0.5 * np.abs(hist - prob).sum()
    assert tv <= TV_TOL, (tv, reads)


def committed_sampler():
    """s2_constrained as committed at COMMIT_2C, loaded from git (reference for the flag-off reproduction)."""
    src = subprocess.run(["git", "show", f"{COMMIT_2C}:optimizer/diagnostics/s2_constrained.py"], cwd=ROOT,
                         check=True, capture_output=True).stdout.decode("utf-8")
    module = types.ModuleType(f"s2_constrained_{COMMIT_2C}")
    exec(compile(src, module.__name__, "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("p", [1, 4, 8])
def test_worldline_off_reproduces_committed_sampler(tiny, p):
    """worldline=False (default) and stats recording give the committed 2C sampler's output bit for bit."""
    old = committed_sampler()
    for model in (tiny_model(tiny), synthetic()):
        q = model["cost"].shape[1]
        beta_g = np.geomspace(0.01, 5.0, 12)
        ell = np.asarray(log_coupling(np.linspace(2.0, 0.05, 12), q)) if p > 1 else np.zeros(12)
        ref = old.anneal(model, beta_g, ell, p, 32, np.random.default_rng(11))
        for kwargs in ({}, {"worldline": False}, {"worldline": False, "stats": []}):
            new = anneal(model, beta_g, ell, p, 32, np.random.default_rng(11), **kwargs)
            assert np.array_equal(ref[0], new[0]) and np.array_equal(ref[1], new[1]), kwargs


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_worldline_loads_consistent_and_reproducible(tiny, beta):
    """anneal asserts incremental loads == recomputed loads after the run and G after every sweep; here also
    energies, totals, reproducibility, and that world-line moves were made (beta 0: every one is accepted)."""
    model = tiny_model(tiny)
    q, n = model["cost"].shape[1], len(tiny["vms"])
    beta_g, ell = np.full(6, beta), np.full(6, float(log_coupling(0.5, q)))
    stats = []
    first = anneal(model, beta_g, ell, 4, 16, np.random.default_rng(7), True, stats)
    again = anneal(model, beta_g, ell, 4, 16, np.random.default_rng(7), True)
    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])
    assert first[0].shape == (16, 4, n) and first[0].min() >= 0 and first[0].max() < q
    assert np.allclose(first[1], energy_of(model, first[0]), rtol=1e-12)
    assert loads(model, first[0]).sum(axis=-1).tolist() == [[model["units"].sum(axis=1).tolist()] * 4] * 16
    assert len(stats) == 6 and all(s["wl_proposed"] == 16 * n for s in stats)
    accepted = sum(s["wl_accepted"] for s in stats)
    assert accepted > 0 and (beta > 0 or accepted == sum(s["wl_proposed"] for s in stats))


def test_constrained_sample_worldline_one_hot_and_budget(tiny):
    *_, (index, y, q, qinfo) = list(all_y(tiny))
    spec = {"method": "csqa", "reads": 8, "sweeps": 5, "trotter": 2, "beta": 50.0, "gamma": 1.0, "worldline": True}
    samples, info = constrained_sample(qinfo, tiny["vms"], tiny["servers"], spec, seed=3)
    assert info["move_attempts"] == 2 * spec["reads"] * spec["sweeps"] * 2 * len(tiny["vms"])
    for row in samples.data(fields=["sample", "energy"]):
        sample = dict(row.sample)
        for v in tiny["vms"]:
            assert sum(sample[x_var(v.vm_id, h)] for h in qinfo["on"]) == 1
        raw = float(q.energy(sample))
        assert abs(raw - float(row.energy)) <= 1e-6 * max(1.0, abs(raw))


def test_worldline_budget_formula():
    """2D.0: a world-line move counts P attempts; csqa sweeps = floor(base x level / 2P); never above B."""
    from diagnostics.s2_sa_vs_sqa import RUN_CONFIG, spec
    base, n, reads = RUN_CONFIG["base_sa_sweeps"], 7, 10
    assert move_attempts({"reads": 2, "sweeps": 3}, 4, 5) == 120
    assert move_attempts({"reads": 2, "sweeps": 3, "worldline": True}, 4, 5) == 240
    for level in (1, 4):
        b = move_attempts(spec("csa", reads, level, {"beta_min": 0.1, "beta_max": 1e3}), 1, n)
        for p in (8, 16):
            s = spec("csqa", reads, level, {"beta": 1e3, "gamma": 1.0, "trotter": p, "worldline": True})
            assert s["sweeps"] == base * level // (2 * p)
            got = move_attempts(s, p, n)
            assert got <= b and b - got < reads * 2 * p * n
            assert (got == b) == ((base * level) % (2 * p) == 0)
    no_wl = spec("csqa", reads, 1, {"beta": 1e3, "gamma": 1.0, "trotter": 8, "worldline": False})
    assert no_wl["sweeps"] == base // 8                    # flag off: the 2C spec


def test_select_rule():
    from diagnostics.s2_sa_vs_sqa import select
    row = lambda order, p, v, g: {"order": order, "p_opt": p, "valid_pct": v, "mean_g": g}  # noqa: E731
    best, by, sat = select([row(0, 0.5, 100, 1.0), row(1, 0.6, 90, 2.0), row(2, 0.6, 95, 3.0)], 1e-9)
    assert (best["order"], by, sat) == (2, "valid_pct", False)
    best, by, sat = select([row(0, 0.6, 95, 2.0), row(1, 0.6 + 1e-12, 95, 1.0)], 1e-9)
    assert (best["order"], by) == (1, "mean_g")
    best, by, sat = select([row(0, 1.0, 100, 1.0), row(1, 1.0, 100, 1.0 + 1e-12)], 1e-9)
    assert (best["order"], by, sat) == (0, "grid order", True)
    best, by, sat = select([row(3, 1.0, 100, 1.0), row(1, 0.9, 100, 0.5)], 1e-9)
    assert (best["order"], by, sat) == (3, "p_opt", False)


def test_decide_rule():
    from diagnostics.s2_sa_vs_sqa import decide
    run = lambda seed, opt, gap: {"seed": seed, "status": "DONE", "pure_and_optimal_count": opt,  # noqa: E731
                                  "sampler": {"reads": 200}, "gap_pct": gap}
    sa = [run(0, 10, 0.0), run(1, 0, 0.5), run(2, 0, None)]
    assert decide([run(0, 20, 0.0), run(1, 0, 0.4), run(2, 0, 3.0)], sa, 1e-9)[0] == "SQA better"
    assert decide([run(0, 5, 0.0), run(1, 0, 0.6), run(2, 0, None)], sa, 1e-9) == (
        "no significant difference at this budget", ["SA", "SA", "tie"])
    assert decide([run(0, 5, 0.0), run(1, 0, 0.6), run(2, 0, None)], sa[:2] + [run(2, 0, 1.0)], 1e-9)[0] == \
        "SA better"
    with pytest.raises(ValueError):
        decide([run(1, 0, 0.0)], [run(0, 0, 0.0)], 1e-9)
