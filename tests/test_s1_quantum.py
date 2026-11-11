"""Step 1.4: numpy statevector QAOA / Grover / GAS checked against analytic formulas and brute force."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.d0_formulation import tiny_problem
from diagnostics.s1_quantum import (apply_mixer, first_hit, gas, grover_state, optimise_qaoa,
                                    probabilities, qaoa_state, walsh_coefficients)
from diagnostics.s1_stage1 import all_costs, stage1_data
from phase2 import config


def test_norm_is_preserved():
    rng = np.random.default_rng(0)
    costs = rng.uniform(size=2 ** 5)
    psi = qaoa_state(costs, rng.uniform(0, 3, 4), rng.uniform(0, 3, 4))
    assert np.linalg.norm(psi) == pytest.approx(1, abs=1e-12)
    psi = grover_state(costs < 0.3, 7)
    assert np.linalg.norm(psi) == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("gamma,beta", [(0.3, 0.2), (1.7, -0.9), (2.5, 1.1)])
def test_one_qubit_p1_matches_analytic(gamma, beta):
    c0, c1 = 0.8, 0.15
    p0 = probabilities(qaoa_state([c0, c1], [gamma], [beta]))[0]
    # |+>, exp(-i g C), exp(-i b X):  P(0) = (1 + sin(2b) sin(g (c0 - c1))) / 2
    assert p0 == pytest.approx((1 + math.sin(2 * beta) * math.sin(gamma * (c0 - c1))) / 2, abs=1e-12)


def test_mixer_acts_on_the_bit_of_the_index():
    psi = np.zeros(8, dtype=complex)
    psi[0] = 1
    apply_mixer(psi, math.pi / 2)                    # exp(-i pi/2 X) = -i X on each qubit
    assert abs(psi[7]) == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("n,marked,rotations", [(6, [3], 0), (6, [3], 4), (6, [1, 9, 40], 2), (8, [5, 6], 9)])
def test_grover_matches_sin_squared(n, marked, rotations):
    mask = np.zeros(2 ** n, dtype=bool)
    mask[marked] = True
    theta = math.asin(math.sqrt(len(marked) / 2 ** n))
    probability = probabilities(grover_state(mask, rotations))[mask].sum()
    assert probability == pytest.approx(math.sin((2 * rotations + 1) * theta) ** 2, abs=1e-12)


def test_walsh_round_trip_and_linear_cost_has_n_plus_1_terms():
    n = 4
    rng = np.random.default_rng(1)
    a, b = rng.uniform(size=n), 0.7
    y = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
    linear = b + y @ a
    w = walsh_coefficients(linear)
    assert np.sum(np.abs(w) > 1e-12) == n + 1
    z = 1 - 2 * y
    rebuilt = [sum(w[s] * np.prod([z[i, h] for h in range(n) if s >> h & 1]) for s in range(2 ** n))
               for i in range(2 ** n)]
    assert np.allclose(rebuilt, linear, atol=1e-12)


def test_gas_counts_calls_and_finds_minimum():
    rng = np.random.default_rng(2)
    costs = rng.permutation(np.linspace(0, 1, 2 ** 7))
    target = int(np.argmin(costs))
    result = gas(costs, np.random.default_rng(3), max_calls=10 ** 4, stop={target})
    calls = [h[0] for h in result["history"]]
    assert calls == sorted(calls) and result["oracle_calls"] == calls[-1]
    assert result["best"] == target
    assert first_hit(result["history"], {target}) == (calls[-1], len(calls))


@pytest.fixture(scope="module")
def tiny_costs():
    vms, servers, _, _, _ = tiny_problem()
    data = stage1_data(vms, servers, config.UNIT_NCU, 1.1)
    return all_costs(data) / data["M"]


def test_tiny_instance_qaoa_finds_bruteforce_optimum_at_larger_p(tiny_costs):
    rows = optimise_qaoa(tiny_costs, p_max=6, seeds=[0, 1, 2], maxiter=400)
    expected = [r["expected"] for r in rows]
    assert all(b <= a + 1e-12 for a, b in zip(expected, expected[1:]))   # layer-wise: never worse
    best = rows[-1]
    probs = probabilities(qaoa_state(tiny_costs, best["gammas"], best["betas"]))
    target = int(np.argmin(tiny_costs))
    assert int(np.argmax(probs)) == target and probs[target] >= 0.5, (probs, tiny_costs)
