"""Step 1.5b: exact CVaR objective for QAOA."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.s1_quantum import cvar, expectation, probabilities, qaoa_state


def test_cvar_hand_values():
    costs, probs = np.array([1.0, 0.0]), np.array([0.5, 0.5])
    assert cvar(costs, probs, 0.25) == pytest.approx(0.0, abs=1e-15)
    assert cvar(costs, probs, 0.75) == pytest.approx(1 / 3, abs=1e-15)   # (0.5*0 + 0.25*1) / 0.75
    assert cvar(costs, probs, 1.0) == pytest.approx(0.5, abs=1e-15)


def test_cvar_alpha_one_is_mean_and_is_monotone_in_alpha():
    rng = np.random.default_rng(4)
    costs = rng.uniform(size=2 ** 5)
    gammas, betas = rng.uniform(0, 3, 2), rng.uniform(0, 3, 2)
    assert expectation(costs, gammas, betas, 1.0) == pytest.approx(expectation(costs, gammas, betas), abs=1e-12)
    probs = probabilities(qaoa_state(costs, gammas, betas))
    values = [cvar(costs, probs, a) for a in (0.01, 0.05, 0.2, 0.5, 1.0)]
    assert all(a <= b + 1e-15 for a, b in zip(values, values[1:]))
    assert values[0] >= costs.min() - 1e-15
