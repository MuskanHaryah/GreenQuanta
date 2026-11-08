import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from phase2.solver_v2 import rank_candidates


def test_lower_energy_repaired_wins_by_default():
    repaired = (True, 1.0, {"vm": "repaired"})
    unrepaired = (False, 2.0, {"vm": "unrepaired"})
    assert rank_candidates([unrepaired, repaired]) is repaired


def test_prefer_unrepaired_preserves_old_ordering():
    repaired = (True, 1.0, {"vm": "repaired"})
    unrepaired = (False, 2.0, {"vm": "unrepaired"})
    assert rank_candidates([unrepaired, repaired], prefer_unrepaired=True) is unrepaired


@pytest.mark.parametrize("delta", [0.0, 5e-13])
@pytest.mark.parametrize("reverse", [False, True])
def test_equal_energy_prefers_unrepaired(delta, reverse):
    repaired = (True, 1.0, {"vm": "repaired"})
    unrepaired = (False, 1.0 + delta, {"vm": "unrepaired"})
    candidates = [repaired, unrepaired]
    assert rank_candidates(candidates[::-1] if reverse else candidates) is unrepaired


@pytest.mark.parametrize("prefer_unrepaired", [False, True])
def test_empty_candidates(prefer_unrepaired):
    assert rank_candidates([], prefer_unrepaired=prefer_unrepaired) is None


def test_lowest_energy_of_three_mixed_candidates():
    candidates = [(False, 3.0, {"vm": "a"}),
                  (True, 1.0, {"vm": "b"}),
                  (True, 2.0, {"vm": "c"})]
    assert rank_candidates(candidates) is candidates[1]
