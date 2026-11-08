import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from phase2 import pipeline_v2


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.setattr(pipeline_v2, "evaluate_allocation",
                        lambda vms, assignment, servers: {"it_energy_kwh": 10.0})
    monkeypatch.setattr(pipeline_v2, "linear_cost",
                        lambda assignment, vms, servers: 1000.5)

    def compute(proven_optimal=True, pure_found=True, exact_available=True,
                include_mip_gap=True):
        exact_info = {"objective_joule": 1000.0,
                      "energy": {"it_energy_kwh": 10.0},
                      "proven_optimal": proven_optimal}
        if include_mip_gap:
            exact_info["mip_gap"] = 0.25
        result = pipeline_v2.gap_metrics_v2(
            {"vm": "final"}, {"vm": "pure"} if pure_found else None,
            object(), object(), {"vm": "exact"} if exact_available else None,
            exact_info, {})
        return result, exact_info

    return compute


def test_proven_optimum_matches_within_tolerance(metrics):
    result, _ = metrics()
    assert result["final_matches_optimum"] is True
    assert result["pure_and_optimal"] is True


def test_unproven_incumbent_is_not_labelled_optimal(metrics):
    result, _ = metrics(proven_optimal=False)
    assert result["final_matches_optimum"] is False
    assert result["pure_and_optimal"] is False


def test_no_exact_assignment(metrics):
    result, _ = metrics(exact_available=False)
    assert result == {"exact_available": False}


def test_no_pure_assignment(metrics):
    result, _ = metrics(pure_found=False)
    assert result["annealer_only_found"] is False
    assert result["pure_and_optimal"] is False


def test_existing_key_set_is_preserved(metrics):
    result, _ = metrics()
    new_keys = {"final_matches_reference_linear", "pure_matches_reference_linear",
                "exact_mip_gap"}
    assert set(result) - new_keys == {
        "exact_available", "exact_proven_optimal", "final_gap_energy_pct",
        "final_matches_optimum", "final_beats_linear_optimum_on_curve",
        "annealer_only_found", "annealer_only_gap_energy_pct",
        "annealer_only_servers", "pure_and_optimal"}


def test_reference_matches_do_not_require_proof(metrics):
    result, _ = metrics(proven_optimal=False)
    assert result["final_matches_reference_linear"] is True
    assert result["pure_matches_reference_linear"] is True


@pytest.mark.parametrize("include_mip_gap", [True, False])
def test_exact_mip_gap_is_forwarded(metrics, include_mip_gap):
    result, exact_info = metrics(include_mip_gap=include_mip_gap)
    assert result["exact_mip_gap"] == exact_info.get("mip_gap")
