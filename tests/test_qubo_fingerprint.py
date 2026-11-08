"""Golden QUBO regression checks; golden creation is deliberately manual."""
import json
from pathlib import Path
import sys

import dimod
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.common import build_problem
from phase2.fingerprint import qubo_fingerprint


GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "qubo_small_v2_unbalanced.json"


@pytest.fixture(scope="module")
def small_problem():
    return build_problem("small_v2")


def test_repeated_build_has_same_hash(small_problem):
    _, _, first, _ = small_problem
    _, _, second, _ = build_problem("small_v2")
    assert qubo_fingerprint(first)["sha256"] == qubo_fingerprint(second)["sha256"]


def test_lambda_changes_hash(small_problem):
    _, _, original, _ = small_problem
    _, _, changed, _ = build_problem("small_v2", lambda_config="L0.06")
    assert qubo_fingerprint(original)["sha256"] != qubo_fingerprint(changed)["sha256"]


def test_center_changes_hash(small_problem):
    _, _, original, _ = small_problem
    _, _, changed, _ = build_problem("small_v2", center=0.03)
    assert qubo_fingerprint(original)["sha256"] != qubo_fingerprint(changed)["sha256"]


def test_matches_golden(small_problem):
    assert GOLDEN_PATH.is_file(), (
        f"Missing QUBO golden: {GOLDEN_PATH}. "
        "Create it explicitly with python tests/make_golden_qubo.py; "
        "the test will never create it."
    )
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    _, _, bqm, info = small_problem
    assert qubo_fingerprint(bqm) == golden["fingerprint"]
    assert {key: info[key] for key in ("num_variables", "bias_range", "e_ref_joule")} == golden["info"]


def test_insertion_order_and_edge_orientation_do_not_change_hash():
    first = dimod.BinaryQuadraticModel({"b": 2.0, "a": 1.0}, {("b", "a"): -0.5}, 0.25, dimod.BINARY)
    second = dimod.BinaryQuadraticModel({"a": 1.0, "b": 2.0}, {("a", "b"): -0.5}, 0.25, dimod.BINARY)
    assert qubo_fingerprint(first) == qubo_fingerprint(second)


def test_offset_changes_hash():
    first = dimod.BinaryQuadraticModel({"a": 1.0}, {}, 0.0, dimod.BINARY)
    second = first.copy()
    second.offset = 0.5
    assert qubo_fingerprint(first)["sha256"] != qubo_fingerprint(second)["sha256"]
