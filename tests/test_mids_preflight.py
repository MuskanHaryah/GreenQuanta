"""Guide 1 step 1.0: the pinned pipeline inputs and R1 references are unchanged."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from demo.preflight import PINNED, check


def test_pipeline_inputs_match_pinned_hashes():
    assert PINNED.exists()
    assert check() == []
