"""Guide 1 step 1.2: the frozen pipeline reproduces the golden snapshot on small_v2 and medium_v2 (no log rows)."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from demo.freeze_check import differences
from diagnostics.s1_end_to_end import hybrid


@pytest.mark.parametrize("size", ["small_v2", "medium_v2"])
@pytest.mark.parametrize("method", ["csqa", "csa"])
def test_pipeline_matches_golden(size, method):
    out, assignment = hybrid(size, "gas", method, "2D", log=False)
    assert differences(size, method, out) == []
    assert assignment is not None and len(assignment) == len(set(assignment))
