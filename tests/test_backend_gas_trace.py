"""Guide 2 step 2.3: the GAS trace replays the pipeline's GAS run faithfully.

A recording wrapper around the run's numpy Generator captures the probability vector s1_quantum.gas sampled every
measurement from; the replay must reproduce each one exactly, and the trace's measured indices must equal gas()'s history.
"""
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import gas_trace
from backend.app import create_app
from diagnostics.s1_end_to_end import RUN_CONFIG
from diagnostics.s1_quantum import gas


class Recorder:
    """Delegates to a Generator and records the p of every choice() call."""
    def __init__(self, rng):
        self.rng, self.p = rng, []

    def integers(self, *args, **kwargs):
        return self.rng.integers(*args, **kwargs)

    def choice(self, n, p):
        self.p.append(p)
        return self.rng.choice(n, p=p)


@pytest.mark.parametrize("size", RUN_CONFIG["sizes"])
@pytest.mark.parametrize("seed", RUN_CONFIG["pipe_gas_seeds"])
def test_replay_is_faithful(size, seed):
    servers, costs, budget, result = gas_trace.run(size, seed)
    rec = Recorder(np.random.default_rng(seed))
    recorded = gas(costs, rec, budget, RUN_CONFIG["gas_growth"])
    assert recorded["history"] == result["history"]
    replayed = list(gas_trace.replay(costs, result["history"]))
    assert len(replayed) == len(rec.p) + 1                        # step 0 is the uniform first sample
    for (_, _, _, p), p_gas in zip(replayed[1:], rec.p):
        assert np.array_equal(p, p_gas)
    t = gas_trace.trace(size, seed)
    assert [s["measured_index"] for s in t["steps"]] == [h[1] for h in result["history"]]
    assert t["totals"] == {"oracle_calls": result["oracle_calls"], "measurements": result["measurements"],
                           "exhaustive": 2 ** len(servers)}
    assert sum(s["rotations"] for s in t["steps"]) == result["oracle_calls"]
    for s, (_, _, _, p) in zip(t["steps"], replayed):
        assert abs(math.fsum(p) - 1) <= 1e-12
        assert len(s["top"]) == min(gas_trace.TOP, len(costs))
        assert [e["p"] for e in s["top"]] == sorted((e["p"] for e in s["top"]), reverse=True)
        assert s["marked_count"] == sum(1 for c in costs if s["threshold"] is not None and c < s["threshold"])


def test_endpoint_and_seed_check():
    client = create_app({"TESTING": True}).test_client()
    body = client.get("/api/gas-trace/small_v2").get_json()
    assert body["seed"] == RUN_CONFIG["pipe_gas_seeds"][0] and body["states"] == 2 ** body["m"]
    assert body["y_star_ids"] == [2, 3]
    for seed in ["99", "-1", "x", "1.5"]:
        r = client.get(f"/api/gas-trace/small_v2?seed={seed}")
        assert r.status_code == 400 and r.get_json()["error"]["code"] == "bad_seed"
    assert client.get("/api/gas-trace/huge_v2").status_code == 404
