"""Step 0.4: slack-mode exact solver vs brute force on the tiny instance; lambda < F counter-example.

Brute force enumerates all 2^15 x/y states. Each state is scored on the genuine slack BQM with
its slack bits at the per-rule minimiser (Lemma, brute-forced in tests/test_s0_slack.py), so the
minimum over states is the minimum over all (x, y, s).
"""
from itertools import product
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.d0_formulation import tiny_problem
from diagnostics.exact_qubo import solve_exact_qubo, with_slack
from phase1.qubo_builder import slack_coefficients, x_var, y_var
from phase2 import config, qubo_builder_v2 as v2
from phase2.validation import validate_allocation


def caps_loads(vms, servers, unit):
    """Documented quantization, written independently of the solver."""
    caps, loads = [], []
    for resource in ("cpu", "mem"):
        demand = [max(1, math.ceil(getattr(vm, f"{resource}_size") / unit - 1e-9)) for vm in vms]
        for server in servers:
            caps.append(math.floor(getattr(server, f"{resource}_limit") / unit + 1e-9))
            loads.append(demand)
    return caps, loads


@pytest.fixture(scope="module")
def tiny():
    vms, servers, _, _, _ = tiny_problem()
    assert config.UNIT_NCU == config.UNIT_NMU
    n, m = len(vms), len(servers)
    unit = config.UNIT_NCU
    caps, loads = caps_loads(vms, servers, unit)
    names = [x_var(vm.vm_id, h) for vm in vms for h in range(m)] + [y_var(h) for h in range(m)]
    states = np.array(list(product((0, 1), repeat=len(names))), dtype=np.int64)
    x = states[:, :n * m].reshape(-1, n, m)
    y = states[:, n * m:]
    feasible = (x.sum(axis=2) == 1).all(axis=1) & (x * (1 - y[:, None, :]) == 0).all(axis=(1, 2))
    for r, (cap, demand) in enumerate(zip(caps, loads)):
        feasible &= x[:, :, r % m] @ np.array(demand) <= cap * y[:, r % m]
    energy, _ = v2.build_qubo(vms, servers, lambda_config="L0", encoding="slack")
    e = states @ np.array([energy.linear[name] for name in names]) + energy.offset
    assert feasible.any() and (e >= 0).all()
    f = float(e[feasible].min())          # best feasible E = tightest F
    return dict(vms=vms, servers=servers, m=m, unit=unit, caps=caps, loads=loads,
                names=names, states=states, feasible=feasible, F=f)


def exhaustive_h(t, lam):
    """H of every x/y state on the genuine slack BQM with optimal slack bits."""
    bqm, _ = v2.build_qubo(t["vms"], t["servers"], lambda_config=f"L{lam!r}", encoding="slack")
    labels = list(bqm.variables)
    rows = np.zeros((len(t["states"]), len(labels)), dtype=np.int8)
    index = {v: i for i, v in enumerate(labels)}
    for s, bits in enumerate(t["states"]):
        full = with_slack(dict(zip(t["names"], map(int, bits))), t["vms"], t["caps"], t["loads"], t["m"])
        assert set(full) == set(labels)
        for var, value in full.items():
            rows[s, index[var]] = value
    return bqm, bqm.energies((rows, labels))


def test_with_slack_writes_every_value_0_to_k():
    vm = SimpleNamespace(vm_id=0)
    for k in range(1, 300):
        coeffs = slack_coefficients(k)
        for value in range(k + 1):
            full = with_slack({x_var(0, 0): 1, y_var(0): 1}, [vm], [k, k], [[k - value], [k - value]], 1)
            for tag in ("sc", "sm"):
                assert sum(c * full[f"{tag}_0_{j}"] for j, c in enumerate(coeffs)) == value


def check_solver(t, lam, bqm, h):
    result = solve_exact_qubo(t["vms"], t["servers"], lambda_config=f"L{lam!r}",
                              time_limit=300, max_iters=50, encoding="slack")
    assert result["status"] == "OPTIMAL", result["reason"]
    assert result["bqm_H"] == pytest.approx(h.min(), abs=1e-9, rel=0)
    assert float(bqm.energy(result["sample"])) == pytest.approx(result["bqm_H"], abs=1e-9, rel=0)
    assert result["encoding"] == "slack"
    return result


def test_bound_holds_global_minimum_feasible_and_energy_optimal(tiny):
    lam = 1.1 * tiny["F"]
    bqm, h = exhaustive_h(tiny, lam)
    # 1e-9 = exact_qubo.TOLERANCE; 1e-12 was below the rounding of lambda*K^2 ~ 6e4 terms (CHANGES.md #9).
    minimisers = np.isclose(h, h.min(), atol=1e-9, rtol=0)
    assert tiny["feasible"][minimisers].all()                 # Theorem (1)
    assert h.min() == pytest.approx(tiny["F"], abs=1e-9)      # Theorem (2): min H = min feasible E
    assert (h[~tiny["feasible"]] > tiny["F"]).all()
    result = check_solver(tiny, lam, bqm, h)
    assert result["validation"]["valid"]
    real = validate_allocation(tiny["vms"], result["assignment"], tiny["servers"])
    quant = validate_allocation(tiny["vms"], result["assignment"], tiny["servers"],
                                quantized_unit=tiny["unit"])
    assert real["valid"] and quant["valid"] and len(result["assignment"]) == len(tiny["vms"])


def test_counterexample_lambda_below_f_minimum_is_invalid(tiny):
    lam = 0.001
    assert lam < tiny["F"]
    bqm, h = exhaustive_h(tiny, lam)
    minimisers = np.isclose(h, h.min(), atol=1e-12, rtol=0)
    assert not tiny["feasible"][minimisers].any()             # the bound matters
    assert h.min() < tiny["F"]
    result = check_solver(tiny, lam, bqm, h)
    assert not result["validation"]["valid"]
