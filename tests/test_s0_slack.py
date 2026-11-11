"""Step 0.3: slack encoding in build_qubo_custom; Lemma (slack elimination) by brute force."""
from itertools import product
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.common import build_problem
from diagnostics.d0_formulation import tiny_problem
from diagnostics.r3c_build import build_qubo_custom
from phase1.qubo_builder import slack_coefficients, x_var, y_var
from phase2 import config, qubo_builder_v2 as v2
from phase2.fingerprint import qubo_fingerprint


@pytest.fixture(scope="module")
def tiny():
    vms, servers, _, _, _ = tiny_problem()
    return vms, servers


@pytest.fixture(scope="module")
def small():
    vms, servers, _, _ = build_problem("small_v2")
    return vms, servers


def test_slack_range_is_exactly_0_to_k():
    """Assumption A3 / L0 in 0.2_proof.md."""
    for k in range(1, 2049):
        coeffs = slack_coefficients(k)
        assert all(isinstance(c, int) and c >= 1 for c in coeffs)
        reachable = {0}
        for c in coeffs:
            reachable |= {r + c for r in reachable}
        assert reachable == set(range(k + 1)), k


def test_default_encoding_is_unbalanced_and_unchanged(tiny):
    vms, servers = tiny
    default, _ = build_qubo_custom(vms, servers, *[0.05] * 4, 0.02, 0.0025)
    explicit, info = build_qubo_custom(vms, servers, *[0.05] * 4, 0.02, 0.0025, "unbalanced")
    assert qubo_fingerprint(default) == qubo_fingerprint(explicit)
    assert info["slack_variables"] == 0 and info["unbalanced_center"] == 0.02


def test_encoding_arguments_are_checked(tiny):
    vms, servers = tiny
    with pytest.raises(ValueError):
        build_qubo_custom(vms, servers, *[0.05] * 4, None, 0.0025, "unbalanced")
    with pytest.raises(ValueError):
        build_qubo_custom(vms, servers, *[0.05] * 4, 0.02, 0.0025, "slack_scaled")
    _, info = build_qubo_custom(vms, servers, *[0.05] * 4, None, 0.0025, "slack")
    assert info["encoding"] == "slack" and info["unbalanced_center"] is None


def test_slack_matches_build_qubo_v2_at_default_unit_and_lambda(small):
    vms, servers = small
    assert config.UNIT_NCU == config.UNIT_NMU
    reference, ref_info = v2.build_qubo(vms, servers, encoding="slack")
    lam = ref_info["lambdas"]
    custom, info = build_qubo_custom(vms, servers, lam["assign"], lam["cpu"], lam["ram"], lam["link"],
                                     None, config.UNIT_NCU, "slack")
    assert qubo_fingerprint(custom) == qubo_fingerprint(reference)
    assert info["slack_variables"] == ref_info["slack_variables"] > 0
    assert info["num_variables"] == ref_info["num_variables"]


@pytest.mark.parametrize("size", ["small_v2", "medium_v2", "large_v2"])
def test_energy_part_is_nonnegative(size):
    """Assumption A4 in 0.2_proof.md: E >= 0 for every binary state (tie-break included)."""
    vms, servers, _, _ = build_problem(size)
    energy, _ = build_qubo_custom(vms, servers, 0, 0, 0, 0, None, config.UNIT_NCU, "slack")
    assert energy.offset == 0
    assert all(b == 0 for b in energy.quadratic.values())
    assert all(b >= 0 for b in energy.linear.values())
    assert any(b > 0 for b in energy.linear.values())


@pytest.mark.parametrize("unit", [0.05, 0.0025])
def test_lemma_slack_elimination_bruteforce(tiny, unit):
    """For every (x_.h, y_h) and every host/resource, min over that rule's slack bits of the
    BQM capacity penalty equals max(0, L - K*y)^2 (Lemma in 0.2_proof.md)."""
    vms, servers = tiny
    energy, _ = build_qubo_custom(vms, servers, 0, 0, 0, 0, None, unit, "slack")
    overloaded_on = cases = 0
    for resource, tag, weights in (("cpu", "sc", [0, 1, 0, 0]), ("mem", "sm", [0, 0, 1, 0])):
        bqm, _ = build_qubo_custom(vms, servers, *weights, None, unit, "slack")
        labels = list(bqm.variables)
        assert set(labels) == set(energy.variables)
        index = {v: i for i, v in enumerate(labels)}
        # Independent quantization (documented formula), not qubo_builder.units.
        a = {vm.vm_id: max(1, math.ceil(getattr(vm, f"{resource}_size") / unit - 1e-9)) for vm in vms}
        for h, server in enumerate(servers):
            k = math.floor(getattr(server, f"{resource}_limit") / unit + 1e-9)
            slack = [f"{tag}_{h}_{j}" for j in range(len(slack_coefficients(k)))]
            assert all(s in index for s in slack)
            block = np.array(list(product((0, 1), repeat=len(slack))), dtype=np.int8)
            main = [x_var(vm.vm_id, h) for vm in vms] + [y_var(h)]
            for bits in product((0, 1), repeat=len(main)):
                rows = np.zeros((len(block), len(labels)), dtype=np.int8)
                for j, var in enumerate(main):
                    rows[:, index[var]] = bits[j]
                for j, var in enumerate(slack):
                    rows[:, index[var]] = block[:, j]
                penalty = bqm.energies((rows, labels)) - energy.energies((rows, labels))
                load = sum(a[vm.vm_id] * b for vm, b in zip(vms, bits[:-1]))
                expected = max(0, load - k * bits[-1]) ** 2
                assert penalty.min() == pytest.approx(expected, abs=1e-6), (resource, h, bits)
                # Integer-valued: every slack setting gives an integer penalty.
                assert np.allclose(penalty, np.round(penalty), atol=1e-6)
                cases += 1
                overloaded_on += int(bits[-1] == 1 and expected > 0)
    assert cases == 2 * len(servers) * 2 ** (len(vms) + 1)
    if unit == 0.0025:
        assert overloaded_on > 0  # the y=1 overload branch of the Lemma is exercised
