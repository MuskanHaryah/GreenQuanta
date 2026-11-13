"""Step 2.4: s2_sqa (OpenJij SQA wrapper) on a tiny random BQM, plus the four open API points of 2.0_openjij_api.md
section 7 as characterisation tests. Hot settings (beta 0.1-0.5, few sweeps) are used where reads must differ."""
from pathlib import Path
import sys

import dimod
import numpy as np
import openjij as oj
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.s2_sqa import read_seeds, rescaled, scale_factor, sqa_sample

N = 10
COLD = dict(sweeps=500, trotter=4, beta=5.0, gamma=1.0)
HOT = dict(sweeps=5, trotter=4, beta=0.1, gamma=1.0)


@pytest.fixture
def bqm():
    rng = np.random.default_rng(3)
    quad = {(i, j): float(rng.normal()) for i in range(N) for j in range(i + 1, N) if rng.random() < 0.5}
    out = dimod.BinaryQuadraticModel({i: float(rng.normal()) for i in range(N)}, quad, 0.7, dimod.BINARY)
    out.scale(37.0)              # max|coef| far from 1, nonzero offset: rescaling and offset both matter
    return out


def run(bqm, params, reads=20, seed=5):
    return sqa_sample(bqm, reads, params["sweeps"], params["trotter"], params["beta"], params["gamma"], seed)


def states(samples, variables):
    return [tuple(int(row.sample[v]) for v in variables) for row in samples.data(fields=["sample"])]


def test_finds_bruteforce_minimum(bqm):
    ground = dimod.ExactSolver().sample(bqm).first.energy
    samples, _ = run(bqm, COLD)
    assert min(samples.record.energy) == pytest.approx(ground, rel=1e-9, abs=1e-9)


def test_same_seed_same_samples_and_reads_differ(bqm):
    first, info = run(bqm, HOT)
    again, _ = run(bqm, HOT)
    assert states(first, bqm.variables) == states(again, bqm.variables)
    assert info["seeds"] == read_seeds(5, 20) and len(set(info["seeds"])) == 20
    assert len(set(states(first, bqm.variables))) > 1


def test_energies_match_bqm_energy(bqm):
    # Open point 3: reported energy x scale vs bqm.energy (offset included), Step 0 CHANGES #12 relative rule.
    samples, _ = run(bqm, HOT)
    for row in samples.data(fields=["sample", "energy"]):
        exact = float(bqm.energy(row.sample))
        assert abs(float(row.energy) - exact) <= 1e-6 * max(1.0, abs(exact))


def test_rescaling_keeps_argmin_and_is_scale_invariant(bqm):
    scaled, scale = rescaled(bqm)
    assert scale == scale_factor(bqm) and scale_factor(scaled) == pytest.approx(1.0, rel=1e-12)
    a, b = dimod.ExactSolver().sample(bqm), dimod.ExactSolver().sample(scaled)
    assert a.first.sample == b.first.sample
    assert a.first.energy == pytest.approx(b.first.energy * scale, rel=1e-12)
    bigger = bqm.copy()
    bigger.scale(10.0)
    assert states(run(bqm, HOT)[0], bqm.variables) == states(run(bigger, HOT)[0], bqm.variables)


def test_spin_update_count(bqm):
    _, info = run(bqm, HOT, reads=3)
    assert info["spin_updates"] == 3 * HOT["sweeps"] * HOT["trotter"] * N


def test_open_point_1_fixed_seed_repeats_reads(bqm):
    # 2.0 section 4: with one seed and num_reads > 1, OpenJij returns identical reads (why s2_sqa seeds per read).
    scaled, _ = rescaled(bqm)
    res = oj.SQASampler().sample(scaled, beta=HOT["beta"], gamma=HOT["gamma"], num_sweeps=HOT["sweeps"],
                                 trotter=HOT["trotter"], num_reads=5, seed=7)
    assert len(set(states(res, bqm.variables))) == 1


def test_open_point_2_solution_is_a_trotter_slice(bqm):
    scaled, _ = rescaled(bqm)
    for seed in read_seeds(11, 5):
        res = oj.SQASampler().sample(scaled, beta=HOT["beta"], gamma=HOT["gamma"], num_sweeps=HOT["sweeps"],
                                     trotter=HOT["trotter"], num_reads=1, seed=seed)
        slices = {tuple(int((s + 1) // 2) for s in row) for row in res.info["system"][0]["trotter_state"]}
        assert len(next(iter(slices))) == N
        assert states(res, sorted(bqm.variables))[0] in slices


def test_tts99():
    from diagnostics.s2_sa_reduced import tts99
    assert tts99(10, 0) is None and tts99(10, 0.0) is None
    assert tts99(10, 1.0) == 10
    assert tts99(10, 0.5) == 70          # ceil(ln 0.01 / ln 0.5) = ceil(6.64) = 7 reads
    assert tts99(10, 0.9) == 20          # ceil(ln 0.01 / ln 0.1) = ceil(2.0000...) — 1 - 0.9 is 0.09999..., so 2


def test_open_point_4_no_internal_rescaling(bqm):
    # Same seeds, coefficients x10 without our rescaling: different statistics => cxxjij uses raw couplings.
    scaled, _ = rescaled(bqm)
    bigger = scaled.copy()
    bigger.scale(10.0)

    def raw(model):
        return [states(oj.SQASampler().sample(model, beta=0.5, gamma=1.0, num_sweeps=5, trotter=4, num_reads=1,
                                              seed=s), bqm.variables)[0] for s in read_seeds(13, 20)]
    assert raw(scaled) != raw(bigger)
