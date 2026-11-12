"""Step 2.4: simulated quantum annealing (path-integral Monte Carlo, OpenJij 0.12.2 SQASampler) on a dimod BQM.

H(s) = s*H_p + Gamma*(1 - s)*sum_i sigma_i^x, beta fixed, default quartic schedule (2.0_openjij_api.md sections 1-3).
1. H_p = BQM / max|coef| (linear and quadratic, offset excluded). A positive scale keeps the argmin (Step 0
   rescaling remark) and gives beta and Gamma the same meaning on every size (OpenJij does not rescale: 2.0 section 1,
   tested in tests/test_s2_sqa.py).
2. One SQASampler.sample call per read, num_reads=1, with its own seed drawn from numpy default_rng(seed): with a
   fixed seed OpenJij reuses the same seed and initial state on every read (2.0 section 4, CHANGES #2).
3. Returns a dimod SampleSet in the ORIGINAL units (reported energy x scale), so the caller scores it on the
   original BQM through the shared path (r3c_run.sample_worker, method "sqa"), which cross-checks the energy.
Budget: spin updates = reads x sweeps x trotter x variables (one single-spin-flip MC step per schedule point visits
every spin of every Trotter slice; inference from cxxjij's updater name, 2.0 section 6).
"""
import dimod
import numpy as np


def scale_factor(bqm):
    """max |coefficient| over linear and quadratic terms."""
    top = max((abs(float(v)) for v in (*bqm.linear.values(), *bqm.quadratic.values())), default=0.)
    if not top > 0:
        raise ValueError("BQM has no nonzero coefficient")
    return top


def rescaled(bqm):
    """(bqm / max|coef|, max|coef|); the offset is scaled too, so energies map back exactly by the factor."""
    scale = scale_factor(bqm)
    out = bqm.copy()
    out.scale(1 / scale)
    return out, scale


def read_seeds(seed, reads):
    return [int(s) for s in np.random.default_rng(seed).integers(0, 2**31 - 1, size=reads)]


def sqa_sample(bqm, reads, sweeps, trotter, beta, gamma, seed, schedule=None):
    """SQA reads on bqm (BINARY or SPIN). Returns (SampleSet in original units, info)."""
    import openjij as oj
    scaled, scale = rescaled(bqm)
    sampler = oj.SQASampler()
    seeds = read_seeds(seed, reads)
    states, energies = [], []
    for s in seeds:
        first = sampler.sample(scaled, beta=beta, gamma=gamma, num_sweeps=sweeps, trotter=trotter,
                               schedule=schedule, num_reads=1, seed=s).first
        states.append({v: int(first.sample[v]) for v in bqm.variables})
        energies.append(float(first.energy) * scale)
    samples = dimod.SampleSet.from_samples(states, bqm.vartype, energies)
    return samples, {"scale": scale, "seeds": seeds, "spin_updates": reads * sweeps * trotter * len(bqm.variables)}
