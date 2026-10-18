"""Local Ocean SDK test. No account needed."""
import dimod
from dwave.samplers import SimulatedAnnealingSampler

# Problem: two switches, exactly one should be ON
bqm = dimod.BinaryQuadraticModel(
    {"x0": -1, "x1": -1},
    {("x0", "x1"): 2},
    0.0,
    dimod.BINARY,
)

sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=10)
print("Best solution:", sampleset.first.sample)
print("Energy:", sampleset.first.energy)
print("Local annealer is working")
