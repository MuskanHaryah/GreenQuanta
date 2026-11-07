"""Ocean SDK ka local test. Koi account nahi chahiye."""
import dimod
from dwave.samplers import SimulatedAnnealingSampler

# Sawal: do switch hain, sirf ek ON hona chahiye
bqm = dimod.BinaryQuadraticModel(
    {"x0": -1, "x1": -1},
    {("x0", "x1"): 2},
    0.0,
    dimod.BINARY,
)

sampleset = SimulatedAnnealingSampler().sample(bqm, num_reads=10)
print("Behtareen jawab:", sampleset.first.sample)
print("Energy:", sampleset.first.energy)
print("Local annealer theek kaam kar raha hai")
