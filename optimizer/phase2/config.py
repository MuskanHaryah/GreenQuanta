"""
GreenQuanta Phase 2 - constants and default inputs.

Phase 2 = static VM allocation (no migration) on HETEROGENEOUS servers.
The workload code (trace loader, VM builder) is reused from Phase 1.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATALOG_JSON = Path(__file__).resolve().parent / "server_catalog.json"
TRACE_DIR = ROOT / "data" / "trace" / "phase2"
RESULTS_DIR = ROOT / "results" / "phase2"

# ---------------------------------------------------------------- scenarios
# size -> trace file and the user's servers (type: count). Same data, nested subsets.
SCENARIOS = {
    "small":  {"trace": "google2019_a_day2_small.csv",  "servers": {"A": 2, "B": 1, "C": 2, "D": 1}},
    "medium": {"trace": "google2019_a_day2_medium.csv", "servers": {"A": 3, "B": 2, "C": 3, "D": 2}},
    "large":  {"trace": "google2019_a_day2_large.csv",  "servers": {"A": 5, "B": 3, "C": 5, "D": 3}},
}
DEFAULT_SIZE = "small"

# ---- Phase 2 update (v2): bigger VMs (16 instances per job) and AUTO server mix.
# "auto": the same number of servers of every type, enough for AUTO_SERVER_FACTOR x the demand.
SCENARIOS.update({
    "small_v2":  {"trace": "google2019_a_day2_v2_small.csv",  "servers": "auto"},
    "medium_v2": {"trace": "google2019_a_day2_v2_medium.csv", "servers": "auto"},
    "large_v2":  {"trace": "google2019_a_day2_v2_large.csv",  "servers": "auto"},
})
AUTO_SERVER_FACTOR = 2.5
DEFAULT_ENCODING = "unbalanced"      # v2 default; "slack" = team formulation
UNBALANCED_CENTER = 0.05             # unbalanced penalty is lowest at 5% free capacity
DESCENT = False                      # steepest descent gave no gain in v2 runs (raw = after descent)

# ---------------------------------------------------------------- workload / SLA (same as Phase 1)
SLA_MAX_VIOLATION_PCT = 5.0
INSTANCES_PER_VM = 4

# ---------------------------------------------------------------- energy & carbon (same as Phase 1)
PUE = 1.54
EMISSION_FACTOR = 0.435

# ---------------------------------------------------------------- QUBO
# One fixed resource unit for ALL servers, so a VM has the same size on every server.
UNIT_NCU = 0.0025
UNIT_NMU = 0.0025
TIE_BREAK = 0.01            # tiny extra cost between identical servers of the same type
DEFAULT_LAMBDA_CONFIG = "A"

# ---------------------------------------------------------------- candidate rule (off by default)
CANDIDATE_REDUCTION = False
CANDIDATE_MARGIN = 0.25

# ---------------------------------------------------------------- annealer
NUM_READS = 200
NUM_SWEEPS = 2000
SEED = 42

# ---------------------------------------------------------------- exact reference solver (SciPy HiGHS)
EXACT_TIME_LIMIT_S = 120
