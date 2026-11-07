"""
GreenQuanta Phase 1 - every constant and default input in one place.

Development inputs are hardcoded here. Any of the "user inputs" can also be
overridden from the command line (see: python run_phase1.py --help).
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]          # .../GreenQuanta
TRACE_CSV = ROOT / "data" / "trace" / "google2019_a_day2_usage.csv"
RESULTS_DIR = ROOT / "results" / "phase1"
CLOUDSIM_INPUT_DIR = RESULTS_DIR / "cloudsim_input"

# ---------------------------------------------------------------- user inputs
# Google 2019 trace units: CPU in NCU (normalized compute units),
# memory in NMU (normalized memory units). Server capacity must use the same units.
NUM_SERVERS = 10                 # how many identical servers the user owns
SERVER_CPU_CAPACITY = 0.591796875    # NCU, most common machine in Google cell a (3346 machines)
SERVER_MEM_CAPACITY = 0.33349609375  # NMU, same machine type
SERVER_IDLE_POWER_W = 100.0      # watts when ON but idle   (placeholder, TBD)
SERVER_PEAK_POWER_W = 250.0      # watts at 100% CPU        (placeholder, TBD)
SAFE_UTILIZATION = 0.85          # a server is never filled above 85%
SLA_MAX_VIOLATION_PCT = 5.0      # user SLA: at most 5% of VM time may be throttled

# ---------------------------------------------------------------- workload -> VMs
INSTANCES_PER_VM = 4             # sibling instances of one job packed into one VM
WINDOW_SECONDS = 300             # Google trace usage window = 5 minutes
WINDOWS_PER_DAY = 288            # 24 h / 5 min
SIMULATION_SECONDS = WINDOW_SECONDS * WINDOWS_PER_DAY   # 86,400 s = 24 hours
MIN_WINDOWS_PRESENT = 280        # an instance must be seen in >= 280 of 288 windows

# ---------------------------------------------------------------- energy & carbon
PUE = 1.54                       # Uptime Institute 2025 (fixed for Phase 1)
EMISSION_FACTOR = 0.435          # kg CO2 per kWh (IEA 2025 global average)

# ---------------------------------------------------------------- QUBO
LAMBDA_ASSIGN = 2.0              # every VM on exactly one server
LAMBDA_CPU = 2.0                 # CPU capacity
LAMBDA_RAM = 2.0                 # memory capacity
LAMBDA_LINK = 2.0                # no VM on a switched-off server
RESOURCE_RESOLUTION = 100        # server capacity split into 100 integer units (1% each)
TIE_BREAK = 0.01                 # tiny extra cost for higher-numbered servers (symmetry breaking)

# Candidate-server reduction: the QUBO only gets as many servers as the workload can need.
# candidates = min(user servers, ceil(lower_bound * (1 + MARGIN)) + EXTRA)
# The other servers stay OFF (0 W). If repair cannot fit the workload, one more server is added.
CANDIDATE_REDUCTION = True
CANDIDATE_MARGIN = 0.25
CANDIDATE_EXTRA = 1

# ---------------------------------------------------------------- simulated annealer
NUM_READS = 200                  # independent annealing runs
NUM_SWEEPS = 2000                # sweeps per run
SEED = 42                        # fixed seed so results can be reproduced
