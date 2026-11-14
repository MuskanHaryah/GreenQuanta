"""Guide 2: Flask backend of the mid-evaluation demo. It serves results/mids/<size>/results.json, the GAS trace and
live pipeline runs; every number comes from the existing optimizer code, which it puts on sys.path here (once)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER = ROOT / "optimizer"
if str(OPTIMIZER) not in sys.path:
    sys.path.insert(0, str(OPTIMIZER))
