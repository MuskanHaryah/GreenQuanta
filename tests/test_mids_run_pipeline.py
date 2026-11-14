"""Guide 2 step 2.1: the live-run CLI (python -m demo.run_pipeline, a subprocess as the backend starts it) reproduces
the golden snapshot on small_v2 and prints its progress lines in pipeline order (appends one log row)."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "optimizer"))

from demo.freeze_check import differences
from demo.run_pipeline import parse_line
from diagnostics.common import build_problem


def test_run_pipeline_small_csqa():
    proc = subprocess.run([sys.executable, "-B", "-u", "-m", "demo.run_pipeline", "--size", "small_v2",
                           "--stage2", "csqa"], cwd=ROOT / "optimizer", capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    events = [e for e in map(parse_line, proc.stdout.splitlines()) if e]
    kinds = [k for k, _ in events]
    assert kinds[0] == "PROPOSALS" and kinds[-2:] == ["LOGGED", "RESULT"] and kinds.count("HYBRID") == 1
    assert kinds.index("STAGE2") < kinds.index("HYBRID")
    result = events[-1][1]
    assert differences("small_v2", "csqa", result["out"]) == []
    vms, _, _, _ = build_problem("small_v2")
    assert sorted(map(int, result["assignment"])) == sorted(v.vm_id for v in vms)
    assert result["energy"]["it_energy_kwh"] > 0 and not any(k.startswith("_") for k in result["energy"])
