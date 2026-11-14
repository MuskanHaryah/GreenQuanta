"""Guide 1 step 1.0: pin the uncommitted pipeline inputs and the R1 references by SHA256.

The first run writes tests/data/mids_inputs_sha256.json; later runs compare against it and exit 1 on any
mismatch or missing file.
    .venv-gq/Scripts/python.exe -B optimizer/demo/preflight.py
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
PINNED = ROOT / "tests" / "data" / "mids_inputs_sha256.json"
FILES = ["docs/reports/step1/1.3_ground_truth.txt", "docs/reports/step1/1.5b_qaoa_cvar.txt",
         "docs/reports/step2/2.5_calibration.txt", "docs/reports/step2/2C.3_calibration.txt",
         "docs/reports/step2/2D.3_calibration.txt", "docs/reports/step2e_raw_output.txt",
         *(f"results/phase2/{s}/assignment_exact.csv" for s in ("small_v2", "medium_v2", "large_v2"))]


def hashes():
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() if (ROOT / f).is_file() else None for f in FILES}


def check():
    """Returns a list of problems (empty = PASS). Writes the pin file when it does not exist yet."""
    now = hashes()
    missing = [f"missing: {f}" for f, h in now.items() if h is None]
    if missing:
        return missing
    if not PINNED.exists():
        PINNED.write_text(json.dumps(now, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {PINNED.relative_to(ROOT)}")
    pinned = json.loads(PINNED.read_text(encoding="utf-8"))
    return [f"mismatch: {f}" for f in sorted(set(pinned) | set(now)) if pinned.get(f) != now.get(f)]


def main():
    problems = check()
    for f, h in hashes().items():
        print(f"{h} {f}")
    print("\n".join(problems) if problems else "PREFLIGHT PASS")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
