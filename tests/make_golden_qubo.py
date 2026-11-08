"""Explicit, manual golden creation; pytest must never generate this file."""
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimizer"))

from diagnostics.common import build_problem
from phase2.fingerprint import qubo_fingerprint


GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "qubo_small_v2_unbalanced.json"


def main():
    _, _, bqm, info = build_problem("small_v2")
    payload = {
        "fingerprint": qubo_fingerprint(bqm),
        "info": {key: info[key] for key in ("num_variables", "bias_range", "e_ref_joule")},
    }
    # Exclusive creation prevents accidentally replacing an established golden.
    with GOLDEN_PATH.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
    print(f"Created {GOLDEN_PATH}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
