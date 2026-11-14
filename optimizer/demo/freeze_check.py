"""Guide 1 step 1.2: the frozen pipeline (GAS -> csqa / csa, --settings 2D) still gives the golden results.

Compares hybrid(size, "gas", method, "2D", log=False) with tests/data/mids_golden.json for every size (large takes
minutes, so this is not a pytest test). Every later guide runs it at its end; exit 1 on any difference.
    .venv-gq/Scripts/python.exe -B optimizer/demo/freeze_check.py [--sizes small_v2 ...]
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.s1_end_to_end import RUN_CONFIG, hybrid

GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "data" / "mids_golden.json"
METHODS = ["csqa", "csa"]


def differences(size, method, out, golden=None):
    """Fields of a HYBRID out dict that differ from the golden record ([] = equal)."""
    g = (golden or json.loads(GOLDEN.read_text(encoding="utf-8")))[size][method]
    bad = [k for k in ("chosen_ids", "real_valid", "quantized_valid") if out.get(k) != g[k]]
    bad += [k for k in ("V", "gap_to_h_min_pct", "linear_joules")
            if out.get(k) is None or not math.isclose(out[k], g[k], rel_tol=1e-12, abs_tol=1e-12)]
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=RUN_CONFIG["sizes"])
    sizes = parser.parse_args().sizes
    failed = []
    for size in sizes:
        for method in METHODS:
            out, _ = hybrid(size, "gas", method, "2D", log=False)
            bad = differences(size, method, out)
            print(f"FREEZE | {size} | {method} | {out.get('chosen_ids')} | V={out.get('V')!r} | "
                  f"joules={out.get('linear_joules')!r} | {'PASS' if not bad else 'FAIL ' + str(bad)}", flush=True)
            failed += [(size, method)] if bad else []
    print("FREEZE CHECK " + ("PASS" if not failed else f"FAIL {failed}"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
