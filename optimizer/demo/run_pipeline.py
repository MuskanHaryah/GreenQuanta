"""Guide 2 step 2.1: one live run of the frozen pipeline (GAS -> csqa / csa, --settings 2D) for the backend.

It calls s1_end_to_end.hybrid (log=True: a live run is an experiment, so one log row is appended) and finally prints
RESULT_JSON={"out", "assignment" {vm_id: host index}, "energy" (evaluate_allocation without its "_" keys)}.
The Stage-2 sampler spawns a child that re-imports __main__, so start it only as a module (cwd optimizer/):
    ../.venv-gq/Scripts/python.exe -B -u -m demo.run_pipeline --size small_v2 --stage2 csqa
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.common import build_problem
from diagnostics.s1_end_to_end import RUN_CONFIG, hybrid
from phase2.energy_model import evaluate_allocation

KINDS = ("PROPOSALS", "SKIP", "STAGE2", "HYBRID", "LOGGED", "RESULT")    # the <KIND>_JSON= lines of a run


def parse_line(line):
    """(kind, data) for a <KIND>_JSON= stdout line of a run, else None."""
    head, sep, body = line.strip().partition("_JSON=")
    return (head, json.loads(body)) if sep and head in KINDS else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=RUN_CONFIG["sizes"], required=True)
    parser.add_argument("--stage2", choices=["csqa", "csa"], required=True)
    args = parser.parse_args()
    out, assignment = hybrid(args.size, "gas", args.stage2, "2D", log=True)
    energy = None
    if assignment is not None:
        vms, servers, _, _ = build_problem(args.size)
        energy = {k: v for k, v in evaluate_allocation(vms, assignment, servers).items() if not k.startswith("_")}
    print("RESULT_JSON=" + json.dumps({"out": out, "assignment": assignment, "energy": energy}, allow_nan=False),
          flush=True)


if __name__ == "__main__":
    main()
