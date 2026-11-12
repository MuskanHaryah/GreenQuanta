"""Step 0.8 protection gates; stdout only.

Compares r3c_run.protected_snapshot() with the R3c BEFORE manifest:
- results/phase2 and results/runs: identical file sets and SHA256;
- results/experiment_log.csv: append-only (its first bytes equal the old file, by SHA256 of the
  prefix of the old length), and the set of appended methods equals --methods (default: Step 0's
  s0_sa_probe; Step 1.8 passes the Step 0 + Step 1 methods).
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.r3c_run import RAW, protected_snapshot
from phase2 import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["s0_sa_probe"])
    allowed = sorted(set(parser.parse_args().methods))
    text = RAW.read_text(encoding="utf-8")
    before = json.loads(next(line[len("BEFORE_JSON="):] for line in text.splitlines()
                             if line.startswith("BEFORE_JSON=")))
    after = protected_snapshot()
    ok = True
    for root in ("results/phase2", "results/runs"):
        same = before[root] == after[root]
        ok &= same
        print(f"{root}: files before={len(before[root])} after={len(after[root])} identical={same}")
    log = config.ROOT / "results/experiment_log.csv"
    old_hash = before["results/experiment_log.csv"]["results/experiment_log.csv"]
    data = log.read_bytes()
    prefix_len = next((n for n in range(len(data) + 1)
                       if data[n - 1:n] in (b"\n",) and hashlib.sha256(data[:n]).hexdigest() == old_hash), None)
    print(f"experiment_log: old prefix found={prefix_len is not None} old_bytes={prefix_len} now_bytes={len(data)}")
    ok &= prefix_len is not None
    if prefix_len is not None:
        rows = list(csv.DictReader(log.open(encoding="utf-8", newline="")))
        old_rows = data[:prefix_len].decode("utf-8").count("\n") - 1
        new = rows[old_rows:]
        methods = sorted({r["method"] for r in new})
        print(f"experiment_log: old_rows={old_rows} appended_rows={len(new)} appended_methods={methods}")
        print(f"experiment_log: rows_per_method={dict(sorted(Counter(r['method'] for r in new).items()))} "
              f"expected_methods={allowed}")
        ok &= methods == allowed
    print("PROTECTION_GATE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
