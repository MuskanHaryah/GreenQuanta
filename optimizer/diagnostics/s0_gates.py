"""Step 0.8 protection gates; stdout only.

Compares r3c_run.protected_snapshot() with the R3c BEFORE manifest:
- results/phase2 and results/runs: identical file sets and SHA256;
- results/experiment_log.csv: append-only (its first bytes equal the old file, by SHA256 of the
  prefix of the old length), and every appended row is method s0_sa_probe.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diagnostics.r3c_run import RAW, protected_snapshot
from phase2 import config


def main():
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
        ok &= methods == ["s0_sa_probe"]
    print("PROTECTION_GATE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
