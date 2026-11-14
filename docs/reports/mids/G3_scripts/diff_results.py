"""G3.00: diff old vs new results/mids/<size>/results.json; ignore generated_at, git_commit and runtime_s; the only
other differences allowed are methods.<id>.reference."""
import json
import sys
from pathlib import Path

OLD = Path(sys.argv[1])
NEW = Path("results/mids")
IGNORE = {"generated_at", "runtime_s", "git_commit"}


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE:
                continue
            if k not in a or k not in b:
                out.append((path + (k,), a.get(k, "<missing>"), b.get(k, "<missing>")))
            else:
                walk(a[k], b[k], path + (k,), out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, path + (i,), out)
    elif a != b:
        out.append((path, a, b))


ok = True
for size in ("small_v2", "medium_v2", "large_v2"):
    old = json.loads((OLD / size / "results.json").read_text(encoding="utf-8"))
    new = json.loads((NEW / size / "results.json").read_text(encoding="utf-8"))
    diffs = []
    walk(old, new, (), diffs)
    refs = [d for d in diffs if len(d[0]) == 3 and d[0][0] == "methods" and d[0][2] == "reference"]
    other = [d for d in diffs if d not in refs]
    print(f"== {size}: {len(diffs)} differences (ignoring {sorted(IGNORE)}); reference: {len(refs)}; other: {len(other)}")
    for p, a, b in refs:
        print(f"  {'.'.join(map(str, p))}\n    old: {a}\n    new: {b}")
    for p, a, b in other:
        print(f"  OTHER {'.'.join(map(str, p))}: {a!r} -> {b!r}")
    ok &= not other
    print(f"  git_commit old {old.get('git_commit')} new {new.get('git_commit')}")
print("RESULTS DIFF PASS (only reference strings changed)" if ok else "RESULTS DIFF FAIL")
sys.exit(0 if ok else 1)
