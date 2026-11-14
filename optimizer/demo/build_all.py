"""Guide 1 step 1.5: build results/mids/<size>/results.json for every size, then run the freeze check.
    .venv-gq/Scripts/python.exe -B optimizer/demo/build_all.py
"""
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.build_results import build
from diagnostics.s1_end_to_end import RUN_CONFIG


def main():
    for size in RUN_CONFIG["sizes"]:
        _, path = build(size)
        print(f"wrote {path}", flush=True)
    sys.exit(subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("freeze_check.py"))]).returncode)


if __name__ == "__main__":
    main()
