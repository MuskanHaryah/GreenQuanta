# Step R3a
Branch: `step3-r3-exact-qubo`; base HEAD: `5da2c94177a2cd5ddb3c68b73248a5a3421d23b9`.
Initial `git status --short`: untracked files only; no tracked changes.
New files: `optimizer/diagnostics/exact_qubo.py`, `tests/test_exact_qubo.py`, this report, `docs/reports/stepR3a_raw_output.txt`.
Changes: integer-residual square epigraph MILP; full/cutting-plane capacity cuts; dual bounds; independent D0/BQM scoring; per-term H; iteration diagnostics; UNPROVEN limit returns.
Settings: `(L0.05, 0.02)`, `(L0.2, 0.07)`, `(L1.0, 0.0)`.
Coverage: 32,768 enumerated states per setting; 1,224 fixed-state solves; four limit-path tests.
Dependencies added: none. Interpreter: `.venv-gq/Scripts/python.exe`; no installs.
No annealer/runner execution or repository experiment-log writes. No commits.
## Commands run
- `git status --short`; `git diff --exit-code`; `git diff --cached --exit-code`; `git switch -c step3-r3-exact-qubo`
- `.venv-gq/Scripts/python.exe -B -m pytest tests/test_exact_qubo.py -v -s -p no:cacheprovider`
- `.venv-gq/Scripts/python.exe -B -m pytest tests/test_exact_qubo.py -v -s -k "unproven or exhausted or interrupted or without_candidate" -p no:cacheprovider`
- `.venv-gq/Scripts/python.exe -B -m pytest tests -v -p no:cacheprovider`
- `.venv-gq/Scripts/python.exe -B -m pytest tests -v -s -p no:cacheprovider`
- SHA256 per-file/path manifests and directory digests, before/after; `git --no-pager diff --exit-code`; `git --no-pager diff --cached --exit-code`; branch/status checks.
## Raw gate output
```text
BEFORE results/phase2 count=105 sha256=fda2cc1416e68370f2392ce53507122535ee1c908ea86d06156a8921fdc53eb7
AFTER results/phase2 count=105 sha256=fda2cc1416e68370f2392ce53507122535ee1c908ea86d06156a8921fdc53eb7
results/phase2 identical=yes
BEFORE results/runs count=24 sha256=d093b45ff696826f54408bab4eade9c6f24eaf21db75b47b49d850339c31dcae
AFTER results/runs count=24 sha256=d093b45ff696826f54408bab4eade9c6f24eaf21db75b47b49d850339c31dcae
results/runs identical=yes
======================== 84 passed in 88.84s (0:01:28) ========================
Command: git --no-pager diff --exit-code
Exit code: 0
```
## Gate table
| Check | Expected | Actual | Pass/fail |
|---|---|---|---|
| Protected SHA256/counts | Identical | yes; 105/24 before and after | PASS |
| Tracked/staged diff | Exit 0 | 0/0 | PASS |
| All tests | All pass | 84 passed; exit 0 | PASS |
| Experiment log SHA256 | Identical | yes | PASS |
| Report length | <=40 lines | 38 lines | PASS |
Full output and per-file hashes: `docs/reports/stepR3a_raw_output.txt`.
Open issues: none.
