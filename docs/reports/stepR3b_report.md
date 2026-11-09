# Step R3b
Branch: `step3-r3-exact-qubo`; only new files created; no commits.
New files: `optimizer/diagnostics/r3_run.py`, `tests/test_r3_run.py`, this report, `stepR3b_raw_output.txt`.
Interpreter: `.venv-gq/Scripts/python.exe`; dependencies added: none; no installs.
No sampling, project runner execution, experiment-log writes, or protected-result writes.
Settings: common.build_problem; L0.05; unbalanced; center=0.02; CPU/RAM unit=0.0025; existing tie-break.
R3/R3v: 600 seconds per solve; max_iters=50; no limits increased.
R3v reuses R3a with hard singleton/link rows and real CPU/RAM <= limit*(1-1e-6); strict revalidation.
Labels compare linear joules with Step 2e R2 (quantized MILP); R1 shown only as information.
## Commands run
- `.venv-gq/Scripts/python.exe -B -m pytest tests/test_exact_qubo.py -v -p no:cacheprovider` (600-second timeout).
- `.venv-gq/Scripts/python.exe -B -c '...'`: common.build_problem + D2 decomposition/saved-assignment safeguard.
- `.venv-gq/Scripts/python.exe -B -u optimizer/diagnostics/r3_run.py --sizes small_v2 medium_v2 large_v2 --audit`.
- Audit executes R3a tests, new tests, exact diagnostics, `pytest tests -v -p no:cacheprovider`, SHA256 manifests, git diff gates.
## Results (all six solves proven optimal)
| Size | Label | H_min | H_valid_min | H_R2 | D3 best valid H (recorded) | R3/R3v seconds |
|---|---|---|---|---|---|---|
| small_v2 | R3_INVALID | 0.310643807345195 | 0.4304366536072653 | 0.4430283491089941 | 0.4304401138135745 | 1.047 / 4.602 |
| medium_v2 | R3_INVALID | 0.3687558638108289 | 0.4032419086959977 | 0.4055256457202229 | 0.5792571166238005 | 8.303 / 3.858 |
| large_v2 | R3_INVALID | 0.4161278529531364 | 0.4428889812708667 | 0.4697070580383528 | 1.09935717248851 | 481.457 / 237.598 |
## Gate table
| Check | Expected | Actual | Pass/fail |
|---|---|---|---|
| R3a tests | Clean pass within 600s | Initial rerun: 13 passed, 0 failed, 84.41s; audit rerun: 13 passed, 0 failed, 59.11s | PASS |
| Saved large_v2 H and five-term split | D2 within 1e-9; D3 H_exact and user rounded values match | Match | PASS |
| New focused tests | All pass | 24 passed, 0 failed, 3.89s | PASS |
| Full pytest tests -v | All pass | 108 passed, 0 failed, 78.89s | PASS |
| Protected SHA256/counts | Identical | phase2 yes, 105/105; runs yes, 24/24 | PASS |
| Tracked/staged diff | Exit 0 | 0 / 0 | PASS |
| Experiment log | Unchanged | identical=yes | PASS |
## Raw gate output
```text
results/phase2 identical=yes counts=105/105
results/runs identical=yes counts=24/24
experiment_log identical=yes
======================= 108 passed in 78.89s (0:01:18) ========================
```
Full raw commands/output, before/after SHA256 manifests, certificates, validators and per-size tables: `docs/reports/stepR3b_raw_output.txt`.
Open issue: D3 stores best-valid H scalars but no best-sample x/y bits; bqm.energy re-evaluation skipped and recorded scalars explicitly identified.
