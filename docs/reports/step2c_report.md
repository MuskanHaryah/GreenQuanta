# Step 2c: proof-gated optimal labels
Branch: `step2c-optimal-label`, created from `step2a-selection`.
Changed only gap_metrics_v2 and new tests; gap calculations and existing keys preserved.
Full raw outputs: [step2c_raw_output.txt](step2c_raw_output.txt).
| Check (.venv-gq) | Expected | Actual | Pass/fail |
| --- | --- | --- | --- |
| Before: `pytest tests/test_gap_metrics.py -v` | a/c/d/e pass; b/f/g fail | 4 passed, 4 failed (g parametrized) | PASS |
| After: `pytest tests -v` | All pass; tests unchanged | 39 passed | PASS |
| `git diff --check` | Exit 0 | Exit 0 | PASS |
```text
========================= 4 failed, 4 passed in 9.39s =========================
============================= 39 passed in 5.59s ==============================
```
`git diff -- optimizer/phase2/pipeline_v2.py`:
```diff
diff --git a/optimizer/phase2/pipeline_v2.py b/optimizer/phase2/pipeline_v2.py
index bb6a2d6e..59d4631f 100644
--- a/optimizer/phase2/pipeline_v2.py
+++ b/optimizer/phase2/pipeline_v2.py
@@ -59,4 +59,9 @@ def gap_metrics_v2(final, pure, vms, servers, exact_assignment, exact_info, stat
         out["pure_and_optimal"] = linear_cost(pure, vms, servers) <= ex_lin * (1 + 1e-3)
     else:
         out["pure_and_optimal"] = False
+    out["final_matches_reference_linear"] = out["final_matches_optimum"]
+    out["pure_matches_reference_linear"] = out["pure_and_optimal"]
+    out["final_matches_optimum"] = out["final_matches_optimum"] and out["exact_proven_optimal"]
+    out["pure_and_optimal"] = out["pure_and_optimal"] and out["exact_proven_optimal"]
+    out["exact_mip_gap"] = exact_info.get("mip_gap")
     return out
```
`git status --short`:
```text
 M optimizer/phase2/pipeline_v2.py
?? .rules
?? docs/
?? results/experiment_log.csv
?? results/runs/
?? tests/test_gap_metrics.py
```
Open issues: none. No dependencies added, runners executed, or results writes.
