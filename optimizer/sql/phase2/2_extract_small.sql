-- GreenQuanta Phase 2, STEP 2 (small): cheap query on OUR table (reads only a few MB).
-- 10 jobs, first 8 instances of each job (the same jobs are inside the bigger sizes).
-- Save the result as CSV:  Save results -> CSV (Google Drive)  if local download is limited,
-- then put it at data/trace/phase2/google2019_a_day2_small.csv
WITH jobs AS (
  SELECT DISTINCT collection_id, job_order FROM cluster_trace.phase2_usage_day2
  ORDER BY job_order LIMIT 10
)
SELECT t.collection_id, t.instance_index, t.window_idx, t.avg_cpu, t.max_cpu, t.avg_mem, t.priority
FROM cluster_trace.phase2_usage_day2 AS t
JOIN jobs USING (collection_id)
WHERE t.rn <= 8
ORDER BY 1, 2, 3;
