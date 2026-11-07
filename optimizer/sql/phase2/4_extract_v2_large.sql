-- GreenQuanta Phase 2 UPDATE (v2), size large: cheap query on OUR table (no new 500 GB scan).
-- 40 jobs, first 16 instances of each job (v1 used 8), so jobs give bigger and more VMs.
-- Same job order as v1, so the v1 jobs are inside the v2 sizes.
-- Save as CSV (Google Drive if local download is limited) at:
--   data/trace/phase2/google2019_a_day2_v2_large.csv
WITH jobs AS (
  SELECT DISTINCT collection_id, job_order FROM cluster_trace.phase2_usage_day2
  ORDER BY job_order LIMIT 40
)
SELECT t.collection_id, t.instance_index, t.window_idx, t.avg_cpu, t.max_cpu, t.avg_mem, t.priority
FROM cluster_trace.phase2_usage_day2 AS t
JOIN jobs USING (collection_id)
WHERE t.rn <= 16
ORDER BY 1, 2, 3;
