-- GreenQuanta Phase 2 v2, optional: rows and instances each v2 size will give (cheap).
WITH ordered AS (
  SELECT collection_id, ROW_NUMBER() OVER (ORDER BY job_order) AS job_rank
  FROM (SELECT DISTINCT collection_id, job_order FROM cluster_trace.phase2_usage_day2)
)
SELECT size, COUNT(DISTINCT CONCAT(CAST(t.collection_id AS STRING), '-', CAST(t.instance_index AS STRING))) AS instances,
       COUNT(*) AS rows
FROM cluster_trace.phase2_usage_day2 AS t
JOIN ordered AS o USING (collection_id)
CROSS JOIN UNNEST([STRUCT('small' AS size, 10 AS n), ('medium', 20), ('large', 40)]) AS s
WHERE o.job_rank <= s.n AND t.rn <= 16
GROUP BY size ORDER BY instances;
