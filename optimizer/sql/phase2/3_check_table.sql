-- GreenQuanta Phase 2, optional check: how big is our table? (cheap)
SELECT COUNT(DISTINCT collection_id) AS jobs,
       COUNT(DISTINCT CONCAT(CAST(collection_id AS STRING), '-', CAST(instance_index AS STRING))) AS instances,
       COUNT(*) AS rows
FROM cluster_trace.phase2_usage_day2;
