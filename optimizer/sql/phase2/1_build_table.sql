-- GreenQuanta Phase 2, STEP 1 (run ONCE): build our own table in BigQuery.
-- Scans ~500 GB of the public trace. Result is SAVED IN BIGQUERY (dataset cluster_trace), not downloaded.
-- Google Cluster Trace 2019, cell a, day 2, production jobs (priority 120-359),
-- instances seen in >= 280 of 288 five-minute windows, 2000 jobs, up to 40 instances per job.
CREATE OR REPLACE TABLE cluster_trace.phase2_usage_day2 AS
WITH params AS (SELECT 87000000000 AS day_start, 300000000 AS window_us),
prod_jobs AS (
  SELECT collection_id, MAX(priority) AS priority
  FROM `google.com:google-cluster-data`.clusterdata_2019_a.collection_events
  WHERE collection_type = 0 AND priority BETWEEN 120 AND 359
  GROUP BY collection_id
),
raw_usage AS (
  SELECT u.collection_id, u.instance_index,
         DIV(u.start_time - p.day_start, p.window_us) AS window_idx,
         LEAST((u.end_time - u.start_time) / p.window_us, 1.0) AS share,
         u.average_usage.cpus AS avg_cpu, u.maximum_usage.cpus AS max_cpu, u.average_usage.memory AS avg_mem
  FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_usage AS u
  CROSS JOIN params AS p
  JOIN prod_jobs AS j ON u.collection_id = j.collection_id
  WHERE u.collection_type = 0
    AND (u.alloc_collection_id IS NULL OR u.alloc_collection_id = 0)
    AND u.start_time >= p.day_start AND u.start_time < p.day_start + 288 * p.window_us
),
per_window AS (
  SELECT collection_id, instance_index, window_idx,
         SUM(avg_cpu * share) AS avg_cpu, MAX(max_cpu) AS max_cpu, SUM(avg_mem * share) AS avg_mem
  FROM raw_usage GROUP BY 1, 2, 3
),
full_day AS (
  SELECT collection_id, instance_index, AVG(avg_cpu) AS inst_mean_cpu, MAX(avg_cpu) AS inst_peak_cpu
  FROM per_window GROUP BY 1, 2 HAVING COUNT(*) >= 280
),
ranked AS (
  SELECT f.*, ROW_NUMBER() OVER (PARTITION BY collection_id ORDER BY instance_index) AS rn,
         COUNT(*) OVER (PARTITION BY collection_id) AS n_inst
  FROM full_day AS f
),
chosen_jobs AS (
  SELECT collection_id FROM ranked WHERE n_inst >= 4 GROUP BY collection_id
  ORDER BY FARM_FINGERPRINT(CAST(collection_id AS STRING)) LIMIT 2000
)
SELECT w.collection_id, j.priority, w.instance_index, r.rn, r.n_inst,
       r.inst_mean_cpu, r.inst_peak_cpu, w.window_idx, w.avg_cpu, w.max_cpu, w.avg_mem,
       FARM_FINGERPRINT(CAST(w.collection_id AS STRING)) AS job_order
FROM per_window AS w
JOIN ranked AS r USING (collection_id, instance_index)
JOIN chosen_jobs AS c USING (collection_id)
JOIN prod_jobs AS j USING (collection_id)
WHERE r.rn <= 40;
