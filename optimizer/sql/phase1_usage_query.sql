-- GreenQuanta Phase 1: real 24-hour usage of long-running production instances
-- Google Cluster Trace 2019, cell a, day 2 of the trace, 288 windows of 5 minutes
-- Output columns: collection_id, instance_index, window_idx, avg_cpu, max_cpu, avg_mem
-- Units: CPU in NCU, memory in NMU (Google's normalized units)
--
-- To scale up later: change LIMIT 10 (number of jobs) and rn <= 8 (instances per job).
-- To use another day: day_start = 600000000 + day_number * 86400000000 (day_number 1 = 2nd day).
-- To use another cell: replace clusterdata_2019_a with clusterdata_2019_b ... _h in BOTH places.

WITH params AS (
  SELECT 87000000000 AS day_start,      -- 600 s trace offset + 1 day, in microseconds
         300000000   AS window_us       -- 5 minutes in microseconds
),
prod_jobs AS (                           -- production tier = long-running services
  SELECT collection_id
  FROM `google.com:google-cluster-data`.clusterdata_2019_a.collection_events
  WHERE collection_type = 0 AND priority BETWEEN 120 AND 359
  GROUP BY collection_id
),
raw_usage AS (
  SELECT u.collection_id, u.instance_index,
         DIV(u.start_time - p.day_start, p.window_us) AS window_idx,
         LEAST((u.end_time - u.start_time) / p.window_us, 1.0) AS share,
         u.average_usage.cpus   AS avg_cpu,
         u.maximum_usage.cpus   AS max_cpu,
         u.average_usage.memory AS avg_mem
  FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_usage AS u
  CROSS JOIN params AS p
  JOIN prod_jobs AS j ON u.collection_id = j.collection_id
  WHERE u.collection_type = 0
    AND (u.alloc_collection_id IS NULL OR u.alloc_collection_id = 0)
    AND u.start_time >= p.day_start
    AND u.start_time <  p.day_start + 288 * p.window_us
),
per_window AS (                          -- time-weighted average if a window has 2 records
  SELECT collection_id, instance_index, window_idx,
         SUM(avg_cpu * share) AS avg_cpu,
         MAX(max_cpu)         AS max_cpu,
         SUM(avg_mem * share) AS avg_mem
  FROM raw_usage
  GROUP BY 1, 2, 3
),
full_day AS (                            -- instance seen in at least 280 of 288 windows
  SELECT collection_id, instance_index
  FROM per_window
  GROUP BY 1, 2
  HAVING COUNT(*) >= 280
),
ranked AS (
  SELECT collection_id, instance_index,
         ROW_NUMBER() OVER (PARTITION BY collection_id ORDER BY instance_index) AS rn,
         COUNT(*)     OVER (PARTITION BY collection_id) AS n_inst
  FROM full_day
),
chosen_jobs AS (                         -- 10 jobs with >= 4 full-day instances, repeatable choice
  SELECT collection_id
  FROM ranked
  WHERE n_inst >= 4
  GROUP BY collection_id
  ORDER BY FARM_FINGERPRINT(CAST(collection_id AS STRING))
  LIMIT 10
)
SELECT w.collection_id, w.instance_index, w.window_idx, w.avg_cpu, w.max_cpu, w.avg_mem
FROM per_window AS w
JOIN ranked      AS r USING (collection_id, instance_index)
JOIN chosen_jobs AS c USING (collection_id)
WHERE r.rn <= 8
ORDER BY 1, 2, 3;
