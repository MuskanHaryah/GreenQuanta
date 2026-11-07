-- GreenQuanta: which server sizes exist in Google cell a?
-- Use the most common row as SERVER_CPU_CAPACITY / SERVER_MEM_CAPACITY in optimizer/phase1/config.py
-- (This is the same query as in Google's official analysis colab.)
SELECT capacity.cpus   AS cpu_capacity_ncu,
       capacity.memory AS mem_capacity_nmu,
       COUNT(DISTINCT machine_id) AS num_machines
FROM `google.com:google-cluster-data`.clusterdata_2019_a.machine_events
WHERE capacity.cpus IS NOT NULL
GROUP BY 1, 2
ORDER BY num_machines DESC
LIMIT 10;
