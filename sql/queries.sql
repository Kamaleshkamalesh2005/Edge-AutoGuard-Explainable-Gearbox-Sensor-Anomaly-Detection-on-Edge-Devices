-- Anomalous windows sorted by highest anomaly score
SELECT id, run_id, window_id, anomaly_score, status
FROM predictions
WHERE status = 'Anomalous'
ORDER BY anomaly_score DESC;

-- Run-wise anomaly summary
SELECT run_id,
       COUNT(*) AS window_count,
       SUM(CASE WHEN status = 'Anomalous' THEN 1 ELSE 0 END) AS anomalous_count,
       AVG(anomaly_score) AS average_anomaly_score
FROM predictions
GROUP BY run_id
ORDER BY anomalous_count DESC;

-- Average anomaly score across all windows
SELECT AVG(anomaly_score) AS average_anomaly_score
FROM predictions;

-- Status distribution
SELECT status, COUNT(*) AS count
FROM predictions
GROUP BY status
ORDER BY count DESC;

-- Latest window for each run
SELECT run_id, MAX(window_id) AS latest_window_id
FROM predictions
GROUP BY run_id;

-- Latest anomaly records by run
SELECT p.*
FROM predictions p
INNER JOIN (
    SELECT run_id, MAX(window_id) AS latest_window_id
    FROM predictions
    GROUP BY run_id
) latest
ON p.run_id = latest.run_id AND p.window_id = latest.latest_window_id
ORDER BY p.run_id;
