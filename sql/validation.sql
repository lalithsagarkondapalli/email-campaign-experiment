-- Data-validation checks. Every row returned is a violation; the pipeline fails if any come back.

WITH raw AS (SELECT * FROM read_csv_auto('{path}'))
SELECT 'null in required column' AS check_name, COUNT(*) AS violations
FROM raw
WHERE segment IS NULL OR visit IS NULL OR conversion IS NULL OR spend IS NULL
UNION ALL
SELECT 'unknown arm', COUNT(*) FROM raw
WHERE segment NOT IN ('No E-Mail', 'Mens E-Mail', 'Womens E-Mail')
UNION ALL
SELECT 'conversion without visit', COUNT(*) FROM raw WHERE conversion = 1 AND visit = 0
UNION ALL
SELECT 'spend without conversion', COUNT(*) FROM raw WHERE spend > 0 AND conversion = 0
UNION ALL
SELECT 'conversion with zero spend', COUNT(*) FROM raw WHERE conversion = 1 AND spend <= 0
UNION ALL
SELECT 'negative history or spend', COUNT(*) FROM raw WHERE history < 0 OR spend < 0
UNION ALL
SELECT 'recency outside 1-12 months', COUNT(*) FROM raw WHERE recency NOT BETWEEN 1 AND 12;
