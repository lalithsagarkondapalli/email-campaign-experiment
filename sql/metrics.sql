-- Metric layer for the e-mail experiment, run in DuckDB over the raw CSV.
-- One row per arm: sample size, share of traffic, and the three outcome metrics.

WITH base AS (
    SELECT
        CASE segment
            WHEN 'No E-Mail'     THEN 'control'
            WHEN 'Mens E-Mail'   THEN 'mens'
            WHEN 'Womens E-Mail' THEN 'womens'
        END                                   AS arm,
        visit,
        conversion,
        spend
    FROM read_csv_auto('{path}')
),
arm_stats AS (
    SELECT
        arm,
        COUNT(*)                              AS n,
        AVG(visit)                            AS visit_rate,
        AVG(conversion)                       AS conversion_rate,
        AVG(spend)                            AS spend_per_customer,
        SUM(spend)                            AS total_spend,
        -- spend among buyers, to check whether lift comes from more buyers or bigger baskets
        AVG(spend) FILTER (WHERE conversion = 1) AS spend_per_buyer
    FROM base
    GROUP BY arm
)
SELECT
    *,
    n * 1.0 / SUM(n) OVER ()                  AS traffic_share,
    visit_rate      - MAX(CASE WHEN arm = 'control' THEN visit_rate END)         OVER () AS visit_lift_abs,
    conversion_rate - MAX(CASE WHEN arm = 'control' THEN conversion_rate END)    OVER () AS conversion_lift_abs,
    spend_per_customer - MAX(CASE WHEN arm = 'control' THEN spend_per_customer END) OVER () AS spend_lift_abs
FROM arm_stats
ORDER BY arm;
