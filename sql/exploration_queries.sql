-- ============================================================================
-- SQL EXPLORATION QUERIES
-- Run these against data/processed/portfolio.db (SQLite) once the pipeline has
-- built it. Five are solved as worked examples; the rest (marked TODO) are
-- yours. Together they cover SELECT, JOIN, GROUP BY, HAVING, CASE WHEN and
-- WINDOW FUNCTIONS - which is most of what internship SQL tests ask.
-- ============================================================================

-- 1. Portfolio size: policy count, total exposure, total claims.  (aggregates)
SELECT
    COUNT(*)                AS n_policies,
    ROUND(SUM(exposure), 1) AS total_exposure,
    SUM(claim_nb)           AS total_claims
FROM policies;

-- 2. Claim frequency by driver-age band.  (CASE WHEN + GROUP BY)
SELECT
    CASE
        WHEN driv_age < 25 THEN '18-24'
        WHEN driv_age < 35 THEN '25-34'
        WHEN driv_age < 45 THEN '35-44'
        WHEN driv_age < 55 THEN '45-54'
        WHEN driv_age < 65 THEN '55-64'
        ELSE '65+'
    END                                            AS age_band,
    SUM(claim_nb)                                  AS claims,
    ROUND(SUM(exposure), 1)                        AS exposure,
    ROUND(1.0 * SUM(claim_nb) / SUM(exposure), 4)  AS frequency
FROM policies
GROUP BY age_band
ORDER BY age_band;

-- 3. Average severity by region.  (multi-table JOIN + GROUP BY)
SELECT
    r.region_name,
    COUNT(c.claim_id)              AS n_claims,
    ROUND(AVG(c.claim_amount), 0)  AS avg_severity
FROM claims c
JOIN policies p ON p.policy_id = c.policy_id
JOIN regions  r ON r.region    = p.region
GROUP BY r.region_name
ORDER BY avg_severity DESC;

-- 4. Credible segments only: vehicle brands with > 10,000 exposure.  (HAVING)
SELECT
    veh_brand,
    ROUND(SUM(exposure), 0)                        AS exposure,
    ROUND(1.0 * SUM(claim_nb) / SUM(exposure), 4)  AS frequency
FROM policies
GROUP BY veh_brand
HAVING SUM(exposure) > 10000
ORDER BY frequency DESC;

-- 5. Rank regions by frequency and compare each to the portfolio average.
--    (WINDOW FUNCTIONS: RANK() and AVG() OVER ())
SELECT
    region,
    ROUND(frequency, 4)                              AS frequency,
    RANK() OVER (ORDER BY frequency DESC)            AS freq_rank,
    ROUND(frequency - AVG(frequency) OVER (), 4)     AS vs_portfolio_avg
FROM (
    SELECT region,
           1.0 * SUM(claim_nb) / SUM(exposure) AS frequency
    FROM policies
    GROUP BY region
);

-- ----- YOUR TURN ------------------------------------------------------------
-- 6.  TODO: Pure premium (SUM(claim_amount) / SUM(exposure)) by vehicle gas type.
SELECT
    p.veh_gas     AS vehicle_gas_type,
    ROUND(SUM(p.exposure),1)    AS exposure,
    ROUND(SUM(c.total_claim_amount), 0) AS total_claim_cost,
    ROUND(SUM(c.total_claim_amount)/SUM(p.exposure), 2)  AS pure_premium
FROM policies p
LEFT JOIN (
    SELECT policy_id,
        SUM(claim_amount) AS total_claim_amount
    FROM claims
    GROUP BY policy_id
) c ON c.policy_id = p.policy_id
GROUP BY p.veh_gas;
    

-- 7.  Top 5 policies by total claim amount.  (JOIN + ORDER BY + LIMIT)
SELECT 
    p.policy_id     AS policy_id,
    p.driv_age      AS driver_age,
    p.veh_brand     AS vehicle_brand,
    ROUND(c.total_claim_amount, 0) AS total_claim_amount
FROM policies p 
JOIN(
    SELECT policy_id,
        SUM(claim_amount) AS total_claim_amount
    FROM claims
    GROUP BY policy_id
) c ON c.policy_id = p.policy_id
ORDER BY c.total_claim_amount DESC
LIMIT 5;


-- 8.  TODO: Per region, running cumulative exposure ordered by driv_age.
--           (WINDOW: SUM(exposure) OVER (PARTITION BY region ORDER BY driv_age))

SELECT
    region,
    driv_age,
    ROUND(exposure, 1)      AS exposure,
    ROUND(SUM(exposure) OVER(PARTITION BY region ORDER BY driv_age),1)      AS total_exposure
FROM(
    SELECT region,
        driv_age,
        SUM(exposure) AS exposure
    FROM policies
    GROUP BY region, driv_age
) AS e
ORDER BY region, driv_age;


-- 9.  TODO: Loss ratio by age band. Needs the technical_premium you add in the
--           pricing stage - revisit after Stage 3 of the blueprint.


-- 10. TODO: Share of total portfolio claims contributed by each region.
--           (claim total per region divided by SUM(...) OVER ())
SELECT
    region,
    SUM(claim_nb) AS claims,
    ROUND(100.0 * SUM(claim_nb) / (SELECT SUM(claim_nb) FROM policies), 1) AS pct_of_portfolio
FROM policies
GROUP BY region
ORDER BY pct_of_portfolio DESC;