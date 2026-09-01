-- ============================================================================
-- DATA QUALITY CHECKS
-- Re-run after any reload of portfolio.db. Findings are written up in
-- docs/data_quality.md; the cleaning decisions are implemented in
-- src/etl/clean.py (Stage 2).
-- ============================================================================

-- 1. ClaimNb in policies vs actual rows in claims.
SELECT 
    claims_per_policies,
    rows_in_claims,
    claims_per_policies - rows_in_claims AS difference

FROM (
    SELECT (SELECT SUM(claim_nb) FROM policies) AS claims_per_policies,
        (SELECT COUNT(*) FROM claims) AS rows_in_claims 
) AS t;

-- 2. Claims referencing a policy_id that isn't in policies.
SELECT COUNT(*) AS orphan_claims, 
    ROUND(SUM(c.claim_amount), 0) AS orphan_amount
FROM claims c
LEFT JOIN policies p ON p.policy_id = c.policy_id
WHERE p.policy_id IS NULL;

-- 3. Exposure above one policy-year (impossible)
SELECT COUNT(*) FROM policies WHERE exposure > 1;

-- 4. Spread of claim amounts.
SELECT MIN(claim_amount) AS min_claim,
       MAX(claim_amount) AS max_claim
FROM claims;

-- 5. Duplicate policy ids 
SELECT COUNT(*) AS duplicate_policy_ids
FROM (SELECT policy_id FROM policies GROUP BY policy_id HAVING COUNT(*) > 1);

-- 6. BonusMalus extremes. French scale: 50 = max no-claims discount, 100 = neutral.
SELECT MAX(bonus_malus) AS max_bonus_malus FROM policies;