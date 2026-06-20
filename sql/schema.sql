-- Star-style schema for the motor pricing platform.
--   policies = fact table (one row per policy)
--   claims   = one row per individual claim
--   regions  = small dimension table, gives you something to JOIN against
--
-- Written for SQLite (the default). It is standard enough to run on PostgreSQL
-- with little or no change.

DROP TABLE IF EXISTS claims;
DROP TABLE IF EXISTS policies;
DROP TABLE IF EXISTS regions;

CREATE TABLE regions (
    region        TEXT PRIMARY KEY,   -- e.g. 'R11'
    region_name   TEXT,               -- human-readable label
    macro_area    TEXT                -- grouping for reporting
);

CREATE TABLE policies (
    policy_id     INTEGER PRIMARY KEY,  -- IDpol
    exposure      REAL    NOT NULL,     -- fraction of the year at risk (0-1)
    claim_nb      INTEGER NOT NULL,     -- number of claims on the policy
    area          TEXT,                 -- population density area code A-F
    veh_power     INTEGER,
    veh_age       INTEGER,
    driv_age      INTEGER,
    bonus_malus   INTEGER,              -- French no-claims coefficient (>100 = penalised)
    veh_brand     TEXT,
    veh_gas       TEXT,                 -- 'Diesel' / 'Regular'
    density       INTEGER,              -- inhabitants per km2 of the city
    region        TEXT REFERENCES regions(region)
);

CREATE TABLE claims (
    claim_id      INTEGER PRIMARY KEY,  -- surrogate key
    policy_id     INTEGER REFERENCES policies(policy_id),
    claim_amount  REAL NOT NULL
);

-- Indexes that speed up the JOIN / GROUP BY queries you'll write.
CREATE INDEX idx_policies_region ON policies(region);
CREATE INDEX idx_claims_policy   ON claims(policy_id);
