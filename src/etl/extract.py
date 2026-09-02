'''
Extract the analysis base table from the SQL database.
One row per policy, exposure, claim amount, total claim amount, risk features
'''

from __future__ import annotations

import logging
import pandas as pd 

from sqlalchemy import create_engine
from src.config import load_config

logger = logging.getLogger(__name__)

# Replace NULL claim with 0
BASE_TABLE_QUERY = """
    SELECT 
        p.policy_id,
        p.exposure,
        p.claim_nb,
        COALESCE(c.total_claim_amount, 0) AS total_claim_amount,
        COALESCE(c.n_claim_rows, 0)       AS n_claim_rows,
        p.area,
        p.veh_power,
        p.veh_age,
        p.driv_age,
        p.bonus_malus,
        p.veh_brand,
        p.veh_gas,
        p.density,
        p.region
    FROM policies p
    LEFT JOIN (
        SELECT policy_id,
            SUM(claim_amount) AS total_claim_amount,
            COUNT(*)          AS n_claim_rows
        FROM claims 
        GROUP BY policy_id
    ) c ON c.policy_id = p.policy_id
"""

def extract_base_table(db_url: str | None = None) -> pd.DataFrame:
    """Return one row per policy, with claim totals joined on."""
    if db_url == None:
        db_url = load_config()["database"]["url"]

    engine = create_engine(db_url)
    df = pd.read_sql(BASE_TABLE_QUERY, engine)
    logger.info("Extracted %d policy rows with %d columns", len(df), df.shape[1])
    return df

if __name__ == "__main__": 
    logging.basicConfig(level=logging.INFO)
    base = extract_base_table()
    print(base.head())
    print(base.shape)

