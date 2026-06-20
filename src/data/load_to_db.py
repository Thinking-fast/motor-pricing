"""Load freMTPL2 CSVs into the SQL database defined in sql/schema.sql.

Builds three tables - policies, claims, regions - using SQLAlchemy, so the same
code targets SQLite (default) or PostgreSQL just by changing the connection URL
in config.yaml.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# A few human-readable region labels as examples. Extend this map yourself.
REGION_NAMES = {
    "R11": "Ile-de-France",
    "R24": "Centre-Val de Loire",
    "R52": "Pays de la Loire",
    "R53": "Bretagne",
    "R82": "Rhone-Alpes",
    "R93": "Provence-Alpes-Cote d'Azur",
}


def _build_regions(freq: pd.DataFrame) -> pd.DataFrame:
    regions = freq[["Region"]].drop_duplicates().rename(columns={"Region": "region"})
    regions["region_name"] = regions["region"].map(REGION_NAMES).fillna(regions["region"])
    regions["macro_area"] = "France"  # TODO: map regions to north/south/etc. macro areas
    return regions


def load_to_db(freq: pd.DataFrame, sev: pd.DataFrame, db_url: str,
               schema_sql: str | Path) -> None:
    """Create the schema and load policies, claims and regions."""
    engine = create_engine(db_url)

    # 1. (Re)create the schema from the DDL file.
    ddl = Path(schema_sql).read_text(encoding="utf-8")
    with engine.begin() as conn:
        for statement in ddl.split(";"):
            if statement.strip():
                conn.execute(text(statement))

    # 2. regions dimension table.
    _build_regions(freq).to_sql("regions", engine, if_exists="append", index=False)

    # 3. policies fact table (rename freMTPL2 columns to the snake_case schema).
    policies = freq.rename(columns={
        "IDpol": "policy_id", "Exposure": "exposure", "ClaimNb": "claim_nb",
        "Area": "area", "VehPower": "veh_power", "VehAge": "veh_age",
        "DrivAge": "driv_age", "BonusMalus": "bonus_malus", "VehBrand": "veh_brand",
        "VehGas": "veh_gas", "Density": "density", "Region": "region",
    })
    keep = ["policy_id", "exposure", "claim_nb", "area", "veh_power", "veh_age",
            "driv_age", "bonus_malus", "veh_brand", "veh_gas", "density", "region"]
    policies[keep].to_sql("policies", engine, if_exists="append", index=False)

    # 4. claims table with a surrogate claim_id.
    claims = sev.rename(columns={"IDpol": "policy_id", "ClaimAmount": "claim_amount"}).copy()
    claims.insert(0, "claim_id", range(1, len(claims) + 1))
    claims[["claim_id", "policy_id", "claim_amount"]].to_sql(
        "claims", engine, if_exists="append", index=False)

    logger.info("Loaded %d policies and %d claims into %s",
                len(policies), len(claims), db_url)
