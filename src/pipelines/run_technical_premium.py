"""Construct technical premiums and produce profitability studies."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analysis.experience_study import add_age_band
from src.config import load_config
from src.etl.clean import clean_base_table
from src.etl.extract import extract_base_table
from src.pricing.technical_premium import (
    add_technical_premium,
    portfolio_pure_premium_rate,
    profitability_study,
)

logger = logging.getLogger(__name__)

FACTORS = [
    "age_band",
    "region",
    "veh_brand",
    "veh_gas",
    "area",
]


def run_technical_premium() -> dict[str, pd.DataFrame]:
    """Build technical premiums and save profitability tables."""
    config = load_config()

    base = extract_base_table()
    base = clean_base_table(base)
    base = add_age_band(base)

    pure_premium_rate = portfolio_pure_premium_rate(base)

    expense_loading = config["pricing"]["expense_loading"]
    profit_loading = config["pricing"]["profit_loading"]

    priced = add_technical_premium(
        base,
        pure_premium_rate=pure_premium_rate,
        expense_loading=expense_loading,
        profit_loading=profit_loading,
    )

    output_directory = Path(config["paths"]["processed_data"])

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables = {}

    for factor in FACTORS:
        table = profitability_study(
            priced,
            by=factor,
        )

        output_path = output_directory / f"profitability_{factor}.csv"

        table.to_csv(
            output_path,
            index=False,
        )

        tables[factor] = table

        logger.info(
            "Saved %s profitability study to %s",
            factor,
            output_path,
        )

    return tables


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_technical_premium()
