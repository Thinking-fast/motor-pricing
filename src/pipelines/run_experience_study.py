"""Generate and save the portfolio experience-study tables."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import load_config
from src.etl.clean import clean_base_table
from src.etl.extract import extract_base_table
from src.analysis.experience_study import (
    add_age_band,
    cap_large_claims,
    experience_study,
    large_loss_loading,
)

logger = logging.getLogger(__name__)


FACTORS = [
    "age_band",
    "region",
    "veh_brand",
    "veh_gas",
    "area",
]


def run_experience_studies() -> dict:
    """Generate and save an experience-study table for each factor."""
    config = load_config()

    # 1. Extract policies from the database.
    base = extract_base_table()

    # 2. Apply the cleaning decisions.
    base = clean_base_table(base)

    # 3. Add the driver-age bands.
    base = add_age_band(base)

    # 4. Cap losses
    large_loss_cap = config["experience_study"]["large_loss_cap"]

    capped_base, total_excess = cap_large_claims(
        base,
        cap=large_loss_cap,
    )

    loading = large_loss_loading(
        total_excess=total_excess,
        total_exposure=base["exposure"].sum(),
    )

    logger.info(
        "Portfolio large-loss loading: %.2f per unit of exposure",
        loading,
    )

    # Check the portfolio reconciliation
    gross_portfolio_rate = base["total_claim_amount"].sum() / base["exposure"].sum()

    capped_portfolio_rate = (
        capped_base["total_claim_amount"].sum() / capped_base["exposure"].sum()
    )

    if abs(gross_portfolio_rate - (capped_portfolio_rate + loading)) > 0.01:
        raise ValueError("Large-loss loading reconciliation failed")

    # 5. Find the processed-data directory.
    output_directory = Path(config["paths"]["processed_data"])

    # 6. Create every experience table.
    tables = {}

    for factor in FACTORS:
        gross_table = experience_study(base, by=factor)
        capped_table = experience_study(capped_base, by=factor)

        gross_difference = (
            gross_table["pure_premium"]
            - gross_table["frequency"] * gross_table["severity"]
        ).abs()

        capped_difference = (
            capped_table["pure_premium"]
            - capped_table["frequency"] * capped_table["severity"]
        ).abs()

        if not gross_difference.dropna().le(0.01).all():
            raise ValueError(f"Gross pure-premium decomposition failed for {factor}")

        if not capped_difference.dropna().le(0.01).all():
            raise ValueError(f"Capped pure-premium decomposition failed for {factor}")

        gross_path = output_directory / f"experience_{factor}.csv"
        capped_path = output_directory / f"experience_{factor}_capped.csv"

        gross_table.to_csv(gross_path, index=False)
        capped_table.to_csv(capped_path, index=False)

        tables[factor] = gross_table
        tables[f"{factor}_capped"] = capped_table

        logger.info(
            "Saved %s gross experience study to %s",
            factor,
            gross_path,
        )

        logger.info(
            "Saved %s capped experience study to %s",
            factor,
            capped_path,
        )

    return tables


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_experience_studies()
