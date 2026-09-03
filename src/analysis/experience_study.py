"""Create one-way actuarial experience studies."""

from __future__ import annotations

import logging

import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)


def add_age_band(df: pd.DataFrame, bins=None, labels=None) -> pd.DataFrame:
    """Add an age_band column based on driver age."""
    df = df.copy()
    if bins is None or labels is None:
        config = load_config()

        if bins is None:
            bins = config["experience_study"]["driv_age_bands"]

        if labels is None:
            labels = config["experience_study"]["labels"]

    if len(labels) != len(bins) - 1:
        raise ValueError("Age-band labels must be one fewer than bin boundaries")

    df["age_band"] = pd.cut(df["driv_age"], bins=bins, labels=labels, right=False)

    missing_age_band = df["age_band"].isna()
    n_missing = missing_age_band.sum()

    logger.info(
        "add_age_band: %d rows received no age band",
        n_missing,
    )

    return df


def cap_large_claims(df: pd.DataFrame, cap: float) -> tuple[pd.DataFrame, float]:
    """Cap each policy's total claim amount and return the excess removed."""
    if cap <= 0:
        raise ValueError("Large-loss cap must be positive")

    df = df.copy()

    over_cap = df["total_claim_amount"] > cap
    n_capped = over_cap.sum()

    original_amount = df["total_claim_amount"].copy()

    df["total_claim_amount"] = original_amount.clip(upper=cap)

    excess = original_amount - df["total_claim_amount"]
    total_excess = float(excess.sum())

    logger.info(
        "cap_large_claims: capped %d policies at %.2f; total excess=%.2f",
        n_capped,
        cap,
        total_excess,
    )

    return df, total_excess


def large_loss_loading(total_excess: float, total_exposure: float) -> float:
    """Return the portfolio large-loss loading per unit of exposure."""
    if total_exposure <= 0:
        raise ValueError("Total exposure must be positive")

    return total_excess / total_exposure


def experience_study(
    df: pd.DataFrame, by: str, min_exposure: float = 1000
) -> pd.DataFrame:
    """Produce a one-way experience study for one rating factor."""

    if by not in df.columns:
        raise ValueError(f"Unknown grouping column: {by}")

    grouped = df.groupby(by, observed=True).agg(
        policies=("policy_id", "count"),
        exposure=("exposure", "sum"),
        claim_nb=("claim_nb", "sum"),
        n_claims_with_amount=("n_claim_rows", "sum"),
        claim_amount=("total_claim_amount", "sum"),
    )

    grouped["frequency"] = grouped["claim_nb"] / grouped["exposure"]
    grouped["severity"] = grouped["claim_amount"] / grouped["claim_nb"]
    grouped["severity_reported"] = (
        grouped["claim_amount"] / grouped["n_claims_with_amount"]
    )
    grouped["pure_premium"] = grouped["claim_amount"] / grouped["exposure"]

    grouped["credible"] = grouped["exposure"] >= min_exposure

    n_low_exposure = (~grouped["credible"]).sum()

    logger.info(
        "experience_study: factor=%s, groups=%d, low-exposure groups=%d",
        by,
        len(grouped),
        n_low_exposure,
    )

    return grouped.reset_index()
