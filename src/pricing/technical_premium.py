"""Construct technical premiums and analyse segment profitability.

The source dataset contains no observed premium. All premiums produced here
are analytical estimates, not actual amounts charged to policyholders.
"""

from __future__ import annotations

import logging

import pandas as pd


logger = logging.getLogger(__name__)

def portfolio_pure_premium_rate(df: pd.DataFrame) -> float:
    """Return total claim cost per unit of portfolio exposure."""

    total_claims = df["total_claim_amount"].sum()
    total_exposure = df["exposure"].sum()

    if total_exposure <= 0:
        raise ValueError("Total exposure must be positive")

    rate = total_claims / total_exposure

    logger.info(
        "Portfolio pure-premium rate: %.2f per unit of exposure",
        rate,
    )

    return float(rate)

def add_technical_premium(
    df: pd.DataFrame,
    pure_premium_rate: float,
    expense_loading: float,
    profit_loading: float,
) -> pd.DataFrame:
    """Add constructed technical-premium columns to each policy."""

    df = df.copy()

    if pure_premium_rate < 0:
        raise ValueError("Pure-premium rate cannot be negative")

    if expense_loading < 0 or profit_loading < 0:
        raise ValueError("Premium loadings cannot be negative")

    loading_factor = 1 + expense_loading + profit_loading

    df["technical_premium_rate"] = (pure_premium_rate * loading_factor)
    df["technical_premium"] = (df["technical_premium_rate"] * df["exposure"])

    logger.info(
        "Constructed technical premium: rate=%.2f, total=%.2f",
        df["technical_premium_rate"].iloc[0],
        df["technical_premium"].sum(),
    )

    return df


def profitability_study(
    df: pd.DataFrame,
    by: str,
    min_exposure: float = 1000,
) -> pd.DataFrame:
    """Aggregate constructed premium and actual claims by segment."""
    required_columns = {
        by,
        "policy_id",
        "exposure",
        "total_claim_amount",
        "technical_premium",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing profitability columns: {sorted(missing)}"
        )

    grouped = df.groupby(by, observed=True).agg(
        policies=("policy_id", "count"),
        exposure=("exposure", "sum"),
        actual_claims=("total_claim_amount", "sum"),
        technical_premium=("technical_premium", "sum"),
    )

    grouped["loss_ratio"] = (grouped["actual_claims"] / grouped["technical_premium"])

    grouped["underwriting_result"] = (grouped["technical_premium"] - grouped["actual_claims"])

    grouped["above_break_even"] = (grouped["loss_ratio"] > 1)

    grouped["break_even_excess"] = (grouped["loss_ratio"] - 1)

    grouped["credible"] = (grouped["exposure"] >= min_exposure)

    return (
        grouped
        .sort_values("loss_ratio", ascending=False)
        .reset_index()
    )