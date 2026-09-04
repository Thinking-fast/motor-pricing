"""Combine frequency and severity into policy-level pure premiums."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_tweedie_deviance


def build_pure_premium_predictions(
    policies: pd.DataFrame,
    frequency_predictions: pd.DataFrame,
    capped_severity: float,
    large_loss_loading: float,
) -> pd.DataFrame:
    """Build annual pure-premium rates and expected policy claim costs."""
    if capped_severity <= 0:
        raise ValueError("Capped severity must be strictly positive")
    if large_loss_loading < 0:
        raise ValueError("Large-loss loading cannot be negative")

    required_policy = {"policy_id", "exposure", "total_claim_amount"}
    required_frequency = {"policy_id", "predicted_frequency"}
    if required_policy - set(policies.columns):
        raise ValueError("Policies are missing pure-premium columns")
    if required_frequency - set(frequency_predictions.columns):
        raise ValueError("Frequency predictions are missing required columns")

    result = policies[["policy_id", "exposure", "total_claim_amount"]].merge(
        frequency_predictions[["policy_id", "predicted_frequency"]],
        on="policy_id",
        how="left",
        validate="one_to_one",
    )
    if result["predicted_frequency"].isna().any():
        raise ValueError("Every policy must have a frequency prediction")
    if (result["exposure"] <= 0).any():
        raise ValueError("Exposure must be strictly positive")
    if (result["predicted_frequency"] <= 0).any():
        raise ValueError("Predicted frequency must be strictly positive")

    result["capped_severity"] = capped_severity
    result["large_loss_loading"] = large_loss_loading
    result["predicted_capped_loss_rate"] = (
        result["predicted_frequency"] * capped_severity
    )
    result["predicted_pure_premium_rate"] = (
        result["predicted_capped_loss_rate"] + large_loss_loading
    )
    result["predicted_claim_cost"] = (
        result["predicted_pure_premium_rate"] * result["exposure"]
    )
    return result


def evaluate_pure_premium(
    predictions: pd.DataFrame,
    tweedie_power: float = 1.5,
) -> dict[str, float]:
    """Return exposure-weighted Tweedie deviance and claim-cost A/E."""
    if not 1 < tweedie_power < 2:
        raise ValueError("Tweedie power must be between 1 and 2")

    actual_rate = predictions["total_claim_amount"] / predictions["exposure"]
    predicted_rate = predictions["predicted_pure_premium_rate"]
    if (actual_rate < 0).any() or (predicted_rate <= 0).any():
        raise ValueError("Tweedie actuals must be nonnegative and predictions positive")

    deviance = mean_tweedie_deviance(
        actual_rate,
        predicted_rate,
        sample_weight=predictions["exposure"],
        power=tweedie_power,
    )
    total_expected = predictions["predicted_claim_cost"].sum()
    return {
        "tweedie_deviance": float(deviance),
        "actual_to_expected": float(
            predictions["total_claim_amount"].sum() / total_expected
        ),
    }


def pure_premium_calibration_by_decile(
    predictions: pd.DataFrame,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Aggregate actual and predicted claim cost by predicted-risk decile."""
    if n_deciles < 2:
        raise ValueError("n_deciles must be at least 2")
    if len(predictions) < n_deciles:
        raise ValueError("Number of predictions must be at least n_deciles")

    df = predictions.copy()
    risk_rank = df["predicted_pure_premium_rate"].rank(method="first")
    df["decile"] = pd.qcut(risk_rank, q=n_deciles, labels=False) + 1
    grouped = df.groupby("decile", observed=True).agg(
        policies=("policy_id", "size"),
        exposure=("exposure", "sum"),
        actual_claim_cost=("total_claim_amount", "sum"),
        predicted_claim_cost=("predicted_claim_cost", "sum"),
    )
    grouped["actual_pure_premium_rate"] = (
        grouped["actual_claim_cost"] / grouped["exposure"]
    )
    grouped["predicted_pure_premium_rate"] = (
        grouped["predicted_claim_cost"] / grouped["exposure"]
    )
    grouped["actual_to_expected"] = (
        grouped["actual_claim_cost"] / grouped["predicted_claim_cost"]
    )
    return grouped.reset_index()


def add_modelled_technical_premium(
    policies: pd.DataFrame,
    pure_premium_predictions: pd.DataFrame,
    expense_loading: float,
    profit_loading: float,
) -> pd.DataFrame:
    """Add constructed modelled technical premium to the policy table."""
    if expense_loading < 0 or profit_loading < 0:
        raise ValueError("Premium loadings cannot be negative")

    priced = policies.merge(
        pure_premium_predictions[
            ["policy_id", "predicted_pure_premium_rate", "predicted_claim_cost"]
        ],
        on="policy_id",
        how="inner",
        validate="one_to_one",
    )
    loading_factor = 1 + expense_loading + profit_loading
    priced["technical_premium_rate"] = (
        priced["predicted_pure_premium_rate"] * loading_factor
    )
    priced["technical_premium"] = priced["technical_premium_rate"] * priced["exposure"]
    return priced
