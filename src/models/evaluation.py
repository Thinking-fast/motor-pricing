"""Evaluation metrics for actuarial frequency models."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_poisson_deviance


def poisson_deviance(
    actual_claims: pd.Series,
    predicted_frequency: pd.Series,
    exposure: pd.Series,
) -> float:
    """Return exposure-weighted mean Poisson deviance."""
    if (exposure <= 0).any():
        raise ValueError("Exposure must be strictly positive")

    if (actual_claims < 0).any():
        raise ValueError("Actual claim counts cannot be negative")

    if (predicted_frequency <= 0).any():
        raise ValueError("Predicted frequency must be strictly positive")

    actual_frequency = actual_claims / exposure

    return float(
        mean_poisson_deviance(
            actual_frequency,
            predicted_frequency,
            sample_weight=exposure,
        )
    )


def actual_to_expected(
    actual_claims: pd.Series,
    predicted_claim_counts: pd.Series,
) -> float:
    """Return total actual claims divided by total expected claims."""
    total_actual = actual_claims.sum()
    total_expected = predicted_claim_counts.sum()

    if total_expected <= 0:
        raise ValueError("Total expected claims must be positive")

    return float(total_actual / total_expected)


def evaluate_frequency_model(
    predictions: pd.DataFrame,
) -> dict[str, float]:
    """Return headline frequency-model performance metrics."""
    deviance = poisson_deviance(
        actual_claims=predictions["claim_nb"],
        predicted_frequency=predictions["predicted_frequency"],
        exposure=predictions["exposure"],
    )

    ae_ratio = actual_to_expected(
        actual_claims=predictions["claim_nb"],
        predicted_claim_counts=predictions["predicted_claim_count"],
    )

    return {
        "poisson_deviance": deviance,
        "actual_to_expected": ae_ratio,
    }


def calibration_by_decile(
    predictions: pd.DataFrame,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Summarize actual and predicted frequency by predicted-risk decile."""
    required_columns = {
        "exposure",
        "claim_nb",
        "predicted_claim_count",
        "predicted_frequency",
    }

    missing = required_columns - set(predictions.columns)

    if missing:
        raise ValueError(f"Missing calibration columns: {sorted(missing)}")

    if n_deciles < 2:
        raise ValueError("n_deciles must be at least 2")

    if len(predictions) < n_deciles:
        raise ValueError("Number of predictions must be at least n_deciles")

    if (predictions["exposure"] <= 0).any():
        raise ValueError("Exposure must be strictly positive")

    df = predictions.copy()

    # rank(method="first") gives tied predictions a stable unique order,
    # allowing qcut to create the requested number of groups.
    prediction_rank = df["predicted_frequency"].rank(method="first")

    df["decile"] = (
        pd.qcut(
            prediction_rank,
            q=n_deciles,
            labels=False,
        )
        + 1
    )

    calibration = df.groupby(
        "decile",
        observed=True,
    ).agg(
        policies=("claim_nb", "size"),
        exposure=("exposure", "sum"),
        actual_claims=("claim_nb", "sum"),
        predicted_claims=(
            "predicted_claim_count",
            "sum",
        ),
    )

    calibration["actual_frequency"] = (
        calibration["actual_claims"] / calibration["exposure"]
    )

    calibration["predicted_frequency"] = (
        calibration["predicted_claims"] / calibration["exposure"]
    )

    calibration["actual_to_expected"] = (
        calibration["actual_claims"] / calibration["predicted_claims"]
    )

    return calibration.reset_index()
