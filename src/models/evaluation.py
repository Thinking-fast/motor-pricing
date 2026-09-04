"""Evaluation metrics for actuarial frequency models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance


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

    gini = normalized_gini(
        actual_claims=predictions["claim_nb"],
        predicted_frequency=predictions["predicted_frequency"],
        exposure=predictions["exposure"],
    )

    return {
        "poisson_deviance": deviance,
        "actual_to_expected": ae_ratio,
        "normalized_gini": gini,
    }


def evaluate_severity_model(predictions: pd.DataFrame) -> dict[str, float]:
    """Return claim-count-weighted Gamma deviance and amount A/E."""
    required = {
        "average_claim_amount",
        "predicted_severity",
        "n_claim_rows",
        "total_claim_amount",
        "predicted_claim_amount",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing severity evaluation columns: {sorted(missing)}")
    if (predictions["average_claim_amount"] <= 0).any():
        raise ValueError("Actual severity must be strictly positive")
    if (predictions["predicted_severity"] <= 0).any():
        raise ValueError("Predicted severity must be strictly positive")
    if (predictions["n_claim_rows"] <= 0).any():
        raise ValueError("Claim weights must be strictly positive")

    deviance = mean_gamma_deviance(
        predictions["average_claim_amount"],
        predictions["predicted_severity"],
        sample_weight=predictions["n_claim_rows"],
    )
    predicted_total = predictions["predicted_claim_amount"].sum()
    if predicted_total <= 0:
        raise ValueError("Total predicted claim amount must be positive")

    return {
        "gamma_deviance": float(deviance),
        "actual_to_expected": float(
            predictions["total_claim_amount"].sum() / predicted_total
        ),
    }


def severity_calibration_by_decile(
    predictions: pd.DataFrame,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Summarize actual and predicted severity by predicted-risk decile."""
    if n_deciles < 2:
        raise ValueError("n_deciles must be at least 2")
    if len(predictions) < n_deciles:
        raise ValueError("Number of predictions must be at least n_deciles")

    df = predictions.copy()
    prediction_rank = df["predicted_severity"].rank(method="first")
    df["decile"] = pd.qcut(prediction_rank, q=n_deciles, labels=False) + 1

    calibration = df.groupby("decile", observed=True).agg(
        policies=("policy_id", "size"),
        claims=("n_claim_rows", "sum"),
        actual_amount=("total_claim_amount", "sum"),
        predicted_amount=("predicted_claim_amount", "sum"),
    )
    calibration["actual_severity"] = (
        calibration["actual_amount"] / calibration["claims"]
    )
    calibration["predicted_severity"] = (
        calibration["predicted_amount"] / calibration["claims"]
    )
    calibration["actual_to_expected"] = (
        calibration["actual_amount"] / calibration["predicted_amount"]
    )
    return calibration.reset_index()


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


def gini_coefficient(
    actual_claims: pd.Series,
    predicted_frequency: pd.Series,
    exposure: pd.Series,
) -> float:
    """Return the exposure-weighted Gini coefficient."""
    if len(actual_claims) != len(predicted_frequency):
        raise ValueError("Actual claims and predictions must have equal length")

    if len(actual_claims) != len(exposure):
        raise ValueError("Actual claims and exposure must have equal length")

    if (exposure <= 0).any():
        raise ValueError("Exposure must be strictly positive")

    if (actual_claims < 0).any():
        raise ValueError("Actual claims cannot be negative")

    if (predicted_frequency <= 0).any():
        raise ValueError("Predicted frequency must be strictly positive")

    if actual_claims.sum() <= 0:
        raise ValueError("At least one actual claim is required")

    data = pd.DataFrame(
        {
            "actual_claims": actual_claims.to_numpy(),
            "predicted_frequency": (predicted_frequency.to_numpy()),
            "exposure": exposure.to_numpy(),
        }
    )

    # Combine policies with exactly equal predictions. This ensures that
    # a constant prediction receives a Gini coefficient of zero.
    ordered = (
        data.groupby(
            "predicted_frequency",
            as_index=False,
            sort=True,
        )
        .agg(
            actual_claims=("actual_claims", "sum"),
            exposure=("exposure", "sum"),
        )
        .sort_values("predicted_frequency")
    )

    cumulative_exposure = ordered["exposure"].cumsum() / ordered["exposure"].sum()

    cumulative_claims = (
        ordered["actual_claims"].cumsum() / ordered["actual_claims"].sum()
    )

    # Include the origin of the Lorenz curve.
    cumulative_exposure = np.concatenate(([0.0], cumulative_exposure.to_numpy()))

    cumulative_claims = np.concatenate(([0.0], cumulative_claims.to_numpy()))

    area_under_curve = np.trapezoid(
        cumulative_claims,
        cumulative_exposure,
    )

    return float(1 - 2 * area_under_curve)


def normalized_gini(
    actual_claims: pd.Series,
    predicted_frequency: pd.Series,
    exposure: pd.Series,
) -> float:
    """Return model Gini divided by the best possible Gini."""
    model_gini = gini_coefficient(
        actual_claims=actual_claims,
        predicted_frequency=predicted_frequency,
        exposure=exposure,
    )

    perfect_ranking = (actual_claims / exposure).clip(lower=1e-12)

    perfect_gini = gini_coefficient(
        actual_claims=actual_claims,
        predicted_frequency=perfect_ranking,
        exposure=exposure,
    )

    if np.isclose(perfect_gini, 0):
        raise ValueError("Normalized Gini is undefined when perfect Gini is zero")

    return float(model_gini / perfect_gini)
