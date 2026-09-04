"""Generalized linear models for motor insurance pricing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)

FREQUENCY_FORMULA = (
    "claim_nb ~ "
    "C(area) + "
    "C(age_band) + "
    "C(veh_brand) + "
    "C(veh_gas) + "
    "C(region) + "
    "veh_power + "
    "veh_age + "
    "bonus_malus + "
    "np.log1p(density)"
)

SEVERITY_FORMULA = FREQUENCY_FORMULA.replace(
    "claim_nb ~ ",
    "average_claim_amount ~ ",
)


def fit_frequency_glm(train: pd.DataFrame):
    """Fit a Poisson frequency GLM with a log-exposure offset."""
    if (train["exposure"] <= 0).any():
        raise ValueError("Frequency GLM requires strictly positive exposure")

    offset = np.log(train["exposure"])

    model = smf.glm(
        formula=FREQUENCY_FORMULA,
        data=train,
        family=sm.families.Poisson(),
        offset=offset,
    )

    result = model.fit()

    logger.info(
        "Fitted frequency GLM on %d policies",
        len(train),
    )

    return result


def predict_frequency_glm(
    model,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return expected claim counts and annual frequencies."""
    if (df["exposure"] <= 0).any():
        raise ValueError("Frequency prediction requires strictly positive exposure")

    predictions = df[
        [
            "policy_id",
            "exposure",
            "claim_nb",
        ]
    ].copy()

    predictions["predicted_claim_count"] = model.predict(
        df,
        offset=np.log(df["exposure"]),
    )

    predictions["predicted_frequency"] = (
        predictions["predicted_claim_count"] / predictions["exposure"]
    )

    logger.info(
        "Generated frequency predictions for %d policies",
        len(predictions),
    )

    return predictions


def prepare_severity_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep positive amount-bearing claims and calculate average severity."""
    required = {"n_claim_rows", "total_claim_amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing severity columns: {sorted(missing)}")

    severity = df.loc[(df["n_claim_rows"] > 0) & (df["total_claim_amount"] > 0)].copy()
    severity["average_claim_amount"] = (
        severity["total_claim_amount"] / severity["n_claim_rows"]
    )
    return severity


def fit_severity_glm(train: pd.DataFrame):
    """Fit a claim-count-weighted Gamma GLM with a log link."""
    severity_train = prepare_severity_data(train)
    if severity_train.empty:
        raise ValueError("Severity GLM requires at least one positive claim amount")

    model = smf.glm(
        formula=SEVERITY_FORMULA,
        data=severity_train,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        freq_weights=severity_train["n_claim_rows"],
    )
    result = model.fit()
    logger.info(
        "Fitted severity GLM on %d policies representing %.0f claims",
        len(severity_train),
        severity_train["n_claim_rows"].sum(),
    )
    return result


def predict_severity_glm(model, df: pd.DataFrame) -> pd.DataFrame:
    """Predict average and total claim amount for amount-bearing policies."""
    severity = prepare_severity_data(df)
    predictions = severity[
        ["policy_id", "n_claim_rows", "total_claim_amount", "average_claim_amount"]
    ].copy()
    predictions["predicted_severity"] = model.predict(severity)
    predictions["predicted_claim_amount"] = (
        predictions["predicted_severity"] * predictions["n_claim_rows"]
    )
    logger.info("Generated severity predictions for %d policies", len(predictions))
    return predictions
