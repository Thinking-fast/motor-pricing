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
