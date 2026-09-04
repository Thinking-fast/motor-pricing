"""Machine-learning frequency models for motor insurance pricing."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


CATEGORICAL_FEATURES = [
    "area",
    "age_band",
    "veh_brand",
    "veh_gas",
    "region",
]

NUMERIC_FEATURES = [
    "veh_power",
    "veh_age",
    "bonus_malus",
    "density",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def fit_frequency_xgboost(
    train: pd.DataFrame,
    model_config: dict,
    random_state: int,
) -> Pipeline:
    """Fit a Poisson XGBoost model to annual claim frequency."""
    if (train["exposure"] <= 0).any():
        raise ValueError("XGBoost requires strictly positive exposure")

    if (train["claim_nb"] < 0).any():
        raise ValueError("Claim counts cannot be negative")

    missing = set(FEATURES) - set(train.columns)

    if missing:
        raise ValueError(f"Missing XGBoost features: {sorted(missing)}")

    target_frequency = train["claim_nb"] / train["exposure"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )

    estimator = XGBRegressor(
        **model_config,
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    model.fit(
        train[FEATURES],
        target_frequency,
        model__sample_weight=train["exposure"],
    )

    logger.info(
        "Fitted frequency XGBoost model on %d policies",
        len(train),
    )

    return model


def predict_frequency_xgboost(
    model,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return XGBoost frequency and claim-count predictions."""
    if (df["exposure"] <= 0).any():
        raise ValueError("XGBoost prediction requires strictly positive exposure")

    missing = set(FEATURES) - set(df.columns)

    if missing:
        raise ValueError(f"Missing XGBoost features: {sorted(missing)}")

    predictions = df[
        [
            "policy_id",
            "exposure",
            "claim_nb",
        ]
    ].copy()

    predicted_frequency = model.predict(df[FEATURES])

    # Poisson deviance requires strictly positive predictions.
    predictions["predicted_frequency"] = np.maximum(
        predicted_frequency,
        1e-12,
    )

    predictions["predicted_claim_count"] = (
        predictions["predicted_frequency"] * predictions["exposure"]
    )

    logger.info(
        "Generated XGBoost predictions for %d policies",
        len(predictions),
    )

    return predictions


def xgboost_feature_importance(
    model: Pipeline,
) -> pd.DataFrame:
    """Return encoded XGBoost feature importances."""
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": estimator.feature_importances_,
        }
    )

    return importance.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)
