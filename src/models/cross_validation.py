"""Cross-validation for actuarial frequency models."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.models.evaluation import evaluate_frequency_model
from src.models.glm import (
    fit_frequency_glm,
    predict_frequency_glm,
)
from src.models.ml import (
    fit_frequency_xgboost,
    predict_frequency_xgboost,
)

logger = logging.getLogger(__name__)


def _constant_baseline_predictions(
    fold_train: pd.DataFrame,
    fold_validation: pd.DataFrame,
) -> pd.DataFrame:
    """Predict the fold-training average frequency."""
    baseline_frequency = fold_train["claim_nb"].sum() / fold_train["exposure"].sum()

    predictions = fold_validation[
        [
            "policy_id",
            "exposure",
            "claim_nb",
        ]
    ].copy()

    predictions["predicted_frequency"] = baseline_frequency

    predictions["predicted_claim_count"] = baseline_frequency * predictions["exposure"]

    return predictions


def cross_validate_frequency_models(
    train: pd.DataFrame,
    model_config: dict,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    """Cross-validate baseline, GLM, and XGBoost frequency models."""
    if n_splits < 2:
        raise ValueError("Cross-validation requires at least 2 folds")

    if len(train) < n_splits:
        raise ValueError("Training rows must be at least the number of folds")

    if (train["exposure"] <= 0).any():
        raise ValueError("Cross-validation requires positive exposure")

    # Claim occurrence is used only to distribute claim and no-claim
    # policies approximately evenly across the folds.
    has_claim = (train["claim_nb"] > 0).astype(int)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    rows = []

    for fold, (fit_indices, validation_indices) in enumerate(
        splitter.split(train, has_claim),
        start=1,
    ):
        fold_train = train.iloc[fit_indices].copy()
        fold_validation = train.iloc[validation_indices].copy()

        logger.info(
            "Cross-validation fold %d/%d: train=%d, validation=%d",
            fold,
            n_splits,
            len(fold_train),
            len(fold_validation),
        )

        # Constant baseline
        baseline_predictions = _constant_baseline_predictions(
            fold_train,
            fold_validation,
        )

        baseline_metrics = evaluate_frequency_model(baseline_predictions)

        rows.append(
            {
                "fold": fold,
                "model": "constant_baseline",
                **baseline_metrics,
            }
        )

        # Poisson GLM
        glm_model = fit_frequency_glm(fold_train)

        glm_predictions = predict_frequency_glm(
            glm_model,
            fold_validation,
        )

        glm_metrics = evaluate_frequency_model(glm_predictions)

        rows.append(
            {
                "fold": fold,
                "model": "poisson_glm",
                **glm_metrics,
            }
        )

        # XGBoost
        xgb_model = fit_frequency_xgboost(
            train=fold_train,
            model_config=model_config,
            random_state=random_state,
        )

        xgb_predictions = predict_frequency_xgboost(
            xgb_model,
            fold_validation,
        )

        xgb_metrics = evaluate_frequency_model(xgb_predictions)

        rows.append(
            {
                "fold": fold,
                "model": "xgboost",
                **xgb_metrics,
            }
        )

    return pd.DataFrame(rows)


def summarize_cross_validation(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Return mean and standard deviation across folds."""
    required_columns = {
        "model",
        "poisson_deviance",
        "actual_to_expected",
        "normalized_gini",
    }

    missing = required_columns - set(results.columns)

    if missing:
        raise ValueError(f"Missing cross-validation columns: {sorted(missing)}")

    summary = results.groupby(
        "model",
        as_index=False,
    ).agg(
        poisson_deviance_mean=(
            "poisson_deviance",
            "mean",
        ),
        poisson_deviance_std=(
            "poisson_deviance",
            "std",
        ),
        actual_to_expected_mean=(
            "actual_to_expected",
            "mean",
        ),
        actual_to_expected_std=(
            "actual_to_expected",
            "std",
        ),
        normalized_gini_mean=(
            "normalized_gini",
            "mean",
        ),
        normalized_gini_std=(
            "normalized_gini",
            "std",
        ),
    )

    return summary.sort_values("poisson_deviance_mean").reset_index(drop=True)
