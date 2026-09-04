import pandas as pd
import pytest

from src.models.cross_validation import (
    cross_validate_frequency_models,
    summarize_cross_validation,
)


def test_cross_validation_requires_at_least_two_folds():
    df = pd.DataFrame(
        {
            "exposure": [1.0],
            "claim_nb": [0],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least 2 folds",
    ):
        cross_validate_frequency_models(
            train=df,
            model_config={},
            n_splits=1,
            random_state=42,
        )


def test_summarize_cross_validation_calculates_means():
    results = pd.DataFrame(
        {
            "fold": [1, 2, 1, 2],
            "model": [
                "poisson_glm",
                "poisson_glm",
                "xgboost",
                "xgboost",
            ],
            "poisson_deviance": [
                0.60,
                0.62,
                0.55,
                0.57,
            ],
            "actual_to_expected": [
                0.98,
                1.02,
                0.99,
                1.01,
            ],
            "normalized_gini": [
                0.24,
                0.26,
                0.32,
                0.34,
            ],
        }
    )

    summary = summarize_cross_validation(results)

    glm = summary.loc[summary["model"] == "poisson_glm"].iloc[0]

    xgboost = summary.loc[summary["model"] == "xgboost"].iloc[0]

    assert glm["poisson_deviance_mean"] == (pytest.approx(0.61))

    assert glm["normalized_gini_mean"] == (pytest.approx(0.25))

    assert xgboost["poisson_deviance_mean"] == (pytest.approx(0.56))

    assert xgboost["normalized_gini_mean"] == (pytest.approx(0.33))


def test_summary_orders_lower_deviance_first():
    results = pd.DataFrame(
        {
            "model": [
                "poisson_glm",
                "poisson_glm",
                "xgboost",
                "xgboost",
            ],
            "poisson_deviance": [
                0.60,
                0.62,
                0.55,
                0.57,
            ],
            "actual_to_expected": [
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            "normalized_gini": [
                0.25,
                0.25,
                0.33,
                0.33,
            ],
        }
    )

    summary = summarize_cross_validation(results)

    assert summary["model"].tolist() == [
        "xgboost",
        "poisson_glm",
    ]
