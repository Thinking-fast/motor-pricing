"""Train and evaluate motor insurance pricing models."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.analysis.experience_study import add_age_band
from src.config import load_config
from src.etl.clean import clean_base_table
from src.etl.extract import extract_base_table
from src.models.evaluation import (
    calibration_by_decile,
    evaluate_frequency_model,
)

from src.models.glm import (
    fit_frequency_glm,
    predict_frequency_glm,
)

from src.models.ml import (
    fit_frequency_xgboost,
    predict_frequency_xgboost,
    xgboost_feature_importance,
)

logger = logging.getLogger(__name__)


def prepare_modelling_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract, clean, enrich, and split the policy data."""
    config = load_config()

    df = extract_base_table()
    df = clean_base_table(df)
    df = add_age_band(df)

    train, test = train_test_split(
        df,
        test_size=config["model"]["test_size"],
        random_state=config["seed"],
    )

    logger.info(
        "Modelling split: train=%d rows, test=%d rows",
        len(train),
        len(test),
    )

    return train, test


def create_constant_baseline_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Predict the same training-set frequency for every test policy."""
    baseline_frequency = train["claim_nb"].sum() / train["exposure"].sum()

    predictions = test[
        [
            "policy_id",
            "exposure",
            "claim_nb",
        ]
    ].copy()

    predictions["predicted_frequency"] = baseline_frequency

    predictions["predicted_claim_count"] = (
        predictions["predicted_frequency"] * predictions["exposure"]
    )

    logger.info(
        "Constant baseline frequency: %.6f",
        baseline_frequency,
    )

    return predictions, float(baseline_frequency)


def run_frequency_models():
    """Train and compare baseline, GLM, and XGBoost models."""
    config = load_config()

    # All models use the exact same split.
    train, test = prepare_modelling_data()

    # Constant baseline
    baseline_predictions, baseline_frequency = create_constant_baseline_predictions(
        train,
        test,
    )

    baseline_metrics = evaluate_frequency_model(baseline_predictions)

    # Poisson GLM
    glm_model = fit_frequency_glm(train)

    glm_predictions = predict_frequency_glm(
        glm_model,
        test,
    )

    glm_metrics = evaluate_frequency_model(glm_predictions)

    glm_calibration = calibration_by_decile(
        glm_predictions,
        n_deciles=10,
    )

    # XGBoost
    xgb_model = fit_frequency_xgboost(
        train=train,
        model_config=config["model"]["xgboost"],
        random_state=config["seed"],
    )

    xgb_predictions = predict_frequency_xgboost(
        xgb_model,
        test,
    )

    xgb_metrics = evaluate_frequency_model(xgb_predictions)

    xgb_calibration = calibration_by_decile(
        xgb_predictions,
        n_deciles=10,
    )

    # GLM coefficients and relativities
    glm_coefficients = pd.DataFrame(
        {
            "term": glm_model.params.index,
            "coefficient": glm_model.params.values,
        }
    )

    glm_coefficients["relativity"] = np.exp(glm_coefficients["coefficient"])

    # XGBoost feature importance
    xgb_importance = xgboost_feature_importance(xgb_model)

    # Headline model comparison
    metrics_table = pd.DataFrame(
        [
            {
                "model": "constant_baseline",
                "baseline_frequency": baseline_frequency,
                **baseline_metrics,
            },
            {
                "model": "poisson_glm",
                "baseline_frequency": None,
                **glm_metrics,
            },
            {
                "model": "xgboost",
                "baseline_frequency": None,
                **xgb_metrics,
            },
        ]
    )

    logger.info(
        "Baseline: deviance=%.6f, A/E=%.4f, Gini=%.4f",
        baseline_metrics["poisson_deviance"],
        baseline_metrics["actual_to_expected"],
        baseline_metrics["normalized_gini"],
    )

    logger.info(
        "GLM: deviance=%.6f, A/E=%.4f, Gini=%.4f",
        glm_metrics["poisson_deviance"],
        glm_metrics["actual_to_expected"],
        glm_metrics["normalized_gini"],
    )

    logger.info(
        "XGBoost: deviance=%.6f, A/E=%.4f, Gini=%.4f",
        xgb_metrics["poisson_deviance"],
        xgb_metrics["actual_to_expected"],
        xgb_metrics["normalized_gini"],
    )

    output_directory = Path(config["paths"]["processed_data"])

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    glm_coefficients_path = output_directory / "frequency_glm_coefficients.csv"

    glm_calibration_path = output_directory / "frequency_glm_calibration.csv"

    xgb_calibration_path = output_directory / "frequency_xgboost_calibration.csv"

    xgb_importance_path = output_directory / "frequency_xgboost_importance.csv"

    metrics_path = output_directory / "frequency_model_metrics.csv"

    glm_coefficients.to_csv(
        glm_coefficients_path,
        index=False,
    )

    glm_calibration.to_csv(
        glm_calibration_path,
        index=False,
    )

    xgb_calibration.to_csv(
        xgb_calibration_path,
        index=False,
    )

    xgb_importance.to_csv(
        xgb_importance_path,
        index=False,
    )

    metrics_table.to_csv(
        metrics_path,
        index=False,
    )

    logger.info(
        "Saved model comparison to %s",
        metrics_path,
    )

    logger.info(
        "Saved XGBoost feature importance to %s",
        xgb_importance_path,
    )

    print("\nModel comparison:")
    print(metrics_table)

    print("\nGLM calibration:")
    print(glm_calibration)

    print("\nXGBoost calibration:")
    print(xgb_calibration)

    print("\nTop 20 XGBoost features:")
    print(xgb_importance.head(20))

    return {
        "glm_model": glm_model,
        "xgboost_model": xgb_model,
        "glm_predictions": glm_predictions,
        "xgboost_predictions": xgb_predictions,
        "metrics": metrics_table,
        "glm_calibration": glm_calibration,
        "xgboost_calibration": xgb_calibration,
        "xgboost_importance": xgb_importance,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_frequency_models()
