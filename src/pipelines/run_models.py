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


def run_frequency_glm():
    """Train and compare the constant baseline and Poisson GLM."""
    config = load_config()

    train, test = prepare_modelling_data()

    # Constant baseline
    baseline_predictions, baseline_frequency = create_constant_baseline_predictions(
        train,
        test,
    )

    baseline_metrics = evaluate_frequency_model(baseline_predictions)

    # Poisson GLM
    model = fit_frequency_glm(train)

    glm_predictions = predict_frequency_glm(
        model,
        test,
    )

    glm_metrics = evaluate_frequency_model(glm_predictions)

    glm_calibration = calibration_by_decile(
        glm_predictions,
        n_deciles=10,
    )

    # Display the comparison
    logger.info(
        "Baseline test Poisson deviance: %.6f",
        baseline_metrics["poisson_deviance"],
    )

    logger.info(
        "Baseline test actual/expected ratio: %.4f",
        baseline_metrics["actual_to_expected"],
    )

    logger.info(
        "Baseline normalized Gini: %.4f",
        baseline_metrics["normalized_gini"],
    )

    logger.info(
        "GLM test Poisson deviance: %.6f",
        glm_metrics["poisson_deviance"],
    )

    logger.info(
        "GLM test actual/expected ratio: %.4f",
        glm_metrics["actual_to_expected"],
    )

    logger.info(
        "GLM normalized Gini: %.4f",
        glm_metrics["normalized_gini"],
    )

    # GLM coefficient table
    coefficients = pd.DataFrame(
        {
            "term": model.params.index,
            "coefficient": model.params.values,
        }
    )

    coefficients["relativity"] = np.exp(coefficients["coefficient"])

    # Model-comparison table
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
        ]
    )

    output_directory = Path(config["paths"]["processed_data"])

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    coefficients_path = output_directory / "frequency_glm_coefficients.csv"

    metrics_path = output_directory / "frequency_model_metrics.csv"

    calibration_path = output_directory / "frequency_glm_calibration.csv"

    coefficients.to_csv(
        coefficients_path,
        index=False,
    )

    metrics_table.to_csv(
        metrics_path,
        index=False,
    )

    glm_calibration.to_csv(
        calibration_path,
        index=False,
    )

    logger.info(
        "Saved GLM coefficients to %s",
        coefficients_path,
    )

    logger.info(
        "Saved model comparison to %s",
        metrics_path,
    )

    logger.info(
        "Saved GLM calibration table to %s",
        calibration_path,
    )

    print(metrics_table)
    print(model.summary())
    print("\nGLM calibration by predicted-risk decile:")
    print(glm_calibration)

    return (
        model,
        glm_predictions,
        metrics_table,
        glm_calibration,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_frequency_glm()
