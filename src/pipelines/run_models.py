"""Train and evaluate motor insurance pricing models."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.analysis.experience_study import (
    add_age_band,
    cap_large_claims,
    large_loss_loading,
)
from src.config import load_config
from src.etl.clean import clean_base_table
from src.etl.extract import extract_base_table
from src.models.evaluation import (
    calibration_by_decile,
    evaluate_frequency_model,
    evaluate_severity_model,
    severity_calibration_by_decile,
)

from src.models.glm import (
    fit_frequency_glm,
    fit_severity_glm,
    predict_frequency_glm,
    predict_severity_glm,
)

from src.models.ml import (
    fit_frequency_xgboost,
    predict_frequency_xgboost,
    xgboost_feature_importance,
)
from src.models.pure_premium import (
    add_modelled_technical_premium,
    build_pure_premium_predictions,
    evaluate_pure_premium,
    pure_premium_calibration_by_decile,
)
from src.pricing.technical_premium import profitability_study

from src.models.cross_validation import (
    cross_validate_frequency_models,
    summarize_cross_validation,
)

logger = logging.getLogger(__name__)

PROFITABILITY_FACTORS = ["age_band", "region", "veh_brand", "veh_gas", "area"]


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


def create_constant_severity_predictions(
    train: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Predict the training-set average reported severity for every claim."""
    eligible_train = train.loc[
        (train["n_claim_rows"] > 0) & (train["total_claim_amount"] > 0)
    ]
    baseline_severity = (
        eligible_train["total_claim_amount"].sum()
        / eligible_train["n_claim_rows"].sum()
    )

    predictions = test_predictions[
        ["policy_id", "n_claim_rows", "total_claim_amount", "average_claim_amount"]
    ].copy()
    predictions["predicted_severity"] = baseline_severity
    predictions["predicted_claim_amount"] = (
        baseline_severity * predictions["n_claim_rows"]
    )
    return predictions, float(baseline_severity)


def run_frequency_models():
    """Train and compare baseline, GLM, and XGBoost models."""
    config = load_config()

    # All models use the exact same split.
    train, test = prepare_modelling_data()

    cv_results = cross_validate_frequency_models(
        train=train,
        model_config=config["model"]["xgboost"],
        n_splits=config["model"]["cv_folds"],
        random_state=config["seed"],
    )

    cv_summary = summarize_cross_validation(cv_results)

    logger.info(
        "Completed %d-fold frequency-model cross-validation",
        config["model"]["cv_folds"],
    )

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

    # Gamma severity GLM
    severity_model = fit_severity_glm(train)
    severity_predictions = predict_severity_glm(severity_model, test)
    severity_metrics = evaluate_severity_model(severity_predictions)
    severity_calibration = severity_calibration_by_decile(severity_predictions)
    severity_baseline_predictions, baseline_severity = (
        create_constant_severity_predictions(train, severity_predictions)
    )
    severity_baseline_metrics = evaluate_severity_model(severity_baseline_predictions)

    severity_metrics_table = pd.DataFrame(
        [
            {
                "model": "constant_baseline",
                "loss_basis": "gross",
                "baseline_severity": baseline_severity,
                **severity_baseline_metrics,
            },
            {
                "model": "gamma_glm",
                "loss_basis": "gross",
                "baseline_severity": None,
                **severity_metrics,
            },
        ]
    )

    # Capped Gamma sensitivity. The configured cap applies to each policy's
    # aggregate claim amount because individual claim rows are not in this table.
    large_loss_cap = config["experience_study"]["large_loss_cap"]
    capped_train, train_excess = cap_large_claims(train, large_loss_cap)
    capped_test, test_excess = cap_large_claims(test, large_loss_cap)
    large_loss_rate = large_loss_loading(train_excess, train["exposure"].sum())

    capped_severity_model = fit_severity_glm(capped_train)
    capped_severity_predictions = predict_severity_glm(
        capped_severity_model,
        capped_test,
    )
    capped_severity_metrics = evaluate_severity_model(capped_severity_predictions)
    capped_severity_calibration = severity_calibration_by_decile(
        capped_severity_predictions
    )
    capped_baseline_predictions, capped_baseline_severity = (
        create_constant_severity_predictions(
            capped_train,
            capped_severity_predictions,
        )
    )
    capped_baseline_metrics = evaluate_severity_model(capped_baseline_predictions)

    # The frequency model uses claim_nb, so the pricing severity must use the
    # same denominator. This makes frequency x severity reproduce capped cost.
    pricing_capped_severity = (
        capped_train["total_claim_amount"].sum() / capped_train["claim_nb"].sum()
    )

    capped_severity_coefficients = pd.DataFrame(
        {
            "term": capped_severity_model.params.index,
            "coefficient": capped_severity_model.params.values,
        }
    )
    capped_severity_coefficients["relativity"] = np.exp(
        capped_severity_coefficients["coefficient"]
    )

    severity_metrics_table = pd.concat(
        [
            severity_metrics_table,
            pd.DataFrame(
                [
                    {
                        "model": "constant_baseline",
                        "loss_basis": "capped",
                        "baseline_severity": capped_baseline_severity,
                        **capped_baseline_metrics,
                    },
                    {
                        "model": "gamma_glm",
                        "loss_basis": "capped",
                        "baseline_severity": None,
                        **capped_severity_metrics,
                    },
                ]
            ),
        ],
        ignore_index=True,
    )

    # Combine each frequency candidate with the same selected severity basis.
    pure_premium_candidates = {
        "constant_baseline": baseline_predictions,
        "poisson_glm": glm_predictions,
        "xgboost": xgb_predictions,
    }
    pure_premium_predictions = {}
    pure_premium_calibrations = {}
    pure_premium_metric_rows = []

    for model_name, frequency_predictions in pure_premium_candidates.items():
        predictions = build_pure_premium_predictions(
            policies=test,
            frequency_predictions=frequency_predictions,
            capped_severity=pricing_capped_severity,
            large_loss_loading=large_loss_rate,
        )
        metrics = evaluate_pure_premium(
            predictions,
            tweedie_power=config["model"]["tweedie_power"],
        )
        calibration = pure_premium_calibration_by_decile(predictions)

        pure_premium_predictions[model_name] = predictions
        pure_premium_calibrations[model_name] = calibration
        pure_premium_metric_rows.append({"model": model_name, **metrics})

    pure_premium_metrics = pd.DataFrame(pure_premium_metric_rows).sort_values(
        "tweedie_deviance"
    )

    # XGBoost frequency is the selected model. Premiums are constructed because
    # the source data contains no premium actually charged to policyholders.
    selected_pure_premium = pure_premium_predictions["xgboost"]
    modelled_priced_test = add_modelled_technical_premium(
        policies=test,
        pure_premium_predictions=selected_pure_premium,
        expense_loading=config["pricing"]["expense_loading"],
        profit_loading=config["pricing"]["profit_loading"],
    )
    modelled_profitability = {
        factor: profitability_study(modelled_priced_test, by=factor)
        for factor in PROFITABILITY_FACTORS
    }

    severity_coefficients = pd.DataFrame(
        {
            "term": severity_model.params.index,
            "coefficient": severity_model.params.values,
        }
    )
    severity_coefficients["relativity"] = np.exp(severity_coefficients["coefficient"])

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

    cv_results_path = output_directory / "frequency_cross_validation_folds.csv"

    cv_summary_path = output_directory / "frequency_cross_validation_summary.csv"

    glm_coefficients_path = output_directory / "frequency_glm_coefficients.csv"

    glm_calibration_path = output_directory / "frequency_glm_calibration.csv"

    xgb_calibration_path = output_directory / "frequency_xgboost_calibration.csv"

    xgb_importance_path = output_directory / "frequency_xgboost_importance.csv"

    metrics_path = output_directory / "frequency_model_metrics.csv"
    severity_metrics_path = output_directory / "severity_model_metrics.csv"
    severity_coefficients_path = output_directory / "severity_glm_coefficients.csv"
    severity_calibration_path = output_directory / "severity_glm_calibration.csv"
    capped_severity_coefficients_path = (
        output_directory / "severity_glm_coefficients_capped.csv"
    )
    capped_severity_calibration_path = (
        output_directory / "severity_glm_calibration_capped.csv"
    )
    large_loss_summary_path = output_directory / "severity_large_loss_summary.csv"
    pure_premium_metrics_path = output_directory / "pure_premium_model_metrics.csv"
    selected_predictions_path = (
        output_directory / "pure_premium_xgboost_predictions.csv"
    )

    cv_results.to_csv(
        cv_results_path,
        index=False,
    )

    cv_summary.to_csv(
        cv_summary_path,
        index=False,
    )

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

    severity_metrics_table.to_csv(
        severity_metrics_path,
        index=False,
    )
    severity_coefficients.to_csv(severity_coefficients_path, index=False)
    severity_calibration.to_csv(severity_calibration_path, index=False)
    capped_severity_coefficients.to_csv(
        capped_severity_coefficients_path,
        index=False,
    )
    capped_severity_calibration.to_csv(
        capped_severity_calibration_path,
        index=False,
    )
    pd.DataFrame(
        [
            {
                "policy_total_cap": large_loss_cap,
                "training_excess": train_excess,
                "training_exposure": train["exposure"].sum(),
                "large_loss_loading_per_exposure": large_loss_rate,
                "pricing_capped_severity_per_incident": pricing_capped_severity,
                "test_excess": test_excess,
            }
        ]
    ).to_csv(large_loss_summary_path, index=False)

    pure_premium_metrics.to_csv(pure_premium_metrics_path, index=False)
    selected_pure_premium.to_csv(selected_predictions_path, index=False)

    for model_name, calibration in pure_premium_calibrations.items():
        calibration.to_csv(
            output_directory / f"pure_premium_{model_name}_calibration.csv",
            index=False,
        )

    for factor, table in modelled_profitability.items():
        table.to_csv(
            output_directory / f"modelled_profitability_{factor}.csv",
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

    logger.info(
        "Saved cross-validation fold results to %s",
        cv_results_path,
    )

    logger.info(
        "Saved cross-validation summary to %s",
        cv_summary_path,
    )

    logger.info(
        "Gamma severity GLM: deviance=%.6f, amount A/E=%.4f",
        severity_metrics["gamma_deviance"],
        severity_metrics["actual_to_expected"],
    )

    logger.info(
        "Capped Gamma GLM: deviance=%.6f, amount A/E=%.4f; "
        "training large-loss loading=%.2f per exposure",
        capped_severity_metrics["gamma_deviance"],
        capped_severity_metrics["actual_to_expected"],
        large_loss_rate,
    )

    selected_metrics = pure_premium_metrics.loc[
        pure_premium_metrics["model"] == "xgboost"
    ].iloc[0]
    logger.info(
        "Selected pure premium: Tweedie deviance=%.6f, A/E=%.4f",
        selected_metrics["tweedie_deviance"],
        selected_metrics["actual_to_expected"],
    )

    print("\nModel comparison:")
    print(metrics_table)

    print("\nGLM calibration:")
    print(glm_calibration)

    print("\nXGBoost calibration:")
    print(xgb_calibration)

    print("\nTop 20 XGBoost features:")
    print(xgb_importance.head(20))

    print("\nCross-validation summary:")
    print(cv_summary)

    print("\nGamma severity model:")
    print(severity_metrics_table)

    print("\nGamma severity calibration:")
    print(severity_calibration)

    print("\nCapped Gamma severity calibration:")
    print(capped_severity_calibration)

    print("\nPure-premium model comparison:")
    print(pure_premium_metrics)

    print("\nSelected XGBoost pure-premium calibration:")
    print(pure_premium_calibrations["xgboost"])

    return {
        "glm_model": glm_model,
        "xgboost_model": xgb_model,
        "glm_predictions": glm_predictions,
        "xgboost_predictions": xgb_predictions,
        "metrics": metrics_table,
        "glm_calibration": glm_calibration,
        "xgboost_calibration": xgb_calibration,
        "xgboost_importance": xgb_importance,
        "cross_validation_folds": cv_results,
        "cross_validation_summary": cv_summary,
        "severity_model": severity_model,
        "severity_predictions": severity_predictions,
        "severity_metrics": severity_metrics,
        "severity_metrics_table": severity_metrics_table,
        "severity_calibration": severity_calibration,
        "capped_severity_model": capped_severity_model,
        "capped_severity_predictions": capped_severity_predictions,
        "capped_severity_metrics": capped_severity_metrics,
        "capped_severity_calibration": capped_severity_calibration,
        "large_loss_loading": large_loss_rate,
        "pure_premium_metrics": pure_premium_metrics,
        "pure_premium_predictions": pure_premium_predictions,
        "pure_premium_calibrations": pure_premium_calibrations,
        "modelled_profitability": modelled_profitability,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_frequency_models()
