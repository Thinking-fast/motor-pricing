import pandas as pd
import pytest

from src.models.evaluation import (
    actual_to_expected,
    calibration_by_decile,
    evaluate_frequency_model,
    evaluate_severity_model,
    gini_coefficient,
    normalized_gini,
    poisson_deviance,
    severity_calibration_by_decile,
)


def test_actual_to_expected_uses_total_claims():
    actual = pd.Series([1.0, 2.0])
    expected = pd.Series([1.0, 1.5])

    result = actual_to_expected(
        actual_claims=actual,
        predicted_claim_counts=expected,
    )

    assert result == pytest.approx(3.0 / 2.5)


def test_poisson_deviance_is_zero_for_perfect_predictions():
    actual_claims = pd.Series([1.0, 2.0, 3.0])
    exposure = pd.Series([1.0, 1.0, 1.0])
    predicted_frequency = pd.Series([1.0, 2.0, 3.0])

    result = poisson_deviance(
        actual_claims=actual_claims,
        predicted_frequency=predicted_frequency,
        exposure=exposure,
    )

    assert result == pytest.approx(0.0)


def test_poisson_deviance_rejects_zero_exposure():
    actual_claims = pd.Series([0.0, 1.0])
    exposure = pd.Series([1.0, 0.0])
    predicted_frequency = pd.Series([0.1, 0.2])

    with pytest.raises(
        ValueError,
        match="Exposure must be strictly positive",
    ):
        poisson_deviance(
            actual_claims=actual_claims,
            predicted_frequency=predicted_frequency,
            exposure=exposure,
        )


def test_poisson_deviance_rejects_nonpositive_predictions():
    actual_claims = pd.Series([0.0, 1.0])
    exposure = pd.Series([1.0, 1.0])
    predicted_frequency = pd.Series([0.1, 0.0])

    with pytest.raises(
        ValueError,
        match="Predicted frequency must be strictly positive",
    ):
        poisson_deviance(
            actual_claims=actual_claims,
            predicted_frequency=predicted_frequency,
            exposure=exposure,
        )


def test_evaluate_frequency_model_returns_both_metrics():
    predictions = pd.DataFrame(
        {
            "claim_nb": [1.0, 2.0],
            "exposure": [1.0, 1.0],
            "predicted_claim_count": [1.0, 2.0],
            "predicted_frequency": [1.0, 2.0],
        }
    )

    result = evaluate_frequency_model(predictions)

    assert result["poisson_deviance"] == pytest.approx(0.0)
    assert result["actual_to_expected"] == pytest.approx(1.0)


def test_calibration_by_decile_aggregates_predictions():
    predictions = pd.DataFrame(
        {
            "claim_nb": [0, 1, 1, 2],
            "exposure": [1.0, 1.0, 2.0, 2.0],
            "predicted_frequency": [
                0.1,
                0.2,
                0.3,
                0.4,
            ],
            "predicted_claim_count": [
                0.1,
                0.2,
                0.6,
                0.8,
            ],
        }
    )

    result = calibration_by_decile(
        predictions,
        n_deciles=2,
    )

    assert result["decile"].tolist() == [1, 2]
    assert result["policies"].tolist() == [2, 2]
    assert result["exposure"].tolist() == pytest.approx([2.0, 4.0])
    assert result["actual_claims"].tolist() == [1, 3]
    assert result["predicted_claims"].tolist() == pytest.approx([0.3, 1.4])

    assert result["actual_frequency"].tolist() == pytest.approx([0.5, 0.75])

    assert result["predicted_frequency"].tolist() == pytest.approx([0.15, 0.35])

    assert result["actual_to_expected"].tolist() == pytest.approx(
        [
            1.0 / 0.3,
            3.0 / 1.4,
        ]
    )


def test_calibration_by_decile_requires_enough_rows():
    predictions = pd.DataFrame(
        {
            "claim_nb": [0, 1],
            "exposure": [1.0, 1.0],
            "predicted_frequency": [0.1, 0.2],
            "predicted_claim_count": [0.1, 0.2],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least n_deciles",
    ):
        calibration_by_decile(
            predictions,
            n_deciles=10,
        )


def test_constant_prediction_has_zero_gini():
    actual_claims = pd.Series([0.0, 1.0, 2.0])
    predicted_frequency = pd.Series([0.1, 0.1, 0.1])
    exposure = pd.Series([1.0, 1.0, 1.0])

    result = gini_coefficient(
        actual_claims=actual_claims,
        predicted_frequency=predicted_frequency,
        exposure=exposure,
    )

    assert result == pytest.approx(0.0)


def test_perfect_ranking_has_normalized_gini_of_one():
    actual_claims = pd.Series([0.0, 1.0, 2.0])
    exposure = pd.Series([1.0, 1.0, 1.0])

    perfect_prediction = actual_claims / exposure

    # The function requires strictly positive predictions. Add a tiny
    # positive value to the zero-claim policy.
    perfect_prediction = perfect_prediction.clip(lower=1e-12)

    result = normalized_gini(
        actual_claims=actual_claims,
        predicted_frequency=perfect_prediction,
        exposure=exposure,
    )

    assert result == pytest.approx(1.0)


def test_reversed_ranking_has_negative_normalized_gini():
    actual_claims = pd.Series([0.0, 1.0, 2.0])
    predicted_frequency = pd.Series([3.0, 2.0, 1.0])
    exposure = pd.Series([1.0, 1.0, 1.0])

    result = normalized_gini(
        actual_claims=actual_claims,
        predicted_frequency=predicted_frequency,
        exposure=exposure,
    )

    assert result < 0


def test_evaluate_perfect_severity_predictions():
    predictions = pd.DataFrame(
        {
            "average_claim_amount": [100.0, 200.0],
            "predicted_severity": [100.0, 200.0],
            "n_claim_rows": [1, 2],
            "total_claim_amount": [100.0, 400.0],
            "predicted_claim_amount": [100.0, 400.0],
        }
    )

    metrics = evaluate_severity_model(predictions)
    assert metrics["gamma_deviance"] == pytest.approx(0.0)
    assert metrics["actual_to_expected"] == pytest.approx(1.0)


def test_severity_calibration_creates_requested_groups():
    predictions = pd.DataFrame(
        {
            "policy_id": range(1, 7),
            "average_claim_amount": [100, 120, 180, 220, 300, 400],
            "predicted_severity": [90, 130, 170, 230, 290, 410],
            "n_claim_rows": [1, 1, 2, 1, 2, 1],
            "total_claim_amount": [100, 120, 360, 220, 600, 400],
            "predicted_claim_amount": [90, 130, 340, 230, 580, 410],
        }
    )

    result = severity_calibration_by_decile(predictions, n_deciles=3)
    assert result["decile"].tolist() == [1, 2, 3]
    assert result["claims"].sum() == 8
