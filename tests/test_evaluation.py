import pandas as pd
import pytest

from src.models.evaluation import (
    actual_to_expected,
    calibration_by_decile,
    evaluate_frequency_model,
    poisson_deviance,
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
