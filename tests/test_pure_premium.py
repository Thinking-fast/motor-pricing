import pandas as pd
import pytest

from src.models.pure_premium import (
    add_modelled_technical_premium,
    build_pure_premium_predictions,
    evaluate_pure_premium,
    pure_premium_calibration_by_decile,
)


def _predictions():
    policies = pd.DataFrame(
        {
            "policy_id": [1, 2, 3, 4],
            "exposure": [1.0, 0.5, 1.0, 0.5],
            "total_claim_amount": [0.0, 100.0, 200.0, 0.0],
            "region": ["A", "A", "B", "B"],
        }
    )
    frequency = pd.DataFrame(
        {"policy_id": [1, 2, 3, 4], "predicted_frequency": [0.1, 0.2, 0.3, 0.4]}
    )
    return policies, build_pure_premium_predictions(
        policies, frequency, capped_severity=1000, large_loss_loading=20
    )


def test_build_pure_premium_combines_frequency_severity_and_loading():
    _, result = _predictions()
    assert result["predicted_pure_premium_rate"].tolist() == pytest.approx(
        [120, 220, 320, 420]
    )
    assert result["predicted_claim_cost"].tolist() == pytest.approx(
        [120, 110, 320, 210]
    )


def test_frequency_and_severity_use_a_consistent_claim_basis():
    claim_count = 4
    exposure = 2
    capped_claim_cost = 600
    frequency = claim_count / exposure
    severity = capped_claim_cost / claim_count

    assert frequency * severity == pytest.approx(capped_claim_cost / exposure)


def test_evaluate_pure_premium_returns_metrics():
    _, predictions = _predictions()
    metrics = evaluate_pure_premium(predictions)
    assert metrics["tweedie_deviance"] >= 0
    assert metrics["actual_to_expected"] == pytest.approx(300 / 760)


def test_pure_premium_calibration_aggregates_deciles():
    _, predictions = _predictions()
    result = pure_premium_calibration_by_decile(predictions, n_deciles=2)
    assert result["decile"].tolist() == [1, 2]
    assert result["policies"].tolist() == [2, 2]


def test_add_modelled_technical_premium_applies_loadings():
    policies, predictions = _predictions()
    priced = add_modelled_technical_premium(
        policies,
        predictions,
        expense_loading=0.25,
        profit_loading=0.05,
    )
    assert priced["technical_premium_rate"].tolist() == pytest.approx(
        [156, 286, 416, 546]
    )
