import pandas as pd
import pytest

from src.pricing.technical_premium import (
    add_technical_premium,
    portfolio_pure_premium_rate,
    profitability_study,
)


def test_portfolio_pure_premium_rate_uses_totals():
    df = pd.DataFrame(
        {
            "exposure": [1.0, 0.5],
            "total_claim_amount": [100.0, 50.0],
        }
    )

    result = portfolio_pure_premium_rate(df)

    assert result == pytest.approx(100.0)

def test_add_technical_premium_applies_exposure_and_loadings():
    df = pd.DataFrame(
        {
            "policy_id": [1, 2],
            "exposure": [1.0, 0.5],
        }
    )

    result = add_technical_premium(
        df,
        pure_premium_rate=100,
        expense_loading=0.25,
        profit_loading=0.05,
    )

    assert result["technical_premium_rate"].tolist() == [
        pytest.approx(130),
        pytest.approx(130),
    ]

    assert result["technical_premium"].tolist() == [
        pytest.approx(130),
        pytest.approx(65),
    ]


def test_profitability_study_aggregates_before_dividing():
    df = pd.DataFrame(
        {
            "policy_id": [1, 2],
            "region": ["A", "A"],
            "exposure": [1.0, 0.5],
            "total_claim_amount": [100.0, 50.0],
            "technical_premium": [130.0, 65.0],
        }
    )

    result = profitability_study(
        df,
        by="region",
        min_exposure=1,
    )

    row = result.iloc[0]

    assert row["actual_claims"] == pytest.approx(150)
    assert row["technical_premium"] == pytest.approx(195)
    assert row["loss_ratio"] == pytest.approx(150 / 195)
    assert row["underwriting_result"] == pytest.approx(45)