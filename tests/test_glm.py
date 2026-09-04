import numpy as np
import pandas as pd
import pytest

from src.models.glm import (
    fit_frequency_glm,
    fit_severity_glm,
    predict_frequency_glm,
    predict_severity_glm,
)


class DummyFrequencyModel:
    """Small fake model used to test prediction calculations."""

    def predict(self, df, offset):
        annual_frequency = 0.2
        return np.exp(offset) * annual_frequency


def test_frequency_glm_rejects_nonpositive_exposure():
    df = pd.DataFrame(
        {
            "exposure": [1.0, 0.0, 0.5],
        }
    )

    with pytest.raises(
        ValueError,
        match="strictly positive exposure",
    ):
        fit_frequency_glm(df)


def test_predict_frequency_glm_returns_counts_and_rates():
    model = DummyFrequencyModel()

    df = pd.DataFrame(
        {
            "policy_id": [101, 102],
            "exposure": [1.0, 0.5],
            "claim_nb": [0, 1],
        }
    )

    result = predict_frequency_glm(
        model,
        df,
    )

    assert result["predicted_claim_count"].tolist() == pytest.approx([0.2, 0.1])

    assert result["predicted_frequency"].tolist() == pytest.approx([0.2, 0.2])


def test_predict_frequency_glm_does_not_modify_input():
    model = DummyFrequencyModel()

    df = pd.DataFrame(
        {
            "policy_id": [101],
            "exposure": [0.5],
            "claim_nb": [0],
        }
    )

    original_columns = df.columns.tolist()

    predict_frequency_glm(model, df)

    assert df.columns.tolist() == original_columns
    assert "predicted_frequency" not in df.columns


def test_predict_frequency_glm_rejects_nonpositive_exposure():
    model = DummyFrequencyModel()

    df = pd.DataFrame(
        {
            "policy_id": [101],
            "exposure": [0.0],
            "claim_nb": [0],
        }
    )

    with pytest.raises(
        ValueError,
        match="strictly positive exposure",
    ):
        predict_frequency_glm(model, df)


def test_severity_glm_returns_positive_predictions():
    rng = np.random.default_rng(42)
    rows = 300
    claim_rows = rng.integers(0, 4, size=rows)
    positive_amounts = rng.gamma(shape=2.0, scale=900.0, size=rows)
    df = pd.DataFrame(
        {
            "policy_id": range(rows),
            "n_claim_rows": claim_rows,
            "total_claim_amount": np.where(
                claim_rows > 0,
                positive_amounts * claim_rows,
                0,
            ),
            "area": rng.choice(["A", "B", "C"], size=rows),
            "age_band": rng.choice(["25-34", "35-44", "45-54"], size=rows),
            "veh_brand": rng.choice(["B1", "B2", "B3"], size=rows),
            "veh_gas": rng.choice(["Diesel", "Regular"], size=rows),
            "region": rng.choice(["R1", "R2", "R3"], size=rows),
            "veh_power": rng.integers(4, 10, size=rows),
            "veh_age": rng.integers(0, 20, size=rows),
            "bonus_malus": rng.integers(50, 150, size=rows),
            "density": rng.uniform(10, 1000, size=rows),
        }
    )

    model = fit_severity_glm(df)
    predictions = predict_severity_glm(model, df)

    assert len(predictions) == (claim_rows > 0).sum()
    assert (predictions["predicted_severity"] > 0).all()
    assert (predictions["predicted_claim_amount"] > 0).all()
