import numpy as np
import pandas as pd
import pytest

from src.models.glm import (
    fit_frequency_glm,
    predict_frequency_glm,
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
