import numpy as np
import pandas as pd
import pytest

from src.models.ml import (
    FEATURES,
    fit_frequency_xgboost,
    predict_frequency_xgboost,
)


class DummyXGBoostModel:
    """Fake model for testing prediction calculations."""

    def predict(self, features):
        return np.array([0.2, 0.3])


def sample_feature_data() -> pd.DataFrame:
    """Return a minimal DataFrame with every required feature."""
    return pd.DataFrame(
        {
            "policy_id": [101, 102],
            "exposure": [1.0, 0.5],
            "claim_nb": [0, 1],
            "area": ["A", "B"],
            "age_band": ["25-34", "35-44"],
            "veh_brand": ["B1", "B2"],
            "veh_gas": ["Diesel", "Regular"],
            "region": ["R11", "R24"],
            "veh_power": [5, 7],
            "veh_age": [3, 8],
            "bonus_malus": [50, 75],
            "density": [100.0, 500.0],
        }
    )


def test_predict_frequency_xgboost_returns_rates_and_counts():
    df = sample_feature_data()
    model = DummyXGBoostModel()

    result = predict_frequency_xgboost(
        model,
        df,
    )

    assert result["predicted_frequency"].tolist() == (pytest.approx([0.2, 0.3]))

    assert result["predicted_claim_count"].tolist() == (pytest.approx([0.2, 0.15]))


def test_predict_frequency_xgboost_does_not_modify_input():
    df = sample_feature_data()
    model = DummyXGBoostModel()

    original_columns = df.columns.tolist()

    predict_frequency_xgboost(
        model,
        df,
    )

    assert df.columns.tolist() == original_columns
    assert "predicted_frequency" not in df.columns


def test_predict_frequency_xgboost_rejects_zero_exposure():
    df = sample_feature_data()
    df.loc[0, "exposure"] = 0.0

    with pytest.raises(
        ValueError,
        match="strictly positive exposure",
    ):
        predict_frequency_xgboost(
            DummyXGBoostModel(),
            df,
        )


def test_fit_frequency_xgboost_rejects_zero_exposure():
    df = sample_feature_data()
    df.loc[0, "exposure"] = 0.0

    with pytest.raises(
        ValueError,
        match="strictly positive exposure",
    ):
        fit_frequency_xgboost(
            train=df,
            model_config={
                "objective": "count:poisson",
                "n_estimators": 2,
                "max_depth": 2,
                "learning_rate": 0.1,
            },
            random_state=42,
        )


def test_sample_data_contains_all_model_features():
    df = sample_feature_data()

    assert set(FEATURES).issubset(df.columns)
