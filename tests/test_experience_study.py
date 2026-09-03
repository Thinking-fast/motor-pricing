import pandas as pd
import numpy as np


import pytest

from src.analysis.experience_study import add_age_band, experience_study


def test_add_age_band_uses_left_inclusive_intervals():
    df = pd.DataFrame(
        {
            "driv_age": [18, 24, 25, 34, 35, 99],
        }
    )

    result = add_age_band(df)

    assert result["age_band"].astype(str).tolist() == [
        "18-24",
        "18-24",
        "25-34",
        "25-34",
        "35-44",
        "75-99",
    ]

def test_experience_study_uses_grouped_totals():
    df = pd.DataFrame(
        {
            "policy_id": [1, 2],
            "region": ["A", "A"],
            "exposure": [1.0, 0.25],
            "claim_nb": [1, 1],
            "n_claim_rows": [1, 1],
            "total_claim_amount": [1000.0, 500.0],
        }
    )

    result = experience_study(df, by="region", min_exposure=1)

    row = result.iloc[0]

    assert row["policies"] == 2
    assert row["exposure"] == 1.25
    assert row["claim_nb"] == 2
    assert row["claim_amount"] == 1500
    assert row["frequency"] == 1.6
    assert row["severity"] == 750
    assert row["pure_premium"] == 1200
    assert bool(row["credible"]) is True


def test_experience_study_flags_low_exposure():
    df = pd.DataFrame(
        {
            "policy_id": [1],
            "region": ["A"],
            "exposure": [40.0],
            "claim_nb": [1],
            "n_claim_rows": [1],
            "total_claim_amount": [500.0],
        }
    )

    result = experience_study(
        df,
        by="region",
        min_exposure=1000,
    )

    assert bool(result.loc[0, "credible"]) is False


def test_experience_study_rejects_unknown_factor():
    df = pd.DataFrame({"region": ["A"]})

    with pytest.raises(ValueError, match="Unknown grouping column"):
        experience_study(df, by="wrong_name")



def test_pure_premium_equals_frequency_times_severity():
    df = pd.DataFrame(
        {
            "policy_id": [1, 2, 3],
            "region": ["A", "A", "B"],
            "exposure": [1.0, 0.5, 1.0],
            "claim_nb": [1, 1, 2],
            "n_claim_rows": [1, 1, 2],
            "total_claim_amount": [500.0, 250.0, 1000.0],
        }
    )

    result = experience_study(df, by="region")

    assert np.allclose(
        result["pure_premium"],
        result["frequency"] * result["severity"],
        equal_nan=True,
    )