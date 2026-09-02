import pandas as pd

from src.etl.clean import (
    cap_bonus_malus,
    cap_exposure,
    clean_base_table,
    drop_zero_exposure,
    flag_claim_mismatch,
)


def test_cap_exposure_caps_values_above_one():
    df = pd.DataFrame({"exposure": [0.5, 1.0, 1.4]})
    result = cap_exposure(df)
    assert result["exposure"].tolist() == [0.5, 1.0, 1.0]


def test_cap_exposure_does_not_modify_original():
    df = pd.DataFrame({"exposure": [0.5, 1.4]})

    result = cap_exposure(df)

    assert df["exposure"].tolist() == [0.5, 1.4]
    assert result["exposure"].tolist() == [0.5, 1.0]


def test_cap_bonus_malus_caps_values_above_limit():
    df = pd.DataFrame({"bonus_malus": [80, 150, 230]})

    result = cap_bonus_malus(df, cap=150)

    assert result["bonus_malus"].tolist() == [80, 150, 150]


def test_drop_zero_exposure_removes_zero_and_negative_rows():
    df = pd.DataFrame(
        {
            "policy_id": [101, 102, 103, 104],
            "exposure": [0.5, 0.0, -0.2, 1.0],
        }
    )

    result = drop_zero_exposure(df)

    assert result["policy_id"].tolist() == [101, 104]
    assert (result["exposure"] <= 0).sum() == 0


def test_flag_claim_mismatch_marks_disagreements():
    df = pd.DataFrame(
        {
            "claim_nb": [0, 2, 1],
            "n_claim_rows": [0, 1, 1],
        }
    )

    result = flag_claim_mismatch(df)

    assert result["claim_count_mismatch"].tolist() == [
        False,
        True,
        False,
    ]


def test_clean_base_table_applies_all_cleaning_rules():
    df = pd.DataFrame(
        {
            "policy_id": [101, 102, 103, 104],
            "exposure": [0.5, 1.4, 0.0, 0.8],
            "bonus_malus": [80, 230, 120, 170],
            "claim_nb": [0, 2, 1, 0],
            "n_claim_rows": [0, 1, 1, 0],
        }
    )

    result = clean_base_table(df)

    assert result["policy_id"].tolist() == [101, 102, 104]
    assert result["exposure"].max() <= 1
    assert result["bonus_malus"].max() <= 150
    assert (result["exposure"] <= 0).sum() == 0
    assert result["claim_count_mismatch"].sum() == 1
