"""Clean the analysis base table.
Each function takes a DataFrame and returns a new one, logging how many rows it
changed. The reasoning behind each decision is in docs/data_quality.md.
"""

from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def cap_exposure(df: pd.DataFrame, max_exposure: float = 1.0) -> pd.DataFrame:
    """Cap exposure at one policy-year. More is physically impossible."""
    df = df.copy()
    n_affected = (df["exposure"] > max_exposure).sum()

    df["exposure"] = df["exposure"].clip(upper=max_exposure)

    logger.info("cap_exposure: capped %d rows at %.2f", n_affected, max_exposure)

    return df


def cap_bonus_malus(df: pd.DataFrame, cap: int = 150) -> pd.DataFrame:
    """Clip bonus_malus at the cap at the supplied maximum."""
    df = df.copy()
    n_affected = (df["bonus_malus"] > cap).sum()

    df["bonus_malus"] = df["bonus_malus"].clip(upper=cap)

    logger.info("cap_bonus_malus: capped %d rows at %d", n_affected, cap)

    return df


def drop_zero_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where exposure <= 0. They can't contribute to any rate
    (division by zero) and represent no risk."""
    df = df.copy()
    invalid_exposure = df["exposure"] <= 0.0
    n_affected = invalid_exposure.sum()
    df = df.loc[~invalid_exposure].copy()

    logger.info("drop_zero_exposure: dropped %d rows with exposure <= 0", n_affected)

    return df


def flag_claim_mismatch(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean column marking policies where claim_nb disagrees with
    n_claim_rows. Flag, don't drop - you need to see the 9,463 downstream."""
    df = df.copy()
    mismatch = df["claim_nb"] != df["n_claim_rows"]
    df["claim_count_mismatch"] = mismatch
    count = mismatch.sum()

    logger.info("flag_claim_mismatch: flagged %d mismatched policies", count)

    return df


def clean_base_table(df: pd.DataFrame) -> pd.DataFrame:
    """All the above in order and return the result."""
    rows_before = len(df)

    df = df.copy()
    df = cap_exposure(df)
    df = cap_bonus_malus(df)
    df = drop_zero_exposure(df)
    df = flag_claim_mismatch(df)

    rows_after = len(df)

    logger.info(
        "clean_base_table: rows before=%d, rows after=%d, dropped=%d",
        rows_before,
        rows_after,
        rows_before - rows_after,
    )

    return df
