"""Clean the analysis base table.
Each function takes a DataFrame and returns a new one, logging how many rows it
changed. The reasoning behind each decision is in docs/data_quality.md.
"""

from __future__ import annotations
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def cap_exposure(df: pd.DataFrame, max_exposure: float = 1.0) -> pd.DataFrame:

    return


def cap_bonus_malus(df, cap=150):
    """TODO: clip bonus_malus at the cap from config.yaml. Your data reaches 230."""


def drop_zero_exposure(df):
    """TODO: remove rows where exposure <= 0. They can't contribute to any rate
    (division by zero) and represent no risk. Log how many you dropped."""


def flag_claim_mismatch(df):
    """TODO: add a boolean column marking policies where claim_nb disagrees with
    n_claim_rows. Flag, don't drop - you need to see the 9,463 downstream."""


def clean_base_table(df):
    """TODO: call the above in order and return the result. Log rows in vs out."""
