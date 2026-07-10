from __future__ import annotations

import pandas as pd

from .models import TRANSACTION_KEY_COLUMNS


def normalized_keys(df: pd.DataFrame) -> pd.MultiIndex:
    key_frame = df.loc[:, list(TRANSACTION_KEY_COLUMNS)].copy()
    return pd.MultiIndex.from_frame(key_frame.astype("string").fillna(""))


def filter_to_transaction_keys(
    details: pd.DataFrame,
    selected_transactions: pd.DataFrame,
) -> pd.DataFrame:
    if selected_transactions.empty:
        return details.iloc[0:0].copy()
    selected_keys = normalized_keys(selected_transactions)
    return details.loc[normalized_keys(details).isin(selected_keys)].copy()
