# -*- coding: utf-8 -*-
"""Summary table builder for the initial-classification (merchant KB) engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from classification_core.category_mapping import to_illion_category


SUMMARY_COLUMNS = [
    "finv_category",
    "counterparty",
    "initial_category",
    "bank_account_id",
    "account_type",
    "application_id",
    "bank",
    "transaction_count",
    "total_amount",
    "average_amount",
    "median_amount",
    "min_amount",
    "max_amount",
    "first_transaction_date",
    "last_transaction_date",
    "matched_keyword",
]


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def build_summary(matched: pd.DataFrame) -> pd.DataFrame:
    """Aggregate matched transactions by category + counterparty."""
    if matched.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    df = matched.copy()
    df["amount_num"] = pd.to_numeric(df["amount"], errors="coerce")
    df["txn_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    group_cols = ["finv_category", "counterparty"]
    # Optionally add account-level grouping columns when they exist.
    for col in ["bank_account_id", "account_type", "application_id", "bank"]:
        if col in df.columns:
            group_cols.append(col)

    rows: list[dict] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_dict = dict(zip(group_cols, keys))

        amounts = group["amount_num"].dropna()
        keyword = group["_matched_keyword"].iloc[0] if "_matched_keyword" in group.columns else ""

        rows.append(
            {
                "finv_category": to_illion_category(
                    str(key_dict.get("finv_category", ""))
                ),
                "counterparty": key_dict.get("counterparty", ""),
                "initial_category": key_dict.get("finv_category", ""),
                "bank_account_id": key_dict.get("bank_account_id", np.nan),
                "account_type": key_dict.get("account_type", np.nan),
                "application_id": key_dict.get("application_id", np.nan),
                "bank": key_dict.get("bank", np.nan),
                "transaction_count": len(group),
                "total_amount": float(amounts.sum()) if not amounts.empty else 0.0,
                "average_amount": float(amounts.mean()) if not amounts.empty else np.nan,
                "median_amount": float(amounts.median()) if not amounts.empty else np.nan,
                "min_amount": float(amounts.min()) if not amounts.empty else np.nan,
                "max_amount": float(amounts.max()) if not amounts.empty else np.nan,
                "first_transaction_date": group["txn_date"].min().date()
                if not group["txn_date"].isna().all()
                else np.nan,
                "last_transaction_date": group["txn_date"].max().date()
                if not group["txn_date"].isna().all()
                else np.nan,
                "matched_keyword": keyword,
            }
        )

    return (
        pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
        .sort_values(["finv_category", "counterparty"])
        .reset_index(drop=True)
    )
