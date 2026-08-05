"""Category-level summary for the final classification output.

A single cross-engine aggregator runs after every engine has committed, so
every finv_category gets one row of basic statistics per bank account.
Statistics are computed from the final winning labels, so the numbers always
match what the downstream consumer sees.  income_summary / liability_summary
are separate stream-level views and do not conflict with this category-level
summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .text import is_blank

CATEGORY_SUMMARY_COLUMNS = [
    "finv_category",
    "bank_account_id",
    "transaction_start_date",
    "transaction_end_date",
    "transaction_count",
    "total_amount",
    "average_amount",
    "median_amount",
    "latest_amount",
]


def build_category_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the final output per finv_category.  Returns an empty frame
    with CATEGORY_SUMMARY_COLUMNS when there is nothing to summarise."""
    if transactions.empty or "finv_category" not in transactions.columns:
        return pd.DataFrame(columns=CATEGORY_SUMMARY_COLUMNS)

    categorized = transactions.loc[~is_blank(transactions["finv_category"])].copy()
    if categorized.empty:
        return pd.DataFrame(columns=CATEGORY_SUMMARY_COLUMNS)

    amount = pd.to_numeric(categorized["amount"], errors="coerce")
    categorized["_amount"] = amount.abs()
    categorized["_date"] = pd.to_datetime(
        categorized["transaction_date"], errors="coerce"
    )

    rows = []
    for (category, bank_account_id), group in categorized.groupby(
        [categorized["finv_category"].str.strip(), categorized["bank_account_id"]],
        dropna=False,
        sort=True,
    ):
        amounts = group["_amount"].dropna()
        dates = group["_date"].dropna()
        if amounts.empty:
            continue
        start_date = dates.min() if not dates.empty else pd.NaT
        end_date = dates.max() if not dates.empty else pd.NaT
        rows.append(
            {
                "finv_category": category,
                "bank_account_id": bank_account_id,
                "transaction_start_date": start_date.date()
                if not pd.isna(start_date)
                else np.nan,
                "transaction_end_date": end_date.date()
                if not pd.isna(end_date)
                else np.nan,
                "transaction_count": int(len(group)),
                "total_amount": float(amounts.sum()),
                "average_amount": float(amounts.mean()),
                "median_amount": float(amounts.median()),
                "latest_amount": float(amounts.iloc[-1]),
            }
        )

    return pd.DataFrame(rows, columns=CATEGORY_SUMMARY_COLUMNS)
