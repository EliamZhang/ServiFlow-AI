from __future__ import annotations

import pandas as pd


SUMMARY_COLUMNS = [
    "finv_category",
    "stream_id",
    "transaction_count",
]


def build_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build a simple aggregate summary grouped by transfer category."""
    transfer = transactions[transactions["is_transfer_pred"].eq(1)].copy()
    if transfer.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary_rows = []
    for category, group in transfer.groupby("finv_category", sort=False):
        summary_rows.append(
            {
                "finv_category": category,
                "stream_id": category,
                "transaction_count": int(len(group)),
            }
        )

    return pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
