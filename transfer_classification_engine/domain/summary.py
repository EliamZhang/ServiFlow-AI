from __future__ import annotations

import pandas as pd


SUMMARY_COLUMNS = [
    "finv_category",
    "stream_id",
    "transaction_count",
]


def build_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build an aggregate summary for transfers, split by finv_category."""
    transfer = transactions[transactions["is_transfer_pred"].eq(1)]
    if transfer.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows = []
    for category, group in transfer.groupby("finv_category"):
        rows.append(
            {
                "finv_category": category,
                "stream_id": category,
                "transaction_count": int(len(group)),
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
