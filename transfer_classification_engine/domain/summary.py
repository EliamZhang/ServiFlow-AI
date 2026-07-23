from __future__ import annotations

import pandas as pd

from classification_core.category_mapping import to_illion_category


SUMMARY_COLUMNS = [
    "finv_category",
    "stream_id",
    "transfer_category",
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
                "finv_category": to_illion_category(str(category)),
                "stream_id": category,
                "transfer_category": category,
                "transaction_count": int(len(group)),
            }
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
