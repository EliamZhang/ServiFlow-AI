from __future__ import annotations

import pandas as pd


SUMMARY_COLUMNS = [
    "finv_category",
    "stream_id",
    "transaction_count",
]


def build_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build a simple aggregate summary for transfers."""
    transfer = transactions[transactions["is_transfer_pred"].eq(1)]
    if transfer.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    return pd.DataFrame(
        [
            {
                "finv_category": "transfer",
                "stream_id": "transfer",
                "transaction_count": int(len(transfer)),
            }
        ],
        columns=SUMMARY_COLUMNS,
    )
