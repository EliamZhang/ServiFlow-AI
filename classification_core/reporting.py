from __future__ import annotations

from pathlib import Path

import pandas as pd

from .excel import format_sheets, format_transactions_sheet
from .models import ClassificationRunResult


# Columns to auto-hide on the transactions sheet (user can manually unhide).
_HIDDEN_COLUMNS = frozenset({
    "sample_datetime", "job_id", "transaction_id", "account_type",
    "credit_limit", "trx_type", "bsb", "account_no",
    "illion_trx_uuid", "balance",
})

# Columns to highlight with a coloured header.
_RED_COLUMNS = frozenset({"category", "third_party"})
_GREEN_COLUMNS = frozenset({"counterparty", "finv_category"})


def write_report(
    result: ClassificationRunResult,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names: set[str] = set()

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        result.transactions.to_excel(writer, sheet_name="transactions", index=False)
        sheet_names.add("transactions")
        for artifact in result.summaries:
            sheet_name = artifact.name[:31]
            if sheet_name in sheet_names:
                raise ValueError(f"Duplicate output sheet name: {sheet_name}")
            artifact.data.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet_names.add(sheet_name)

        # ── per-sheet formatting ──
        for name in sheet_names:
            ws = writer.book[name]
            if name == "transactions":
                format_transactions_sheet(
                    ws,
                    hidden_cols=_HIDDEN_COLUMNS,
                    red_cols=_RED_COLUMNS,
                    green_cols=_GREEN_COLUMNS,
                )
            else:
                format_sheets(writer.book, [name])
