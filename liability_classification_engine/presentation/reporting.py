from __future__ import annotations

from pathlib import Path

import pandas as pd

from serviflow.excel import format_sheets
from serviflow.models import PipelineResult

from ..domain.summary import build_summary, ensure_finv_category
from ..pipeline import DEFAULT_RESOURCES_DIR


TRANSACTIONS_SHEET_NAME = "transactions"
SUMMARY_SHEET_NAME = "liability_summary"
INTERNAL_TRANSACTION_COLUMNS = [
    "product_type",
    "is_dishonours",
    "stream_id",
]


def write_report(
    result: PipelineResult,
    output_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the standard liability transaction and summary sheets."""
    workbook_path = Path(output_path)
    workbook_path.parent.mkdir(parents=True, exist_ok=True)

    transactions = ensure_finv_category(result.transactions)
    summary = build_summary(
        transactions,
        limits_file=DEFAULT_RESOURCES_DIR / "bnpl_maximum_limits.csv",
    )
    transactions_export = _prepare_transactions(transactions)
    summary_export = summary.copy()

    try:
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            transactions_export.to_excel(
                writer,
                index=False,
                sheet_name=TRANSACTIONS_SHEET_NAME,
            )
            summary_export.to_excel(
                writer,
                index=False,
                sheet_name=SUMMARY_SHEET_NAME,
            )
            format_sheets(
                writer.book,
                [TRANSACTIONS_SHEET_NAME, SUMMARY_SHEET_NAME],
            )
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot update {workbook_path}. Close the workbook and rerun."
        ) from exc

    return transactions_export, summary_export


def _prepare_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    return transactions.drop(
        columns=[
            column
            for column in INTERNAL_TRANSACTION_COLUMNS
            if column in transactions.columns
        ],
    )
