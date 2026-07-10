from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from serviflow.excel import format_sheets
from serviflow.models import PipelineResult

from .summary import build_summary


TRANSACTIONS_SHEET_NAME = "transactions"
DETAIL_SHEET_NAME = "transactions_detail"
SUMMARY_SHEET_NAME = "income_summary"
EXCEL_MAX_ROWS = 1_048_576
EXCEL_DATA_ROWS_PER_SHEET = EXCEL_MAX_ROWS - 1

PUBLISHED_TRANSACTION_EXTRA_COLUMNS = [
    "counterparty",
    "finv_category",
]

SUMMARY_PRIORITY_COLUMNS = [
    "finv_category",
    "stream_id",
    "bank_account_id",
    "bank",
    "account_type",
    "credit_limit",
    "application_id",
    "counterparty",
    "transaction_start_date",
    "transaction_end_date",
]

AUDIT_DETAIL_COLUMNS = [
    "user_id",
    "sample_datetime",
    "application_id",
    "job_id",
    "transaction_id",
    "bank_account_id",
    "account_type",
    "transaction_date",
    "amount",
    "dr_cr",
    "category",
    "illion_trx_uuid",
    "balance",
    "text",
    "counterparty",
    "income_type_pred",
    "centrelink_payment_type",
    "is_income_pred",
    "is_wages_pred",
    "stream_id",
    "finv_category",
    "income_type_rule_name",
    "income_type_pred_reason",
    "wages_rule_name",
    "wages_pred_reason",
]

def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required column(s): {', '.join(missing)}")


def _select_transaction_columns(
    result_df: pd.DataFrame,
    original_columns: tuple[str, ...],
) -> pd.DataFrame:
    _require_columns(result_df, PUBLISHED_TRANSACTION_EXTRA_COLUMNS, "transactions sheet")
    selected_columns = [
        column for column in original_columns if column in result_df.columns
    ]
    for column in PUBLISHED_TRANSACTION_EXTRA_COLUMNS:
        if column not in selected_columns:
            selected_columns.append(column)
    return result_df[selected_columns].copy()


def _select_audit_detail_columns(result_df: pd.DataFrame) -> pd.DataFrame:
    existing_columns = [col for col in AUDIT_DETAIL_COLUMNS if col in result_df.columns]
    return result_df[existing_columns].copy()


def _filter_report_detail_rows(detail_output: pd.DataFrame) -> pd.DataFrame:
    _require_columns(detail_output, ["is_income_pred"], "detail sheet")
    return detail_output[detail_output["is_income_pred"].eq(1)].copy()


def _format_summary_columns(summary_df: pd.DataFrame, include_centrelink_detail: bool) -> pd.DataFrame:
    output = summary_df.copy()
    if not include_centrelink_detail:
        output = output.drop(columns=["centrelink_payment_type"], errors="ignore")

    required_columns = [col for col in SUMMARY_PRIORITY_COLUMNS if col in output.columns]
    remaining_columns = [col for col in output.columns if col not in required_columns]
    return output[required_columns + remaining_columns]


def _write_income_detail_sheets(writer: pd.ExcelWriter, income_detail_df: pd.DataFrame) -> None:
    if len(income_detail_df) <= EXCEL_DATA_ROWS_PER_SHEET:
        income_detail_df.to_excel(writer, sheet_name=DETAIL_SHEET_NAME, index=False)
        return

    sheet_count = math.ceil(len(income_detail_df) / EXCEL_DATA_ROWS_PER_SHEET)
    for idx in range(sheet_count):
        start = idx * EXCEL_DATA_ROWS_PER_SHEET
        end = start + EXCEL_DATA_ROWS_PER_SHEET
        sheet_name = f"detail_{idx + 1:02d}"
        income_detail_df.iloc[start:end].to_excel(writer, sheet_name=sheet_name, index=False)

    print(
        "Detail report exceeded Excel row limit; "
        f"split income transaction detail across {sheet_count} sheets."
    )


def write_report(
    result: PipelineResult,
    output_path: str | Path,
    full: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = _format_summary_columns(
        build_summary(result.transactions),
        include_centrelink_detail=full,
    )

    if not full:
        transactions_df = _select_transaction_columns(
            result.transactions,
            result.original_columns,
        )

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            transactions_df.to_excel(writer, sheet_name=TRANSACTIONS_SHEET_NAME, index=False)
            summary_df.to_excel(writer, sheet_name=SUMMARY_SHEET_NAME, index=False)
            format_sheets(
                writer.book,
                [TRANSACTIONS_SHEET_NAME, SUMMARY_SHEET_NAME],
            )

        print(
            f"Income report workbook saved with {len(summary_df)} summary rows and "
            f"{len(transactions_df)} transaction rows."
        )
        return transactions_df, summary_df

    income_detail_df = _filter_report_detail_rows(
        _select_audit_detail_columns(result.transactions)
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name=SUMMARY_SHEET_NAME, index=False)
        _write_income_detail_sheets(writer, income_detail_df)
        detail_sheets = [
            name for name in writer.book.sheetnames if name != SUMMARY_SHEET_NAME
        ]
        format_sheets(writer.book, [SUMMARY_SHEET_NAME, *detail_sheets])

    print(
        f"Income report workbook saved with {len(summary_df)} summary rows and "
        f"{len(income_detail_df)} income-detail rows."
    )

    return income_detail_df, summary_df
