from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

from .models import ClassificationRunResult


def write_excel_report(
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
        _format_sheets(writer.book, sheet_names)


def write_transactions_csv(
    result: ClassificationRunResult,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.transactions.to_csv(path, index=False, encoding="utf-8-sig")


def _format_sheets(workbook, sheet_names: set[str]) -> None:
    header_fill = PatternFill(fill_type="solid", fgColor="000000")
    header_font = Font(color="FFFFFF", bold=True)
    for sheet_name in sheet_names:
        worksheet = workbook[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
