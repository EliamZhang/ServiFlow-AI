from __future__ import annotations

from pathlib import Path

import pandas as pd

from .excel import format_sheets
from .models import ClassificationRunResult


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
        format_sheets(writer.book, sheet_names)
