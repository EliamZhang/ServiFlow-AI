from __future__ import annotations

from collections.abc import Iterable

from openpyxl.styles import Alignment, Font, PatternFill


DEFAULT_FONT_NAME = "Microsoft YaHei"
HEADER_FILL = "FF000000"
HEADER_FONT_COLOR = "FFFFFFFF"


def format_sheets(workbook, sheet_names: Iterable[str]) -> None:
    for sheet_name in sheet_names:
        worksheet = workbook[sheet_name]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        default_font = Font(name=DEFAULT_FONT_NAME)
        header_font = Font(
            name=DEFAULT_FONT_NAME,
            color=HEADER_FONT_COLOR,
            bold=True,
        )
        header_fill = PatternFill(fill_type="solid", fgColor=HEADER_FILL)

        for row in worksheet.iter_rows():
            for cell in row:
                cell.font = default_font
                cell.alignment = Alignment(vertical="center")

        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
