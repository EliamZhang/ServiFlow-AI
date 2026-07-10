"""Build the single-file HTML credit review dashboard."""

from __future__ import annotations

import json
from functools import lru_cache
from datetime import date, datetime, timedelta
from typing import Any, Dict, List

import pandas as pd


DATE_COLUMNS = {
    "sample_datetime",
    "transaction_date",
    "transaction_start_date",
    "transaction_end_date",
    "predicted_closing_date",
}

AMOUNT_COLUMNS = {
    "amount",
    "balance",
    "credit_limit",
    "funded_amount",
    "repaid_amount",
    "repayment_amount",
    "recent_fn_repay_amount",
}

DEFAULT_NA_TEXT = {
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
}


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in DEFAULT_NA_TEXT:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def excel_serial_to_date(value: float) -> str:
    """Convert Excel serial date to YYYY-MM-DD. Excel day 0 is treated as 1899-12-30."""
    dt = datetime(1899, 12, 30) + timedelta(days=float(value))
    return dt.strftime("%Y-%m-%d")


@lru_cache(maxsize=20000)
def normalize_date_text(text: str) -> str:
    """Normalize a text date with caching to keep large transaction files fast."""
    text = text.strip()
    if not text:
        return ""

    # Date values can sometimes arrive as numeric strings.
    try:
        numeric_value = float(text)
        if 20000 <= numeric_value <= 80000:
            return excel_serial_to_date(numeric_value)
    except Exception:
        pass

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    return parsed.strftime("%Y-%m-%d")


def normalize_date(value: Any) -> str:
    """Normalize Excel dates, pandas timestamps, Python dates, and date-like strings."""
    if is_blank(value):
        return ""

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, (int, float)) and 20000 <= float(value) <= 80000:
        return excel_serial_to_date(value)

    return normalize_date_text(str(value))


def normalize_id_like(value: Any) -> Any:
    """Keep ID-like values readable by avoiding unnecessary .0 suffix when possible."""
    if is_blank(value):
        return ""

    if isinstance(value, (int,)):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return value

    if hasattr(value, "item"):
        try:
            item = value.item()
            return normalize_id_like(item)
        except Exception:
            pass

    return value


def normalize_general_value(value: Any) -> Any:
    """Normalize non-special columns while keeping JSON output clean."""
    if is_blank(value):
        return ""

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert DataFrame to clean JSON-ready records with column-level normalization."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    for col in df.columns:
        col_lower = col.lower()
        series = df[col]

        if col_lower in DATE_COLUMNS or "date" in col_lower or "datetime" in col_lower:
            df[col] = series.map(normalize_date)
            continue

        if col_lower in AMOUNT_COLUMNS:
            numeric = pd.to_numeric(series, errors="coerce")
            fallback = series.astype("object").map(lambda value: "" if is_blank(value) else str(value).strip())
            df[col] = numeric.where(numeric.notna(), fallback).astype("object")
            df.loc[series.isna(), col] = ""
            continue

        if col_lower.endswith("_id") or col_lower in {"application_id", "user_id", "job_id"}:
            df[col] = series.map(normalize_id_like)
            continue

        df[col] = series.map(normalize_general_value)

    return df.to_dict(orient="records")


def build_html(data: Dict[str, Any]) -> str:
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    data_json = data_json.replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Liability Classification Review Dashboard</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --surface: rgba(255, 255, 255, 0.86);
      --card: #ffffff;
      --text: #111827;
      --muted: #64748b;
      --subtle: #94a3b8;
      --line: #e2e8f0;
      --line-strong: #cbd5e1;
      --primary: #2563eb;
      --primary-dark: #1d4ed8;
      --primary-soft: #dbeafe;
      --green: #16a34a;
      --green-soft: #dcfce7;
      --amber: #d97706;
      --amber-soft: #fef3c7;
      --red: #dc2626;
      --red-soft: #fee2e2;
      --purple: #7c3aed;
      --purple-soft: #ede9fe;
      --shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
      --shadow-soft: 0 8px 22px rgba(15, 23, 42, 0.06);
      --radius-lg: 24px;
      --radius-md: 18px;
      --radius-sm: 12px;
      --table-header: #f8fafc;
      --page-padding: 10px;
      --sidebar-width: 320px;
      --content-gap: 14px;
    }

    * { box-sizing: border-box; }

    html { scroll-behavior: smooth; }

    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.12), transparent 34rem),
        radial-gradient(circle at top right, rgba(124, 58, 237, 0.10), transparent 30rem),
        linear-gradient(180deg, #f8fafc 0%, var(--bg) 100%);
      color: var(--text);
    }

    .page {
      width: 100%;
      max-width: none;
      margin: 0;
      padding: var(--page-padding) var(--page-padding) 32px;
    }

    .hero {
      position: relative;
      overflow: hidden;
      color: #fff;
      border-radius: 18px;
      padding: 12px 18px;
      box-shadow: var(--shadow-soft);
      margin-bottom: 10px;
      background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 64, 175, 0.92)),
        radial-gradient(circle at 90% 20%, rgba(59, 130, 246, 0.38), transparent 20rem);
    }

    .hero::after {
      content: "";
      position: absolute;
      inset: auto -110px -210px auto;
      width: 320px;
      height: 320px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      filter: blur(1px);
    }

    .hero-top {
      position: relative;
      z-index: 1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      height: 20px;
      padding: 0 9px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.12);
      border: 1px solid rgba(255, 255, 255, 0.15);
      color: rgba(255, 255, 255, 0.86);
      font-size: 10px;
      font-weight: 800;
      margin-bottom: 6px;
    }

    .hero h1 {
      margin: 0 0 3px;
      font-size: clamp(22px, 1.75vw, 28px);
      line-height: 1.05;
      letter-spacing: -0.04em;
    }

    .hero p {
      margin: 0;
      color: rgba(255, 255, 255, 0.74);
      font-size: 12px;
      line-height: 1.35;
    }

    .hero-meta {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: repeat(3, minmax(108px, 1fr));
      gap: 8px;
      min-width: 340px;
    }

    .hero-stat {
      border-radius: 12px;
      padding: 8px 10px;
      background: rgba(255, 255, 255, 0.10);
      border: 1px solid rgba(255, 255, 255, 0.15);
      backdrop-filter: blur(10px);
    }

    .hero-stat .label {
      color: rgba(255, 255, 255, 0.66);
      font-size: 10px;
      font-weight: 800;
      margin-bottom: 5px;
    }

    .hero-stat .value {
      color: #fff;
      font-size: 15px;
      font-weight: 900;
      letter-spacing: -0.02em;
      white-space: nowrap;
    }

    .panel {
      background: var(--surface);
      border: 1px solid rgba(226, 232, 240, 0.86);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-soft);
      padding: 20px;
      margin-bottom: 18px;
      backdrop-filter: blur(8px);
    }

    .panel.compact { padding: 16px 18px; }

    .review-toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 270px) minmax(0, 1fr);
      align-items: center;
      gap: 6px 14px;
      padding: 10px 16px;
      margin-bottom: 10px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.92);
    }

    .review-toolbar .section-title {
      margin: 0;
      min-width: 0;
    }

    .review-toolbar .section-title h2 {
      font-size: 16px;
      white-space: nowrap;
    }

    .review-toolbar .section-title .hint {
      display: none;
    }

    .review-toolbar .icon-dot {
      width: 24px;
      height: 24px;
      border-radius: 9px;
      font-size: 13px;
    }

    .review-toolbar .search-panel {
      grid-template-columns: minmax(220px, 420px) auto auto minmax(220px, 1fr);
      align-items: center;
      gap: 8px;
    }

    .review-toolbar label {
      display: none;
    }

    .review-toolbar input,
    .review-toolbar select {
      height: 36px;
      border-radius: 12px;
      font-size: 13px;
    }

    .review-toolbar .btn {
      height: 36px;
      border-radius: 12px;
      padding: 0 16px;
      font-size: 13px;
    }

    .review-toolbar .active-app-box {
      justify-content: flex-end;
      font-size: 12px;
      gap: 8px;
    }

    .review-toolbar .active-app-pill {
      height: 32px;
      padding: 0 11px;
      font-size: 12px;
    }

    .review-toolbar .message {
      grid-column: 2;
      min-height: 0;
      margin: 0;
      line-height: 1.25;
      font-size: 12px;
    }

    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 16px;
    }

    .section-title h2 {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 0;
      font-size: 18px;
      letter-spacing: -0.02em;
    }

    .section-title .hint {
      color: var(--muted);
      font-size: 13px;
      text-align: right;
    }

    .icon-dot {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 28px;
      height: 28px;
      border-radius: 10px;
      background: var(--primary-soft);
      color: var(--primary);
      font-size: 15px;
      flex: 0 0 auto;
    }

    .search-panel {
      display: grid;
      grid-template-columns: minmax(280px, 460px) auto auto 1fr;
      align-items: end;
      gap: 12px;
    }

    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 7px;
      font-weight: 750;
    }

    select, input {
      width: 100%;
      height: 44px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 0 13px;
      background: #fff;
      color: var(--text);
      outline: none;
      transition: border-color 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease;
    }

    input::placeholder { color: #a8b3c3; }

    select:focus, input:focus {
      border-color: rgba(37, 99, 235, 0.52);
      box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12);
    }

    .btn {
      height: 44px;
      border: 0;
      border-radius: 14px;
      padding: 0 18px;
      cursor: pointer;
      font-weight: 800;
      letter-spacing: -0.01em;
      transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.16s ease;
      white-space: nowrap;
    }

    .btn:hover { transform: translateY(-1px); }
    .btn:active { transform: translateY(0); }

    .btn-primary {
      color: #fff;
      background: linear-gradient(135deg, var(--primary), var(--primary-dark));
      box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22);
    }

    .btn-secondary {
      color: #334155;
      background: #fff;
      border: 1px solid var(--line);
      box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
    }

    .active-app-box {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 10px;
      min-width: 0;
      color: var(--muted);
      font-size: 13px;
    }

    .active-app-pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      max-width: 100%;
      height: 36px;
      padding: 0 12px;
      border-radius: 999px;
      background: #eff6ff;
      color: #1d4ed8;
      font-weight: 850;
      border: 1px solid #bfdbfe;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .message {
      margin-top: 10px;
      min-height: 20px;
      color: var(--red);
      font-size: 13px;
      font-weight: 650;
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(180px, 1fr));
      gap: 14px;
    }

    .kpi-card {
      position: relative;
      overflow: hidden;
      background: #fff;
      border: 1px solid rgba(226, 232, 240, 0.95);
      border-radius: var(--radius-md);
      padding: 18px;
      min-height: 116px;
      box-shadow: 0 8px 18px rgba(15, 23, 42, 0.035);
    }

    .kpi-card::after {
      content: "";
      position: absolute;
      right: -42px;
      top: -42px;
      width: 120px;
      height: 120px;
      border-radius: 999px;
      background: var(--primary-soft);
      opacity: 0.55;
    }

    .kpi-card.money::after { background: var(--green-soft); }
    .kpi-card.warning::after { background: var(--amber-soft); }
    .kpi-card.danger::after { background: var(--red-soft); }
    .kpi-card.purple::after { background: var(--purple-soft); }

    .kpi-label {
      position: relative;
      z-index: 1;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
      margin-bottom: 10px;
      text-transform: none;
    }

    .kpi-value {
      position: relative;
      z-index: 1;
      font-size: clamp(22px, 2.1vw, 30px);
      line-height: 1.08;
      font-weight: 900;
      letter-spacing: -0.045em;
      overflow-wrap: anywhere;
    }

    .kpi-footnote {
      position: relative;
      z-index: 1;
      margin-top: 8px;
      color: var(--subtle);
      font-size: 12px;
    }

    .distribution-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(240px, 1fr));
      gap: 14px;
      margin-top: 14px;
    }

    .dist-card {
      background: #fff;
      border: 1px solid rgba(226, 232, 240, 0.95);
      border-radius: var(--radius-md);
      padding: 18px;
      box-shadow: 0 8px 18px rgba(15, 23, 42, 0.035);
    }

    .dist-title {
      color: var(--muted);
      font-size: 12px;
      font-weight: 850;
      margin-bottom: 12px;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .chip {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 28px;
      padding: 5px 10px;
      border-radius: 999px;
      color: #334155;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      font-size: 12px;
      font-weight: 750;
    }

    .chip .num {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 22px;
      height: 22px;
      padding: 0 7px;
      border-radius: 999px;
      background: #e2e8f0;
      color: #0f172a;
      font-weight: 900;
    }

    .toolbar {
      display: grid;
      grid-template-columns: repeat(7, minmax(145px, 1fr));
      gap: 12px;
      align-items: end;
      margin-bottom: 14px;
    }

    .quick-filter-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 12px;
    }

    .quick-filter-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 34px;
      padding: 7px 12px;
      border-radius: 999px;
      color: #334155;
      background: #fff;
      border: 1px solid var(--line);
      cursor: pointer;
      font-size: 12px;
      font-weight: 900;
      transition: background 0.16s ease, border-color 0.16s ease, color 0.16s ease, transform 0.16s ease;
    }

    .quick-filter-btn:hover {
      transform: translateY(-1px);
      background: #eff6ff;
      border-color: #bfdbfe;
      color: #1d4ed8;
    }

    .quick-filter-btn.active {
      color: #fff;
      background: linear-gradient(135deg, var(--primary), var(--primary-dark));
      border-color: var(--primary);
      box-shadow: 0 8px 18px rgba(37, 99, 235, 0.18);
    }

    .table-shell {
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #fff;
      overflow: hidden;
      box-shadow: 0 8px 18px rgba(15, 23, 42, 0.03);
    }

    .table-wrap {
      overflow: auto;
      max-height: 640px;
    }

    table {
      width: 100%;
      min-width: 1240px;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 13px;
    }

    th, td {
      border-bottom: 1px solid var(--line);
      padding: 11px 12px;
      vertical-align: top;
      text-align: left;
      white-space: nowrap;
    }

    th {
      position: sticky;
      top: 0;
      z-index: 2;
      color: #475569;
      background: var(--table-header);
      font-size: 12px;
      font-weight: 900;
      cursor: pointer;
      user-select: none;
      border-bottom: 1px solid var(--line-strong);
    }

    td {
      color: #1f2937;
      background: #fff;
    }

    tbody tr:hover td { background: #f8fbff; }
    tbody tr:last-child td { border-bottom: 0; }

    td.text-cell {
      white-space: normal;
      min-width: 420px;
      max-width: 720px;
      line-height: 1.45;
      color: #334155;
    }

    .tag {
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      background: #f1f5f9;
      color: #334155;
      border: 1px solid #e2e8f0;
      font-size: 12px;
      font-weight: 750;
    }

    /* Color system: blue = navigation/product, green = ongoing, slate = closed/neutral, amber = needs review, red = real risk only. */
    .tag.green { color: #047857; background: #ecfdf5; border-color: #a7f3d0; }
    .tag.amber { color: #b45309; background: #fffbeb; border-color: #fde68a; }
    .tag.red { color: #b91c1c; background: #fef2f2; border-color: #fecaca; }
    .tag.blue { color: #1d4ed8; background: #eff6ff; border-color: #bfdbfe; }
    .tag.purple { color: #6d28d9; background: #f5f3ff; border-color: #ddd6fe; }
    .tag.gray { color: #475569; background: #f8fafc; border-color: #e2e8f0; }
    .tag.risk { color: #b91c1c; background: #fef2f2; border-color: #fecaca; }

    .table-jump-link {
      color: #0f172a;
      font-weight: 850;
      text-decoration: none;
    }

    .table-jump-link.tag {
      color: #0f172a;
    }

    .table-jump-link:hover {
      text-decoration: underline;
    }

    .amount-col { text-align: right; font-variant-numeric: tabular-nums; }

    .tx-sticky-col {
      position: sticky;
      background: #fff;
      z-index: 3;
      box-shadow: 1px 0 0 var(--line);
    }

    th.tx-sticky-col {
      background: var(--table-header);
      z-index: 6;
    }

    .tx-sticky-1 { left: 0; min-width: 122px; width: 122px; }
    .tx-sticky-2 { left: 122px; min-width: 112px; width: 112px; }
    .tx-sticky-3 { left: 234px; min-width: 190px; width: 190px; }

    tbody tr:hover td.tx-sticky-col { background: #f8fbff; }

    .text-clamp {
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      white-space: normal;
    }


    .empty {
      padding: 36px 22px;
      color: var(--muted);
      text-align: center;
      font-weight: 700;
    }

    .count-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 26px;
      height: 26px;
      border-radius: 999px;
      padding: 0 9px;
      background: #eef2ff;
      color: #3730a3;
      font-weight: 900;
      font-size: 12px;
    }

    .muted { color: var(--muted); }

    .mini-summary {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-top: 14px;
    }

    .mini-item {
      border-radius: 14px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      padding: 12px;
    }

    .mini-item .label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      margin-bottom: 6px;
    }

    .mini-item .value {
      color: var(--text);
      font-size: 15px;
      font-weight: 900;
      overflow-wrap: anywhere;
    }

    .overview-table-block {
      margin-top: 14px;
      padding: 14px;
      border: 1px solid rgba(226, 232, 240, 0.95);
      border-radius: var(--radius-md);
      background: #fff;
      box-shadow: 0 8px 18px rgba(15, 23, 42, 0.035);
    }

    .overview-table-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }

    .overview-table-title {
      color: #0f172a;
      font-size: 14px;
      font-weight: 950;
      letter-spacing: -0.01em;
    }

    .overview-table-hint {
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
      text-align: right;
    }

    .dashboard-layout {
      display: block;
      position: relative;
      min-width: 0;
    }

    .dashboard-main {
      min-width: 0;
      margin-left: calc(var(--sidebar-width) + var(--content-gap));
      width: calc(100vw - var(--sidebar-width) - var(--content-gap) - var(--page-padding) * 2);
    }

    .dashboard-main > .panel:last-child {
      margin-bottom: 0;
    }

    #liabilityOverviewSection,
    #streamFlatSection,
    #allBankTransactionsSection {
      scroll-margin-top: 18px;
    }

    .stream-directory {
      position: fixed;
      left: var(--page-padding);
      top: 168px;
      width: var(--sidebar-width);
      z-index: 30;
      max-height: calc(100vh - 178px);
      overflow-y: auto;
      overflow-x: hidden;
      overscroll-behavior: contain;
      scrollbar-width: none;
      -ms-overflow-style: none;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #fff;
      box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
      will-change: top, max-height;
    }

    .stream-directory::-webkit-scrollbar {
      width: 0;
      height: 0;
      display: none;
    }

    .directory-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      color: var(--text);
      font-size: 13px;
      font-weight: 950;
      margin-bottom: 10px;
    }

    .directory-title .small {
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      white-space: nowrap;
    }

    .directory-current-app {
      display: grid;
      gap: 4px;
      margin-bottom: 10px;
      padding: 10px 11px;
      border-radius: 14px;
      background: linear-gradient(180deg, #eff6ff, #ffffff);
      border: 1px solid #bfdbfe;
    }

    .directory-current-app span {
      color: var(--muted);
      font-size: 10px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }

    .directory-current-app strong {
      color: #1d4ed8;
      font-size: 14px;
      font-weight: 950;
      overflow-wrap: anywhere;
    }

    .directory-actions {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      margin-bottom: 12px;
    }

    .directory-actions a {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 34px;
      padding: 7px 10px;
      border-radius: 999px;
      color: #1d4ed8;
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      font-size: 12px;
      font-weight: 900;
      text-decoration: none;
    }

    .directory-actions a:hover {
      background: #dbeafe;
      border-color: #93c5fd;
    }

    .directory-actions a.active {
      color: #1d4ed8;
      background: #dbeafe;
      border-color: #93c5fd;
      box-shadow: inset 3px 0 0 #2563eb;
    }

    .directory-product {
      padding: 11px 0;
      border-top: 1px solid #f1f5f9;
    }

    .directory-product:first-of-type { border-top: 0; padding-top: 0; }

    .directory-product > a {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      min-height: 34px;
      padding: 7px 9px;
      border: 1px solid transparent;
      border-radius: 10px;
      color: #0f172a;
      font-size: 13px;
      font-weight: 950;
      text-decoration: none;
      overflow-wrap: anywhere;
      line-height: 1.35;
    }

    .directory-product > a:hover,
    .directory-streams a:hover {
      color: #1d4ed8;
      background: #eff6ff;
      border-color: #bfdbfe;
    }

    .directory-product > a.active {
      color: #1d4ed8;
      background: #eff6ff;
      border-color: #93c5fd;
      box-shadow: inset 3px 0 0 #2563eb;
    }

    .directory-product.contains-active > a {
      color: #1d4ed8;
    }

    .directory-product.contains-active .directory-product-count {
      color: #1d4ed8;
      background: #dbeafe;
    }

    .directory-product-count {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 24px;
      height: 22px;
      padding: 0 7px;
      border-radius: 999px;
      background: #f1f5f9;
      color: #334155;
      font-size: 11px;
      font-weight: 950;
      flex: 0 0 auto;
    }

    .directory-streams {
      display: grid;
      gap: 6px;
      margin-top: 8px;
      padding-left: 8px;
    }

    .directory-streams a {
      display: block;
      min-height: 48px;
      padding: 8px 9px;
      border-radius: 10px;
      color: #334155;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      font-size: 12px;
      font-weight: 850;
      text-decoration: none;
      overflow-wrap: anywhere;
    }

    .directory-stream-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }

    .directory-stream-top .directory-stream-id {
      flex: 1 1 auto;
    }

    .directory-stream-meta {
      display: block;
      margin-top: 4px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.3;
      font-weight: 800;
      white-space: normal;
      overflow-wrap: anywhere;
    }

    .directory-stream-meta .directory-status {
      color: #475569;
      font-weight: 900;
    }

    .directory-risk-dot {
      display: inline-flex;
      width: 8px;
      height: 8px;
      border-radius: 999px;
      flex: 0 0 auto;
      background: #94a3b8;
      box-shadow: 0 0 0 3px rgba(148, 163, 184, 0.13);
    }

    .directory-risk-dot.green { background: #10b981; box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.12); }
    .directory-risk-dot.blue { background: #64748b; box-shadow: 0 0 0 3px rgba(100, 116, 139, 0.12); }
    .directory-risk-dot.amber { background: #f59e0b; box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.13); }
    .directory-risk-dot.red { background: #dc2626; box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.12); }
    .directory-risk-dot.gray { background: #94a3b8; box-shadow: 0 0 0 3px rgba(148, 163, 184, 0.13); }

    .directory-status.green { color: #047857; }
    .directory-status.blue { color: #475569; }
    .directory-status.amber { color: #b45309; }
    .directory-status.red { color: #b91c1c; }
    .directory-status.gray { color: #64748b; }

    .directory-risk-badge {
      display: inline-flex;
      align-items: center;
      min-height: 18px;
      padding: 2px 6px;
      margin-left: 6px;
      border-radius: 999px;
      color: #b91c1c;
      background: #fef2f2;
      border: 1px solid #fecaca;
      font-size: 10px;
      font-weight: 950;
      white-space: nowrap;
    }

    .directory-streams a.active {
      color: #1d4ed8;
      background: #eff6ff;
      border-color: #93c5fd;
      box-shadow: inset 3px 0 0 #2563eb;
    }

    .directory-streams a.active .directory-stream-meta,
    .directory-streams a.active .directory-stream-count {
      color: #1d4ed8;
    }

    .directory-stream-id {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .directory-stream-count {
      color: var(--muted);
      font-size: 11px;
      font-weight: 900;
      white-space: nowrap;
      flex: 0 0 auto;
    }

    .stream-groups {
      min-width: 0;
    }

    .product-group {
      margin-top: 18px;
      padding-top: 6px;
      scroll-margin-top: 18px;
    }

    .product-group:first-child { margin-top: 0; padding-top: 0; }

    .product-header {
      position: sticky;
      top: 0;
      z-index: 4;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin: 0 0 12px;
      padding: 14px 16px;
      border-radius: var(--radius-md);
      background: linear-gradient(135deg, #0f172a, #1e3a8a);
      color: #fff;
      box-shadow: 0 12px 24px rgba(15, 23, 42, 0.12);
    }

    .product-header h3 {
      margin: 0;
      font-size: 18px;
      letter-spacing: -0.02em;
      overflow-wrap: anywhere;
    }

    .product-header .meta {
      color: rgba(255, 255, 255, 0.78);
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }

    .stream-card {
      margin: 12px 0 18px;
      overflow: hidden;
      scroll-margin-top: 18px;
      border: 1px solid rgba(226, 232, 240, 0.95);
      border-radius: var(--radius-lg);
      background: #fff;
      box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
    }

    .stream-card.has-dishonours {
      border-color: #fecaca;
      box-shadow: 0 12px 30px rgba(220, 38, 38, 0.08);
    }

    .stream-card.has-dishonours .stream-card-header {
      background: linear-gradient(180deg, #fff 0%, #f8fafc 100%);
      box-shadow: inset 4px 0 0 #dc2626;
    }

    .stream-card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
      padding: 16px 18px;
      background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
      border-bottom: 1px solid var(--line);
    }

    .stream-title-block {
      min-width: 0;
    }

    .stream-title {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
      font-size: 16px;
      font-weight: 950;
      letter-spacing: -0.02em;
    }

    .stream-counterparty-title {
      color: #0f172a;
      font-size: clamp(20px, 2vw, 26px);
      font-weight: 950;
      line-height: 1.1;
      letter-spacing: -0.035em;
      overflow-wrap: anywhere;
    }

    .stream-id-line {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin: 8px 0 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }

    .stream-id-line strong {
      color: #334155;
      font-weight: 950;
    }

    .stream-highlight {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 8px 0 8px;
    }

    .stream-highlight-card {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 42px;
      padding: 8px 12px;
      border-radius: 14px;
      background: #fff;
      border: 1px solid #dbeafe;
      box-shadow: 0 6px 14px rgba(15, 23, 42, 0.04);
    }

    .stream-highlight-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: 0.02em;
    }

    .stream-highlight-value {
      color: #0f172a;
      font-size: 17px;
      font-weight: 950;
      letter-spacing: -0.02em;
      overflow-wrap: anywhere;
    }

    .stream-highlight-card.counterparty .stream-highlight-value {
      font-size: 18px;
      color: #1d4ed8;
    }

    .stream-status-badge {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 4px 10px;
      border-radius: 999px;
      background: #f1f5f9;
      color: #334155;
      border: 1px solid #e2e8f0;
      font-size: 13px;
      font-weight: 950;
    }

    .stream-status-badge.green { color: #047857; background: #ecfdf5; border-color: #a7f3d0; }
    .stream-status-badge.amber { color: #b45309; background: #fffbeb; border-color: #fde68a; }
    .stream-status-badge.red { color: #b91c1c; background: #fef2f2; border-color: #fecaca; }
    .stream-status-badge.blue { color: #475569; background: #f8fafc; border-color: #cbd5e1; }
    .stream-status-badge.purple { color: #6d28d9; background: #f5f3ff; border-color: #ddd6fe; }
    .stream-status-badge.gray { color: #64748b; background: #f8fafc; border-color: #e2e8f0; }

    .stream-risk-badge {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      padding: 4px 10px;
      border-radius: 999px;
      color: #b91c1c;
      background: #fef2f2;
      border: 1px solid #fecaca;
      font-size: 13px;
      font-weight: 950;
    }

    .stream-meta-line {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 750;
    }

    .stream-stats {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
      flex: 0 0 auto;
    }

    .stream-stat-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 30px;
      padding: 5px 10px;
      border-radius: 999px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      color: #334155;
      font-size: 12px;
      font-weight: 850;
    }

    .stream-stat-pill.danger {
      color: #991b1b;
      background: var(--red-soft);
      border-color: #fecaca;
    }

    .stream-stat-pill.warning {
      color: #92400e;
      background: var(--amber-soft);
      border-color: #fde68a;
    }

    .stream-body {
      padding: 16px 18px 18px;
    }

    .subsection-label {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 0 0 8px;
      color: #334155;
      font-size: 13px;
      font-weight: 950;
    }

    .stream-table-shell {
      margin-bottom: 14px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #fff;
      overflow: hidden;
    }

    .stream-table-wrap {
      overflow: auto;
    }

    .stream-table-wrap table {
      min-width: 1180px;
    }

    .stream-table-wrap.compact table {
      min-width: 980px;
    }

    .stream-table-wrap th {
      position: sticky;
      top: 0;
    }

    .stream-empty {
      padding: 18px;
      color: var(--muted);
      text-align: center;
      font-size: 13px;
      font-weight: 750;
      background: #f8fafc;
    }

    .back-to-top {
      position: fixed;
      right: 24px;
      bottom: 24px;
      z-index: 80;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      padding: 0 16px;
      border: 0;
      border-radius: 999px;
      color: #fff;
      background: linear-gradient(135deg, var(--primary), var(--primary-dark));
      box-shadow: 0 14px 28px rgba(37, 99, 235, 0.28);
      cursor: pointer;
      font-size: 13px;
      font-weight: 950;
      opacity: 0;
      pointer-events: none;
      transform: translateY(10px);
      transition: opacity 0.18s ease, transform 0.18s ease;
    }

    .back-to-top.show {
      opacity: 1;
      pointer-events: auto;
      transform: translateY(0);
    }

    @media (max-width: 1180px) {
      .hero-top { flex-direction: column; }
      .hero-meta { width: 100%; min-width: 0; }
      .kpi-grid { grid-template-columns: repeat(2, minmax(180px, 1fr)); }
      .toolbar { grid-template-columns: repeat(2, minmax(150px, 1fr)); }
      .review-toolbar { grid-template-columns: 1fr; }
      .review-toolbar .message { grid-column: 1; }
      .search-panel { grid-template-columns: 1fr auto auto; }
      .active-app-box { grid-column: 1 / -1; justify-content: flex-start; }
      .mini-summary { grid-template-columns: repeat(2, 1fr); }
      :root { --sidebar-width: 290px; --content-gap: 12px; }
    }

    @media (max-width: 1080px) {
      .dashboard-layout { display: block; }
      .dashboard-main { margin-left: 0; width: 100%; }
      .stream-directory {
        position: relative;
        left: auto;
        top: auto !important;
        width: 100%;
        max-height: 380px !important;
        margin-bottom: 14px;
      }
    }

    @media (max-width: 720px) {
      .page { width: 100%; padding: 8px; }
      .hero { padding: 12px 14px; }
      .hero-meta { grid-template-columns: 1fr; }
      .panel { padding: 16px; border-radius: 18px; }
      .section-title { align-items: flex-start; flex-direction: column; }
      .section-title .hint { text-align: left; }
      .search-panel, .toolbar, .kpi-grid, .distribution-grid, .mini-summary { grid-template-columns: 1fr; }
      .btn { width: 100%; }
      .product-header, .stream-card-header { align-items: flex-start; flex-direction: column; }
      .stream-stats { justify-content: flex-start; }
      .back-to-top { right: 14px; bottom: 14px; min-height: 40px; padding: 0 14px; }
      .tx-sticky-1, .tx-sticky-2, .tx-sticky-3 { left: auto; min-width: auto; width: auto; position: static; }
    }
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="hero-top">
        <div>
          <div class="eyebrow">● Liability Analytics</div>
          <h1>Liability Classification Review</h1>
          <p id="metaInfo">Credit review workspace by application_id</p>
        </div>
        <div class="hero-meta">
          <div class="hero-stat">
            <div class="label">Applications</div>
            <div class="value" id="heroApplicationCount">-</div>
          </div>
          <div class="hero-stat">
            <div class="label">Liability Summary Rows</div>
            <div class="value" id="heroLiabilityRows">-</div>
          </div>
          <div class="hero-stat">
            <div class="label">Transaction Rows</div>
            <div class="value" id="heroTxRows">-</div>
          </div>
        </div>
      </div>
    </header>

    <section class="panel compact review-toolbar">
      <div class="section-title">
        <h2><span class="icon-dot">⌕</span>Application Search</h2>
        <span class="hint">Search by typing or selecting an application_id. Data updates only after clicking Search.</span>
      </div>
      <div class="search-panel">
        <div>
          <label for="applicationSearch">application_id</label>
          <input id="applicationSearch" list="applicationOptions" placeholder="application_id" autocomplete="off" />
          <datalist id="applicationOptions"></datalist>
        </div>
        <button class="btn btn-primary" id="applyApplication">Search</button>
        <button class="btn btn-secondary" id="clearApplication">Clear</button>
        <div class="active-app-box">
          <span>Current Application</span>
          <span class="active-app-pill" id="activeApplicationPill">-</span>
        </div>
      </div>
      <div class="message" id="applicationMessage"></div>
    </section>

    <div class="dashboard-layout" id="dashboardLayout">
      <aside class="stream-directory dashboard-directory" id="streamDirectory"></aside>

      <main class="dashboard-main">
        <section class="panel" id="liabilityOverviewSection">
          <div class="section-title">
            <h2><span class="icon-dot">↗</span>Liability Overview</h2>
            <span class="hint" id="overviewSubtitle"></span>
          </div>
          <div class="kpi-grid" id="kpiContainer"></div>
          <div class="distribution-grid">
            <div class="dist-card">
              <div class="dist-title">finv_category Distribution</div>
              <div class="chips" id="productDistribution"></div>
            </div>
            <div class="dist-card">
              <div class="dist-title">status Distribution</div>
              <div class="chips" id="statusDistribution"></div>
            </div>
          </div>
          <div class="overview-table-block">
            <div class="overview-table-header">
              <div class="overview-table-title">Application Liability Summary</div>
              <div class="overview-table-hint">Mirrors the liability_summary sheet for the selected application. One row per detected stream.</div>
            </div>
            <div id="liabilityOverviewTable"></div>
          </div>
          <div class="mini-summary" id="miniSummary"></div>
        </section>

        <section class="panel" id="streamFlatSection">
          <div class="section-title">
            <h2><span class="icon-dot">≡</span>Liability Stream Summary <span class="count-badge" id="streamGroupCount">0</span></h2>
            <span class="hint">Grouped by finv_category and stream_id. Each stream shows its summary followed by its own transaction details.</span>
          </div>
          <div class="stream-groups" id="groupedLiabilityStreams"></div>
        </section>

        <section class="panel" id="allBankTransactionsSection">
          <div class="section-title">
            <h2><span class="icon-dot">≋</span>All Bank Transactions <span class="count-badge" id="allTxRowCount">0</span></h2>
            <span class="hint">All transaction rows for the selected application, including rows not attached to a liability stream.</span>
          </div>
          <div class="quick-filter-row" id="allTxQuickFilters">
            <button class="quick-filter-btn active" type="button" data-quick-filter="all">All</button>
            <button class="quick-filter-btn" type="button" data-quick-filter="dishonours">Dishonours only</button>
            <button class="quick-filter-btn" type="button" data-quick-filter="debits">Debits only</button>
            <button class="quick-filter-btn" type="button" data-quick-filter="credits">Credits only</button>
            <button class="quick-filter-btn" type="button" data-quick-filter="ongoing">Ongoing streams</button>
            <button class="quick-filter-btn" type="button" data-quick-filter="bnpl">BNPL</button>
            <button class="quick-filter-btn" type="button" data-quick-filter="wage_advance">Wage advance</button>
          </div>
          <div class="toolbar" id="allTxToolbar">
            <div>
              <label for="allTxStreamFilter">stream_id</label>
              <select id="allTxStreamFilter"></select>
            </div>
            <div>
              <label for="allTxProductFilter">finv_category</label>
              <select id="allTxProductFilter"></select>
            </div>
            <div>
              <label for="allTxDrCrFilter">dr_cr</label>
              <select id="allTxDrCrFilter"></select>
            </div>
            <div>
              <label for="allTxCategoryFilter">category</label>
              <select id="allTxCategoryFilter"></select>
            </div>
            <div>
              <label for="allTxCounterpartyFilter">counterparty</label>
              <select id="allTxCounterpartyFilter"></select>
            </div>
            <div>
              <label for="allTxTextSearch">text Search</label>
              <input id="allTxTextSearch" placeholder="Search text / third_party / counterparty" />
            </div>
            <div>
              <label>&nbsp;</label>
              <button class="btn btn-secondary" id="resetAllTxFilters">Reset Filters</button>
            </div>
          </div>
          <div id="allBankTransactionsTable"></div>
        </section>
      </main>
    </div>
  </div>

  <button class="back-to-top" id="backToTop" type="button">↑ Back to Top</button>

  <script>
    const DATA = __DATA_JSON__;

    const overviewColumns = [
      { key: 'counterparty', label: 'counterparty', type: 'streamJump' },
      { key: 'finv_category', label: 'finv_category', type: 'product' },
      { key: 'stream_id', label: 'stream_id', type: 'tag' },
      { key: 'bank_account_id', label: 'bank_account_id', type: 'tag' },
      { key: 'account_type', label: 'account_type' },
      { key: 'bank', label: 'bank' },
      { key: 'credit_limit', label: 'credit_limit', type: 'amount' },
      { key: 'transaction_start_date', label: 'transaction_start_date', type: 'date' },
      { key: 'transaction_end_date', label: 'transaction_end_date', type: 'date' },
      { key: 'status', label: 'status', type: 'status' },
      { key: 'funded_amount', label: 'funded_amount', type: 'amount' },
      { key: 'repaid_amount', label: 'repaid_amount', type: 'amount' },
      { key: 'repayment_amount', label: 'repayment_amount', type: 'amount' },
      { key: 'recent_fn_repay_amount', label: 'recent_fn_repay_amount', type: 'amount' },
      { key: 'frequency', label: 'frequency' },
      { key: 'frequency_day', label: 'frequency_day' },
      { key: 'predicted_closing_date', label: 'predicted_closing_date', type: 'date' }
    ];

    const summaryColumns = [
      { key: 'stream_id', label: 'stream_id', type: 'tag' },
      { key: 'bank_account_id', label: 'bank_account_id', type: 'tag' },
      { key: 'account_type', label: 'account_type' },
      { key: 'bank', label: 'bank' },
      { key: 'credit_limit', label: 'credit_limit', type: 'amount' },
      { key: 'finv_category', label: 'finv_category', type: 'product' },
      { key: 'counterparty', label: 'counterparty' },
      { key: 'transaction_start_date', label: 'transaction_start_date', type: 'date' },
      { key: 'transaction_end_date', label: 'transaction_end_date', type: 'date' },
      { key: 'status', label: 'status', type: 'status' },
      { key: 'funded_amount', label: 'funded_amount', type: 'amount' },
      { key: 'repaid_amount', label: 'repaid_amount', type: 'amount' },
      { key: 'repayment_amount', label: 'repayment_amount', type: 'amount' },
      { key: 'recent_fn_repay_amount', label: 'recent_fn_repay_amount', type: 'amount' },
      { key: 'frequency', label: 'frequency' },
      { key: 'frequency_day', label: 'frequency_day' },
      { key: 'predicted_closing_date', label: 'predicted_closing_date', type: 'date' }
    ];

    const txColumns = [
      { key: 'transaction_date', label: 'transaction_date', type: 'date' },
      { key: 'amount', label: 'amount', type: 'amount' },
      { key: 'counterparty', label: 'counterparty' },
      { key: 'dr_cr', label: 'dr_cr', type: 'drcr' },
      { key: 'category', label: 'category' },
      { key: 'finv_category', label: 'finv_category', type: 'product' },
      { key: 'stream_id', label: 'stream_id', type: 'tag' },
      { key: 'bank_account_id', label: 'bank_account_id', type: 'tag' },
      { key: 'account_type', label: 'account_type' },
      { key: 'bank', label: 'bank' },
      { key: 'credit_limit', label: 'credit_limit', type: 'amount' },
      { key: 'is_dishonours', label: 'is_dishonours', type: 'flag' },
      { key: 'balance', label: 'balance', type: 'amount' },
      { key: 'text', label: 'text', type: 'text' },
      { key: 'third_party', label: 'third_party' },
      { key: 'product_type', label: 'product_type', type: 'product' }
    ];

    const summarySort = { key: 'stream_id', dir: 'asc' };
    const txSort = { key: 'transaction_date', dir: 'desc' };
    const EMPTY_LABEL = 'N/A';

    let allApplicationIds = [];
    let activeApplicationId = '';
    let activeQuickFilter = 'all';
    let scrollSpyTimer = null;
    let scrollSpyTargetsCache = [];
    let directoryLinksCache = [];
    let lastActiveDirectoryTarget = '';
    let activeLiabilities = [];
    let activeTxs = [];
    let applicationIdSet = new Set();
    let liabilityRowsByApplication = new Map();
    let txRowsByApplication = new Map();
    let activeLiabilityStatusByStream = new Map();
    const SCROLL_SPY_DELAY_MS = 160;

    function norm(value) {
      return String(value ?? '').trim();
    }

    function isEmpty(value) {
      return norm(value) === '';
    }

    function sameId(a, b) {
      return norm(a) === norm(b);
    }

    function escapeHtml(value) {
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }

    function uniqueValues(rows, key) {
      const values = rows.map(r => norm(r[key])).filter(v => v !== '');
      return Array.from(new Set(values)).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    }

    function fillSelect(id, values, allLabel) {
      const select = document.getElementById(id);
      select.innerHTML = '';
      select.appendChild(new Option(allLabel, ''));
      values.forEach(v => select.appendChild(new Option(v, v)));
    }

    function addRowToAppIndex(indexMap, row) {
      const appId = norm(row.application_id);
      if (!appId) return;
      applicationIdSet.add(appId);
      if (!indexMap.has(appId)) indexMap.set(appId, []);
      indexMap.get(appId).push(row);
    }

    function buildAllTxSearchText(row) {
      return [
        row.text,
        row.third_party,
        row.counterparty,
        row.category,
        row.product_type,
        row.finv_category,
        row.stream_id
      ].map(value => norm(value).toLowerCase()).join(' | ');
    }

    function buildDataIndexes() {
      liabilityRowsByApplication = new Map();
      txRowsByApplication = new Map();
      applicationIdSet = new Set();

      DATA.liabilitySummary.forEach(row => addRowToAppIndex(liabilityRowsByApplication, row));
      DATA.transactions.forEach(row => {
        row.__search_text = buildAllTxSearchText(row);
        addRowToAppIndex(txRowsByApplication, row);
      });
    }

    function rebuildActiveRows() {
      activeLiabilities = liabilityRowsByApplication.get(activeApplicationId) || [];
      activeTxs = txRowsByApplication.get(activeApplicationId) || [];

      const rowsByStream = new Map();
      activeLiabilities.forEach(row => {
        const sid = norm(row.stream_id);
        if (!sid) return;
        if (!rowsByStream.has(sid)) rowsByStream.set(sid, []);
        rowsByStream.get(sid).push(row);
      });

      activeLiabilityStatusByStream = new Map();
      rowsByStream.forEach((rows, sid) => {
        activeLiabilityStatusByStream.set(sid, mostCommonValue(rows, 'status'));
      });
    }

    function getLiabilityRowsForApp() {
      return activeLiabilities;
    }

    function getTxRowsForApp() {
      return activeTxs;
    }

    function parseNumber(value) {
      if (value === null || value === undefined || value === '') return 0;
      const n = Number(String(value).replace(/,/g, ''));
      return Number.isFinite(n) ? n : 0;
    }

    function formatInteger(value) {
      const n = Number(value);
      if (!Number.isFinite(n)) return escapeHtml(value);
      return n.toLocaleString(undefined, { maximumFractionDigits: 0 });
    }

    function formatAmount(value) {
      if (value === null || value === undefined || value === '') return EMPTY_LABEL;
      const n = Number(String(value).replace(/,/g, ''));
      if (!Number.isFinite(n)) return escapeHtml(value);
      return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function formatDate(value) {
      const text = norm(value);
      if (!text) return EMPTY_LABEL;
      if (/^\d{4}-\d{2}-\d{2}/.test(text)) return text.slice(0, 10);
      const d = new Date(text);
      if (!Number.isNaN(d.getTime())) return d.toISOString().slice(0, 10);
      return text;
    }

    function truthyFlag(value) {
      const text = norm(value).toLowerCase();
      return ['1', 'true', 'yes', 'y', 'dishonour', 'dishonours'].includes(text);
    }

    function tagClass(value, type) {
      const text = norm(value).toLowerCase();
      if (type === 'status') {
        if (text.includes('ongoing')) return 'green';
        if (text.includes('closing')) return 'amber';
        if (text.includes('closed')) return 'blue';
        if (text === '' || text === 'n/a') return 'gray';
        return 'purple';
      }
      if (type === 'product') {
        if (text.includes('sacc')) return 'purple';
        if (text.includes('bnpl')) return 'blue';
        if (text.includes('bank')) return 'green';
        if (text.includes('unknown')) return 'amber';
        return '';
      }
      if (type === 'drcr') {
        if (text === 'cr' || text === 'credit') return 'green';
        if (text === 'dr' || text === 'debit') return 'red';
        return '';
      }
      if (type === 'flag') {
        return truthyFlag(value) ? 'red' : 'green';
      }
      return '';
    }

    function formatCell(row, col, tableKind) {
      const value = row[col.key];
      const displayValue = col.type === 'flag'
        ? (truthyFlag(value) ? 'Yes' : 'No')
        : (isEmpty(value) ? EMPTY_LABEL : value);

      if (col.type === 'amount') return formatAmount(value);
      if (col.type === 'date') return formatDate(value);
      if (col.type === 'streamJump') {
        const text = isEmpty(value) ? EMPTY_LABEL : value;
        const streamId = norm(row.stream_id);
        if (!streamId) return escapeHtml(text);
        const targetId = safeAnchorId('stream', streamId);
        const cls = tagClass(row.finv_category, 'product');
        return `<a class="table-jump-link tag ${cls}" href="#${targetId}">${escapeHtml(text)}</a>`;
      }
      if (col.type === 'text') {
        const text = isEmpty(value) ? EMPTY_LABEL : value;
        return `<span class="text-clamp" title="${escapeHtml(text)}">${escapeHtml(text)}</span>`;
      }
      if (col.type === 'status' || col.type === 'product' || col.type === 'drcr' || col.type === 'tag' || col.type === 'flag') {
        const cls = tagClass(value, col.type);
        return `<span class="tag ${cls}">${escapeHtml(displayValue)}</span>`;
      }
      if (value === null || value === undefined || value === '') return EMPTY_LABEL;
      return escapeHtml(value);
    }

    function countDistinct(rows, key) {
      return uniqueValues(rows, key).length;
    }

    function countByMap(rows, key) {
      const result = new Map();
      rows.forEach(row => {
        const value = norm(row[key]) || EMPTY_LABEL;
        result.set(value, (result.get(value) || 0) + 1);
      });
      return Array.from(result.entries())
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], undefined, { numeric: true }));
    }

    function sumBy(rows, key) {
      return rows.reduce((sum, row) => sum + parseNumber(row[key]), 0);
    }

    function minDate(rows, key) {
      const dates = rows.map(r => Date.parse(r[key])).filter(t => Number.isFinite(t));
      if (!dates.length) return EMPTY_LABEL;
      return new Date(Math.min(...dates)).toISOString().slice(0, 10);
    }

    function maxDate(rows, key) {
      const dates = rows.map(r => Date.parse(r[key])).filter(t => Number.isFinite(t));
      if (!dates.length) return EMPTY_LABEL;
      return new Date(Math.max(...dates)).toISOString().slice(0, 10);
    }

    function renderDistribution(containerId, rows, key, type) {
      const container = document.getElementById(containerId);
      const data = countByMap(rows, key);
      if (!data.length) {
        container.innerHTML = `<span class="muted">${EMPTY_LABEL}</span>`;
        return;
      }
      container.innerHTML = data.map(([label, count]) => {
        const cls = tagClass(label, type);
        return `<span class="chip ${cls}"><span>${escapeHtml(label)}</span><span class="num">${formatInteger(count)}</span></span>`;
      }).join('');
    }

    function renderKpis() {
      const liabilities = getLiabilityRowsForApp();
      const txs = getTxRowsForApp();
      const baseRows = liabilities.length ? liabilities : txs;
      const streamCount = countDistinct(baseRows, 'stream_id');
      const counterpartyCount = countDistinct(liabilities.concat(txs), 'counterparty');
      const hasDishonours = txs.some(row => truthyFlag(row.is_dishonours));

      const cards = [
        { label: 'Liability Stream Count', value: formatInteger(streamCount), note: 'distinct stream_id', cls: 'purple' },
        { label: 'Counterparty Count', value: formatInteger(counterpartyCount), note: 'distinct counterparty', cls: '' },
        { label: 'Total funded_amount', value: formatAmount(sumBy(liabilities, 'funded_amount')), note: 'liability_summary total', cls: 'money' },
        { label: 'Total repaid_amount', value: formatAmount(sumBy(liabilities, 'repaid_amount')), note: 'liability_summary total', cls: 'money' },
        { label: 'recent_fn_repay_amount', value: formatAmount(sumBy(liabilities, 'recent_fn_repay_amount')), note: 'recent repayment total', cls: 'money' },
        { label: 'Has Dishonours', value: hasDishonours ? 'Yes' : 'No', note: 'from transaction details', cls: hasDishonours ? 'danger' : 'money' },
        { label: 'Liability Start Date', value: minDate(liabilities, 'transaction_start_date'), note: 'earliest transaction_start_date', cls: '' },
        { label: 'Liability End Date', value: maxDate(liabilities, 'transaction_end_date'), note: 'latest transaction_end_date', cls: '' }
      ];

      document.getElementById('overviewSubtitle').textContent =
        `liability_summary: ${formatInteger(liabilities.length)} rows / transactions: ${formatInteger(txs.length)} rows`;

      document.getElementById('kpiContainer').innerHTML = cards.map(card => `
        <div class="kpi-card ${card.cls}">
          <div class="kpi-label">${escapeHtml(card.label)}</div>
          <div class="kpi-value">${card.value}</div>
          <div class="kpi-footnote">${escapeHtml(card.note)}</div>
        </div>
      `).join('');

      renderDistribution('productDistribution', baseRows, 'finv_category', 'product');
      renderDistribution('statusDistribution', liabilities, 'status', 'status');

      document.getElementById('miniSummary').innerHTML = [
        { label: 'Application ID', value: activeApplicationId || EMPTY_LABEL },
        { label: 'Transaction Date Range', value: `${minDate(txs, 'transaction_date')} → ${maxDate(txs, 'transaction_date')}` },
        { label: 'Debit Amount', value: formatAmount(sumBy(txs.filter(r => norm(r.dr_cr).toLowerCase().startsWith('d')), 'amount')) },
        { label: 'Credit Amount', value: formatAmount(sumBy(txs.filter(r => norm(r.dr_cr).toLowerCase().startsWith('c')), 'amount')) }
      ].map(item => `
        <div class="mini-item">
          <div class="label">${escapeHtml(item.label)}</div>
          <div class="value">${escapeHtml(item.value)}</div>
        </div>
      `).join('');
    }

    function comparable(value, col) {
      if (col?.type === 'amount') return parseNumber(value);
      if (col?.type === 'date') {
        const time = Date.parse(value);
        return Number.isNaN(time) ? 0 : time;
      }
      return norm(value).toLowerCase();
    }

    function isPersonalLoanUnknown(value) {
      return norm(value).toLowerCase() === 'personal_loan_unknown';
    }

    function productComparator(a, b) {
      const au = isPersonalLoanUnknown(a);
      const bu = isPersonalLoanUnknown(b);
      if (au && !bu) return 1;
      if (!au && bu) return -1;
      return norm(a).localeCompare(norm(b), undefined, { numeric: true });
    }

    function liabilityProductRank(row) {
      return isPersonalLoanUnknown(row.finv_category) ? 1 : 0;
    }

    function sortRows(rows, columns, sortState, tableKind) {
      const col = columns.find(c => c.key === sortState.key);
      const multiplier = sortState.dir === 'asc' ? 1 : -1;
      return [...rows].sort((a, b) => {
        // Liability Stream Summary should always keep personal_loan_unknown at the bottom.
        if (tableKind === 'liability') {
          const ar = liabilityProductRank(a);
          const br = liabilityProductRank(b);
          if (ar !== br) return ar - br;
        }

        const av = comparable(a[sortState.key], col);
        const bv = comparable(b[sortState.key], col);
        if (av < bv) return -1 * multiplier;
        if (av > bv) return 1 * multiplier;
        return 0;
      });
    }

    function columnClass(col, index, tableKind) {
      const classes = [];
      if (col.type === 'text') classes.push('text-cell');
      if (col.type === 'amount') classes.push('amount-col');
      if (tableKind === 'tx' && index <= 2) {
        classes.push('tx-sticky-col', `tx-sticky-${index + 1}`);
      }
      return classes.join(' ');
    }

    function tableHtml(rows, columns, sortState, tableKind, compact = false) {
      if (!rows.length) {
        return '<div class="stream-empty">No matching data</div>';
      }

      const sortedRows = sortRows(rows, columns, sortState, tableKind);
      const headerHtml = columns.map((col, index) => {
        const className = columnClass(col, index, tableKind);
        return `<th${className ? ` class="${className}"` : ''}>${escapeHtml(col.label)}</th>`;
      }).join('');
      const bodyHtml = sortedRows.map(row => {
        const cells = columns.map((col, index) => {
          const className = columnClass(col, index, tableKind);
          return `<td${className ? ` class="${className}"` : ''}>${formatCell(row, col, tableKind)}</td>`;
        }).join('');
        return `<tr>${cells}</tr>`;
      }).join('');

      const compactClass = compact ? ' compact' : '';
      return `
        <div class="stream-table-shell">
          <div class="stream-table-wrap${compactClass}">
            <table><thead><tr>${headerHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>
          </div>
        </div>
      `;
    }

    function mostCommonValue(rows, key) {
      const counts = new Map();
      rows.forEach(row => {
        const value = norm(row[key]);
        if (!value) return;
        counts.set(value, (counts.get(value) || 0) + 1);
      });
      if (!counts.size) return '';
      return Array.from(counts.entries())
        .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], undefined, { numeric: true }))[0][0];
    }

    function nonEmptyJoined(rows, key, limit = 4) {
      const values = uniqueValues(rows, key);
      if (!values.length) return EMPTY_LABEL;
      if (values.length <= limit) return values.join(', ');
      return `${values.slice(0, limit).join(', ')} +${values.length - limit} more`;
    }

    function firstNonEmptyValue(rows, key) {
      const row = rows.find(item => !isEmpty(item[key]));
      return row ? row[key] : '';
    }

    function getStreamAccountMeta(group) {
      const rows = group.liabilities.concat(group.txs);
      return {
        bankAccountId: nonEmptyJoined(rows, 'bank_account_id', 2),
        accountType: nonEmptyJoined(rows, 'account_type', 2),
        bank: nonEmptyJoined(rows, 'bank', 2),
        creditLimit: formatAmount(firstNonEmptyValue(rows, 'credit_limit'))
      };
    }


    function renderLiabilityOverviewTable() {
      const liabilities = getLiabilityRowsForApp();
      const subtitle = liabilities.length
        ? `${formatInteger(liabilities.length)} liability_summary rows for application_id ${activeApplicationId}`
        : `No liability_summary rows for application_id ${activeApplicationId || EMPTY_LABEL}`;

      document.querySelector('.overview-table-hint').textContent = subtitle;
      document.getElementById('liabilityOverviewTable').innerHTML = tableHtml(liabilities, overviewColumns, summarySort, 'liability', true);
    }

    function safeAnchorId(prefix, value) {
      const slug = norm(value)
        .toLowerCase()
        .replace(/[^a-z0-9_-]+/g, '-')
        .replace(/^-+|-+$/g, '') || 'blank';
      return `${prefix}-${slug}`;
    }

    function getGroupedStreamsForApp() {
      const liabilities = getLiabilityRowsForApp();
      const txs = getTxRowsForApp();
      const streamMap = new Map();

      function ensureStream(streamId) {
        const sid = norm(streamId) || 'NO_STREAM_ID';
        if (!streamMap.has(sid)) {
          streamMap.set(sid, {
            stream_id: sid,
            finv_category: '',
            liabilities: [],
            txs: []
          });
        }
        return streamMap.get(sid);
      }

      liabilities.forEach(row => {
        const group = ensureStream(row.stream_id);
        group.liabilities.push(row);
        if (!group.finv_category && norm(row.finv_category)) {
          group.finv_category = norm(row.finv_category);
        }
      });

      txs.forEach(row => {
        const group = ensureStream(row.stream_id);
        group.txs.push(row);
      });

      streamMap.forEach(group => {
        if (!group.finv_category) {
          group.finv_category = mostCommonValue(group.txs, 'finv_category') || 'Unknown product type';
        }
      });

      // Keep the Liability Stream Summary focused on real detected liability streams.
      // Transactions without stream_id are still available in the All Bank Transactions section,
      // but are not shown as an artificial "Unknown product type / NO_STREAM_ID" group.
      const visibleGroups = Array.from(streamMap.values()).filter(group => norm(group.stream_id) !== 'NO_STREAM_ID');

      const productMap = new Map();
      visibleGroups.forEach(group => {
        const product = group.finv_category || 'Unknown product type';
        if (!productMap.has(product)) productMap.set(product, []);
        productMap.get(product).push(group);
      });

      const products = Array.from(productMap.keys()).sort(productComparator);
      products.forEach(product => {
        productMap.get(product).sort((a, b) => norm(a.stream_id).localeCompare(norm(b.stream_id), undefined, { numeric: true }));
      });

      return { products, productMap, totalStreams: visibleGroups.length, liabilityRows: liabilities.length, txRows: txs.length };
    }

    function getStreamDisplayMeta(group) {
      const accountMeta = getStreamAccountMeta(group);
      return {
        counterparty: nonEmptyJoined(group.liabilities.concat(group.txs), 'counterparty', 2),
        status: nonEmptyJoined(group.liabilities, 'status', 2),
        accountMeta
      };
    }

    function groupHasDishonours(group) {
      return group.txs.some(row => truthyFlag(row.is_dishonours));
    }

    function groupPrimaryStatus(group) {
      return mostCommonValue(group.liabilities, 'status') || '';
    }

    function statusVisualClass(status) {
      const text = norm(status).toLowerCase();
      if (!text || text === 'n/a') return 'gray';
      if (text.includes('ongoing')) return 'green';
      if (text.includes('closed')) return 'blue';
      if (text.includes('closing')) return 'amber';
      return 'amber';
    }

    function renderStreamDirectory(products, productMap) {
      const container = document.getElementById('streamDirectory');
      if (!products.length) {
        container.innerHTML = `
          <div class="directory-current-app">
            <span>Current Application</span>
            <strong>${escapeHtml(activeApplicationId || EMPTY_LABEL)}</strong>
          </div>
          <div class="directory-title">
            <span>Directory</span>
            <span class="small">0 streams</span>
          </div>
          <div class="directory-actions">
            <a href="#allBankTransactionsSection" data-nav-target="allBankTransactionsSection" data-section-nav="allBankTransactionsSection">All Bank Transactions</a>
          </div>
          <div class="empty">No stream data for this application</div>
        `;
        return;
      }

      const totalStreams = products.reduce((sum, product) => sum + productMap.get(product).length, 0);
      container.innerHTML = `
        <div class="directory-current-app">
          <span>Current Application</span>
          <strong>${escapeHtml(activeApplicationId || EMPTY_LABEL)}</strong>
        </div>
        <div class="directory-title">
          <span>Directory</span>
          <span class="small">${formatInteger(totalStreams)} streams</span>
        </div>
        <div class="directory-actions">
          <a href="#liabilityOverviewSection" data-nav-target="liabilityOverviewSection" data-section-nav="liabilityOverviewSection">Liability Overview</a>
          <a href="#streamFlatSection" data-nav-target="streamFlatSection" data-section-nav="streamFlatSection">Top of streams</a>
          <a href="#allBankTransactionsSection" data-nav-target="allBankTransactionsSection" data-section-nav="allBankTransactionsSection">All Bank Transactions</a>
        </div>
        ${products.map(product => {
          const productId = safeAnchorId('product', product);
          const streams = productMap.get(product);
          return `
            <div class="directory-product">
              <a href="#${productId}" data-nav-target="${productId}" data-product-nav="${escapeHtml(product)}">
                <span>${escapeHtml(product)}</span>
                <span class="directory-product-count">${formatInteger(streams.length)}</span>
              </a>
              <div class="directory-streams">
                ${streams.map(group => {
                  const meta = getStreamDisplayMeta(group);
                  const statusClass = statusVisualClass(groupPrimaryStatus(group));
                  const riskBadge = groupHasDishonours(group)
                    ? '<span class="directory-risk-badge">Dishonour</span>'
                    : '';
                  return `
                    <a href="#${safeAnchorId('stream', group.stream_id)}" data-nav-target="${safeAnchorId('stream', group.stream_id)}" data-stream-nav="${escapeHtml(group.stream_id)}" data-product-nav="${escapeHtml(product)}">
                      <div class="directory-stream-top">
                        <span class="directory-risk-dot ${statusClass}" title="Status: ${escapeHtml(meta.status)}"></span>
                        <span class="directory-stream-id">${escapeHtml(group.stream_id)}</span>
                        <span class="directory-stream-count">${formatInteger(group.txs.length)} tx</span>
                      </div>
                      <div class="directory-stream-meta">
                        <span class="directory-counterparty">${escapeHtml(meta.counterparty)}</span>
                        <span> · </span>
                        <span class="directory-status ${statusClass}">${escapeHtml(meta.status)}</span>
                        ${riskBadge}
                      </div>
                    </a>
                  `;
                }).join('')}
              </div>
            </div>
          `;
        }).join('')}
      `;
    }

    function getDirectoryLinks() {
      if (directoryLinksCache.length) return directoryLinksCache;
      return Array.from(document.querySelectorAll('#streamDirectory a[data-nav-target]'));
    }

    function ensureDirectoryLinkVisible(link, directory) {
      // The sidebar scrollbar is visually hidden. Keep the auto-adjustment minimal and
      // only move the directory when the active item is outside the visible area.
      if (directory.scrollHeight <= directory.clientHeight + 2) return;

      const linkTop = link.offsetTop;
      const linkBottom = linkTop + link.offsetHeight;
      const visibleTop = directory.scrollTop + 72;
      const visibleBottom = directory.scrollTop + directory.clientHeight - 24;

      if (linkTop < visibleTop) {
        directory.scrollTop = Math.max(0, linkTop - 72);
      } else if (linkBottom > visibleBottom) {
        directory.scrollTop = linkBottom - directory.clientHeight + 24;
      }
    }

    function setActiveDirectoryTarget(targetId, shouldScrollIntoView = true) {
      const directory = document.getElementById('streamDirectory');
      if (!directory || !targetId) return;
      const links = getDirectoryLinks();
      const activeLink = links.find(link => link.dataset.navTarget === targetId);
      if (!activeLink) return;

      if (lastActiveDirectoryTarget === targetId && activeLink.classList.contains('active')) return;
      lastActiveDirectoryTarget = targetId;

      links.forEach(link => {
        link.classList.remove('active');
        link.removeAttribute('aria-current');
      });
      directory.querySelectorAll('.directory-product').forEach(item => item.classList.remove('contains-active'));

      activeLink.classList.add('active');
      activeLink.setAttribute('aria-current', 'location');

      const parentProduct = activeLink.closest('.directory-product');
      if (parentProduct) parentProduct.classList.add('contains-active');

      if (shouldScrollIntoView) {
        ensureDirectoryLinkVisible(activeLink, directory);
      }
    }

    function refreshScrollSpyTargets() {
      const targetElements = [
        document.getElementById('liabilityOverviewSection'),
        document.getElementById('streamFlatSection'),
        ...Array.from(document.querySelectorAll('.product-group')),
        ...Array.from(document.querySelectorAll('.stream-card')),
        document.getElementById('allBankTransactionsSection')
      ].filter(Boolean);

      scrollSpyTargetsCache = targetElements
        .map(element => ({ id: element.id, top: element.offsetTop }))
        .sort((a, b) => a.top - b.top);
    }

    function updateDirectoryHighlightFromScroll() {
      if (!scrollSpyTargetsCache.length) refreshScrollSpyTargets();
      if (!scrollSpyTargetsCache.length) return;

      const probeY = window.scrollY + Math.max(110, Math.round(window.innerHeight * 0.16));
      let activeTarget = scrollSpyTargetsCache[0];

      // Cached offsetTop is much cheaper than calling getBoundingClientRect on every
      // stream card during every scroll frame.
      for (const target of scrollSpyTargetsCache) {
        if (target.top <= probeY) activeTarget = target;
        else break;
      }

      if (activeTarget) {
        setActiveDirectoryTarget(activeTarget.id, true);
      }
    }

    function scheduleDirectoryScrollSpy() {
      if (scrollSpyTimer) return;
      scrollSpyTimer = window.setTimeout(() => {
        scrollSpyTimer = null;
        updateDirectoryHighlightFromScroll();
      }, SCROLL_SPY_DELAY_MS);
    }

    function handleDirectoryResize() {
      updateFixedSidebarPosition();
      refreshScrollSpyTargets();
      updateDirectoryHighlightFromScroll();
    }


    function updateFixedSidebarPosition() {
      const directory = document.getElementById('streamDirectory');
      const layout = document.getElementById('dashboardLayout');
      if (!directory || !layout) return;

      if (window.innerWidth <= 1080) {
        directory.style.top = '';
        directory.style.maxHeight = '';
        return;
      }

      const minTop = 10;
      const bottomGap = 10;
      const layoutTop = layout.getBoundingClientRect().top;
      const top = Math.max(minTop, Math.round(layoutTop));
      directory.style.top = `${top}px`;
      directory.style.maxHeight = `calc(100vh - ${top + bottomGap}px)`;
    }

    function initDirectoryNavigation() {
      directoryLinksCache = Array.from(document.querySelectorAll('#streamDirectory a[data-nav-target]'));
      lastActiveDirectoryTarget = '';
      const links = getDirectoryLinks();
      if (!links.length) return;

      links.forEach(link => {
        link.addEventListener('click', () => {
          setActiveDirectoryTarget(link.dataset.navTarget, false);
        });
      });

      if (scrollSpyTimer) {
        window.clearTimeout(scrollSpyTimer);
        scrollSpyTimer = null;
      }
      window.removeEventListener('scroll', scheduleDirectoryScrollSpy);
      window.removeEventListener('scroll', updateFixedSidebarPosition);
      window.removeEventListener('resize', handleDirectoryResize);
      window.addEventListener('scroll', scheduleDirectoryScrollSpy, { passive: true });
      window.addEventListener('scroll', updateFixedSidebarPosition, { passive: true });
      window.addEventListener('resize', handleDirectoryResize);
      updateFixedSidebarPosition();
      refreshScrollSpyTargets();
      updateDirectoryHighlightFromScroll();
    }

    function renderStreamCard(group) {
      const meta = getStreamDisplayMeta(group);
      const counterparty = meta.counterparty;
      const status = meta.status;
      const primaryStatus = groupPrimaryStatus(group);
      const debitTotal = sumBy(group.txs.filter(r => norm(r.dr_cr).toLowerCase().startsWith('d')), 'amount');
      const creditTotal = sumBy(group.txs.filter(r => norm(r.dr_cr).toLowerCase().startsWith('c')), 'amount');
      const hasDishonours = groupHasDishonours(group);
      const streamId = safeAnchorId('stream', group.stream_id);
      const cardClass = hasDishonours ? 'stream-card has-dishonours' : 'stream-card';
      const riskBadge = hasDishonours ? '<span class="stream-risk-badge">Dishonours</span>' : '';

      return `
        <article class="${cardClass}" id="${streamId}">
          <div class="stream-card-header">
            <div class="stream-title-block">
              <div class="stream-title">
                <span class="stream-counterparty-title">${escapeHtml(counterparty)}</span>
                <span class="stream-status-badge ${statusVisualClass(primaryStatus)}">${escapeHtml(status)}</span>
                ${riskBadge}
                <span class="tag ${tagClass(group.finv_category, 'product')}">${escapeHtml(group.finv_category)}</span>
              </div>
              <div class="stream-id-line">
                <span>stream_id: <strong>${escapeHtml(group.stream_id)}</strong></span>
                <span>·</span>
                <span>${formatInteger(group.txs.length)} transactions</span>
                <span>·</span>
                <span>date range: ${escapeHtml(`${minDate(group.txs, 'transaction_date')} → ${maxDate(group.txs, 'transaction_date')}`)}</span>
              </div>
              <div class="stream-highlight">
                <div class="stream-highlight-card">
                  <span class="stream-highlight-label">bank_account_id</span>
                  <span class="stream-highlight-value">${escapeHtml(meta.accountMeta.bankAccountId)}</span>
                </div>
                <div class="stream-highlight-card">
                  <span class="stream-highlight-label">account_type</span>
                  <span class="stream-highlight-value">${escapeHtml(meta.accountMeta.accountType)}</span>
                </div>
                <div class="stream-highlight-card">
                  <span class="stream-highlight-label">credit_limit</span>
                  <span class="stream-highlight-value">${escapeHtml(meta.accountMeta.creditLimit)}</span>
                </div>
              </div>
            </div>
            <div class="stream-stats">
              <span class="stream-stat-pill">Summary rows: ${formatInteger(group.liabilities.length)}</span>
              <span class="stream-stat-pill">Debit: ${formatAmount(debitTotal)}</span>
              <span class="stream-stat-pill">Credit: ${formatAmount(creditTotal)}</span>
              <span class="stream-stat-pill ${hasDishonours ? 'danger' : ''}">Dishonours: ${hasDishonours ? 'Yes' : 'No'}</span>
            </div>
          </div>
          <div class="stream-body">
            <div class="subsection-label">Liability Summary</div>
            ${tableHtml(group.liabilities, summaryColumns, summarySort, 'liability', true)}
            <div class="subsection-label">Transaction Details</div>
            ${tableHtml(group.txs, txColumns, txSort, 'tx')}
          </div>
        </article>
      `;
    }

    function renderGroupedLiabilityStreams() {
      const grouped = getGroupedStreamsForApp();
      document.getElementById('streamGroupCount').textContent = `${formatInteger(grouped.totalStreams)} streams`;
      renderStreamDirectory(grouped.products, grouped.productMap);

      const container = document.getElementById('groupedLiabilityStreams');
      if (!grouped.products.length) {
        container.innerHTML = '<div class="empty">No stream data for this application</div>';
        initDirectoryNavigation();
        return;
      }

      container.innerHTML = grouped.products.map(product => {
        const groups = grouped.productMap.get(product);
        const totalTxRows = groups.reduce((sum, group) => sum + group.txs.length, 0);
        const productId = safeAnchorId('product', product);
        return `
          <section class="product-group" id="${productId}">
            <div class="product-header">
              <h3>${escapeHtml(product)}</h3>
              <div class="meta">${formatInteger(groups.length)} streams / ${formatInteger(totalTxRows)} transactions</div>
            </div>
            ${groups.map(renderStreamCard).join('')}
          </section>
        `;
      }).join('');
      initDirectoryNavigation();
    }

    function buildApplicationSearch() {
      allApplicationIds = Array.from(applicationIdSet)
        .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

      const datalist = document.getElementById('applicationOptions');
      datalist.innerHTML = '';
      allApplicationIds.forEach(id => {
        const option = document.createElement('option');
        option.value = id;
        datalist.appendChild(option);
      });

      activeApplicationId = allApplicationIds[0] || '';
      rebuildActiveRows();
      document.getElementById('applicationSearch').value = activeApplicationId;
      document.getElementById('activeApplicationPill').textContent = activeApplicationId || '-';
      document.getElementById('heroApplicationCount').textContent = formatInteger(allApplicationIds.length);
      document.getElementById('heroLiabilityRows').textContent = formatInteger(DATA.meta.liabilitySummaryRows);
      document.getElementById('heroTxRows').textContent = formatInteger(DATA.meta.transactionRows);
    }

    function clearApplicationSearch() {
      document.getElementById('applicationSearch').value = '';
      document.getElementById('applicationMessage').textContent = '';
      document.getElementById('applicationSearch').focus();
    }

    function applyApplication() {
      const message = document.getElementById('applicationMessage');
      const nextApp = norm(document.getElementById('applicationSearch').value);

      if (!nextApp) {
        message.textContent = 'Please type or select an application_id.';
        return;
      }

      if (!applicationIdSet.has(nextApp)) {
        message.textContent = `application_id not found: ${nextApp}`;
        return;
      }

      message.textContent = '';
      activeApplicationId = nextApp;
      rebuildActiveRows();
      document.getElementById('activeApplicationPill').textContent = activeApplicationId;
      activeQuickFilter = 'all';
      updateQuickFilterButtons();
      clearAllTxFilterValues();
      renderAll();
    }

    function setSelectOptionsPreserveValue(id, values, allLabel) {
      const select = document.getElementById(id);
      const previousValue = select.value;
      select.innerHTML = '';
      select.appendChild(new Option(allLabel, ''));
      values.forEach(value => select.appendChild(new Option(value, value)));
      select.value = previousValue && values.includes(previousValue) ? previousValue : '';
    }

    function getAllTxFilters() {
      return {
        stream_id: document.getElementById('allTxStreamFilter').value,
        finv_category: document.getElementById('allTxProductFilter').value,
        dr_cr: document.getElementById('allTxDrCrFilter').value,
        category: document.getElementById('allTxCategoryFilter').value,
        counterparty: document.getElementById('allTxCounterpartyFilter').value,
        text: norm(document.getElementById('allTxTextSearch').value).toLowerCase()
      };
    }

    function allTxTextHaystack(row) {
      if (row.__search_text !== undefined) return row.__search_text;
      row.__search_text = buildAllTxSearchText(row);
      return row.__search_text;
    }

    function rowMatchesAllTxFilters(row, filters, excludeKey = null) {
      if (excludeKey !== 'stream_id' && filters.stream_id && norm(row.stream_id) !== filters.stream_id) return false;
      if (excludeKey !== 'finv_category' && filters.finv_category && norm(row.finv_category) !== filters.finv_category) return false;
      if (excludeKey !== 'dr_cr' && filters.dr_cr && norm(row.dr_cr) !== filters.dr_cr) return false;
      if (excludeKey !== 'category' && filters.category && norm(row.category) !== filters.category) return false;
      if (excludeKey !== 'counterparty' && filters.counterparty && norm(row.counterparty) !== filters.counterparty) return false;
      if (excludeKey !== 'text' && filters.text && !allTxTextHaystack(row).includes(filters.text)) return false;
      return true;
    }

    function streamStatusForTxRow(row) {
      const sid = norm(row.stream_id);
      if (!sid) return '';
      return activeLiabilityStatusByStream.get(sid) || '';
    }

    function rowMatchesQuickFilter(row) {
      const filter = activeQuickFilter;
      const drCr = norm(row.dr_cr).toLowerCase();
      const product = norm(row.finv_category || row.product_type).toLowerCase();
      if (filter === 'all') return true;
      if (filter === 'dishonours') return truthyFlag(row.is_dishonours);
      if (filter === 'debits') return drCr.startsWith('d');
      if (filter === 'credits') return drCr.startsWith('c');
      if (filter === 'ongoing') return norm(streamStatusForTxRow(row)).toLowerCase().includes('ongoing');
      if (filter === 'bnpl') return product.includes('bnpl');
      if (filter === 'wage_advance') return product.includes('wage_advance') || product.includes('wage advance');
      return true;
    }

    function getFilteredAllBankTransactions() {
      const filters = getAllTxFilters();
      return getTxRowsForApp().filter(row => rowMatchesQuickFilter(row) && rowMatchesAllTxFilters(row, filters));
    }

    function rebuildAllTxFilters() {
      const txs = getTxRowsForApp();
      const filters = getAllTxFilters();
      const config = [
        { id: 'allTxStreamFilter', key: 'stream_id', label: 'All stream_id' },
        { id: 'allTxProductFilter', key: 'finv_category', label: 'All finv_category' },
        { id: 'allTxDrCrFilter', key: 'dr_cr', label: 'All dr_cr' },
        { id: 'allTxCategoryFilter', key: 'category', label: 'All category' },
        { id: 'allTxCounterpartyFilter', key: 'counterparty', label: 'All counterparty' }
      ];

      config.forEach(item => {
        // Cascading filter logic: each dropdown only shows values that still match the other active filters.
        const candidateRows = txs.filter(row => rowMatchesQuickFilter(row) && rowMatchesAllTxFilters(row, filters, item.key));
        const values = uniqueValues(candidateRows, item.key);
        setSelectOptionsPreserveValue(item.id, values, item.label);
      });
    }

    function clearAllTxFilterValues() {
      document.getElementById('allTxStreamFilter').value = '';
      document.getElementById('allTxProductFilter').value = '';
      document.getElementById('allTxDrCrFilter').value = '';
      document.getElementById('allTxCategoryFilter').value = '';
      document.getElementById('allTxCounterpartyFilter').value = '';
      document.getElementById('allTxTextSearch').value = '';
    }

    function resetAllTxFilters() {
      activeQuickFilter = 'all';
      updateQuickFilterButtons();
      clearAllTxFilterValues();
      renderAllBankTransactions();
    }

    function updateQuickFilterButtons() {
      document.querySelectorAll('#allTxQuickFilters .quick-filter-btn').forEach(button => {
        button.classList.toggle('active', button.dataset.quickFilter === activeQuickFilter);
      });
    }

    function applyQuickFilter(filter) {
      activeQuickFilter = filter || 'all';
      updateQuickFilterButtons();
      clearAllTxFilterValues();
      renderAllBankTransactions();
    }

    function renderAllBankTransactions() {
      rebuildAllTxFilters();
      const txs = getFilteredAllBankTransactions();
      document.getElementById('allTxRowCount').textContent = `${formatInteger(txs.length)} / ${formatInteger(getTxRowsForApp().length)}`;
      document.getElementById('allBankTransactionsTable').innerHTML = tableHtml(txs, txColumns, txSort, 'tx');
      refreshScrollSpyTargets();
    }

    function renderAll() {
      const liabilities = getLiabilityRowsForApp();
      const txs = getTxRowsForApp();

      renderKpis();
      renderLiabilityOverviewTable();
      renderGroupedLiabilityStreams();
      renderAllBankTransactions();
      updateFixedSidebarPosition();
      document.getElementById('overviewSubtitle').textContent =
        `liability_summary: ${formatInteger(liabilities.length)} rows / transactions: ${formatInteger(txs.length)} rows`;
      scheduleDirectoryScrollSpy();
    }

    document.getElementById('applyApplication').addEventListener('click', applyApplication);
    document.getElementById('clearApplication').addEventListener('click', clearApplicationSearch);
    document.getElementById('applicationSearch').addEventListener('keydown', event => {
      if (event.key === 'Enter') applyApplication();
    });

    ['allTxStreamFilter', 'allTxProductFilter', 'allTxDrCrFilter', 'allTxCategoryFilter', 'allTxCounterpartyFilter'].forEach(id => {
      document.getElementById(id).addEventListener('change', renderAllBankTransactions);
    });

    document.getElementById('allTxTextSearch').addEventListener('input', renderAllBankTransactions);
    document.querySelectorAll('#allTxQuickFilters .quick-filter-btn').forEach(button => {
      button.addEventListener('click', () => applyQuickFilter(button.dataset.quickFilter));
    });

    document.getElementById('resetAllTxFilters').addEventListener('click', resetAllTxFilters);

    const backToTopButton = document.getElementById('backToTop');
    window.addEventListener('scroll', () => {
      if (window.scrollY > 420) {
        backToTopButton.classList.add('show');
      } else {
        backToTopButton.classList.remove('show');
      }
    });
    backToTopButton.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    document.getElementById('metaInfo').textContent =
      `Source file: ${DATA.meta.sourceFile} | Generated at: ${DATA.meta.generatedAt}`;

    buildDataIndexes();
    buildApplicationSearch();
    renderAll();
  </script>
</body>
</html>
'''
