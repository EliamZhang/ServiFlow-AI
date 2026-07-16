# -*- coding: utf-8 -*-
"""Compare our classification output against illion labels.

Usage::

    python compare_labels.py [--output output/classification_report.xlsx]

The illion labels are ~95 % accurate but *not* ground truth —
disagreement does not always mean we are wrong.

Output (xlsx)
-------------
1. **Coverage** — who classified more rows?
2. **Confusion** — our finv_category × illion category pivot
3. **Counterparty** — coverage + exact-match rate
4. **Gaps** — rows only we / only illion classified
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = PROJECT_ROOT / "sample.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "classification_report.xlsx"
DEFAULT_RESULT = PROJECT_ROOT / "output" / "compare_result.xlsx"

# ── category name mapping: our finv_category → illion category ──────────
# Only pairs where names differ; exact name matches work automatically.

OUR_TO_ILLION: dict[str, str] = {
    "fee":                "Fees",
    "Internal Transfer":  "Internal Transfer",
    "External Transfers": "External Transfers",
    "salary_payg":        "Wages",
    "salary_packaging":   "Wages",
    "self_employed_gig":  "Wages",
    "centrelink":         "Centrelink",
    "bnpl":               "Non SACC Loans",
    "wage_advance":       "Non SACC Loans",
    "personal_loan_sacc": "SACC Loans",
    "personal_loan_non_sacc": "Non SACC Loans",
    "personal_loan_unknown":  "Non SACC Loans",
    "contract_loan":      "Non SACC Loans",
    "loc":                "Non SACC Loans",
    "bank":               "Non SACC Loans",
}

# Our categories that legitimately span MULTIPLE illion categories.
OUR_TO_ILLION_MULTI: dict[str, frozenset[str]] = {
    "Internal Transfer": frozenset({"Internal Transfer", "External Transfers",
                                     "All Other Credits", "Credit Card Repayments"}),
    "External Transfers": frozenset({"Internal Transfer", "External Transfers",
                                      "All Other Credits",
                                      "Credit Card Repayments"}),
    "bnpl":     frozenset({"Non SACC Loans", "SACC Loans"}),
    "bank":     frozenset({"Non SACC Loans", "SACC Loans",
                            "Credit Card Repayments",
                            "Internal Transfer", "External Transfers"}),
    "personal_loan_non_sacc": frozenset({"Non SACC Loans", "SACC Loans"}),
    "loc":      frozenset({"Non SACC Loans", "Credit Card Repayments"}),
}

# Columns to hide in the detail/confusion sheets.
_HIDDEN_COLS = frozenset({
    "sample_datetime", "job_id", "transaction_id", "account_type",
    "credit_limit", "trx_type", "bsb", "account_no",
    "illion_trx_uuid", "balance",
})


# ── data loading ─────────────────────────────────────────────────────────

def _load(report_path: Path) -> pd.DataFrame:
    our = pd.read_excel(report_path, sheet_name="transactions")
    illion = pd.read_csv(SAMPLE_CSV, encoding="utf-8-sig")
    df = our.copy()
    df["_illion_cat"] = illion["category"].values
    df["_illion_cp"] = illion["third_party"].values
    return df


# ── helpers ──────────────────────────────────────────────────────────────

def _map(our_cat: str) -> str:
    c = str(our_cat).strip()
    return OUR_TO_ILLION.get(c, c)


def _agree(our: str, illion_cat: str) -> bool:
    """True when our category is consistent with illion's category."""
    our, illion_cat = str(our).strip(), str(illion_cat).strip()
    if _map(our) == illion_cat:
        return True
    if our in OUR_TO_ILLION_MULTI and illion_cat in OUR_TO_ILLION_MULTI[our]:
        return True
    return False


def _is_classified(series: pd.Series) -> pd.Series:
    return (
        series.notna()
        & series.astype(str).str.strip().ne("")
        & series.astype(str).str.strip().ne("unclassified")
    )


# ── excel helpers ────────────────────────────────────────────────────────

def _auto_width(worksheet, min_width: int = 8, max_width: int = 40) -> None:
    """Set column widths based on content."""
    for col_cells in worksheet.columns:
        col_letter = col_cells[0].column_letter
        max_chars = 0
        for cell in col_cells:
            if cell.value is not None:
                max_chars = max(max_chars, len(str(cell.value)))
        width = max(min_width, min(max_chars + 2, max_width))
        worksheet.column_dimensions[col_letter].width = width


def _hide_columns(worksheet) -> None:
    """Auto-hide noisy detail columns on the transactions sheet."""
    col_map: dict[str, int] = {}
    for cell in worksheet[1]:
        if cell.value is not None:
            col_map[str(cell.value)] = cell.column

    for col_name in _HIDDEN_COLS:
        col_idx = col_map.get(col_name)
        if col_idx is not None:
            worksheet.column_dimensions[
                worksheet.cell(row=1, column=col_idx).column_letter
            ].hidden = True


def _write_sheet(writer, name: str, df: pd.DataFrame) -> None:
    """Write a DataFrame as a sheet and apply basic formatting."""
    df.to_excel(writer, sheet_name=name, index=True)
    ws = writer.book[name]
    ws.freeze_panes = "B2" if df.index.name else "A2"
    ws.auto_filter.ref = ws.dimensions
    _auto_width(ws)
    if name == "disagreement_detail":
        _hide_columns(ws)


# ── main entry point ─────────────────────────────────────────────────────

def run(
    report_path: str | Path = DEFAULT_OUTPUT,
    result_path: str | Path = DEFAULT_RESULT,
) -> Path:
    report_path = Path(report_path)
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load(report_path)

    our_fc = df["finv_category"]
    illion_cat = df["_illion_cat"]
    our_cp = df["counterparty"]
    illion_cp = df["_illion_cp"]

    our_fc_ok = _is_classified(our_fc)
    illion_ok = _is_classified(illion_cat)
    our_cp_ok = _is_classified(our_cp)
    illion_cp_ok = _is_classified(illion_cp)

    n = len(df)
    both_mask = our_fc_ok & illion_ok

    # ── 1. Coverage stats ─────────────────────────────────────────────────
    coverage_rows = [
        {
            "指标": "finv_category / category",
            "我们已分类": f"{our_fc_ok.sum()} ({our_fc_ok.sum()/n*100:.1f}%)",
            "illion已分类": f"{illion_ok.sum()} ({illion_ok.sum()/n*100:.1f}%)",
            "总行数": n,
        },
        {
            "指标": "counterparty / third_party",
            "我们已分类": f"{our_cp_ok.sum()} ({our_cp_ok.sum()/n*100:.1f}%)",
            "illion已分类": f"{illion_cp_ok.sum()} ({illion_cp_ok.sum()/n*100:.1f}%)",
            "总行数": n,
        },
        {
            "指标": "双方都未分类",
            "我们已分类": "",
            "illion已分类": "",
            "总行数": int((~our_fc_ok & ~illion_ok).sum()),
        },
    ]
    coverage_df = pd.DataFrame(coverage_rows)

    # ── 2. Agreement summary ──────────────────────────────────────────────
    n_both = int(both_mask.sum())
    agree_count = sum(
        _agree(our_fc[i], illion_cat[i]) for i in df.index[both_mask]
    )
    agreement_rows = [
        {"指标": "双方都有标签的行", "值": f"{n_both:,}"},
        {"指标": "一致 (含1:N映射)", "值": f"{agree_count:,} / {n_both:,} = {agree_count/n_both*100:.1f}%"},
    ]
    agreement_df = pd.DataFrame(agreement_rows)

    # ── 3. Confusion pivot ────────────────────────────────────────────────
    pivot_rows = []
    for our_cat_name in sorted(our_fc[our_fc_ok].unique()):
        mask = our_fc_ok & (our_fc == our_cat_name)
        row: dict = {"finv_category": our_cat_name, "total": int(mask.sum())}
        both_for_cat = mask & illion_ok
        if both_for_cat.sum() > 0:
            ag = sum(
                _agree(our_cat_name, illion_cat[i])
                for i in df.index[both_for_cat]
            )
            row["agree_pct"] = round(ag / both_for_cat.sum() * 100, 1)
        else:
            row["agree_pct"] = None
        for illion_val, cnt in (
            df.loc[mask & illion_ok, "_illion_cat"].value_counts().items()
        ):
            row[str(illion_val)] = cnt
        pivot_rows.append(row)
    pivot = (
        pd.DataFrame(pivot_rows)
        .set_index("finv_category")
        .fillna(0)
    )
    meta_cols = ["total", "agree_pct"]
    cat_cols = sorted(
        [c for c in pivot.columns if c not in meta_cols],
        key=lambda x: pivot[x].sum(),
        reverse=True,
    )
    pivot = pivot[meta_cols + cat_cols]
    # Convert float counts to int.
    for c in cat_cols:
        pivot[c] = pivot[c].astype(int)
    pivot["total"] = pivot["total"].astype(int)

    # ── 4. Counterparty comparison ────────────────────────────────────────
    cp_both = our_cp_ok & illion_cp_ok
    n_cp_both = int(cp_both.sum())
    cp_agree = (
        our_cp[cp_both].astype(str).str.strip()
        == illion_cp[cp_both].astype(str).str.strip()
    ).sum()
    counterparty_rows = [
        {"指标": "双方都有counterparty", "值": f"{n_cp_both:,}"},
        {"指标": "完全一致", "值": f"{cp_agree:,} / {n_cp_both:,} = {cp_agree/n_cp_both*100:.1f}%"},
        {"指标": "我们独有", "值": int(our_cp_ok.sum() - n_cp_both)},
        {"指标": "illion独有", "值": int(illion_cp_ok.sum() - n_cp_both)},
    ]
    counterparty_df = pd.DataFrame(counterparty_rows)

    # ── 5. Coverage gaps ──────────────────────────────────────────────────
    our_only = our_fc[our_fc_ok & ~illion_ok].value_counts().reset_index()
    our_only.columns = ["finv_category", "count"]
    illion_only = (
        illion_cat[~our_fc_ok & illion_ok].value_counts().reset_index()
    )
    illion_only.columns = ["illion_category", "count"]

    # ── 6. Per-category agreement breakdown ───────────────────────────────
    cat_summary_rows = []
    for our_cat_name in pivot.index:
        r = pivot.loc[our_cat_name]
        agree_str = (
            f"{r['agree_pct']:.0f}%"
            if r["agree_pct"] and r["agree_pct"] > 0
            else "—"
        )
        total_n = int(r["total"])
        illion_top = []
        for c in cat_cols:
            v = int(r[c])
            if v > 0:
                illion_top.append(f"{c}({v})")
                if len(illion_top) >= 3:
                    break
        cat_summary_rows.append({
            "我们的分类": our_cat_name,
            "数量": total_n,
            "一致率": agree_str,
            "illion对应标签 (top 3)": ", ".join(illion_top) if illion_top else "—",
        })
    cat_summary_df = pd.DataFrame(cat_summary_rows)

    # ── 7. Disagreement detail ────────────────────────────────────────────
    disagree_mask = both_mask & ~pd.Series(
        [
            _agree(our_fc[i], illion_cat[i])
            for i in range(len(df))
        ],
        index=df.index,
    )
    disagree_cols = [
        "user_id", "application_id", "transaction_date", "amount",
        "dr_cr", "text",
        "finv_category", "_illion_cat",
        "counterparty", "_illion_cp",
        "classification_rule_id",
    ]
    available = [c for c in disagree_cols if c in df.columns]
    disagree_df = df.loc[disagree_mask, available].copy()
    disagree_df.rename(columns={
        "_illion_cat": "illion_category",
        "_illion_cp": "illion_counterparty",
    }, inplace=True)
    disagree_df = disagree_df.sort_values(
        ["finv_category", "illion_category"]
    )

    # ═══════════════════════════════════════════════════════════════════
    #  WRITE XLSX
    # ═══════════════════════════════════════════════════════════════════
    with pd.ExcelWriter(result_path, engine="openpyxl") as writer:
        _write_sheet(writer, "coverage", coverage_df.set_index("指标"))
        _write_sheet(writer, "agreement", agreement_df.set_index("指标"))
        _write_sheet(writer, "confusion", pivot)
        _write_sheet(writer, "category_summary", cat_summary_df.set_index("我们的分类"))
        _write_sheet(writer, "counterparty", counterparty_df.set_index("指标"))
        if len(our_only) > 0:
            _write_sheet(writer, "our_only", our_only.set_index("finv_category"))
        if len(illion_only) > 0:
            _write_sheet(writer, "illion_only", illion_only.set_index("illion_category"))
        if len(disagree_df) > 0:
            _write_sheet(writer, "disagreement_detail", disagree_df)

    print(f"→ 对比报告已写入: {result_path}")
    print(f"  覆盖: 我们 {our_fc_ok.sum()/n*100:.1f}% | illion {illion_ok.sum()/n*100:.1f}%")
    print(f"  一致率: {agree_count/n_both*100:.1f}% ({agree_count:,}/{n_both:,})")
    return result_path


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Compare our labels against illion labels"
    )
    p.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to classification_report.xlsx (our output)",
    )
    p.add_argument(
        "--result",
        default=str(DEFAULT_RESULT),
        help="Path to write compare_result.xlsx",
    )
    args = p.parse_args()
    run(args.output, args.result)


if __name__ == "__main__":
    main()
