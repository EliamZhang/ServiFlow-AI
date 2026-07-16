# -*- coding: utf-8 -*-
"""Compare our classification output against third-party labels.

Usage::

    python compare_labels.py [--output output/classification_report.xlsx]

The third-party labels are ~95 % accurate but *not* ground truth —
disagreement does not always mean we are wrong.

Core outputs
------------
1. **Coverage** — who classified more rows?
2. **Confusion pivot** — our finv_category × their category (the most useful view)
3. **Counterparty** — simple coverage + exact-match rate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = PROJECT_ROOT / "sample.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "classification_report.xlsx"

# ── category name mapping: our finv_category → their category ────────────
# Only pairs where names differ; exact name matches work automatically.

OUR_TO_THEIR: dict[str, str] = {
    "fee":                "Fees",
    "transfer":           "Internal Transfer",
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

# Our categories that legitimately span MULTIPLE third-party categories.
OUR_TO_THEIR_MULTI: dict[str, frozenset[str]] = {
    "transfer": frozenset({"Internal Transfer", "External Transfers",
                           "All Other Credits", "Credit Card Repayments"}),
    "bnpl":     frozenset({"Non SACC Loans", "SACC Loans"}),
    "bank":     frozenset({"Non SACC Loans", "SACC Loans", "Credit Card Repayments",
                           "Internal Transfer", "External Transfers"}),
    "personal_loan_non_sacc": frozenset({"Non SACC Loans", "SACC Loans"}),
    "loc":      frozenset({"Non SACC Loans", "Credit Card Repayments"}),
}


# ── data loading ─────────────────────────────────────────────────────────

def _load(report_path: Path) -> pd.DataFrame:
    our = pd.read_excel(report_path, sheet_name="transactions")
    third = pd.read_csv(SAMPLE_CSV, encoding="utf-8-sig")
    df = our.copy()
    df["_their_cat"] = third["category"].values
    df["_their_cp"] = third["third_party"].values
    return df


# ── helpers ──────────────────────────────────────────────────────────────

def _map(our_cat: str) -> str:
    c = str(our_cat).strip()
    return OUR_TO_THEIR.get(c, c)


def _agree(our: str, their: str) -> bool:
    """True when our category is consistent with their category."""
    our, their = str(our).strip(), str(their).strip()
    if _map(our) == their:
        return True
    if our in OUR_TO_THEIR_MULTI and their in OUR_TO_THEIR_MULTI[our]:
        return True
    return False


def _is_classified(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("") & series.astype(str).str.strip().ne("unclassified")


# ── main entry point ─────────────────────────────────────────────────────

def run(report_path: str | Path = DEFAULT_OUTPUT) -> None:
    report_path = Path(report_path)
    df = _load(report_path)

    our_fc = df["finv_category"]
    their_cat = df["_their_cat"]
    our_cp = df["counterparty"]
    their_cp = df["_their_cp"]

    our_fc_ok = _is_classified(our_fc)
    their_ok = _is_classified(their_cat)
    our_cp_ok = _is_classified(our_cp)
    their_cp_ok = _is_classified(their_cp)

    n = len(df)

    # ── 1. finv_category coverage + agreement ──────────────────────────
    both_mask = our_fc_ok & their_ok
    agree_count = sum(_agree(our_fc[i], their_cat[i]) for i in df.index[both_mask])
    n_both = int(both_mask.sum())

    # ── 2. counterparty coverage + agreement ────────────────────────────
    cp_both = our_cp_ok & their_cp_ok
    cp_agree = (our_cp[cp_both].astype(str).str.strip()
                == their_cp[cp_both].astype(str).str.strip()).sum()
    n_cp_both = int(cp_both.sum())

    # ── 3. confusion pivot ──────────────────────────────────────────────
    pivot_rows = []
    for our_cat_name in sorted(our_fc[our_fc_ok].unique()):
        mask = our_fc_ok & (our_fc == our_cat_name)
        row = {"finv_category": our_cat_name, "total": int(mask.sum())}
        # agreement %
        both_for_cat = mask & their_ok
        if both_for_cat.sum() > 0:
            ag = sum(_agree(our_cat_name, their_cat[i]) for i in df.index[both_for_cat])
            row["agree_pct"] = round(ag / both_for_cat.sum() * 100, 1)
        else:
            row["agree_pct"] = None
        # count by third-party category
        for their_val, cnt in df.loc[mask & their_ok, "_their_cat"].value_counts().items():
            row[str(their_val)] = cnt
        pivot_rows.append(row)
    pivot = pd.DataFrame(pivot_rows).set_index("finv_category").fillna(0)
    # Move total + agree_pct to front
    meta_cols = ["total", "agree_pct"]
    cat_cols = sorted([c for c in pivot.columns if c not in meta_cols],
                      key=lambda x: pivot[x].sum(), reverse=True)
    pivot = pivot[meta_cols + cat_cols]

    # ── 4. coverage gaps ────────────────────────────────────────────────
    our_only = our_fc[our_fc_ok & ~their_ok].value_counts()
    their_only = their_cat[~our_fc_ok & their_ok].value_counts()

    # ═══════════════════════════════════════════════════════════════════
    #  CONSOLE REPORT
    # ═══════════════════════════════════════════════════════════════════
    def H(s: str) -> None:
        print(f"\n{'─'*72}\n  {s}\n{'─'*72}")

    H("1. 覆盖率")
    print(f"  {'':>25s} {'我们':>12s} {'第三方':>12s}  {'合计':>8s}")
    print(f"  {'finv_category / category':>25s} {our_fc_ok.sum():>8d} ({our_fc_ok.sum()/n*100:4.1f}%)"
          f"  {their_ok.sum():>8d} ({their_ok.sum()/n*100:4.1f}%)  {n:>8d}")
    print(f"  {'counterparty / third_party':>25s} {our_cp_ok.sum():>8d} ({our_cp_ok.sum()/n*100:4.1f}%)"
          f"  {their_cp_ok.sum():>8d} ({their_cp_ok.sum()/n*100:4.1f}%)")
    print(f"  {'  双方都未分类':>25s} {(~our_fc_ok & ~their_ok).sum():>20d}")

    H("2. finv_category 一致性")
    print(f"  双方都有标签: {n_both:,} 行")
    print(f"  一致 (含1:N映射): {agree_count:,} / {n_both:,} = {agree_count/n_both*100:.1f}%")
    print(f"  (映射规则: 我们的 transfer→Internal/External Transfers, bnpl→Non SACC Loans 等)")

    # Compact per-category summary
    print(f"\n  {'类别':<28s} {'数量':>6s}  {'一致率':>6s}  {'第三方主要标签'}")
    print(f"  {'-'*28} {'-'*6}  {'-'*6}  {'-'*30}")
    for our_cat_name in pivot.index:
        r = pivot.loc[our_cat_name]
        agree_str = f"{r['agree_pct']:.0f}%" if r['agree_pct'] > 0 else "—"
        total_n = int(r["total"])
        # top 3 their labels
        their_top = []
        for c in cat_cols:
            v = int(r[c])
            if v > 0:
                their_top.append(f"{c}({v})")
                if len(their_top) >= 2:
                    break
        their_str = ", ".join(their_top) if their_top else "—"
        print(f"  {our_cat_name:<28s} {total_n:>6d}  {agree_str:>6s}  {their_str}")

    H("3. counterparty 一致性")
    print(f"  双方都有值: {n_cp_both:,} 行")
    print(f"  完全一致:   {cp_agree:,} / {n_cp_both:,} = {cp_agree/n_cp_both*100:.1f}%")
    print(f"  (counterparty 格式差异较大——fee引擎输出费用类型，transfer引擎输出转账类型)")

    if len(our_only) > 0:
        print(f"\n  我们独有标签 (top 5):")
        for cat, cnt in our_only.head(5).items():
            print(f"    {cat:<30s} {cnt:>5d}")

    if len(their_only) > 0:
        print(f"  第三方独有标签 (top 5):")
        for cat, cnt in their_only.head(5).items():
            print(f"    {str(cat):<30s} {cnt:>5d}")

    print(f"\n{'─'*72}\n  → 完成\n{'─'*72}")


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Compare labels against third-party")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Path to classification_report.xlsx")
    args = p.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
