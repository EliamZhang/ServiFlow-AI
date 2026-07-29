# -*- coding: utf-8 -*-
"""
Rule-based fee classification for bank transactions.

Classifies transactions into:
- Overdrawn (checked first — overrides generic fee)
- fee

Uses regex rules loaded from ``resources/fee_classification_rules.csv``,
applied in priority order to identify fee transactions from text alone.
Overdrawn-related fees (overdrawn, overlimit, overdraft, overdraw, debit excess
interest) are checked FIRST so they override the generic "fee" category.

Fee types include: overdrawn/overlimit fees, international transaction fees,
ATM operator fees, bank account fees, dishonour fees, late payment fees,
cash advance fees, and third-party maintenance/membership fees.

This module is invoked by the unified engine pipeline as the LAST engine
(priority 500), after all other classification engines.
"""

import csv
import re
import warnings
from pathlib import Path

import pandas as pd

from classification_core.reasons import format_classification_reason


# =============================================================================
# Paths
# =============================================================================

_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
_RULES_FILE = _RESOURCES_DIR / "fee_classification_rules.csv"


# =============================================================================
# Rule loading from CSV
# =============================================================================

def load_fee_rules(rules_file: str | Path | None = None) -> list[dict]:
    """Load fee classification rules from CSV.

    Returns a list of rule dicts with keys:
        priority, rule_name, category, pattern, counterparty,
        match_type, zero_amount_reject, description

    Rules are sorted by priority ascending (lower = checked first).
    """
    if rules_file is None:
        rules_file = _RULES_FILE

    rules: list[dict] = []
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pattern = str(row.get("pattern", "")).strip()
            if not pattern:
                continue
            try:
                compiled = re.compile(pattern)
            except re.error:
                continue
            rules.append({
                "priority": int(row.get("priority", 0)),
                "rule_name": str(row.get("rule_name", "")).strip(),
                "category": str(row.get("category", "")).strip(),
                "pattern": compiled,
                "counterparty": str(row.get("counterparty", "")).strip(),
                "zero_amount_reject": str(row.get("zero_amount_reject", "false")).strip().lower() == "true",
            })

    rules.sort(key=lambda r: r["priority"])
    return rules


# =============================================================================
# Text normalization
# =============================================================================

def normalize_text(value: object) -> str:
    """Normalize text for stable rule matching — preserve original case."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


# =============================================================================
# Pipeline — lazy-loaded rules (following liability engine pattern)
# =============================================================================

_RULES: list[dict] | None = None
_ZERO_REJECT: set[str] | None = None


def _get_rules() -> list[dict]:
    global _RULES, _ZERO_REJECT
    if _RULES is None:
        _RULES = load_fee_rules()
        _ZERO_REJECT = {r["rule_name"] for r in _RULES if r["zero_amount_reject"]}
    return _RULES


def classify_fees(df: pd.DataFrame) -> pd.DataFrame:
    """Apply fee classification rules and produce output columns (vectorised).

    Rules are applied in priority order — first match wins.  Matched rows are
    removed from consideration before the next rule runs.
    """
    rules = _get_rules()
    output = df.copy()
    output["text_norm"] = output.get("text", pd.Series("", index=output.index)).apply(normalize_text)
    output["is_fee_pred"] = 0
    output["finv_category"] = ""
    output["counterparty"] = ""
    output["fee_rule_name"] = ""

    _warn_msg = "This pattern is interpreted as a regular expression"

    remaining = pd.Series(True, index=output.index)
    for rule in rules:
        if not remaining.any():
            break
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_warn_msg)
            matched = remaining & output.loc[remaining, "text_norm"].str.contains(
                rule["pattern"], na=False, regex=True
            )
        if not matched.any():
            continue
        output.loc[matched, "is_fee_pred"] = 1
        output.loc[matched, "finv_category"] = rule["category"]
        output.loc[matched, "counterparty"] = rule["counterparty"]
        output.loc[matched, "fee_rule_name"] = rule["rule_name"]
        remaining[matched] = False

    # ── Post-processing: reject $0 informational fee lines ──────────
    _reject_zero_amount(output)

    output["fee_pred_reason"] = output.apply(
        lambda row: format_classification_reason(
            category=row["finv_category"] if row["is_fee_pred"] else "not_fee",
            rule=row["fee_rule_name"] if row["is_fee_pred"] else "no_fee_rule_matched",
            evidence=(
                [f"counterparty={row['counterparty']}"] if row["is_fee_pred"] else []
            ),
        ),
        axis=1,
    )
    output["stream_id"] = output["finv_category"].where(output["is_fee_pred"].eq(1), "")
    return output.drop(columns=["text_norm"])


def _reject_zero_amount(df: pd.DataFrame) -> None:
    """Unset fee predictions whose amount is $0 and rule is informational."""
    if "amount" not in df.columns:
        return
    fee_mask = df["is_fee_pred"].eq(1)
    if not fee_mask.any():
        return
    amount = pd.to_numeric(df["amount"], errors="coerce")
    reject = fee_mask & (amount.abs() < 0.001) & df["fee_rule_name"].isin(_ZERO_REJECT or set())
    if reject.any():
        df.loc[reject, ["is_fee_pred", "finv_category", "counterparty", "fee_rule_name"]] = [0, "", "", ""]
