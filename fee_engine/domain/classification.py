# -*- coding: utf-8 -*-
"""
Rule-based fee classification for bank transactions.

Loads regex rules from a CSV file and applies them in priority order.

Classifies transactions into:
- Overdrawn (checked first — overrides generic fee)
- fee

Rules are loaded from *fee_classification_rules.csv* — no hardcoded patterns.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from classification_core.reasons import format_classification_reason


# Category value normalisation — CSV uses lowercase "fee", engine outputs "Fees".
_CATEGORY_MAP = {"fee": "Fees"}


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class FeePrediction:
    is_fee: bool
    category: str | None
    counterparty: str | None
    rule_name: str | None
    unclassified_only: bool = False
    dr_cr: str = ""


# =============================================================================
# Text normalization
# =============================================================================

def normalize_text(value: object) -> str:
    """Normalize text for stable rule matching — preserve original case."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


# =============================================================================
# Rule loader
# =============================================================================

def load_fee_rules(
    rules_file: str | Path,
) -> tuple[
    list[tuple[str, str, re.Pattern, str, bool, str]], set[str]
]:
    """Load fee classification rules from CSV.

    Returns:
        rules: list of ``(rule_name, category, compiled_pattern, counterparty,
               unclassified_only, dr_cr)`` sorted by priority ascending (lower
               priority = matched first).
        zero_amount_reject: set of *rule_name* values whose matches should be
                            discarded when the transaction amount is $0.
    """
    raw: list[tuple[int, str, str, str, str, bool, bool, str]] = []

    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            priority = int(str(row.get("priority", "0")).strip())
            rule_name = str(row.get("rule_name", "")).strip()
            category_raw = str(row.get("category", "")).strip()
            pattern = str(row.get("pattern", "")).strip()
            counterparty = str(row.get("counterparty", "")).strip()
            zero_reject = (
                str(row.get("zero_amount_reject", "false")).strip().lower() == "true"
            )
            unclassified_only = (
                str(row.get("unclassified_only", "false")).strip().lower() == "true"
            )
            dr_cr = str(row.get("dr_cr", "")).strip().lower()

            if not rule_name or not pattern:
                continue

            category = _CATEGORY_MAP.get(category_raw, category_raw)
            raw.append(
                (
                    priority,
                    rule_name,
                    category,
                    pattern,
                    counterparty,
                    zero_reject,
                    unclassified_only,
                    dr_cr,
                )
            )

    # Sort by priority ascending — lower number = higher priority = checked first.
    raw.sort(key=lambda r: r[0])

    rules: list[tuple[str, str, re.Pattern, str, bool, str]] = []
    zero_amount_reject: set[str] = set()

    for _, rule_name, category, pattern, counterparty, zero_reject, unclassified_only, dr_cr in raw:
        try:
            # Case-insensitive: bank statement text mixes ALL CAPS / Title Case /
            # lowercase for the same fee phrase (e.g. "ANNUAL FEE" vs "Annual Fee").
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            continue
        rules.append(
            (rule_name, category, compiled, counterparty, unclassified_only, dr_cr)
        )
        if zero_reject:
            zero_amount_reject.add(rule_name)

    return rules, zero_amount_reject


# =============================================================================
# Fee classifier
# =============================================================================

class FeeClassifier:
    """Apply fee classification rules in priority order.

    Rules are pre-loaded via :func:`load_fee_rules` and passed to the
    constructor — this class only handles matching, not I/O.
    """

    def __init__(
        self, rules: list[tuple[str, str, re.Pattern, str, bool, str]]
    ) -> None:
        self.rules = rules

    def predict(self, text: str) -> FeePrediction:
        """Apply rules in priority order — first match wins."""
        for rule_name, category, pattern, counterparty, unclassified_only, dr_cr in self.rules:
            if pattern.search(text):
                return FeePrediction(
                    is_fee=True,
                    category=category,
                    counterparty=counterparty,
                    rule_name=rule_name,
                    unclassified_only=unclassified_only,
                    dr_cr=dr_cr,
                )
        return FeePrediction(
            is_fee=False,
            category=None,
            counterparty=None,
            rule_name=None,
        )


# =============================================================================
# Pipeline entry point
# =============================================================================

def classify_fees(
    df: pd.DataFrame,
    rules_file: str | Path | None = None,
) -> pd.DataFrame:
    """Apply fee classification rules and produce output columns.

    Parameters
    ----------
    df: Transaction DataFrame.
    rules_file: Path to ``fee_classification_rules.csv``.  Defaults to
        ``fee_engine/resources/fee_classification_rules.csv``.
    """
    if rules_file is None:
        rules_file = (
            Path(__file__).resolve().parent.parent
            / "resources"
            / "fee_classification_rules.csv"
        )

    rules, zero_amount_reject = load_fee_rules(rules_file)
    classifier = FeeClassifier(rules)

    output = df.copy()
    raw_text = output.get("text", pd.Series("", index=output.index))
    output["_text_original"] = raw_text
    output["text_norm"] = raw_text.apply(normalize_text)

    n = len(output)
    is_fee = np.zeros(n, dtype=bool)
    categories = np.empty(n, dtype=object)
    counterparties = np.empty(n, dtype=object)
    rule_names = np.empty(n, dtype=object)
    uncl_only_flags = np.zeros(n, dtype=bool)

    texts = output["text_norm"].values
    dr_cr_vals = (
        output.get("dr_cr", pd.Series("", index=output.index))
        .fillna("").astype(str).str.lower().values
    )

    # Per-rule vectorised matching, first rule that matches a row wins.
    # Rules with a dr_cr constraint only apply to rows of that direction
    # (e.g. bare "Interest" is a credit — interest earned, not a fee).
    for rule_name, category, pattern, counterparty, unclassified_only, dr_cr in classifier.rules:
        remain = np.where(~is_fee)[0]
        if len(remain) == 0:
            break
        hits = np.fromiter(
            (pattern.search(texts[i]) is not None for i in remain),
            dtype=bool,
            count=len(remain),
        )
        if dr_cr:
            hits &= dr_cr_vals[remain] == dr_cr
        if not hits.any():
            continue
        matched_idx = remain[hits]
        is_fee[matched_idx] = True
        categories[matched_idx] = category
        counterparties[matched_idx] = counterparty
        rule_names[matched_idx] = rule_name
        uncl_only_flags[matched_idx] = unclassified_only

    # Preserve any pre-existing classification on non-fee rows.
    existing_cat = (
        output["finv_category"]
        if "finv_category" in output.columns
        else pd.Series("", index=output.index)
    )
    existing_cp = (
        output["counterparty"]
        if "counterparty" in output.columns
        else pd.Series("", index=output.index)
    )
    output["is_fee_pred"] = is_fee.astype(int)
    output["finv_category"] = [
        cat if f else prev
        for f, cat, prev in zip(is_fee, categories, existing_cat)
    ]
    output["counterparty"] = [
        cp if f else prev
        for f, cp, prev in zip(is_fee, counterparties, existing_cp)
    ]
    output["fee_rule_name"] = [
        rn if f else "" for f, rn in zip(is_fee, rule_names)
    ]
    output["fee_unclassified_only"] = uncl_only_flags

    # ── Post-processing: reject $0 informational fee lines ──────────
    _reject_zero_amount_informational(output, zero_amount_reject)

    output["fee_pred_reason"] = output.apply(_build_reason, axis=1)

    # stream_id (legacy value "fee" — keep as-is for baseline parity)
    output["stream_id"] = output["finv_category"].map(
        {"Fees": "fee"}
    ).where(output["is_fee_pred"].eq(1), "")

    # Drop internal columns
    output = output.drop(columns=["text_norm", "_text_original"])

    return output


def _reject_zero_amount_informational(
    df: pd.DataFrame, zero_amount_reject: set[str]
) -> None:
    """Unset fee predictions whose amount is $0 and rule is informational.

    Certain fee-rule patterns match informational line-items (e.g. "Includes
    Foreign Currency Conversion Fee $0.81") where the transaction amount is
    $0.00 — these are notes attached to other transactions, NOT real fee
    charges.  This function unsets the prediction so the row can be picked
    up by a later engine or left unclassified.
    """
    if "amount" not in df.columns:
        return

    # Only consider rows currently marked as fee.
    fee_mask = df["is_fee_pred"].eq(1)
    if not fee_mask.any():
        return

    # Find rows where the rule is in our reject set AND amount is zero.
    amount_col = pd.to_numeric(df["amount"], errors="coerce")
    zero_amount_mask = (
        amount_col.abs() < 0.001  # near-zero — informational lines
    )
    reject_rule_mask = df["fee_rule_name"].isin(zero_amount_reject)

    reject_mask = fee_mask & zero_amount_mask & reject_rule_mask
    if not reject_mask.any():
        return

    # Unset the prediction columns for rejected rows.
    df.loc[reject_mask, "is_fee_pred"] = 0
    df.loc[reject_mask, "finv_category"] = ""
    df.loc[reject_mask, "counterparty"] = ""
    df.loc[reject_mask, "fee_rule_name"] = ""


def _build_reason(row: pd.Series) -> str:
    if int(row.get("is_fee_pred", 0)) != 1:
        return format_classification_reason(
            category="not_fee",
            rule="no_fee_rule_matched",
            evidence=[],
        )

    rule_name = str(row.get("fee_rule_name", ""))
    category = str(row.get("finv_category", ""))
    counterparty = str(row.get("counterparty", ""))

    return format_classification_reason(
        category=category,
        rule=rule_name,
        evidence=[f"counterparty={counterparty}"],
    )
