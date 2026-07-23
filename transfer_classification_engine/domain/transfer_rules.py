"""Data-driven rule loading for transfer classification (Internal vs External).

Provides two CSV-backed rule sets that supplement the hardcoded regex rules:

1. **transfer_classification_rules.csv**
   Priority-ordered rules that help distinguish Internal Transfer from
   External Transfers.  Checked *before* the hardcoded regex rules so
   they can correct known misclassification patterns.

2. **transfer_pairing_exclusions.csv**
   Text patterns that indicate a transaction is person-to-person (P2P)
   and should NOT be paired as Internal Transfer.  These supplement the
   hardcoded ``_EXCLUDED_PAIRING_PATTERNS``.

Both CSVs follow the same conventions as the liability engine's rule
files: keywords are semicolon-separated, match_type is ``"keyword"``
(simple substring) or ``"regex"``, and rules are applied in priority
order (highest-first within each category).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _normalize_text(value: object) -> str:
    """Collapse whitespace — used for keyword/substring matching."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_upper(value: object) -> str:
    """Uppercase + collapse whitespace for case-insensitive keyword matching."""
    return _normalize_text(value).upper()


# ---------------------------------------------------------------------------
# Rule data types
# ---------------------------------------------------------------------------


class ClassificationRule:
    """A single classification rule (Internal / External)."""

    __slots__ = (
        "keyword_raw",
        "target_category",
        "dr_cr",
        "match_type",
        "priority",
        "description",
        "_compiled",
    )

    def __init__(
        self,
        keyword: str,
        target_category: str,
        dr_cr: str = "",
        match_type: str = "keyword",
        priority: int = 0,
        description: str = "",
    ) -> None:
        self.keyword_raw = (keyword or "").strip()
        self.target_category = (target_category or "").strip()
        self.dr_cr = (dr_cr or "").strip().lower()
        self.match_type = (match_type or "keyword").strip().lower()
        self.priority = priority
        self.description = (description or "").strip()
        self._compiled: re.Pattern | None = None
        if self.match_type == "regex" and self.keyword_raw:
            try:
                self._compiled = re.compile(self.keyword_raw, re.IGNORECASE)
            except re.error:
                self._compiled = None

    def matches(self, text: str) -> bool:
        """Return True if *text* matches this rule's pattern.

        For regex rules: compiled pattern is tested against raw text.
        For keyword rules: each semicolon-separated keyword is checked
        case-insensitively as a substring of the normalised text.
        """
        if not text or not self.keyword_raw:
            return False
        if self._compiled is not None:
            return bool(self._compiled.search(text))
        # keyword match: split on ';', then case-insensitive substring
        text_upper = _normalize_upper(text)
        for kw in self.keyword_raw.split(";"):
            kw = _normalize_upper(kw)
            if kw and kw in text_upper:
                return True
        return False

    def __repr__(self) -> str:
        return (
            f"ClassificationRule(cat={self.target_category!r}, "
            f"kw={self.keyword_raw[:40]!r}, "
            f"pri={self.priority})"
        )


class ExclusionRule:
    """A single pairing-exclusion rule."""

    __slots__ = (
        "keyword_raw",
        "match_type",
        "exclusion_reason",
        "priority",
        "description",
        "_compiled",
    )

    def __init__(
        self,
        keyword: str,
        match_type: str = "keyword",
        exclusion_reason: str = "",
        priority: int = 0,
        description: str = "",
    ) -> None:
        self.keyword_raw = (keyword or "").strip()
        self.match_type = (match_type or "keyword").strip().lower()
        self.exclusion_reason = (exclusion_reason or "").strip()
        self.priority = priority
        self.description = (description or "").strip()
        self._compiled: re.Pattern | None = None
        if self.match_type == "regex" and self.keyword_raw:
            try:
                self._compiled = re.compile(self.keyword_raw, re.IGNORECASE)
            except re.error:
                self._compiled = None

    def matches(self, text: str) -> bool:
        """Return True if *text* matches this exclusion rule.

        For regex rules: compiled pattern is tested against raw text.
        For keyword rules: each semicolon-separated keyword is checked
        case-insensitively as a substring of the normalised text.
        """
        if not text or not self.keyword_raw:
            return False
        if self._compiled is not None:
            return bool(self._compiled.search(text))
        # keyword match: split on ';', then case-insensitive substring
        text_upper = _normalize_upper(text)
        for kw in self.keyword_raw.split(";"):
            kw = _normalize_upper(kw)
            if kw and kw in text_upper:
                return True
        return False

    def __repr__(self) -> str:
        return (
            f"ExclusionRule(reason={self.exclusion_reason!r}, "
            f"kw={self.keyword_raw[:40]!r})"
        )


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_classification_rules(
    rules_file: str | Path,
) -> list[ClassificationRule]:
    """Load classification rules from a CSV file, sorted by priority (desc).

    Columns: keyword, target_category, dr_cr, match_type, priority, description
    """
    rules: list[ClassificationRule] = []
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            kw = (row.get("keyword") or "").strip()
            cat = (row.get("target_category") or "").strip()
            if not kw or not cat:
                continue
            rules.append(
                ClassificationRule(
                    keyword=kw,
                    target_category=cat,
                    dr_cr=row.get("dr_cr", ""),
                    match_type=row.get("match_type", "keyword"),
                    priority=_parse_int(row.get("priority"), 0),
                    description=row.get("description", ""),
                )
            )
    rules.sort(key=lambda r: -r.priority)
    return rules


def load_exclusion_rules(
    rules_file: str | Path,
) -> list[ExclusionRule]:
    """Load pairing-exclusion rules from a CSV file, sorted by priority (desc).

    Columns: keyword, match_type, exclusion_reason, priority, description
    """
    rules: list[ExclusionRule] = []
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            kw = (row.get("keyword") or "").strip()
            if not kw:
                continue
            rules.append(
                ExclusionRule(
                    keyword=kw,
                    match_type=row.get("match_type", "keyword"),
                    exclusion_reason=row.get("exclusion_reason", ""),
                    priority=_parse_int(row.get("priority"), 0),
                    description=row.get("description", ""),
                )
            )
    rules.sort(key=lambda r: -r.priority)
    return rules


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Batch matching
# ---------------------------------------------------------------------------


def apply_classification_rules(
    df: pd.DataFrame,
    rules: list[ClassificationRule],
) -> pd.DataFrame:
    """Apply classification rules to rows not yet classified.

    Only touches rows where ``is_transfer_pred == 0``.  For each rule, writes
    ``finv_category`` and prediction metadata on first match.  dr_cr
    constraints are respected (blank = match any direction).

    Returns a copy of *df* with matched rows updated.
    """
    output = df.copy()
    remaining_mask = output["is_transfer_pred"] == 0
    if not remaining_mask.any() or not rules:
        return output

    text_col = output.get("text_norm", output.get("text", pd.Series("", index=output.index)))
    dr_cr_col = output.get("dr_cr", pd.Series("", index=output.index))

    for rule in rules:
        # Build a boolean mask of unclassified rows that match this rule.
        match_mask = pd.Series(False, index=output.index)
        for idx in output[remaining_mask].index:
            text = str(text_col.loc[idx] if idx in text_col.index else "")
            if not rule.matches(text):
                continue
            if rule.dr_cr:
                row_dr_cr = str(dr_cr_col.loc[idx] if idx in dr_cr_col.index else "").strip().lower()
                if row_dr_cr != rule.dr_cr:
                    continue
            match_mask[idx] = True

        if not match_mask.any():
            continue

        output.loc[match_mask, "is_transfer_pred"] = 1
        output.loc[match_mask, "finv_category"] = rule.target_category
        output.loc[match_mask, "predicted_category"] = rule.target_category
        output.loc[match_mask, "prediction_confidence"] = "high"
        output.loc[match_mask, "prediction_rule"] = (
            f"kb_{rule.target_category.replace(' ', '_').lower()}"
        )
        output.loc[match_mask, "prediction_dr_cr_used"] = bool(rule.dr_cr)

        remaining_mask = output["is_transfer_pred"] == 0
        if not remaining_mask.any():
            break

    return output


def matches_any_exclusion(
    text: str,
    exclusion_rules: list[ExclusionRule],
) -> bool:
    """Return True if *text* matches any exclusion rule.

    Used during Internal Transfer pairing: if any row in a candidate group
    matches an exclusion rule, the entire group is skipped.
    """
    if not text or pd.isna(text):
        return False
    text_str = str(text)
    for rule in exclusion_rules:
        if rule.matches(text_str):
            return True
    return False


def matches_any_exclusion_in_group(
    texts: pd.Series,
    exclusion_rules: list[ExclusionRule],
) -> bool:
    """Return True if any text in the group matches an exclusion rule."""
    for _, text in texts.items():
        if pd.isna(text) or not str(text).strip():
            continue
        if matches_any_exclusion(str(text), exclusion_rules):
            return True
    return False
