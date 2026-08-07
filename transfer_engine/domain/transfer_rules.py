"""Data-driven exclusion rule loading for transfer classification.

Provides ``transfer_pairing_exclusions.csv``-backed rules: text patterns that
indicate a transaction is person-to-person (P2P) and should NOT be paired as
Internal Transfer.  These supplement the hardcoded ``_EXCLUDED_PAIRING_PATTERNS``.

The CSV follows the same conventions as the liability engine's rule files:
keywords are semicolon-separated, match_type is ``"keyword"`` (simple
substring) or ``"regex"``, and rules are applied in priority order.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Rule data types
# ---------------------------------------------------------------------------


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

    def __repr__(self) -> str:
        return (
            f"ExclusionRule(reason={self.exclusion_reason!r}, "
            f"kw={self.keyword_raw[:40]!r})"
        )


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


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
