"""Knowledge-base driven counterparty extraction for transfer transactions.

Matches transaction text against a CSV rule table to derive standardised
counterparty (third_party) labels for External Transfers.

The CSV file (``transfer_counterparty_rules.csv``) uses the same format as the
liability engine's ``counterparty_keyword_rules.csv``::

    keyword,counterparty,match_type
    Osko;DEPOSIT-OSKO,Osko,keyword
    COMMBANK APP,CBA Funds Transfer,keyword
    ...

Keywords are semicolon-separated and matched case-insensitively against
normalised (uppercase, collapsed whitespace) transaction text.  Rules are
applied in CSV row order — first match wins.

Rows that match no rule fall back to ``"Miscellaneous Funds Transfer"``.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Text normalisation (mirrors liability engine conventions)
# ---------------------------------------------------------------------------

def _normalize_match_text(value: object) -> str:
    """Uppercase + collapse whitespace, the same as the liability engine."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().upper())


def _split_keywords(raw: str, separator: str = ";") -> list[str]:
    """Split a semicolon-separated keyword string into individual terms."""
    terms: list[str] = []
    for term in (raw or "").split(separator):
        term = _normalize_match_text(term)
        if term:
            terms.append(term)
    return terms


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------

def load_counterparty_rules(
    rules_file: str | Path,
) -> list[tuple[list[str], str]]:
    """Load counterparty rules from a CSV file.

    Returns a list of ``(keywords, counterparty)`` tuples in file order.
    Each *keywords* entry is a list of uppercase strings to match against
    normalised transaction text.

    The CSV must have columns ``keyword`` and ``counterparty``.
    A ``match_type`` column is accepted for compatibility but currently
    only ``"keyword"`` matching is supported.
    """
    rules: list[tuple[list[str], str]] = []
    with open(rules_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            counterparty = (row.get("counterparty") or "").strip()
            raw_keywords = row.get("keyword") or ""
            if not counterparty or not raw_keywords.strip():
                continue
            keywords = _split_keywords(raw_keywords)
            if keywords:
                rules.append((keywords, counterparty))
    return rules


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_counterparty(
    text: str,
    rules: list[tuple[list[str], str]],
    fallback: str = "Miscellaneous Funds Transfer",
) -> str:
    """Return the counterparty label for *text* using the first matching rule.

    Parameters
    ----------
    text : str
        Raw transaction text.
    rules : list[tuple[list[str], str]]
        Loaded rule list (keyword-lists → counterparty), in priority order.
    fallback : str
        Counterparty name returned when no rule matches.

    Returns
    -------
    str
        The counterparty (third_party) label.
    """
    if not text or pd.isna(text):
        return fallback

    normalized = _normalize_match_text(text)
    if not normalized:
        return fallback

    for keywords, counterparty in rules:
        for kw in keywords:
            if kw in normalized:
                return counterparty

    return fallback


def derive_counterparty_series(
    text_series: pd.Series,
    rules: list[tuple[list[str], str]],
    fallback: str = "Miscellaneous Funds Transfer",
) -> pd.Series:
    """Vectorised counterparty derivation for a Series of text values.

    Returns a Series of counterparty labels with the same index as
    *text_series*.
    """
    return text_series.apply(lambda t: match_counterparty(t, rules, fallback=fallback))
