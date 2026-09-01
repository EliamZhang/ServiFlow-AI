"""Shared text / value helpers used across engines."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

import pandas as pd

_CLEAN_RE = re.compile(r"[^A-Z0-9]+")
_DIGIT_LETTER_SEAM_RE = re.compile(r"(?<=\d)(?=[A-Z])|(?<=[A-Z])(?=\d)")


def clean_text(value: object) -> str:
    """Normalise text for keyword matching: uppercase, alphanumerics only."""
    if pd.isna(value):
        return ""
    text = str(value).upper()
    text = _CLEAN_RE.sub(" ", text)
    return " ".join(text.split())


# Payment-channel prefixes that should not be treated as merchant names.
# Stripped from the beginning of transaction text before keyword matching so the
# actual counterparty name can be matched at position 0.  Mirrors the private
# _CHANNEL_PREFIX_RE in initial_engine/domain/classification.py — keep in sync.
_CHANNEL_PREFIX_RE = re.compile(
    r"^(?:"
    r"BILL\s*PAY(?:MENT)?\s+|"
    r"VISA\s+(?:WDL|PURCHASE|DEBIT)\s+(?:PURCHASE\s+)?(?:\d+\w+\s+)?\s*|"
    r"VISA\s+CREDIT\s+\d*\s*|"
    r"EFTPOS\s+(?:DEBIT|WDL)(?:\s+\d+)?\s+|"
    r"MISCELLANEOUS\s+DEBIT\s+V\d+\s+\d+\s+\d+\s+|"
    r"DEBIT\s+CARD\s+PURCHASE\s+|"
    r"EFT\s+Dep\s+"
    r")",
    re.IGNORECASE,
)

# EFTPOS receipt timestamps ("EFTPOS DEBIT EFTPOS 21/05 10:14") are glued to the
# merchant name in the raw text ("...10:14Subway").  They must be stripped
# *before* non-alphanumerics become spaces — after that step the minute field is
# glued to the merchant ("14SUBWAY") and the whole-word boundary check rejects the
# merchant keyword.  Mirrors the private _EFTPOS_TS_RE in
# initial_engine/domain/classification.py — keep in sync.
_EFTPOS_TS_RE = re.compile(
    r"^EFTPOS\s+DEBIT\s+(?:EFTPOS\s+)?\d{1,2}/\d{1,2}\s+\d{1,2}:\d{1,2}",
    re.IGNORECASE,
)


def clean_text_with_channel_prefix(value: object) -> str:
    """clean_text() plus payment-channel prefix stripping.

    Used by the rent engine's institution layer so merchant-KB keywords match
    the same texts the initial engine matched before the rent categories moved
    out of the KB: BILL PAY / VISA / EFTPOS / EFT Dep prefixes are removed from
    the start of the transaction text (after the EFTPOS-timestamp strip), letting
    the institution name match at position 0.
    """
    if pd.isna(value):
        return ""
    text = str(value).upper()
    text = _EFTPOS_TS_RE.sub("", text)
    text = _CLEAN_RE.sub(" ", text)
    text = _CHANNEL_PREFIX_RE.sub("", text)
    text = _CLEAN_RE.sub(" ", text)
    return " ".join(text.split())


def clean_text_with_seams(value: object) -> str:
    """Income-engine variant of :func:`clean_text` that also inserts a space
    at digit-letter seams ("3780.7Salary" -> "3780 7 SALARY", "IBMAUPAY986915"
    -> "IBMAUPAY 986915") so word-boundary regexes like ``\\bSALARY\\b`` do not
    fail on concatenated bank descriptions.

    Only the income engine uses this variant.  A global change to clean_text
    altered other engines' matching behaviour (e.g. rent engine's "TEN00083"
    tenant reference was split and lost its match, producing Rent ->
    External Transfers regressions), so the seam-splitting stays scoped here.
    """
    if pd.isna(value):
        return ""
    return _DIGIT_LETTER_SEAM_RE.sub(" ", clean_text(value))


def is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def parse_decimal_amount(value: object) -> Decimal | None:
    if pd.isna(value):
        return None

    text = str(value).strip().replace(",", "")
    if not text:
        return None

    try:
        return Decimal(text)
    except InvalidOperation:
        return None
