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
