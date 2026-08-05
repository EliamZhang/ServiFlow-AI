"""Shared text / value helpers used across engines."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

import pandas as pd

_CLEAN_RE = re.compile(r"[^A-Z0-9]+")


def clean_text(value: object) -> str:
    """Normalise text for keyword matching: uppercase, alphanumerics only."""
    if pd.isna(value):
        return ""
    text = str(value).upper()
    text = _CLEAN_RE.sub(" ", text)
    return " ".join(text.split())


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
