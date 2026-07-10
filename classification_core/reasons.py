from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _clean_part(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _clean_evidence(values: Iterable[object], limit: int) -> list[str]:
    evidence: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_part(value)
        if not text or text in seen:
            continue
        evidence.append(text)
        seen.add(text)
        if len(evidence) >= limit:
            break
    return evidence


def format_classification_reason(
    *,
    category: object,
    rule: object,
    evidence: Iterable[object] = (),
    evidence_limit: int = 3,
) -> str:
    """Return a concise, consistent reason string for engine predictions."""

    parts = [
        f"category={_clean_part(category)}",
        f"rule={_clean_part(rule)}",
    ]
    evidence_parts = _clean_evidence(evidence, evidence_limit)
    if evidence_parts:
        parts.append(f"evidence={', '.join(evidence_parts)}")
    return "; ".join(part for part in parts if part)
