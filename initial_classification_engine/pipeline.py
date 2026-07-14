"""Merchant-keyword classification pipeline.

Orchestrates KB loading and batch matching for a set of transactions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from classification_core.models import PipelineResult

from .domain.classification import get_cached_automaton, match_transactions

DEFAULT_KB_PATH = Path(__file__).resolve().parent / "sample_merchant_kb.csv"


def run_pipeline(
    transactions: pd.DataFrame,
    kb_path: str | Path = DEFAULT_KB_PATH,
) -> PipelineResult:
    """Classify transactions via keyword matching against the merchant KB."""
    automaton = get_cached_automaton(kb_path)
    output = match_transactions(transactions, automaton)
    return PipelineResult(
        transactions=output,
        original_columns=tuple(transactions.columns),
        diagnostics={
            "keyword_count": automaton.keyword_count,
        },
    )
