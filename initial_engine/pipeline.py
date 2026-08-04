"""Merchant-keyword classification pipeline.

Orchestrates KB loading and batch matching for a set of transactions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from classification_core.models import PipelineResult

from .domain.classification import get_cached_automaton, match_transactions

DEFAULT_KB_PATH = Path(__file__).resolve().parent / "merchant_kb.csv"


def run_pipeline(
    transactions: pd.DataFrame,
    kb_path: str | Path = DEFAULT_KB_PATH,
) -> PipelineResult:
    """Classify transactions via keyword matching against the merchant KB."""
    automaton = get_cached_automaton(kb_path)
    output = match_transactions(transactions, automaton)

    # Debt Collection / Debt Consolidation are now owned by the liability engine.
    # Clear them here so the initial engine does not fail ownership validation.
    _liability_owned = output["finv_category"].isin(["Debt Collection", "Debt Consolidation"])
    output.loc[_liability_owned, "finv_category"] = ""
    # Financial Institutions is handled by liability/dishonour engines.
    _fi_mask = output["finv_category"] == "Financial Institutions"
    output.loc[_fi_mask, "finv_category"] = ""

    return PipelineResult(
        transactions=output,
        original_columns=tuple(transactions.columns),
        diagnostics={
            "keyword_count": automaton.keyword_count,
        },
    )
