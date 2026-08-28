from __future__ import annotations

import pandas as pd

from classification_core.models import PipelineResult
from classification_core.text import clean_text_with_seams
from initial_engine.domain.classification import get_cached_automaton

from .domain.classification import (
    add_income_type_rules,
    add_wages_features,
    apply_wages_rules,
    prepare_input,
    reorder_output_columns,
)
from .domain.summary import add_income_streams


def _add_kb_counterparty(transactions: pd.DataFrame) -> pd.DataFrame:
    """Match income transactions against the merchant KB for counterparty.

    For every row that the income engine predicts as income, attempt to find a
    matching keyword in the merchant knowledge base.  When found, the keyword
    is stored in ``_kb_counterparty`` and later used by ``derive_counterparty``
    in preference to the regex-based payer extraction.
    """
    automaton = get_cached_automaton()
    out = transactions.copy()
    texts = out["text"].apply(clean_text_with_seams)
    kb_counterparties: list[str] = []

    for text_clean in texts:
        hits = automaton.search(str(text_clean))
        if hits:
            best_kw, _merchant, _cat = max(hits, key=lambda h: len(h[0]))
            kb_counterparties.append(best_kw.title())
        else:
            kb_counterparties.append("")

    out["_kb_counterparty"] = kb_counterparties
    return out


def run_pipeline(transactions: pd.DataFrame) -> PipelineResult:
    """Classify an in-memory transaction dataframe."""
    output = prepare_input(transactions)
    original_columns = list(output.columns)

    output = add_wages_features(output)
    output = apply_wages_rules(output)
    output = add_income_type_rules(output)
    output = _add_kb_counterparty(output)
    output = add_income_streams(output)
    output = reorder_output_columns(output, original_columns)
    return PipelineResult(
        transactions=output,
    )
