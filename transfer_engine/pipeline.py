from __future__ import annotations

import pandas as pd

from classification_core.models import PipelineResult

from .domain.classification import classify_transfers


def run_pipeline(transactions: pd.DataFrame, *, all_rows: pd.DataFrame | None = None) -> PipelineResult:
    """Classify an in-memory transaction dataframe for transfer types.

    Parameters
    ----------
    transactions : pd.DataFrame
        Candidate rows the engine may write to.
    all_rows : pd.DataFrame or None
        If provided, used as the full dataset for Internal Transfer
        pairing detection (so pairs are found even when one side has
        already been classified by an earlier pipeline engine).
    """
    output = classify_transfers(transactions, all_rows=all_rows)
    return PipelineResult(
        transactions=output,
        diagnostics={},
    )
