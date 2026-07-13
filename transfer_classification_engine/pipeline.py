from __future__ import annotations

import pandas as pd

from classification_core.models import PipelineResult

from .domain.classification import classify_transfers


def run_pipeline(transactions: pd.DataFrame) -> PipelineResult:
    """Classify an in-memory transaction dataframe for transfer types."""
    output = classify_transfers(transactions)
    return PipelineResult(
        transactions=output,
        diagnostics={},
    )
