from __future__ import annotations

from pathlib import Path

import pandas as pd

from classification_core.models import PipelineResult

from .domain.classification import classify_fees

DEFAULT_RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


def run_pipeline(transactions: pd.DataFrame) -> PipelineResult:
    """Classify an in-memory transaction dataframe for fee types."""
    output = classify_fees(
        transactions,
        DEFAULT_RESOURCES_DIR / "fee_classification_rules.csv",
    )
    return PipelineResult(
        transactions=output,
        diagnostics={},
    )
