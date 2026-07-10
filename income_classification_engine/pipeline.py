from __future__ import annotations

import pandas as pd

from classification_core.models import PipelineResult

from .domain.classification import (
    add_income_type_rules,
    add_wages_features,
    apply_wages_rules,
    prepare_input,
    reorder_output_columns,
)
from .domain.summary import add_income_streams


def run_pipeline(transactions: pd.DataFrame) -> PipelineResult:
    """Classify an in-memory transaction dataframe."""
    output = prepare_input(transactions)
    original_columns = list(output.columns)

    output = add_wages_features(output)
    output = apply_wages_rules(output)
    output = add_income_type_rules(output)
    output = add_income_streams(output)
    output = reorder_output_columns(output, original_columns)
    return PipelineResult(
        transactions=output,
        original_columns=tuple(original_columns),
    )
