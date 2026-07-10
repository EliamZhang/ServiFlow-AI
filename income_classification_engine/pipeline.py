from __future__ import annotations

from pathlib import Path

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


ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parent


def run_pipeline(
    transactions: pd.DataFrame,
    include_centrelink_payment_type: bool = False,
) -> PipelineResult:
    """Classify an in-memory transaction dataframe."""
    output = prepare_input(transactions)
    original_columns = list(output.columns)

    output = add_wages_features(output)
    output = apply_wages_rules(output)
    output = add_income_type_rules(
        output,
        include_centrelink_payment_type=include_centrelink_payment_type,
    )
    output = add_income_streams(output)
    output = reorder_output_columns(output, original_columns)
    return PipelineResult(
        transactions=output,
        original_columns=tuple(original_columns),
    )
