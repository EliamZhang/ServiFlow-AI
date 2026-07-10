from __future__ import annotations

from pathlib import Path

import pandas as pd

from classification_core.models import PipelineResult

from .domain.counterparty import apply_credit_card_rules, apply_counterparty_rules
from .domain.dishonours import apply_dishonour_rules
from .domain.special_rules import apply_special_rules
from .domain.streams import add_finv_category, identify_streams


ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parent
DEFAULT_RESOURCES_DIR = ENGINE_DIR / "resources"


def run_pipeline(
    transactions: pd.DataFrame,
    resources_dir: str | Path = DEFAULT_RESOURCES_DIR,
) -> PipelineResult:
    """Classify liabilities in an in-memory transaction dataframe."""
    resources_path = Path(resources_dir)
    output = apply_counterparty_rules(
        transactions,
        resources_path / "counterparty_keyword_rules.csv",
    )
    output = apply_credit_card_rules(output, resources_path / "credit_card_rules.csv")
    output = apply_dishonour_rules(
        output,
        resources_path / "dishonours_rules.csv",
    )
    output = apply_special_rules(output)
    output = identify_streams(output, reset_stream_ids=True)
    output = add_finv_category(output)
    return PipelineResult(
        transactions=output,
        original_columns=tuple(transactions.columns),
    )
