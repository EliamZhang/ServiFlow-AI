from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .apply_special_rules import apply_special_rules
from .detect_dishonours import apply_dishonour_rules
from .match_counterparty import apply_cc_rules, apply_counterparty_rules
from .match_stream import add_finv_category, identify_streams


ENGINE_DIR = Path(__file__).resolve().parent
DEFAULT_RESOURCES_DIR = ENGINE_DIR / "resources"


@dataclass(frozen=True)
class LiabilityClassificationResult:
    transactions: pd.DataFrame
    diagnostics: dict[str, int]


def classify_liability_transactions(
    transactions: pd.DataFrame,
    resources_dir: str | Path = DEFAULT_RESOURCES_DIR,
) -> LiabilityClassificationResult:
    """Classify liabilities in an in-memory transaction dataframe."""
    resources_path = Path(resources_dir)
    output = apply_counterparty_rules(
        transactions,
        resources_path / "counterparty_keyword_rules.csv",
    )
    output = apply_cc_rules(output, resources_path / "cc_rules.csv")
    output = apply_dishonour_rules(
        output,
        resources_path / "dishonours_rules.csv",
    )
    output = apply_special_rules(output)
    output, diagnostics = identify_streams(output, reset_stream_ids=True)
    output = add_finv_category(output)
    return LiabilityClassificationResult(output, diagnostics)
