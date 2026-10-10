from __future__ import annotations

from pathlib import Path

import pandas as pd

from classification_core.models import PipelineResult

from .domain.counterparty import (
    apply_counterparty_rules,
    apply_credit_card_rules,
    apply_debt_collection_flag,
    apply_debt_consolidation_flag,
    apply_generic_loan_catchall,
    apply_home_loan_car_loan_flags,
    apply_overdrawn_flag,
)
from .domain.dishonours import apply_dishonour_rules
from .domain.special_rules import apply_special_rules
from .domain.streams import (
    add_bscat,
    align_product_type_with_stream,
    assign_generic_catchall_stream_ids,
    identify_streams,
    renumber_stream_ids_uniform,
)


DEFAULT_RESOURCES_DIR = Path(__file__).resolve().parent / "resources"


def run_pipeline(
    transactions: pd.DataFrame,
    resources_dir: str | Path = DEFAULT_RESOURCES_DIR,
    *,
    prior_claims: pd.DataFrame | None = None,
) -> PipelineResult:
    """Classify liabilities in an in-memory transaction dataframe.

    ``prior_claims`` (optional) is passed to the final stream-numbering
    stage, which must not spend a stream id on a stream whose rows a prior
    engine finally owns (gambling, 特例5) -- see
    ``domain.streams._PRIOR_CLAIM_ENGINES_STREAM_EXCLUDES``.
    """
    resources_path = Path(resources_dir)
    output = apply_counterparty_rules(
        transactions,
        resources_path / "counterparty_keyword_rules.csv",
    )
    output = apply_home_loan_car_loan_flags(
        output,
        resources_path / "home_loan_car_loan_rules.csv",
    )
    output = apply_credit_card_rules(output, resources_path / "credit_card_rules.csv")
    output = apply_dishonour_rules(
        output,
        resources_path / "dishonours_rules.csv",
    )
    output = apply_special_rules(output)
    output = apply_overdrawn_flag(output, resources_path / "overdrawn_rules.csv")
    output = apply_debt_collection_flag(
        output,
        resources_path / "debt_collection_rules.csv",
    )
    output = apply_debt_consolidation_flag(
        output,
        resources_path / "debt_consolidation_rules.csv",
    )
    output = identify_streams(output, reset_stream_ids=True)
    # One stream must own exactly one product_type before any later consumer
    # keys off it (the summary layer selects its builders by product_type).
    output = align_product_type_with_stream(output)
    output = add_bscat(output)
    output = apply_generic_loan_catchall(output)
    # The catchall manufactures its rows after stream identification ran, so
    # they never entered a product rule; give each application's group the
    # stream id it lacks before the final numbering stage.
    output = assign_generic_catchall_stream_ids(output)
    # Must stay last: bscat derivation (add_bscat) and every
    # product/type check inside identify_streams read the stream_id prefix.
    output = renumber_stream_ids_uniform(output, prior_claims=prior_claims)
    return PipelineResult(
        transactions=output,
    )
