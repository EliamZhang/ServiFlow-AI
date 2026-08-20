from __future__ import annotations

import pandas as pd

from classification_core.models import (
    EngineContext,
    EngineResult,
    SummaryArtifact,
    TRANSACTION_KEY_COLUMNS,
)

from .pipeline import run_pipeline


def _prior_claim_keys(context: EngineContext) -> set[tuple[str, str]]:
    """Keys (application_id, transaction_id) already claimed by earlier engines."""
    claims = context.prior_claims
    if claims is None or claims.empty:
        return set()
    key_frame = claims.loc[:, list(TRANSACTION_KEY_COLUMNS)].copy()
    key_frame = key_frame.astype("string").fillna("")
    return {tuple(row) for row in key_frame.itertuples(index=False, name=None)}


class FeeEngine:
    """Classify fee transactions by matching text against regex patterns.

    The engine examines each transaction's ``text`` field and matches against
    ~70 ordered regex rules covering international transaction fees, ATM operator
    fees, bank account fees, overdrawn/dishonour/late-payment fees, cash advance
    fees, and third-party maintenance/membership fees.

    Execution order is defined by the engine's ``priority`` in
    ``configs/pipeline.json`` (fee = 500), not by this file.
    """

    engine_id = "fee"
    engine_version = "1.0"

    # ------------------------------------------------------------------
    # ClassificationEngine protocol
    # ------------------------------------------------------------------

    def classify(self, context: EngineContext) -> EngineResult:
        result = run_pipeline(context.candidates)
        details = result.transactions

        matched = details[details["is_fee_pred"].eq(1)].copy()

        # Rules flagged ``unclassified_only`` (the fee keyword catch-all) only
        # apply to rows no earlier engine has claimed — otherwise they would
        # override correct classifications (school fees → Education,
        # "Body Corp Fees" in a rent transfer, DISHONOUR FEE → Dishonours, …).
        if "fee_unclassified_only" in matched.columns:
            uncl_only = (
                matched["fee_unclassified_only"].fillna(False)
            )
            if uncl_only.any():
                prior_keys = _prior_claim_keys(context)
                key_frame = matched.loc[:, list(TRANSACTION_KEY_COLUMNS)].astype(
                    "string"
                ).fillna("")
                key_tuples = [
                    tuple(row)
                    for row in key_frame.itertuples(index=False, name=None)
                ]
                drop_mask = uncl_only & pd.Series(
                    [k in prior_keys for k in key_tuples], index=matched.index
                )
                matched = matched[~drop_mask]
        predictions = pd.DataFrame(
            {
                **{col: matched[col].values for col in TRANSACTION_KEY_COLUMNS},
                "matched": True,
                "counterparty": matched["counterparty"].values,
                "finv_category": matched["finv_category"].values,
                "stream_id": matched["stream_id"].values,
                "classification_rule_id": matched["fee_rule_name"].values,
                "classification_reason": matched["fee_pred_reason"].values,
            }
        )

        return EngineResult(
            predictions=predictions,
            transactions=details,
            diagnostics=result.diagnostics,
        )

    def summarize(
        self,
        context: EngineContext,
        result: EngineResult,
        accepted_predictions: pd.DataFrame,
    ) -> list[SummaryArtifact]:
        # Fee engine is simple — no detailed summary needed beyond counts.
        return []
