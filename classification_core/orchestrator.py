from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from uuid import uuid4

import pandas as pd

from .category_mapping import to_illion_category
from .config import PipelineConfig
from .engine import ClassificationEngine
from .models import (
    ClassificationRunResult,
    EngineContext,
    EngineExecution,
    EngineResult,
    SummaryArtifact,
    TRANSACTION_KEY_COLUMNS,
)
from .registry import build_engine


PREDICTION_REQUIRED_COLUMNS = {
    *TRANSACTION_KEY_COLUMNS,
    "matched",
    "counterparty",
    "finv_category",
}

_UNCLASSIFIED_SENTINEL = "unclassified"


# ── key helpers ─────────────────────────────────────────────────────────────

def _key_tuples(df: pd.DataFrame) -> list[tuple[str, str]]:
    key_frame = df.loc[:, list(TRANSACTION_KEY_COLUMNS)].copy()
    key_frame = key_frame.astype("string").fillna("")
    return [tuple(row) for row in key_frame.itertuples(index=False, name=None)]


def _is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


# ── orchestrator ────────────────────────────────────────────────────────────

class ClassificationOrchestrator:
    def __init__(
        self,
        config: PipelineConfig,
        category_owners: dict[str, str],
        engine_factory: Callable[[str], ClassificationEngine] = build_engine,
    ) -> None:
        self.config = config
        self.category_owners = category_owners
        self.engine_factory = engine_factory

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def run(self, transactions: pd.DataFrame) -> ClassificationRunResult:
        original = self._prepare_input(transactions)
        run_id = str(uuid4())
        output = self._initialize_output(original)
        key_to_index = dict(zip(_key_tuples(original), output.index))
        summaries: list[SummaryArtifact] = []
        executions: list[EngineExecution] = []

        for spec in self.config.enabled_engines:
            engine = self.engine_factory(spec.engine_id)

            # All engines see all transactions; later engines overwrite earlier ones
            # at the row level (finv_category + counterparty as a pair).
            candidate_mask = pd.Series(True, index=output.index)

            context = EngineContext(
                run_id=run_id,
                all_transactions=original.copy(),
                candidates=original.copy(),
                prior_claims=output.loc[
                    output["classification_status"].ne(_UNCLASSIFIED_SENTINEL),
                    [
                        *TRANSACTION_KEY_COLUMNS,
                        "counterparty",
                        "finv_category",
                        "classification_engine",
                    ],
                ].copy(),
            )

            engine_started = perf_counter()
            engine_result = engine.classify(context)
            accepted = self._validate_predictions(engine, context, engine_result)
            self._commit(
                output=output,
                key_to_index=key_to_index,
                engine=engine,
                priority=spec.priority,
                predictions=accepted,
            )
            engine_summaries = engine.summarize(context, engine_result, accepted)
            engine_seconds = perf_counter() - engine_started
            summaries.extend(engine_summaries)
            executions.append(
                EngineExecution(
                    engine_id=engine.engine_id,
                    engine_version=engine.engine_version,
                    priority=spec.priority,
                    candidate_count=int(candidate_mask.sum()),
                    prediction_count=len(engine_result.predictions),
                    accepted_count=len(accepted),
                    duration_seconds=engine_seconds,
                    diagnostics=engine_result.diagnostics,
                )
            )

        return ClassificationRunResult(
            run_id=run_id,
            transactions=output,
            summaries=summaries,
            executions=executions,
        )

    # ------------------------------------------------------------------
    # input / initialisation
    # ------------------------------------------------------------------

    def _prepare_input(self, transactions: pd.DataFrame) -> pd.DataFrame:
        missing = [
            column
            for column in TRANSACTION_KEY_COLUMNS
            if column not in transactions.columns
        ]
        if missing:
            raise ValueError(
                "Input transactions are missing key column(s): "
                + ", ".join(missing)
            )

        original = transactions.reset_index(drop=True).copy()
        keys = _key_tuples(original)
        if any(not all(key) for key in keys):
            raise ValueError("Transaction key columns cannot contain blank values.")
        if len(keys) != len(set(keys)):
            raise ValueError(
                "Each (application_id, transaction_id) pair must be unique."
            )
        return original

    def _initialize_output(self, original: pd.DataFrame) -> pd.DataFrame:
        output = original.copy()
        output["counterparty"] = pd.NA
        output["finv_category"] = pd.NA
        output["classification_status"] = _UNCLASSIFIED_SENTINEL
        output["classification_engine"] = pd.NA
        output["classification_engine_version"] = pd.NA
        output["classification_priority"] = pd.NA
        output["classification_rule_id"] = pd.NA
        output["classification_reason"] = pd.NA
        output["stream_id"] = pd.NA
        return output

    # ------------------------------------------------------------------
    # validation (unchanged semantics)
    # ------------------------------------------------------------------

    def _validate_predictions(
        self,
        engine: ClassificationEngine,
        context: EngineContext,
        result: EngineResult,
    ) -> pd.DataFrame:
        if not isinstance(result, EngineResult):
            raise TypeError(
                f"Engine {engine.engine_id!r} must return EngineResult."
            )
        predictions = result.predictions.copy()
        missing = sorted(PREDICTION_REQUIRED_COLUMNS.difference(predictions.columns))
        if missing:
            raise ValueError(
                f"Engine {engine.engine_id!r} predictions are missing: "
                + ", ".join(missing)
            )

        invalid_matched_values = ~predictions["matched"].isin([True, False])
        if invalid_matched_values.any():
            raise ValueError(
                f"Engine {engine.engine_id!r} returned a non-boolean matched value."
            )
        predictions = predictions[predictions["matched"]].copy()
        prediction_keys = _key_tuples(predictions)
        if len(prediction_keys) != len(set(prediction_keys)):
            raise ValueError(
                f"Engine {engine.engine_id!r} returned duplicate transaction keys."
            )

        candidate_keys = set(_key_tuples(context.candidates))
        outside_candidates = set(prediction_keys).difference(candidate_keys)
        if outside_candidates:
            raise ValueError(
                f"Engine {engine.engine_id!r} attempted to claim a transaction "
                "outside its candidate set."
            )

        blank_core = _is_blank(predictions["counterparty"])
        if blank_core.any():
            raise ValueError(
                f"Engine {engine.engine_id!r} returned blank counterparty for "
                f"{int(blank_core.sum())} matched transaction(s)."
            )

        categories = predictions["finv_category"].astype(str)
        invalid_categories = sorted(
            {
                category
                for category in categories
                if category
                and engine.engine_id not in self.category_owners.get(category, "").replace(" ", "").split(",")
            }
        )
        if invalid_categories:
            raise ValueError(
                f"Engine {engine.engine_id!r} returned unowned category/categories: "
                + ", ".join(invalid_categories)
            )
        return predictions

    # ------------------------------------------------------------------
    # commit (row-level — later engines overwrite earlier ones entirely)
    # ------------------------------------------------------------------

    @staticmethod
    def _commit(
        output: pd.DataFrame,
        key_to_index: dict[tuple[str, str], int],
        engine: ClassificationEngine,
        priority: int,
        predictions: pd.DataFrame,
    ) -> None:
        """Write predictions to output, overwriting both finv_category and
        counterparty as a pair.  Later engines always win at the row level."""
        for (_, prediction), key in zip(
            predictions.iterrows(),
            _key_tuples(predictions),
        ):
            row_index = key_to_index[key]

            output.at[row_index, "finv_category"] = to_illion_category(
                prediction["finv_category"]
            )
            output.at[row_index, "classification_status"] = "classified"
            output.at[row_index, "classification_engine"] = engine.engine_id
            output.at[row_index, "classification_engine_version"] = (
                engine.engine_version
            )
            output.at[row_index, "classification_priority"] = priority
            output.at[row_index, "classification_rule_id"] = prediction.get(
                "classification_rule_id", pd.NA
            )
            output.at[row_index, "classification_reason"] = prediction.get(
                "classification_reason", pd.NA
            )
            output.at[row_index, "stream_id"] = prediction.get(
                "stream_id", pd.NA
            )

            if pd.notna(prediction.get("counterparty")):
                output.at[row_index, "counterparty"] = prediction["counterparty"]
