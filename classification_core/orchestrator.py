from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from uuid import uuid4

import pandas as pd

from .config import FieldPolicy, PipelineConfig
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

# Internal tracking columns (prefixed with _ to signal "private").
_FC_SET_BY = "_fc_set_by"
_CP_SET_BY = "_cp_set_by"

_UNCLASSIFIED_SENTINEL = "unclassified"


# ── key helpers ─────────────────────────────────────────────────────────────

def _key_tuples(df: pd.DataFrame) -> list[tuple[str, str]]:
    key_frame = df.loc[:, list(TRANSACTION_KEY_COLUMNS)].copy()
    key_frame = key_frame.astype("string").fillna("")
    return [tuple(row) for row in key_frame.itertuples(index=False, name=None)]


def _is_blank(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().eq("")


def _is_value_blank(value: object) -> bool:
    """Check a scalar value for blankness (NA, empty, or unclassified sentinel)."""
    if pd.isna(value):
        return True
    stripped = str(value).strip()
    return stripped == "" or stripped == _UNCLASSIFIED_SENTINEL


def _compute_field_mask(
    output: pd.DataFrame,
    engine_id: str,
    set_by_column: str,
    policy: FieldPolicy | None,
) -> pd.Series:
    """Return a boolean mask of rows where *engine_id* may write this field.

    When *policy* is ``None`` the field is treated as first-write-wins
    (the engine can write only if the field has never been set).
    """
    set_by: pd.Series = output[set_by_column]
    n_rows = len(output)

    if policy is None:
        # Classic: engine can write only when the field is unset.
        return set_by.isna()

    immutable = policy.immutable_when_set_by

    # ── immutable guard ──
    if immutable:
        can_write = ~set_by.isin(immutable)  # NA → False → ~True
    else:
        can_write = pd.Series(True, index=output.index)

    # ── fill-blank-only guard ──
    if policy.fill_blank_only:
        # Determine which rows have a non-blank value.
        value_col = (
            output["finv_category"]
            if set_by_column == _FC_SET_BY
            else output["counterparty"]
        )
        has_value = ~(value_col.isna() | value_col.astype("string").str.strip().eq("").values | (value_col == _UNCLASSIFIED_SENTINEL))

        # Rows with a value are blocked … unless an explicit override applies.
        blocked = has_value.copy()
        for rule in policy.override_rules:
            if rule.by == engine_id:
                overridable = set_by.isin(rule.when_set_by)  # NA → False
                blocked = blocked & ~overridable
        can_write = can_write & ~blocked

    return can_write


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

        fc_policy = self.config.field_policies.get("finv_category")
        cp_policy = self.config.field_policies.get("counterparty")

        for spec in self.config.enabled_engines:
            engine = self.engine_factory(spec.engine_id)

            # ── build candidate mask: rows where this engine may write
            #     finv_category *or* counterparty (whichever is still open) ──
            candidate_mask = self._build_candidate_mask(
                output, engine.engine_id, fc_policy, cp_policy,
            )
            claimed_mask = ~candidate_mask

            context = EngineContext(
                run_id=run_id,
                all_transactions=original.copy(),
                candidates=original.loc[candidate_mask].copy(),
                prior_claims=output.loc[
                    claimed_mask,
                    [
                        *TRANSACTION_KEY_COLUMNS,
                        "counterparty",
                        "finv_category",
                        "classification_engine",
                    ],
                ].copy() if claimed_mask.any() else pd.DataFrame(
                    columns=[
                        *TRANSACTION_KEY_COLUMNS,
                        "counterparty",
                        "finv_category",
                        "classification_engine",
                    ]
                ),
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
                field_policies=self.config.field_policies,
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

        summaries.append(self._build_run_summary(executions, summaries))
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
        output["finv_category"] = self.config.unclassified_category
        output["classification_status"] = _UNCLASSIFIED_SENTINEL
        output["classification_engine"] = pd.NA
        output["classification_engine_version"] = pd.NA
        output["classification_priority"] = pd.NA
        output["classification_rule_id"] = pd.NA
        output["classification_reason"] = pd.NA
        output["stream_id"] = pd.NA
        # Internal tracking: which engine last wrote each field.
        output[_FC_SET_BY] = pd.NA
        output[_CP_SET_BY] = pd.NA
        return output

    # ------------------------------------------------------------------
    # candidate selection (field-level — union of both fields)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_candidate_mask(
        output: pd.DataFrame,
        engine_id: str,
        fc_policy: FieldPolicy | None,
        cp_policy: FieldPolicy | None,
    ) -> pd.Series:
        """Return a boolean mask of rows where *engine_id* may write at least
        one of ``finv_category`` or ``counterparty``.

        When neither policy is configured this falls back to the classic
        row-level ``classification_status == "unclassified"`` check.
        """
        if fc_policy is None and cp_policy is None:
            return output["classification_status"].eq(_UNCLASSIFIED_SENTINEL)

        can_set_fc = _compute_field_mask(
            output, engine_id, _FC_SET_BY, fc_policy,
        )
        can_set_cp = _compute_field_mask(
            output, engine_id, _CP_SET_BY, cp_policy,
        )
        return can_set_fc | can_set_cp

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

        blank_core = _is_blank(predictions["counterparty"]) | _is_blank(
            predictions["finv_category"]
        )
        if blank_core.any():
            raise ValueError(
                f"Engine {engine.engine_id!r} returned blank core fields for "
                f"{int(blank_core.sum())} matched transaction(s)."
            )

        categories = predictions["finv_category"].astype(str)
        invalid_categories = sorted(
            {
                category
                for category in categories
                if self.category_owners.get(category) != engine.engine_id
            }
        )
        if invalid_categories:
            raise ValueError(
                f"Engine {engine.engine_id!r} returned unowned category/categories: "
                + ", ".join(invalid_categories)
            )
        return predictions

    # ------------------------------------------------------------------
    # commit (field-level)
    # ------------------------------------------------------------------

    @staticmethod
    def _commit(
        output: pd.DataFrame,
        key_to_index: dict[tuple[str, str], int],
        engine: ClassificationEngine,
        priority: int,
        predictions: pd.DataFrame,
        field_policies: dict[str, FieldPolicy],
    ) -> None:
        fc_policy = field_policies.get("finv_category")
        cp_policy = field_policies.get("counterparty")

        for (_, prediction), key in zip(
            predictions.iterrows(),
            _key_tuples(predictions),
        ):
            row_index = key_to_index[key]

            # ── resolve current ownership ──
            current_fc_set_by: str | None = output.at[row_index, _FC_SET_BY]
            if pd.isna(current_fc_set_by):
                current_fc_set_by = None

            current_cp_set_by: str | None = output.at[row_index, _CP_SET_BY]
            if pd.isna(current_cp_set_by):
                current_cp_set_by = None

            current_fc_value = output.at[row_index, "finv_category"]
            current_cp_value = output.at[row_index, "counterparty"]

            # ── check field-level permissions ──
            can_set_fc = ClassificationOrchestrator._can_set_field(
                fc_policy,
                engine.engine_id,
                current_value=current_fc_value,
                current_set_by=current_fc_set_by,
            )
            can_set_cp = ClassificationOrchestrator._can_set_field(
                cp_policy,
                engine.engine_id,
                current_value=current_cp_value,
                current_set_by=current_cp_set_by,
            )

            if not can_set_fc and not can_set_cp:
                # Engine has nothing to contribute to this row.
                continue

            # ── write finv_category ──
            if can_set_fc:
                output.at[row_index, "finv_category"] = prediction["finv_category"]
                output.at[row_index, _FC_SET_BY] = engine.engine_id
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

            # ── write counterparty ──
            if can_set_cp and pd.notna(prediction.get("counterparty")):
                output.at[row_index, "counterparty"] = prediction["counterparty"]
                output.at[row_index, _CP_SET_BY] = engine.engine_id

    # ------------------------------------------------------------------
    # field-policy helper
    # ------------------------------------------------------------------

    @staticmethod
    def _can_set_field(
        policy: FieldPolicy | None,
        engine_id: str,
        *,
        current_value: object,
        current_set_by: str | None,
    ) -> bool:
        """Return *True* if *engine_id* is permitted to write this field.

        When *policy* is ``None`` (no policy configured) the classic
        first-write-wins behaviour applies: the field can be written as long as
        it has never been set before.
        """
        if policy is None:
            return current_set_by is None

        return policy.can_overwrite(
            engine_id,
            current_set_by=current_set_by,
            current_is_blank=_is_value_blank(current_value),
        )

    # ------------------------------------------------------------------
    # run summary
    # ------------------------------------------------------------------

    @staticmethod
    def _build_run_summary(
        executions: list[EngineExecution],
        summaries: list[SummaryArtifact],
    ) -> SummaryArtifact:
        summary_rows_by_engine = {
            artifact.name.removesuffix("_summary"): len(artifact.data)
            for artifact in summaries
            if artifact.name.endswith("_summary")
        }
        rows = [
            {
                "engine_id": execution.engine_id,
                "engine_version": execution.engine_version,
                "priority": execution.priority,
                "candidate_count": execution.candidate_count,
                "prediction_count": execution.prediction_count,
                "accepted_count": execution.accepted_count,
                "summary_row_count": summary_rows_by_engine.get(
                    execution.engine_id, 0
                ),
            }
            for execution in executions
        ]
        return SummaryArtifact(
            "run_summary",
            "1.0",
            pd.DataFrame(rows),
        )
