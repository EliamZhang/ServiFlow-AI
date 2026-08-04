from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


TRANSACTION_KEY_COLUMNS = ("application_id", "transaction_id")


@dataclass(frozen=True)
class PipelineResult:
    transactions: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)
    original_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class EngineContext:
    run_id: str
    all_transactions: pd.DataFrame
    candidates: pd.DataFrame
    prior_claims: pd.DataFrame


@dataclass
class EngineResult:
    predictions: pd.DataFrame
    transactions: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SummaryArtifact:
    name: str
    data: pd.DataFrame


@dataclass
class EngineExecution:
    engine_id: str
    engine_version: str
    priority: int
    candidate_count: int
    prediction_count: int
    accepted_count: int
    duration_seconds: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    claims: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ClassificationRunResult:
    run_id: str
    transactions: pd.DataFrame
    summaries: list[SummaryArtifact]
    executions: list[EngineExecution]
