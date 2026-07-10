from __future__ import annotations

from typing import Protocol

import pandas as pd

from .models import EngineContext, PredictionBatch, SummaryArtifact


class ClassificationEngine(Protocol):
    engine_id: str
    engine_version: str

    def classify(self, context: EngineContext) -> PredictionBatch:
        """Return classification proposals for context.candidates only."""

    def summarize(
        self,
        context: EngineContext,
        batch: PredictionBatch,
        accepted_predictions: pd.DataFrame,
    ) -> list[SummaryArtifact]:
        """Build summaries from proposals accepted by the orchestrator."""
