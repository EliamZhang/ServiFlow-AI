from __future__ import annotations

try:
    from typing import Protocol
except ImportError:  # Python 3.7
    try:
        from typing_extensions import Protocol
    except ImportError:  # Python 3.7 without typing_extensions installed
        class Protocol:  # type: ignore[no-redef]
            """Fallback: ClassificationEngine is type-annotation only, never checked at runtime."""

            pass

import pandas as pd

from .models import EngineContext, EngineResult, SummaryArtifact


class ClassificationEngine(Protocol):
    engine_id: str
    engine_version: str

    def classify(self, context: EngineContext) -> EngineResult:
        """Return classification proposals for context.candidates only."""

    def summarize(
        self,
        context: EngineContext,
        result: EngineResult,
        accepted_predictions: pd.DataFrame,
    ) -> list[SummaryArtifact]:
        """Build summaries from proposals accepted by the orchestrator."""
