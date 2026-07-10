from __future__ import annotations

from collections.abc import Callable

from income_classification_engine.engine import IncomeEngine
from liability_classification_engine.engine import LiabilityEngine

from .engine import ClassificationEngine


ENGINE_FACTORIES: dict[str, Callable[[], ClassificationEngine]] = {
    "income": IncomeEngine,
    "liability": LiabilityEngine,
}


def build_engine(engine_id: str) -> ClassificationEngine:
    try:
        factory = ENGINE_FACTORIES[engine_id]
    except KeyError as exc:
        available = ", ".join(sorted(ENGINE_FACTORIES))
        raise ValueError(
            f"Unknown engine {engine_id!r}. Registered engines: {available}"
        ) from exc
    return factory()
