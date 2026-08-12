from __future__ import annotations

from collections.abc import Callable

from all_other_credit_engine import AllOtherCreditEngine
from catch_all_engine import CatchAllEngine
from dishonour_engine import DishonourEngine
from fee_engine import FeeEngine
from income_engine import IncomeEngine
from initial_engine import InitialClassificationEngine
from liability_engine import LiabilityEngine
from rent_engine import RentEngine
from transfer_engine import TransferEngine

from .engine import ClassificationEngine


ENGINE_FACTORIES: dict[str, Callable[[], ClassificationEngine]] = {
    "all_other_credit": AllOtherCreditEngine,
    "catch_all": CatchAllEngine,
    "dishonour": DishonourEngine,
    "fee": FeeEngine,
    "initial": InitialClassificationEngine,
    "income": IncomeEngine,
    "liability": LiabilityEngine,
    "rent": RentEngine,
    "transfer": TransferEngine,
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
