"""Income classification engine package."""

from .wages_detector import IncomeClassificationResult, classify_income_transactions

__all__ = ["IncomeClassificationResult", "classify_income_transactions"]
