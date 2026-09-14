"""Shared model-service business logic: input JSON dict -> run pipeline -> output JSON dict.

Used by two entry points:
- model_main.py: production inference entry point (PredictMain.predict).
- verify_model.py: local verification script (CLI).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    DEFAULT_CATEGORY_CATALOG,
    DEFAULT_PIPELINE_CONFIG,
    load_category_owners,
    load_pipeline_config,
)
from .models import (
    ClassificationRunResult,
    TRANSACTION_KEY_COLUMNS,
)
from .orchestrator import ClassificationOrchestrator

# Internal orchestrator output columns that are not business results; excluded when serializing
_INTERNAL_OUTPUT_COLUMNS = frozenset(
    {
        "classification_status",
        "classification_engine",
        "classification_engine_version",
        "classification_priority",
        "classification_rule_id",
        "classification_reason",
    }
)

# Top-level input fields echoed back: {input key: output key}. Output keys are
# snake_case to match the Excel report column names (user_id / application_id /
# sample_datetime); values are carried through unchanged.
_INPUT_ECHO_TOP_KEYS = {
    "userId": "user_id",
    "applicationId": "application_id",
    "flowTime": "sample_datetime",
}

# Account metadata stays at the top-level bank_accounts only; not repeated per row
_ACCOUNT_METADATA_COLUMNS = frozenset({"account_type", "bank", "credit_limit"})
# Record keys are the frame column names verbatim (snake_case) so the JSON output
# field names line up with the Excel report columns written by
# classification_core/reporting.write_report.


class ModelService:
    """Model service entry point: loads configuration once, runs inference per single-application input dict."""

    def __init__(
        self,
        pipeline_config_path: str | Path = DEFAULT_PIPELINE_CONFIG,
        category_catalog_path: str | Path = DEFAULT_CATEGORY_CATALOG,
    ) -> None:
        self.orchestrator = ClassificationOrchestrator(
            config=load_pipeline_config(pipeline_config_path),
            category_owners=load_category_owners(category_catalog_path),
        )

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run the pipeline on a single application's input dict; return a JSON-serializable result dict."""
        try:
            transactions = build_transactions_frame(payload)
        except Exception as exc:
            return _build_error_output(payload, str(exc))
        result = self.orchestrator.run(transactions)
        return serialize_result(result, payload)


def _serialize_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    # Engine summary layers turn empty values into empty strings via normalize_text
    # (e.g. credit_limit); serialize those as null
    if isinstance(value, str) and value == "":
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (pd.Timedelta, type(pd.NaT))):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (float, int, str, bool)):
        return value
    return str(value)


def _serialize_records(
    frame: pd.DataFrame,
    exclude: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        record: dict[str, Any] = {}
        for column in frame.columns:
            if column in exclude:
                continue
            record[column] = _serialize_value(row[column])
        records.append(record)
    return records


def build_transactions_frame(payload: dict) -> pd.DataFrame:
    transactions = payload.get("illion_raw_transactions")
    if not isinstance(transactions, list):
        raise ValueError(
            "Input JSON must contain an 'illion_raw_transactions' list."
        )
    if not transactions:
        # Zero-transaction input (empty bank card / statement arrays) is legal:
        # return an empty frame that still carries the key columns and
        # transaction_date so the orchestrator short-circuit and the stats
        # builder can produce a structurally correct empty success result.
        return pd.DataFrame(
            {
                column: pd.Series(dtype="object")
                for column in (*TRANSACTION_KEY_COLUMNS, "transaction_date")
            }
        )
    frame = pd.DataFrame(transactions)
    application_id = payload.get("applicationId")
    if application_id is not None and "application_id" not in frame.columns:
        frame["application_id"] = application_id

    bank_accounts = payload.get("bank_accounts", [])
    if bank_accounts:
        accounts_frame = pd.DataFrame(
            [
                {
                    "bank_account_id": account.get("bank_account_id"),
                    "account_type": account.get("account_type"),
                    "bank": account.get("bank"),
                    "credit_limit": account.get("credit_limit"),
                }
                for account in bank_accounts
                if isinstance(account, dict)
            ]
        )
        for column in ("account_type", "bank", "credit_limit"):
            if column not in frame.columns:
                frame = frame.merge(
                    accounts_frame[["bank_account_id", column]],
                    on="bank_account_id",
                    how="left",
                    suffixes=("", f"_{column}"),
                )
    return frame


def build_bank_accounts(payload: dict) -> list[dict[str, Any]]:
    bank_accounts = payload.get("bank_accounts", [])
    return [
        {
            "bank_account_id": account.get("bank_account_id"),
            "account_type": account.get("account_type"),
            "bank": account.get("bank"),
            "credit_limit": account.get("credit_limit"),
        }
        for account in bank_accounts
        if isinstance(account, dict)
    ]


def _max_date(values: list[Any]) -> str | None:
    dates = [v for v in values if v is not None and str(v).strip()]
    return max(dates) if dates else None


def build_stats(
    transactions: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "txn_raw_input_cnt": len(transactions),
        "transaction_date_max": _max_date(
            transactions["transaction_date"].tolist()
        ),
    }


def serialize_result(
    result: ClassificationRunResult,
    payload: dict,
) -> dict:
    output: dict[str, Any] = {}
    for input_key, output_key in _INPUT_ECHO_TOP_KEYS.items():
        if input_key in payload:
            output[output_key] = payload[input_key]
    output["run_id"] = result.run_id
    output["status"] = "success"
    output["error"] = None
    output["stats"] = build_stats(result.transactions)
    output["bank_accounts"] = build_bank_accounts(payload)

    transactions_frame = result.transactions
    transactions = _serialize_records(
        transactions_frame,
        exclude=_INTERNAL_OUTPUT_COLUMNS | _ACCOUNT_METADATA_COLUMNS,
    )
    output["transactions"] = transactions

    summaries: dict[str, list[dict[str, Any]]] = {}
    for artifact in result.summaries:
        summaries[artifact.name] = _serialize_records(
            artifact.data,
            exclude=_ACCOUNT_METADATA_COLUMNS,
        )
    output["summaries"] = summaries
    return output


def _build_error_output(payload: dict, error: str) -> dict:
    output: dict[str, Any] = {
        "run_id": None,
        "status": "failed",
        "error": error,
        "stats": {
            "txn_raw_input_cnt": 0,
            "transaction_date_max": None,
        },
    }
    for input_key, output_key in _INPUT_ECHO_TOP_KEYS.items():
        if input_key in payload:
            output[output_key] = payload[input_key]
    output["transactions"] = []
    output["summaries"] = {}
    return output
