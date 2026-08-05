from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from classification_core.config import (
    DEFAULT_CATEGORY_CATALOG,
    DEFAULT_PIPELINE_CONFIG,
    load_category_owners,
    load_pipeline_config,
)
from classification_core.models import ClassificationRunResult
from classification_core.orchestrator import ClassificationOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

# orchestrator 输出列中不属于业务结果的内部列，序列化时排除
_INTERNAL_OUTPUT_COLUMNS = frozenset({"classification_status"})

# 与 model_output.json 样例一致的输入顶层字段
_INPUT_ECHO_TOP_KEYS = ("userId", "applicationId")

# 输入 original 中可能存在的、序列化时需转换命名的列
_TRANSACTION_OUTPUT_MAP = {
    "bank_account_id": "bankAccountId",
    "transaction_id": "transactionId",
    "transaction_date": "transactionDate",
    "dr_cr": "drCr",
    "third_party": "thirdParty",
    "trx_type": "trxType",
    "illion_trx_uuid": "illionTrxUuid",
}

_SUMMARY_OUTPUT_MAP = {
    "finv_category": "finvCategory",
    "income_category": "incomeCategory",
    "liability_category": "liabilityCategory",
    "bank_account_id": "bankAccountId",
    "transaction_start_date": "transactionStartDate",
    "transaction_end_date": "transactionEndDate",
    "total_income_amount": "totalIncomeAmount",
    "average_income_amount": "averageIncomeAmount",
    "median_income_amount": "medianIncomeAmount",
    "latest_income_amount": "latestIncomeAmount",
    "estimated_monthly_income": "estimatedMonthlyIncome",
    "frequency_day": "frequencyDay",
    "predicted_next_income_date": "predictedNextIncomeDate",
    "funded_amount": "fundedAmount",
    "repaid_amount": "repaidAmount",
    "repayment_amount": "repaymentAmount",
    "recent_fn_repay_amount": "recentFnRepayAmount",
    "predicted_closing_date": "predictedClosingDate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a single application from JSON and write JSON output."
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "model_input.json"),
        help="Input application JSON path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. Default: output/model_output_{applicationId}_"
            "{YYYYMMDD_HHMMSS}.json"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_PIPELINE_CONFIG),
        help="Pipeline JSON configuration path.",
    )
    parser.add_argument(
        "--category-catalog",
        default=str(DEFAULT_CATEGORY_CATALOG),
        help="Category catalog JSON path.",
    )
    return parser.parse_args()


def _to_camel(snake: str) -> str:
    head, *rest = snake.split("_")
    return head + "".join(part.capitalize() for part in rest)


def _serialize_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    # 引擎汇总层用 normalize_text 把空值转成空串（如 credit_limit），序列化为 null
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
    field_map: dict[str, str] | None = None,
    exclude: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        record: dict[str, Any] = {}
        for column in frame.columns:
            if column in exclude:
                continue
            name = (
                field_map.get(column, _to_camel(column))
                if field_map is not None
                else _to_camel(column)
            )
            record[name] = _serialize_value(row[column])
        records.append(record)
    return records


def load_input(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    return payload


def build_transactions_frame(payload: dict) -> pd.DataFrame:
    transactions = payload.get("illion_raw_transactions")
    if not isinstance(transactions, list) or not transactions:
        raise ValueError(
            "Input JSON must contain a non-empty 'illion_raw_transactions' list."
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
            "bankAccountId": account.get("bank_account_id"),
            "accountType": account.get("account_type"),
            "bank": account.get("bank"),
            "creditLimit": account.get("credit_limit"),
        }
        for account in bank_accounts
        if isinstance(account, dict)
    ]


def _resolve_output_path(output_arg: str | None, payload: dict) -> Path:
    if output_arg:
        return Path(output_arg)
    application_id = payload.get("applicationId")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"model_output_{application_id}_{timestamp}.json"


def serialize_result(
    result: ClassificationRunResult,
    payload: dict,
) -> dict:
    output: dict[str, Any] = {}
    for key in _INPUT_ECHO_TOP_KEYS:
        if key in payload:
            output[key] = payload[key]
    output["runId"] = result.run_id
    output["status"] = "success"
    output["error"] = None
    output["bankAccounts"] = build_bank_accounts(payload)

    transactions_frame = result.transactions
    transactions = _serialize_records(
        transactions_frame,
        field_map=_TRANSACTION_OUTPUT_MAP,
        exclude=_INTERNAL_OUTPUT_COLUMNS,
    )
    output["transactions"] = transactions

    summaries: dict[str, list[dict[str, Any]]] = {}
    for artifact in result.summaries:
        summaries[artifact.name] = _serialize_records(
            artifact.data,
            field_map=_SUMMARY_OUTPUT_MAP,
        )
    output["summaries"] = summaries
    return output


def main() -> None:
    args = parse_args()
    payload = load_input(args.input)
    output_path = _resolve_output_path(args.output, payload)

    try:
        transactions = build_transactions_frame(payload)
        orchestrator = ClassificationOrchestrator(
            config=load_pipeline_config(args.config),
            category_owners=load_category_owners(args.category_catalog),
        )
        result = orchestrator.run(transactions)
    except Exception as exc:
        output = {
            "runId": None,
            "status": "failed",
            "error": str(exc),
        }
        for key in _INPUT_ECHO_TOP_KEYS:
            if key in payload:
                output[key] = payload[key]
        output["transactions"] = []
        output["summaries"] = {}
        print(json.dumps(output, ensure_ascii=False, indent=2))
        raise

    output = serialize_result(result, payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
