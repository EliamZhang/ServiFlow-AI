"""模型服务的公共业务逻辑：输入 JSON dict → 运行流水线 → 输出 JSON dict。

供两个入口共用：
- model_main.py：生产环境推理入口（PredictMain.predict）。
- verify_model.py：本地验证脚本（CLI）。
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
from .models import ClassificationRunResult
from .orchestrator import ClassificationOrchestrator

# orchestrator 输出列中不属于业务结果的内部列，序列化时排除
_INTERNAL_OUTPUT_COLUMNS = frozenset(
    {
        "classification_status",
        "classification_engine",
        "classification_engine_version",
        "classification_priority",
        "classification_rule_id",
        "classification_reason",
        "stream_id",
    }
)

# 与 model_output.json 样例一致的输入顶层字段
_INPUT_ECHO_TOP_KEYS = ("userId", "applicationId", "flowTime")

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

# 账户元数据只保留在顶层 bankAccounts，行级不重复输出
_ACCOUNT_METADATA_COLUMNS = frozenset({"account_type", "bank", "credit_limit"})

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


class ModelService:
    """模型服务入口：加载一次配置，逐次对单个 application 的输入 dict 执行推理。"""

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
        """对单个 application 的输入 dict 执行流水线，返回可 JSON 序列化的结果 dict。"""
        try:
            transactions = build_transactions_frame(payload)
        except Exception as exc:
            return _build_error_output(payload, str(exc))
        result = self.orchestrator.run(transactions)
        return serialize_result(result, payload)


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


def _max_date(values: list[Any]) -> str | None:
    dates = [v for v in values if v is not None and str(v).strip()]
    return max(dates) if dates else None


def build_stats(
    transactions: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "txnRawInputCnt": len(transactions),
        "transactionDateMax": _max_date(
            transactions["transaction_date"].tolist()
        ),
    }


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
    output["stats"] = build_stats(result.transactions)
    output["bankAccounts"] = build_bank_accounts(payload)

    transactions_frame = result.transactions
    transactions = _serialize_records(
        transactions_frame,
        field_map=_TRANSACTION_OUTPUT_MAP,
        exclude=_INTERNAL_OUTPUT_COLUMNS | _ACCOUNT_METADATA_COLUMNS,
    )
    output["transactions"] = transactions

    summaries: dict[str, list[dict[str, Any]]] = {}
    for artifact in result.summaries:
        summaries[artifact.name] = _serialize_records(
            artifact.data,
            field_map=_SUMMARY_OUTPUT_MAP,
            exclude=_ACCOUNT_METADATA_COLUMNS,
        )
    output["summaries"] = summaries
    return output


def _build_error_output(payload: dict, error: str) -> dict:
    output: dict[str, Any] = {
        "runId": None,
        "status": "failed",
        "error": error,
        "stats": {
            "txnRawInputCnt": 0,
            "transactionDateMax": None,
        },
    }
    for key in _INPUT_ECHO_TOP_KEYS:
        if key in payload:
            output[key] = payload[key]
    output["transactions"] = []
    output["summaries"] = {}
    return output
