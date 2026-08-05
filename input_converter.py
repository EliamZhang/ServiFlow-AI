"""从 sample.csv 提取指定 application，转换为 model_predict.py 的 JSON 入参结构。

用法：python input_converter.py [--input sample.csv] [--output output/]
运行后列出 CSV 中所有 application（user_id + application_id + 交易数），
输入 application_id 后生成对应 JSON 入参文件。
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "sample.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


def _to_iso_date(value: str) -> str:
    if not value or pd.isna(value):
        return value
    return datetime.strptime(str(value).strip(), "%Y/%m/%d").strftime("%Y-%m-%d")


def _to_amount(value: str, dr_cr: str) -> float:
    amount = float(value)
    return -amount if dr_cr == "debit" else amount


def _to_credit_limit(value: str) -> float | None:
    if value == "null" or not value or pd.isna(value):
        return None
    return float(value)


def build_payload(frame: pd.DataFrame) -> dict:
    app_id = str(int(frame.iloc[0]["application_id"]))
    user_id = str(int(frame.iloc[0]["user_id"]))
    flow_time = str(frame.iloc[0]["sample_datetime"]).strip()
    if flow_time and "." not in flow_time:
        flow_time += ".0"

    accounts: list[dict] = []
    seen = set()
    for _, row in frame.iterrows():
        account_id = str(int(row["bank_account_id"]))
        if account_id in seen:
            continue
        seen.add(account_id)
        accounts.append(
            {
                "bank_account_id": int(account_id),
                "account_type": str(row["account_type"]),
                "bank": str(row["bank"]),
                "credit_limit": _to_credit_limit(row["credit_limit"]),
            }
        )

    transactions = []
    for _, row in frame.iterrows():
        transactions.append(
            {
                "amount": _to_amount(row["amount"], row["dr_cr"]),
                "balance": float(row["balance"]),
                "bank_account_id": int(row["bank_account_id"]),
                "category": str(row["category"]),
                "dr_cr": str(row["dr_cr"]),
                "illion_trx_uuid": str(row["illion_trx_uuid"]),
                "text": str(row["text"]),
                "third_party": str(row["third_party"]),
                "transaction_date": _to_iso_date(row["transaction_date"]),
                "transaction_id": int(row["transaction_id"]),
                "trx_type": str(row["trx_type"]),
            }
        )

    return {
        "userId": int(user_id),
        "applicationId": int(app_id),
        "flowTime": flow_time,
        "bank_accounts": accounts,
        "illion_raw_transactions": transactions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8-sig")

    apps: dict[str, set[str]] = defaultdict(set)
    for _, row in frame.iterrows():
        apps[str(int(row["application_id"]))].add(str(int(row["user_id"])))

    apps = dict(sorted(apps.items(), key=lambda item: int(item[0])))
    print("CSV 中的 application 列表：")
    for app_id, user_ids in apps.items():
        count = (frame["application_id"] == int(app_id)).sum()
        print(f"  application_id={app_id}  user_id={','.join(sorted(user_ids))}  交易数={count}")

    while True:
        chosen = input("\n输入要转换的 application_id（输入 q 退出）：").strip()
        if chosen.lower() in ("q", "quit"):
            return
        if chosen not in apps:
            print(f"未找到 application_id={chosen}，请重新输入")
            continue
        break

    selected = frame[frame["application_id"] == int(chosen)].copy()
    payload = build_payload(selected)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"model_input_{chosen}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已生成 {out_path}（{len(payload['illion_raw_transactions'])} 笔交易）")


if __name__ == "__main__":
    main()
