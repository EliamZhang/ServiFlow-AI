import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from apply_special_rules import apply_special_rules
from detect_dishonours import apply_dishonour_rules
from loan_dashboard import build_html, dataframe_to_records
from loan_summary import build_loan_summary, write_loan_summary_workbook_from_dataframe
from match_counterparty import apply_cc_rules, apply_counterparty_rules
from match_stream import add_final_product_type, identify_streams


FINAL_WORKBOOK = Path("output/sample_with_counterparty.xlsx")
FINAL_DASHBOARD = Path("output/loan_dashboard.html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run liability classification output generation."
    )
    parser.add_argument(
        "--with-dashboard",
        action="store_true",
        help="Also generate the HTML dashboard after writing Excel.",
    )
    return parser.parse_args()


def build_transactions() -> pd.DataFrame:
    transactions = pd.read_csv("sample.csv", encoding="utf-8-sig")
    transactions = apply_counterparty_rules(
        transactions,
        "resources/counterparty_keyword_rules.csv",
    )
    transactions = apply_cc_rules(
        transactions,
        "resources/cc_rules.csv",
    )
    transactions = apply_dishonour_rules(
        transactions,
        "resources/dishonours_rules.csv",
    )
    transactions = apply_special_rules(transactions)
    transactions = identify_streams(transactions, reset_stream_ids=True)
    return add_final_product_type(transactions)


def build_dashboard_data(
    transactions: pd.DataFrame,
    source_file: Path,
) -> dict[str, object]:
    summary = build_loan_summary(transactions)
    return {
        "loanSummary": dataframe_to_records(summary),
        "transactions": dataframe_to_records(transactions),
        "meta": {
            "sourceFile": source_file.name,
            "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "loanSummaryRows": len(summary),
            "transactionRows": len(transactions),
        },
    }


def write_dashboard_html(
    transactions: pd.DataFrame,
    workbook_file: Path,
    dashboard_file: Path,
) -> None:
    data = build_dashboard_data(transactions, workbook_file)
    html = build_html(data)
    dashboard_file.parent.mkdir(parents=True, exist_ok=True)
    dashboard_file.write_text(html, encoding="utf-8")


def main(with_dashboard: bool = False) -> None:
    transactions = build_transactions()
    write_loan_summary_workbook_from_dataframe(
        transactions,
        FINAL_WORKBOOK,
    )
    if with_dashboard:
        write_dashboard_html(transactions, FINAL_WORKBOOK, FINAL_DASHBOARD)


if __name__ == "__main__":
    args = parse_args()
    main(with_dashboard=args.with_dashboard)
