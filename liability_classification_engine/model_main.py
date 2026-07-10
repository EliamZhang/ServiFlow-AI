import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from .loan_dashboard import build_html, dataframe_to_records
from .loan_summary import build_loan_summary, write_loan_summary_workbook_from_dataframe
from .pipeline import (
    DEFAULT_RESOURCES_DIR,
    ENGINE_DIR,
    classify_liability_transactions,
)


DEFAULT_INPUT = ENGINE_DIR / "sample.csv"
DEFAULT_WORKBOOK = ENGINE_DIR / "output" / "sample_with_counterparty.xlsx"
DEFAULT_DASHBOARD = ENGINE_DIR / "output" / "loan_dashboard.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run liability classification output generation."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_WORKBOOK))
    parser.add_argument(
        "--with-dashboard",
        action="store_true",
        help="Also generate the HTML dashboard after writing Excel.",
    )
    parser.add_argument(
        "--dashboard-output",
        default=str(DEFAULT_DASHBOARD),
    )
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    input_file = Path(args.input)
    output_file = Path(args.output)
    transactions = pd.read_csv(input_file, encoding="utf-8-sig")
    result = classify_liability_transactions(transactions)
    transactions = result.transactions
    write_loan_summary_workbook_from_dataframe(
        transactions,
        output_file,
        limits_file=DEFAULT_RESOURCES_DIR / "bnpl_maximum_limits.csv",
    )
    if args.with_dashboard:
        write_dashboard_html(
            transactions,
            output_file,
            Path(args.dashboard_output),
        )


if __name__ == "__main__":
    main()
