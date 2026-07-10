import argparse
from pathlib import Path

import pandas as pd

from .income_report_builder import build_income_workbook
from .wages_detector import (
    classify_income_transactions,
    print_income_type_summary,
    print_optional_validation,
)


ENGINE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = ENGINE_DIR / "sample.csv"
DEFAULT_OUTPUT_XLSX = ENGINE_DIR / "output" / "income_report.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run income classification and build the Excel income report.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_CSV), help="Input transaction CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_XLSX), help="Output Excel workbook path.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Write income audit detail and Centrelink subtypes instead of the compact transaction report.",
    )
    parser.add_argument(
        "--predictions-csv",
        help="Optional row-level prediction CSV output path. Omit to write only the Excel workbook.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input)
    output_xlsx = Path(args.output)
    include_full_detail = args.full
    predictions_csv = Path(args.predictions_csv) if args.predictions_csv else None

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Running income detection for: {input_csv} "
        f"(report_mode={'full' if include_full_detail else 'summary'})"
    )
    raw_df = pd.read_csv(input_csv, encoding="utf-8-sig")
    result = classify_income_transactions(
        raw_df,
        include_centrelink_payment_type=include_full_detail,
    )
    print(f"Predicted wages rows: {int(result.transactions['is_wages_pred'].sum())}")
    print(f"Predicted income rows: {int(result.transactions['is_income_pred'].sum())}")
    print_income_type_summary(result.transactions)
    print_optional_validation(result.transactions)
    if predictions_csv is not None:
        predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        result.transactions.to_csv(
            predictions_csv,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"Saved row-level predictions to: {predictions_csv}")
    print(f"Building income report: {output_xlsx}")
    build_income_workbook(
        result,
        output_xlsx,
        include_full_detail=include_full_detail,
    )
    print(f"Saved income report to: {output_xlsx}")


if __name__ == "__main__":
    main()
