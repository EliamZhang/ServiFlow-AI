import argparse
from pathlib import Path

import pandas as pd

from .reporting import write_report
from .pipeline import (
    run_pipeline,
    print_income_type_summary,
    print_optional_validation,
)


ENGINE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = ENGINE_DIR / "sample.csv"
DEFAULT_OUTPUT = ENGINE_DIR / "output" / "income_report.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run income classification and build the Excel income report.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input transaction CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output Excel workbook path.")
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
    input_file = Path(args.input)
    output_file = Path(args.output)
    predictions_csv = Path(args.predictions_csv) if args.predictions_csv else None

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Running income classification for: {input_file} "
        f"(output_mode={'full' if args.full else 'compact'})"
    )
    transactions = pd.read_csv(input_file, encoding="utf-8-sig")
    result = run_pipeline(
        transactions,
        include_centrelink_payment_type=args.full,
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
    print(f"Building income report: {output_file}")
    write_report(
        result,
        output_file,
        full=args.full,
    )
    print(f"Saved income report to: {output_file}")


if __name__ == "__main__":
    main()
