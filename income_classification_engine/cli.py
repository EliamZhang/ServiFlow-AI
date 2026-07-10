import argparse
from pathlib import Path

import pandas as pd

from .pipeline import ENGINE_DIR, PROJECT_ROOT, run_pipeline
from .presentation.reporting import write_report


DEFAULT_INPUT = PROJECT_ROOT / "sample.csv"
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
    transactions = pd.read_csv(input_file, encoding="utf-8-sig")
    result = run_pipeline(
        transactions,
        include_centrelink_payment_type=args.full,
    )
    if predictions_csv is not None:
        predictions_csv.parent.mkdir(parents=True, exist_ok=True)
        result.transactions.to_csv(
            predictions_csv,
            index=False,
            encoding="utf-8-sig",
        )
    write_report(
        result,
        output_file,
        full=args.full,
    )


if __name__ == "__main__":
    main()
