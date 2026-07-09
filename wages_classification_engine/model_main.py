import argparse
from pathlib import Path

from income_report_builder import build_income_workbook
from wages_detector import detect_income


DEFAULT_INPUT_CSV = Path("sample.csv")
DEFAULT_OUTPUT_XLSX = Path("output") / "income_report.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run income classification and build the Excel income report.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT_CSV), help="Input transaction CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_XLSX), help="Output Excel workbook path.")
    parser.add_argument(
        "--report-mode",
        choices=["summary", "full"],
        default="summary",
        help="summary writes published transactions plus income summary; full writes income summary plus audit detail.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Shortcut for --report-mode full, kept for compatibility with the previous entry point.",
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
    include_full_detail = args.full or args.report_mode == "full"
    predictions_csv = Path(args.predictions_csv) if args.predictions_csv else None

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running income detection for: {input_csv} (report_mode={'full' if include_full_detail else 'summary'})")
    result_df = detect_income(
        input_csv,
        output_path=predictions_csv,
        include_centrelink_payment_type=include_full_detail,
        save_csv=predictions_csv is not None,
    )
    print(f"Building income report: {output_xlsx}")
    build_income_workbook(result_df, output_xlsx, include_full_detail=include_full_detail)
    print(f"Saved income report to: {output_xlsx}")


if __name__ == "__main__":
    main()
