import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from classification_core.models import PipelineResult

from .domain.summary import build_summary
from .pipeline import DEFAULT_RESOURCES_DIR, ENGINE_DIR, PROJECT_ROOT, run_pipeline
from .presentation.dashboard import build_html, dataframe_to_records
from .presentation.reporting import write_report


DEFAULT_INPUT = PROJECT_ROOT / "sample.csv"
DEFAULT_OUTPUT = ENGINE_DIR / "output" / "liability_report.xlsx"
DEFAULT_DASHBOARD = ENGINE_DIR / "output" / "liability_dashboard.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run liability classification output generation."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
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
    result: PipelineResult,
    source_file: Path,
) -> dict[str, object]:
    transactions = result.transactions
    summary = build_summary(
        transactions,
        limits_file=DEFAULT_RESOURCES_DIR / "bnpl_maximum_limits.csv",
    )
    return {
        "liabilitySummary": dataframe_to_records(summary),
        "transactions": dataframe_to_records(transactions),
        "meta": {
            "sourceFile": source_file.name,
            "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "liabilitySummaryRows": len(summary),
            "transactionRows": len(transactions),
        },
    }


def write_dashboard(
    result: PipelineResult,
    workbook_file: Path,
    dashboard_file: Path,
) -> None:
    data = build_dashboard_data(result, workbook_file)
    html = build_html(data)
    dashboard_file.parent.mkdir(parents=True, exist_ok=True)
    dashboard_file.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_file = Path(args.input)
    output_file = Path(args.output)
    transactions = pd.read_csv(input_file, encoding="utf-8-sig")
    result = run_pipeline(transactions)
    write_report(result, output_file)
    if args.with_dashboard:
        write_dashboard(
            result,
            output_file,
            Path(args.dashboard_output),
        )


if __name__ == "__main__":
    main()
