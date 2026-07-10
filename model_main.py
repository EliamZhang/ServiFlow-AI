from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from serviflow.config import (
    DEFAULT_CATEGORY_CATALOG,
    DEFAULT_PIPELINE_CONFIG,
    load_category_owners,
    load_pipeline_config,
)
from serviflow.orchestrator import ClassificationOrchestrator
from serviflow.reporting import write_excel_report, write_transactions_csv


DEFAULT_OUTPUT = Path("output") / "serviflow_report.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all enabled ServiFlow classification engines in priority order."
    )
    parser.add_argument("--input", required=True, help="Input transaction CSV path.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Unified Excel report path.",
    )
    parser.add_argument(
        "--transactions-csv",
        help="Optional unified row-level CSV output path.",
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


def run_pipeline(
    input_file: str | Path,
    output_file: str | Path,
    config_file: str | Path = DEFAULT_PIPELINE_CONFIG,
    category_catalog_file: str | Path = DEFAULT_CATEGORY_CATALOG,
    transactions_csv: str | Path | None = None,
):
    transactions = pd.read_csv(input_file, encoding="utf-8-sig")
    orchestrator = ClassificationOrchestrator(
        config=load_pipeline_config(config_file),
        category_owners=load_category_owners(category_catalog_file),
    )
    result = orchestrator.run(transactions)
    write_excel_report(result, output_file)
    if transactions_csv is not None:
        write_transactions_csv(result, transactions_csv)
    return result


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        input_file=args.input,
        output_file=args.output,
        config_file=args.config,
        category_catalog_file=args.category_catalog,
        transactions_csv=args.transactions_csv,
    )
    print(f"Classification run: {result.run_id}")
    for execution in result.executions:
        print(
            f"{execution.engine_id}: candidates={execution.candidate_count}, "
            f"accepted={execution.accepted_count}"
        )
    print(f"Saved unified report to: {args.output}")


if __name__ == "__main__":
    main()
