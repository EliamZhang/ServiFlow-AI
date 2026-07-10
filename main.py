from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd

from classification_core.config import (
    DEFAULT_CATEGORY_CATALOG,
    DEFAULT_PIPELINE_CONFIG,
    load_category_owners,
    load_pipeline_config,
)
from classification_core.models import ClassificationRunResult
from classification_core.orchestrator import ClassificationOrchestrator
from classification_core.reporting import write_report, write_transactions_csv


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "sample.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "classification_report.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all enabled transaction classification engines in priority order."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input transaction CSV path.",
    )
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


def run_classification(
    input_file: str | Path = DEFAULT_INPUT,
    output_file: str | Path = DEFAULT_OUTPUT,
    config_file: str | Path = DEFAULT_PIPELINE_CONFIG,
    category_catalog_file: str | Path = DEFAULT_CATEGORY_CATALOG,
    transactions_csv: str | Path | None = None,
) -> ClassificationRunResult:
    result, _ = _execute_classification(
        input_file=input_file,
        output_file=output_file,
        config_file=config_file,
        category_catalog_file=category_catalog_file,
        transactions_csv=transactions_csv,
    )
    return result


def _execute_classification(
    input_file: str | Path,
    output_file: str | Path,
    config_file: str | Path,
    category_catalog_file: str | Path,
    transactions_csv: str | Path | None,
) -> tuple[ClassificationRunResult, dict[str, float]]:
    total_started = perf_counter()

    stage_started = perf_counter()
    transactions = pd.read_csv(input_file, encoding="utf-8-sig")
    read_seconds = perf_counter() - stage_started

    stage_started = perf_counter()
    orchestrator = ClassificationOrchestrator(
        config=load_pipeline_config(config_file),
        category_owners=load_category_owners(category_catalog_file),
    )
    result = orchestrator.run(transactions)
    classify_seconds = perf_counter() - stage_started

    stage_started = perf_counter()
    write_report(result, output_file)
    if transactions_csv is not None:
        write_transactions_csv(result, transactions_csv)
    output_seconds = perf_counter() - stage_started

    return result, {
        "read": read_seconds,
        "classify": classify_seconds,
        "output": output_seconds,
        "total": perf_counter() - total_started,
    }


def main() -> None:
    args = parse_args()
    _, timings = _execute_classification(
        input_file=args.input,
        output_file=args.output,
        config_file=args.config,
        category_catalog_file=args.category_catalog,
        transactions_csv=args.transactions_csv,
    )
    print(
        f"Timing | read {timings['read']:.2f}s | "
        f"classify {timings['classify']:.2f}s | "
        f"output {timings['output']:.2f}s | total {timings['total']:.2f}s"
    )


if __name__ == "__main__":
    main()
