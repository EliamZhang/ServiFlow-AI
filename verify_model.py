"""Local verification script: run the pipeline on a single application JSON input and write JSON output.

Accepts either supported input contract (illion v1 / wagego v2) and writes the
matching output contract; see classification_core/service.py.  The output carries the
echoed input identifiers (userId / applicationId / flowTime), `bscat_stats`,
`bank_accounts`, `bscat_transactions` and `bscat_summaries` -- no run envelope: this
script re-raises instead of returning the failed shape.

Usage: python verify_model.py [--input model_input.json] [--output ...]
[--config configs/pipeline.json] [--category-catalog configs/category_catalog.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from classification_core.config import (
    DEFAULT_CATEGORY_CATALOG,
    DEFAULT_PIPELINE_CONFIG,
)
from classification_core.service import (
    ModelService,
    application_id_from_payload,
    build_input_frame,
    serialize_output,
)

# Windows forbids these in filenames; control chars for good measure
_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_MAX_APPLICATION_ID_IN_FILENAME = 80

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a single application from JSON and write JSON output."
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "model_input.json"),
        help="Input application JSON path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. Default: output/model_output_{applicationId}_"
            "{YYYYMMDD_HHMMSS}.json"
        ),
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


def load_input(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _safe_filename_part(value: Any) -> str:
    """Render an application id safe to embed in a filename.

    v2 flowIds look like "1000008272@1002818@10001@<uuid>" (already safe); the
    sanitizing is defensive for arbitrary upstream ids.
    """
    if value is None:
        return "unknown"
    text = _UNSAFE_FILENAME_CHARS.sub("_", str(value)).strip().strip(".")
    return text[:_MAX_APPLICATION_ID_IN_FILENAME] or "unknown"


def _resolve_output_path(output_arg: str | None, payload: dict) -> Path:
    if output_arg:
        return Path(output_arg)
    application_id = application_id_from_payload(payload)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / (
        f"model_output_{_safe_filename_part(application_id)}_{timestamp}.json"
    )


def main() -> None:
    args = parse_args()
    payload = load_input(args.input)
    output_path = _resolve_output_path(args.output, payload)

    started = perf_counter()
    try:
        transactions = build_input_frame(payload)
        service = ModelService(
            pipeline_config_path=args.config,
            category_catalog_path=args.category_catalog,
        )
        result = service.orchestrator.run(transactions)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise

    output = serialize_output(result, payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    total_seconds = perf_counter() - started
    stats = output.get("bscat_stats", {})
    print(
        f"application_no={application_id_from_payload(payload)} | "
        f"product={stats.get('product')} | "
        f"transactions={stats.get('txn_raw_input_cnt')} | "
        f"date_max={stats.get('transaction_date_max')}"
    )
    engine_times = " | ".join(
        f"{execution.engine_id} {execution.duration_seconds:.2f}s"
        for execution in result.executions
    )
    if engine_times:
        print(f"Engines | {engine_times}")
    print(f"Total | {total_seconds:.2f}s")
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
