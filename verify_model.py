"""Local verification script: run the pipeline on a single application JSON input and write JSON output.

Usage: python verify_model.py [--input model_input.json] [--output ...]
[--config configs/pipeline.json] [--category-catalog configs/category_catalog.json]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from classification_core.config import (
    DEFAULT_CATEGORY_CATALOG,
    DEFAULT_PIPELINE_CONFIG,
)
from classification_core.service import (
    ModelService,
    _build_error_output,
    build_transactions_frame,
    serialize_result,
)

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


def _resolve_output_path(output_arg: str | None, payload: dict) -> Path:
    if output_arg:
        return Path(output_arg)
    application_id = payload.get("applicationId")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"model_output_{application_id}_{timestamp}.json"


def main() -> None:
    args = parse_args()
    payload = load_input(args.input)
    output_path = _resolve_output_path(args.output, payload)

    try:
        transactions = build_transactions_frame(payload)
        service = ModelService(
            pipeline_config_path=args.config,
            category_catalog_path=args.category_catalog,
        )
        result = service.orchestrator.run(transactions)
    except Exception as exc:
        output = _build_error_output(payload, str(exc))
        print(json.dumps(output, ensure_ascii=False, indent=2))
        raise

    output = serialize_result(result, payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
