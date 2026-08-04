"""Baseline regression comparison for the classification pipeline.

Usage:
    python baseline.py save [--input sample.csv] [--baseline baseline/sample_baseline.csv]
    python baseline.py diff [--input sample.csv] [--baseline baseline/sample_baseline.csv]

``save`` runs the full pipeline and stores each transaction's final
classification (finv_category, counterparty, winning engine, stream_id) as a
baseline CSV.  ``diff`` reruns the pipeline and reports every transaction whose
classification changed versus the baseline, plus per-engine claim count
changes.  Exit code: 0 = no differences, 1 = differences found, 2 = error.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from classification_core.config import (
    DEFAULT_CATEGORY_CATALOG,
    DEFAULT_PIPELINE_CONFIG,
    load_category_owners,
    load_pipeline_config,
)
from classification_core.models import ClassificationRunResult
from classification_core.orchestrator import ClassificationOrchestrator

DEFAULT_INPUT = Path("sample.csv")
DEFAULT_BASELINE = Path("baseline/sample_baseline.csv")

KEY_COLUMNS = ("application_id", "transaction_id")
RESULT_COLUMNS = ("finv_category", "counterparty", "classification_engine", "stream_id")
BASELINE_COLUMNS = (*KEY_COLUMNS, *RESULT_COLUMNS)


def run_pipeline(input_file: str | Path) -> ClassificationRunResult:
    transactions = pd.read_csv(input_file, encoding="utf-8-sig")
    orchestrator = ClassificationOrchestrator(
        config=load_pipeline_config(DEFAULT_PIPELINE_CONFIG),
        category_owners=load_category_owners(DEFAULT_CATEGORY_CATALOG),
    )
    return orchestrator.run(transactions)


def _normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise types so CSV round-trips compare identically to in-memory data."""
    output = df[list(BASELINE_COLUMNS)].copy()
    for col in KEY_COLUMNS:
        output[col] = output[col].astype(str)
    for col in RESULT_COLUMNS:
        output[col] = output[col].fillna("").astype(str)
    return output.reset_index(drop=True)


def extract_baseline(result: ClassificationRunResult) -> pd.DataFrame:
    return _normalise_frame(result.transactions)


def load_baseline(path: str | Path) -> pd.DataFrame:
    return _normalise_frame(pd.read_csv(path, encoding="utf-8-sig"))


@dataclass
class TransactionChange:
    kind: str  # CHANGED / NEW / GONE
    application_id: str
    transaction_id: str
    old: dict[str, str]
    new: dict[str, str]


@dataclass
class CompareReport:
    changes: list[TransactionChange] = field(default_factory=list)
    engine_deltas: list[tuple[str, int, int]] = field(default_factory=list)
    engine_versions: dict[str, str] = field(default_factory=dict)

    @property
    def has_differences(self) -> bool:
        return bool(self.changes) or bool(self.engine_deltas)


def compare_transactions(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    engine_versions: dict[str, str] | None = None,
) -> CompareReport:
    """Compare two baseline-shaped frames, aligned on (application_id, transaction_id).

    Keys are compared as strings so int/str type mismatches across a CSV
    round-trip do not produce false positives.  A row is CHANGED when either
    side is classified and any result column differs; NEW when the baseline
    was unclassified; GONE when the current run is unclassified.
    """
    b = baseline.set_index(list(KEY_COLUMNS))
    c = current.set_index(list(KEY_COLUMNS))
    changes: list[TransactionChange] = []

    for key in b.index.union(c.index):
        old_row = b.loc[key] if key in b.index else None
        new_row = c.loc[key] if key in c.index else None

        old = {col: (old_row[col] if old_row is not None else "") for col in RESULT_COLUMNS}
        new = {col: (new_row[col] if new_row is not None else "") for col in RESULT_COLUMNS}

        old_classified = any(old.values())
        new_classified = any(new.values())
        if not old_classified and not new_classified:
            continue
        if not old_classified and new_classified:
            changes.append(TransactionChange("NEW", *key, old, new))
            continue
        if old_classified and not new_classified:
            changes.append(TransactionChange("GONE", *key, old, new))
            continue
        if any(old[col] != new[col] for col in RESULT_COLUMNS):
            changes.append(TransactionChange("CHANGED", *key, old, new))

    old_counts = Counter(b["classification_engine"] for _, b in b.iterrows())
    new_counts = Counter(c["classification_engine"] for _, c in c.iterrows())
    engine_deltas = [
        (engine, old_counts.get(engine, 0), new_counts.get(engine, 0))
        for engine in sorted(set(old_counts) | set(new_counts))
        if old_counts.get(engine, 0) != new_counts.get(engine, 0)
    ]

    return CompareReport(
        changes=changes,
        engine_deltas=engine_deltas,
        engine_versions=engine_versions or {},
    )


def _print_engine_claims(result: ClassificationRunResult) -> None:
    counts = result.transactions["classification_engine"].value_counts()
    total = len(result.transactions)
    unclassified = int(
        (result.transactions["classification_status"] == "unclassified").sum()
    )
    print(f"共 {total} 笔 | 未分类 {unclassified} 笔")
    for engine in result.executions:
        print(f"  {engine.engine_id:<18} {counts.get(engine.engine_id, 0)}")


def _engine_versions(result: ClassificationRunResult) -> dict[str, str]:
    return {execution.engine_id: execution.engine_version for execution in result.executions}


def _format_value(change: TransactionChange, col: str) -> str:
    old, new = change.old[col], change.new[col]
    if old == new:
        return ""
    return f"{old or '未分类'} → {new or '未分类'}"


def print_diff(report: CompareReport) -> None:
    if report.changes:
        print(f"=== 分类变化 ({len(report.changes)} 笔) ===")
        for change in report.changes:
            detail = " | ".join(
                _format_value(change, col)
                for col in RESULT_COLUMNS
                if _format_value(change, col)
            )
            print(f"  [{change.kind}] app={change.application_id} tx={change.transaction_id}: {detail}")

    if report.engine_deltas:
        print("\n=== 引擎认领数变化 ===")
        for engine, old, new in report.engine_deltas:
            print(f"  {engine:<18} {old} → {new} ({new - old:+d})")

    versions = ", ".join(
        f"{engine} {version}" for engine, version in sorted(report.engine_versions.items())
    )
    print(f"\n=== 汇总 ===\n差异 {len(report.changes)} 笔 | 引擎版本: {versions}")


def cmd_save(args: argparse.Namespace) -> int:
    result = run_pipeline(args.input)
    baseline_path = Path(args.baseline)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    extract_baseline(result).to_csv(baseline_path, index=False)
    print(f"基线已保存: {baseline_path}")
    _print_engine_claims(result)
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"错误: 基线文件不存在: {baseline_path}（先运行 save）", file=sys.stderr)
        return 2

    baseline = load_baseline(baseline_path)
    result = run_pipeline(args.input)
    report = compare_transactions(baseline, extract_baseline(result), _engine_versions(result))
    print_diff(report)

    if report.has_differences:
        print("\n结论: 与基线存在差异（exit 1）")
        return 1
    print("\n结论: 与基线一致（exit 0）")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the classification pipeline and compare against a saved baseline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (("save", "run pipeline and write baseline"), ("diff", "rerun pipeline and compare to baseline")):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--input", default=str(DEFAULT_INPUT), help="Input transaction CSV path.")
        sub.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="Baseline CSV path.")
        sub.set_defaults(func=cmd_save if name == "save" else cmd_diff)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
