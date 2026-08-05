"""Baseline regression comparison for the classification pipeline.

Usage:
    python baseline.py save [--input sample.csv] [--baseline baseline/sample_baseline.csv]
    python baseline.py diff [--input sample.csv] [--baseline baseline/sample_baseline.csv]

``save`` runs the full pipeline and stores each transaction's final
classification (finv_category, counterparty, winning engine, stream_id) as a
baseline CSV, plus per-engine claim snapshots (engine_claims.csv) and pipeline
config/engine versions (run_meta.json).  ``diff`` reruns the pipeline and
reports every transaction whose classification changed versus the baseline,
per-engine claim changes, claim-count changes per rule, and any pipeline
config / engine version drift.  Exit code: 0 = no differences, 1 =
differences found, 2 = error.
"""

from __future__ import annotations

import argparse
import json
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
from classification_core.registry import build_engine

DEFAULT_INPUT = Path("sample.csv")
DEFAULT_BASELINE = Path("baseline/sample_baseline.csv")
DEFAULT_ENGINE_BASELINE = Path("baseline/engine_claims.csv")
DEFAULT_RUN_META = Path("baseline/run_meta.json")

KEY_COLUMNS = ("application_id", "transaction_id")
RESULT_COLUMNS = ("finv_category", "counterparty", "classification_engine", "stream_id")
BASELINE_COLUMNS = (*KEY_COLUMNS, *RESULT_COLUMNS)

ENGINE_CLAIM_COLUMNS = (
    "engine_id",
    *KEY_COLUMNS,
    "finv_category",
    "counterparty",
    "classification_rule_id",
    "classification_reason",
    "stream_id",
    "priority",
)

CLAIM_RESULT_COLUMNS = (
    "finv_category",
    "counterparty",
    "classification_rule_id",
    "classification_reason",
    "stream_id",
)


def run_pipeline(input_file: str | Path) -> ClassificationRunResult:
    transactions = pd.read_csv(input_file, encoding="utf-8-sig")
    orchestrator = ClassificationOrchestrator(
        config=load_pipeline_config(DEFAULT_PIPELINE_CONFIG),
        category_owners=load_category_owners(DEFAULT_CATEGORY_CATALOG),
    )
    return orchestrator.run(transactions)


# ── run metadata (pipeline config + engine versions) ────────────────────────

def _run_meta() -> dict:
    """Snapshot pipeline config and engine versions to detect config-level
    changes (priority, enable/disable, version bumps) that the per-row
    comparisons cannot see."""
    config = load_pipeline_config(DEFAULT_PIPELINE_CONFIG)
    return {
        "engines": [
            {"engine_id": spec.engine_id, "priority": spec.priority, "enabled": spec.enabled}
            for spec in config.engines
        ],
        "on_engine_error": config.on_engine_error,
        "engine_versions": dict(sorted(_engine_versions_of_config(config).items())),
    }


def _engine_versions_of_config(config) -> dict[str, str]:
    """Resolve each enabled engine's version string without running the pipeline."""
    versions: dict[str, str] = {}
    for spec in config.enabled_engines:
        engine = build_engine(spec.engine_id)
        versions[engine.engine_id] = engine.engine_version
    return versions


def save_run_meta(path: str | Path, meta: dict) -> None:
    Path(path).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_run_meta(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def compare_run_meta(baseline_meta: dict, current_meta: dict) -> list[str]:
    """Return human-readable config/version differences; empty means identical."""
    differences: list[str] = []

    baseline_engines = {(e["engine_id"], e["priority"], e["enabled"]) for e in baseline_meta.get("engines", [])}
    current_engines = {(e["engine_id"], e["priority"], e["enabled"]) for e in current_meta.get("engines", [])}
    if baseline_engines != current_engines:
        differences.append(
            f"引擎配置变化: {sorted(baseline_engines)} -> {sorted(current_engines)}"
        )

    if baseline_meta.get("on_engine_error") != current_meta.get("on_engine_error"):
        differences.append(
            f"on_engine_error 变化: {baseline_meta.get('on_engine_error')!r} -> {current_meta.get('on_engine_error')!r}"
        )

    baseline_versions = baseline_meta.get("engine_versions", {})
    current_versions = current_meta.get("engine_versions", {})
    version_changes = {
        engine: (baseline_versions.get(engine, ""), current_versions.get(engine, ""))
        for engine in sorted(set(baseline_versions) | set(current_versions))
        if baseline_versions.get(engine, "") != current_versions.get(engine, "")
    }
    for engine, (old, new) in version_changes.items():
        differences.append(f"引擎版本变化: {engine} {old or '(缺失)'} -> {new or '(缺失)'}")

    return differences


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


def extract_engine_claims(result: ClassificationRunResult) -> pd.DataFrame:
    """Concatenate every engine's claim snapshot into one baseline frame.

    Each engine may produce a different subset of the claim columns, so the
    output is normalised to ENGINE_CLAIM_COLUMNS with blanks filled."""
    frames = []
    for execution in result.executions:
        claims = execution.claims.copy()
        for col in ENGINE_CLAIM_COLUMNS:
            if col not in claims.columns:
                claims[col] = pd.NA
        claims["engine_id"] = execution.engine_id
        frames.append(claims[list(ENGINE_CLAIM_COLUMNS)])
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ENGINE_CLAIM_COLUMNS)
    for col in (*KEY_COLUMNS, "engine_id"):
        output[col] = output[col].astype(str)
    for col in ENGINE_CLAIM_COLUMNS:
        if col not in KEY_COLUMNS and col != "engine_id":
            output[col] = output[col].fillna("").astype(str)
    return output.reset_index(drop=True)


def load_engine_claims(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    for col in ENGINE_CLAIM_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
        frame[col] = frame[col].fillna("").astype(str)
    return frame[list(ENGINE_CLAIM_COLUMNS)]


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
class EngineClaimChange:
    kind: str  # CHANGED / NEW / GONE
    engine_id: str
    application_id: str
    transaction_id: str
    old: dict[str, str]
    new: dict[str, str]


@dataclass
class CompareReport:
    changes: list[TransactionChange] = field(default_factory=list)
    engine_deltas: list[tuple[str, int, int]] = field(default_factory=list)
    engine_rule_deltas: list[tuple[str, str, int, int]] = field(default_factory=list)
    claim_changes: list[EngineClaimChange] = field(default_factory=list)
    engine_versions: dict[str, str] = field(default_factory=dict)
    run_meta_differences: list[str] = field(default_factory=list)
    row_count_mismatches: list[str] = field(default_factory=list)

    @property
    def has_differences(self) -> bool:
        return bool(
            self.changes
            or self.engine_deltas
            or self.engine_rule_deltas
            or self.claim_changes
            or self.run_meta_differences
            or self.row_count_mismatches
        )


def compare_transactions(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    engine_versions: dict[str, str] | None = None,
    engine_baseline: pd.DataFrame | None = None,
    engine_current: pd.DataFrame | None = None,
    baseline_meta: dict | None = None,
    current_meta: dict | None = None,
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

    if engine_baseline is not None and engine_current is not None:
        claim_changes = compare_engine_claims(engine_baseline, engine_current)
        rule_deltas = compare_engine_rule_counts(engine_baseline, engine_current)
    else:
        claim_changes, rule_deltas = [], []

    if baseline_meta is not None and current_meta is not None:
        run_meta_differences = compare_run_meta(baseline_meta, current_meta)
    else:
        run_meta_differences = []

    row_count_mismatches: list[str] = []
    if len(current) != len(baseline):
        row_count_mismatches.append(
            f"交易行数变化: 基线 {len(baseline)} 行 -> 当前 {len(current)} 行"
        )
    if engine_current is not None and engine_baseline is not None:
        if len(engine_current) != len(engine_baseline):
            row_count_mismatches.append(
                f"引擎认领行数变化: 基线 {len(engine_baseline)} 行 -> 当前 {len(engine_current)} 行"
            )

    return CompareReport(
        changes=changes,
        engine_deltas=engine_deltas,
        engine_rule_deltas=rule_deltas,
        claim_changes=claim_changes,
        engine_versions=engine_versions or {},
        run_meta_differences=run_meta_differences,
        row_count_mismatches=row_count_mismatches,
    )


CLAIM_RESULT_COLUMNS = (
    "finv_category",
    "counterparty",
    "classification_rule_id",
    "classification_reason",
    "stream_id",
)


def compare_engine_claims(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
) -> list[EngineClaimChange]:
    """Compare per-engine claim snapshots, aligned on (engine_id, key).

    Unlike the final-output comparison, this catches regressions inside an
    engine even when a later engine overwrites the row in the final output."""
    b = baseline.set_index(["engine_id", *KEY_COLUMNS])
    c = current.set_index(["engine_id", *KEY_COLUMNS])
    claim_changes: list[EngineClaimChange] = []

    for key in b.index.union(c.index):
        engine_id, app_id, tx_id = key
        old_row = b.loc[key] if key in b.index else None
        new_row = c.loc[key] if key in c.index else None

        old = {col: (old_row[col] if old_row is not None else "") for col in CLAIM_RESULT_COLUMNS}
        new = {col: (new_row[col] if new_row is not None else "") for col in CLAIM_RESULT_COLUMNS}

        old_classified = any(old.values())
        new_classified = any(new.values())
        if not old_classified and not new_classified:
            continue
        if not old_classified and new_classified:
            claim_changes.append(EngineClaimChange("NEW", engine_id, app_id, tx_id, old, new))
            continue
        if old_classified and not new_classified:
            claim_changes.append(EngineClaimChange("GONE", engine_id, app_id, tx_id, old, new))
            continue
        if any(old[col] != new[col] for col in CLAIM_RESULT_COLUMNS):
            claim_changes.append(EngineClaimChange("CHANGED", engine_id, app_id, tx_id, old, new))

    return claim_changes


def compare_engine_rule_counts(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
) -> list[tuple[str, str, int, int]]:
    """Per-engine (engine_id, rule_id) claim-count deltas."""
    def _counts(frame: pd.DataFrame) -> Counter[tuple[str, str]]:
        return Counter(zip(frame["engine_id"], frame["classification_rule_id"]))
    old_counts, new_counts = _counts(baseline), _counts(current)
    return [
        (engine, rule, old_counts.get((engine, rule), 0), new_counts.get((engine, rule), 0))
        for (engine, rule) in sorted(set(old_counts) | set(new_counts))
        if old_counts.get((engine, rule), 0) != new_counts.get((engine, rule), 0)
    ]


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

    if report.claim_changes:
        print(f"\n=== 引擎认领变化 ({len(report.claim_changes)} 笔) ===")
        for change in report.claim_changes:
            detail = " | ".join(
                _format_value(change, col)
                for col in CLAIM_RESULT_COLUMNS
                if _format_value(change, col)
            )
            print(
                f"  [{change.kind}] {change.engine_id} "
                f"app={change.application_id} tx={change.transaction_id}: {detail}"
            )

    if report.engine_deltas:
        print("\n=== 引擎认领数变化 ===")
        for engine, old, new in report.engine_deltas:
            print(f"  {engine:<18} {old} → {new} ({new - old:+d})")

    if report.engine_rule_deltas:
        print("\n=== 引擎规则认领数变化 ===")
        for engine, rule, old, new in report.engine_rule_deltas:
            print(f"  {engine:<18} {rule:<36} {old} → {new} ({new - old:+d})")

    if report.run_meta_differences:
        print("\n=== 配置/版本变化 ===")
        for diff in report.run_meta_differences:
            print(f"  {diff}")

    if report.row_count_mismatches:
        print("\n=== 行数检查 ===")
        for mismatch in report.row_count_mismatches:
            print(f"  {mismatch}")

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
    engine_baseline_path = Path(args.engine_baseline)
    extract_engine_claims(result).to_csv(engine_baseline_path, index=False)
    print(f"引擎认领基线已保存: {engine_baseline_path}")
    run_meta_path = Path(args.run_meta)
    save_run_meta(run_meta_path, _run_meta())
    print(f"运行元数据已保存: {run_meta_path}")
    _print_engine_claims(result)
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"错误: 基线文件不存在: {baseline_path}（先运行 save）", file=sys.stderr)
        return 2

    baseline = load_baseline(baseline_path)
    engine_baseline_path = Path(args.engine_baseline)
    if engine_baseline_path.exists():
        engine_baseline = load_engine_claims(engine_baseline_path)
    else:
        print(
            f"注意: 引擎认领基线不存在: {engine_baseline_path}（跳过每引擎对比）",
            file=sys.stderr,
        )
        engine_baseline = None

    run_meta_path = Path(args.run_meta)
    if run_meta_path.exists():
        baseline_meta = load_run_meta(run_meta_path)
    else:
        print(
            f"注意: 运行元数据不存在: {run_meta_path}（跳过配置/版本对比）",
            file=sys.stderr,
        )
        baseline_meta = None

    result = run_pipeline(args.input)
    engine_current = extract_engine_claims(result) if engine_baseline is not None else None
    report = compare_transactions(
        baseline,
        extract_baseline(result),
        _engine_versions(result),
        engine_baseline=engine_baseline,
        engine_current=engine_current,
        baseline_meta=baseline_meta,
        current_meta=_run_meta(),
    )
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
        sub.add_argument("--engine-baseline", default=str(DEFAULT_ENGINE_BASELINE), help="Per-engine claim baseline CSV path.")
        sub.add_argument("--run-meta", default=str(DEFAULT_RUN_META), help="Pipeline config/version metadata path.")
        sub.set_defaults(func=cmd_save if name == "save" else cmd_diff)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
