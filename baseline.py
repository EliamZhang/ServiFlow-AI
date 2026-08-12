"""Baseline regression comparison for the classification pipeline.

Usage:
    python baseline.py save [--input sample.csv] [--baseline baseline/sample_baseline.csv]
    python baseline.py diff [--input sample.csv] [--baseline baseline/sample_baseline.csv]

``save`` runs the full pipeline and stores each transaction's final
classification (finv_category, counterparty, winning engine, stream_id) as a
baseline CSV, plus per-engine claim snapshots (engine_claims.csv), pipeline
config/engine versions (run_meta.json), and deterministic summary metrics
(baseline/summaries/: category_summary all columns, liability_summary amount
columns).  ``diff`` reruns the pipeline and reports every transaction whose
classification changed versus the baseline, per-engine claim changes,
claim-count changes per rule, pipeline config / engine version drift, and
summary metric changes.  Exit code: 0 = no differences, 1 = differences
found, 2 = error.
"""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_SUMMARIES_DIR = Path("baseline/summaries")
PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_FORMAT_VERSION = 3

# Only deterministic summary artifacts are compared.  liability_summary rows
# mix time-sensitive fields (status, predicted_closing_date, frequency) with
# stable amounts; comparing the amounts alone avoids drift noise when the
# sample window changes.
SUMMARY_ARTIFACTS = (
    ("category_summary", None),
    ("liability_summary", ("funded_amount", "repaid_amount", "repayment_amount", "recent_fn_repay_amount")),
)

SUMMARY_KEY_COLUMNS = (
    "application_id",
    "bank_account_id",
    "finv_category",
    "stream_id",
)

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


def run_pipeline(
    input_file: str | Path,
    config_file: str | Path = DEFAULT_PIPELINE_CONFIG,
    category_catalog_file: str | Path = DEFAULT_CATEGORY_CATALOG,
) -> ClassificationRunResult:
    transactions = pd.read_csv(input_file, encoding="utf-8-sig")
    orchestrator = ClassificationOrchestrator(
        config=load_pipeline_config(config_file),
        category_owners=load_category_owners(category_catalog_file),
    )
    return orchestrator.run(transactions)


# ── run metadata (pipeline config + engine versions) ────────────────────────

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Return a portable project-relative path where possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _rule_files(
    config_file: str | Path,
    category_catalog_file: str | Path,
) -> list[Path]:
    """Return all data files that define pipeline classification rules."""
    files = [Path(config_file), Path(category_catalog_file)]
    files.extend(PROJECT_ROOT.glob("*_engine/resources/**/*.csv"))
    files.append(PROJECT_ROOT / "initial_engine" / "merchant_kb.csv")
    return sorted({path.resolve() for path in files if path.is_file()})


def _fingerprints(
    input_file: str | Path,
    config_file: str | Path,
    category_catalog_file: str | Path,
) -> dict:
    input_path = Path(input_file)
    return {
        "input": {"path": _display_path(input_path), "sha256": _sha256(input_path)},
        "rule_files": {
            _display_path(path): _sha256(path)
            for path in _rule_files(config_file, category_catalog_file)
        },
    }


def _run_meta(
    input_file: str | Path,
    config_file: str | Path,
    category_catalog_file: str | Path,
) -> dict:
    """Snapshot pipeline config and engine versions to detect config-level
    changes (priority, enable/disable, version bumps) that the per-row
    comparisons cannot see."""
    config = load_pipeline_config(config_file)
    return {
        "baseline_format_version": BASELINE_FORMAT_VERSION,
        "engines": [
            {"engine_id": spec.engine_id, "priority": spec.priority, "enabled": spec.enabled}
            for spec in config.engines
        ],
        "on_engine_error": config.on_engine_error,
        "engine_versions": dict(sorted(_engine_versions_of_config(config).items())),
        "fingerprints": _fingerprints(
            input_file, config_file, category_catalog_file
        ),
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
            f"Engine configuration changed: {sorted(baseline_engines)} -> {sorted(current_engines)}"
        )

    if baseline_meta.get("on_engine_error") != current_meta.get("on_engine_error"):
        differences.append(
            f"on_engine_error changed: {baseline_meta.get('on_engine_error')!r} -> {current_meta.get('on_engine_error')!r}"
        )

    baseline_versions = baseline_meta.get("engine_versions", {})
    current_versions = current_meta.get("engine_versions", {})
    version_changes = {
        engine: (baseline_versions.get(engine, ""), current_versions.get(engine, ""))
        for engine in sorted(set(baseline_versions) | set(current_versions))
        if baseline_versions.get(engine, "") != current_versions.get(engine, "")
    }
    for engine, (old, new) in version_changes.items():
        differences.append(f"Engine version changed: {engine} {old or '(missing)'} -> {new or '(missing)'}")

    if baseline_meta.get("baseline_format_version") != current_meta.get(
        "baseline_format_version"
    ):
        differences.append(
            "Baseline format changed: "
            f"{baseline_meta.get('baseline_format_version', '(missing)')} -> "
            f"{current_meta.get('baseline_format_version', '(missing)')}; "
            "review and rebuild with save --replace"
        )

    baseline_fingerprints = baseline_meta.get("fingerprints")
    current_fingerprints = current_meta.get("fingerprints")
    if baseline_fingerprints is None:
        differences.append("Baseline is missing input/rule fingerprints; review and rebuild with save --replace")
    elif baseline_fingerprints != current_fingerprints:
        if baseline_fingerprints.get("input") != current_fingerprints.get("input"):
            differences.append("Input file fingerprint changed")
        if baseline_fingerprints.get("rule_files") != current_fingerprints.get("rule_files"):
            differences.append("Rule file fingerprint changed")

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


def _normalise_summary_values(series: pd.Series) -> pd.Series:
    """Canonicalise numeric summaries so harmless float tail noise is ignored."""
    non_blank = series.dropna().astype(str).str.strip()
    if non_blank.empty:
        return series.fillna("").astype(str)
    numeric = pd.to_numeric(non_blank, errors="coerce")
    if numeric.isna().any():
        return series.fillna("").astype(str)
    return series.map(
        lambda value: ""
        if pd.isna(value)
        else f"{float(value):.10f}".rstrip("0").rstrip(".")
    )


def extract_summary_baseline(
    result: ClassificationRunResult,
    artifact_name: str,
    amount_columns: tuple[str, ...] | None,
) -> pd.DataFrame:
    """Extract a deterministic summary snapshot from a pipeline run.

    Only the given amount columns are kept for ``liability_summary`` (the
    stable, date-independent values); ``None`` keeps all columns."""
    artifacts = {artifact.name: artifact for artifact in result.summaries}
    artifact = artifacts.get(artifact_name)
    if artifact is None or artifact.data.empty:
        return pd.DataFrame(columns=SUMMARY_KEY_COLUMNS + (amount_columns or ()))

    data = artifact.data
    key_columns = [col for col in SUMMARY_KEY_COLUMNS if col in data.columns]
    value_columns = (
        [col for col in data.columns if col not in key_columns]
        if amount_columns is None
        else [col for col in amount_columns if col in data.columns]
    )
    compare_columns = [*key_columns, *value_columns]

    output = data[compare_columns].copy()
    for col in key_columns:
        output[col] = output[col].fillna("").astype(str)
    for col in value_columns:
        output[col] = _normalise_summary_values(output[col])
    if amount_columns is not None:
        for col in amount_columns:
            if col not in output.columns:
                output[col] = ""
    # Multiple streams can share one (bank_account_id, finv_category) key in
    # category_summary; keep the first row so keys stay unique for alignment.
    return output.drop_duplicates(subset=key_columns, keep="first").reset_index(drop=True)


def load_summary_baseline(
    path: str | Path,
    amount_columns: tuple[str, ...] | None,
) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    key_columns = [col for col in SUMMARY_KEY_COLUMNS if col in frame.columns]
    for col in key_columns:
        frame[col] = frame[col].fillna("").astype(str)
    value_columns = (
        [col for col in frame.columns if col not in key_columns]
        if amount_columns is None
        else list(amount_columns)
    )
    if amount_columns is not None:
        for col in amount_columns:
            if col not in frame.columns:
                frame[col] = ""
    for col in value_columns:
        frame[col] = _normalise_summary_values(frame[col])
    return (
        frame[[*key_columns, *value_columns]]
        .drop_duplicates(subset=key_columns, keep="first")
        .reset_index(drop=True)
    )


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
class SummaryChange:
    artifact_name: str
    kind: str  # CHANGED / NEW / GONE
    row_key: dict[str, str]
    old: dict[str, str]
    new: dict[str, str]


@dataclass
class CompareReport:
    changes: list[TransactionChange] = field(default_factory=list)
    engine_deltas: list[tuple[str, int, int]] = field(default_factory=list)
    engine_rule_deltas: list[tuple[str, str, int, int]] = field(default_factory=list)
    claim_changes: list[EngineClaimChange] = field(default_factory=list)
    summary_changes: list[SummaryChange] = field(default_factory=list)
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
            or self.summary_changes
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
    summary_baselines: dict[str, pd.DataFrame] | None = None,
    summary_currents: dict[str, pd.DataFrame] | None = None,
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

    summary_changes: list[SummaryChange] = []
    if summary_baselines is not None and summary_currents is not None:
        for artifact_name, amount_columns in SUMMARY_ARTIFACTS:
            sb = summary_baselines.get(artifact_name)
            sc = summary_currents.get(artifact_name)
            if sb is not None and sc is not None:
                summary_changes.extend(compare_summaries(sb, sc, artifact_name))

    row_count_mismatches: list[str] = []
    if len(current) != len(baseline):
        row_count_mismatches.append(
            f"Transaction row count changed: baseline {len(baseline)} rows -> current {len(current)} rows"
        )
    if engine_current is not None and engine_baseline is not None:
        if len(engine_current) != len(engine_baseline):
            row_count_mismatches.append(
                f"Engine claim row count changed: baseline {len(engine_baseline)} rows -> current {len(engine_current)} rows"
            )

    return CompareReport(
        changes=changes,
        engine_deltas=engine_deltas,
        engine_rule_deltas=rule_deltas,
        claim_changes=claim_changes,
        summary_changes=summary_changes,
        engine_versions=engine_versions or {},
        run_meta_differences=run_meta_differences,
        row_count_mismatches=row_count_mismatches,
    )


def compare_summaries(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    artifact_name: str,
) -> list[SummaryChange]:
    """Compare deterministic summary snapshots, aligned on the key columns.

    Rows are matched on the key columns present in the artifact; value columns
    are compared as strings after CSV normalisation."""
    key_columns = [col for col in SUMMARY_KEY_COLUMNS if col in baseline.columns]
    value_columns = [col for col in baseline.columns if col not in key_columns]

    b = baseline.set_index(list(key_columns))
    c = current.set_index(list(key_columns))
    changes: list[SummaryChange] = []

    for key in b.index.union(c.index):
        old_row = b.loc[key] if key in b.index else None
        new_row = c.loc[key] if key in c.index else None
        row_key = dict(zip(key_columns, key))

        old = {col: (old_row[col] if old_row is not None else "") for col in value_columns}
        new = {col: (new_row[col] if new_row is not None else "") for col in value_columns}

        if not old_row is not None and not new_row is not None:
            continue
        if old_row is None:
            changes.append(SummaryChange(artifact_name, "NEW", row_key, {}, new))
            continue
        if new_row is None:
            changes.append(SummaryChange(artifact_name, "GONE", row_key, old, {}))
            continue
        if any(old[col] != new[col] for col in value_columns):
            changes.append(SummaryChange(artifact_name, "CHANGED", row_key, old, new))

    return changes


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
    print(f"Total {total} transactions | unclassified {unclassified}")
    for engine in result.executions:
        print(f"  {engine.engine_id:<18} {counts.get(engine.engine_id, 0)}")


def _engine_versions(result: ClassificationRunResult) -> dict[str, str]:
    return {execution.engine_id: execution.engine_version for execution in result.executions}


def _format_value(change: TransactionChange, col: str) -> str:
    old, new = change.old[col], change.new[col]
    if old == new:
        return ""
    return f"{old or 'unclassified'} -> {new or 'unclassified'}"


def _format_summary_change(change: SummaryChange) -> str:
    key = ", ".join(f"{k}={v}" for k, v in change.row_key.items() if v)
    if change.kind == "NEW":
        detail = " | ".join(f"{col}={new}" for col, new in change.new.items() if new)
        return f"[NEW] {key}: {detail}"
    if change.kind == "GONE":
        detail = " | ".join(f"{col}={old}" for col, old in change.old.items() if old)
        return f"[GONE] {key}: {detail}"
    detail = " | ".join(
        f"{col}={old} -> {new}"
        for col, old in change.old.items()
        for new in [change.new.get(col, "")]
        if old != new
    )
    return f"[CHANGED] {key}: {detail}"


def print_diff(report: CompareReport) -> None:
    if report.changes:
        print(f"=== Classification changes ({len(report.changes)} rows) ===")
        for change in report.changes:
            detail = " | ".join(
                _format_value(change, col)
                for col in RESULT_COLUMNS
                if _format_value(change, col)
            )
            print(f"  [{change.kind}] app={change.application_id} tx={change.transaction_id}: {detail}")

    if report.claim_changes:
        print(f"\n=== Engine claim changes ({len(report.claim_changes)} rows) ===")
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
        print("\n=== Engine claim count changes ===")
        for engine, old, new in report.engine_deltas:
            print(f"  {engine:<18} {old} → {new} ({new - old:+d})")

    if report.engine_rule_deltas:
        print("\n=== Engine rule claim count changes ===")
        for engine, rule, old, new in report.engine_rule_deltas:
            print(f"  {engine:<18} {rule:<36} {old} → {new} ({new - old:+d})")

    if report.summary_changes:
        print(f"\n=== Summary metric changes ({len(report.summary_changes)} rows) ===")
        for change in report.summary_changes:
            print(f"  {change.artifact_name}: {_format_summary_change(change)}")

    if report.run_meta_differences:
        print("\n=== Config/version changes ===")
        for diff in report.run_meta_differences:
            print(f"  {diff}")

    if report.row_count_mismatches:
        print("\n=== Row count check ===")
        for mismatch in report.row_count_mismatches:
            print(f"  {mismatch}")

    versions = ", ".join(
        f"{engine} {version}" for engine, version in sorted(report.engine_versions.items())
    )
    print(f"\n=== Summary ===\n{len(report.changes)} differences | engine versions: {versions}")


def cmd_save(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline)
    engine_baseline_path = Path(args.engine_baseline)
    run_meta_path = Path(args.run_meta)
    summaries_dir = Path(args.summaries_dir)
    artifact_paths = [
        baseline_path,
        engine_baseline_path,
        run_meta_path,
        *(summaries_dir / f"{artifact_name}.csv" for artifact_name, _ in SUMMARY_ARTIFACTS),
    ]
    existing_paths = [path for path in artifact_paths if path.exists()]
    if existing_paths and not args.replace:
        paths = ", ".join(str(path) for path in existing_paths)
        print(
            "Error: baseline artifacts already exist, refusing to overwrite: " + paths
            + ". Review the differences, then use --replace --reason <reason> to explicitly rebuild.",
            file=sys.stderr,
        )
        return 2
    if args.replace and not args.reason:
        print("Error: --replace requires --reason.", file=sys.stderr)
        return 2

    result = run_pipeline(args.input, args.config, args.category_catalog)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    extract_baseline(result).to_csv(baseline_path, index=False)
    print(f"Baseline saved: {baseline_path}")
    extract_engine_claims(result).to_csv(engine_baseline_path, index=False)
    print(f"Engine claim baseline saved: {engine_baseline_path}")
    save_run_meta(
        run_meta_path,
        _run_meta(args.input, args.config, args.category_catalog),
    )
    print(f"Run metadata saved: {run_meta_path}")
    summaries_dir.mkdir(parents=True, exist_ok=True)
    for artifact_name, amount_columns in SUMMARY_ARTIFACTS:
        extract_summary_baseline(result, artifact_name, amount_columns).to_csv(
            summaries_dir / f"{artifact_name}.csv", index=False
        )
    print(f"Summary baselines saved: {summaries_dir}")
    _print_engine_claims(result)
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"Error: baseline file not found: {baseline_path} (run save first)", file=sys.stderr)
        return 2

    baseline = load_baseline(baseline_path)
    engine_baseline_path = Path(args.engine_baseline)
    if engine_baseline_path.exists():
        engine_baseline = load_engine_claims(engine_baseline_path)
    else:
        print(
            f"Note: engine claim baseline not found: {engine_baseline_path} (skipping per-engine comparison)",
            file=sys.stderr,
        )
        engine_baseline = None

    run_meta_path = Path(args.run_meta)
    if run_meta_path.exists():
        baseline_meta = load_run_meta(run_meta_path)
    else:
        print(
            f"Note: run metadata not found: {run_meta_path} (skipping config/version comparison)",
            file=sys.stderr,
        )
        baseline_meta = None

    summaries_dir = Path(args.summaries_dir)
    summary_baselines: dict[str, pd.DataFrame] = {}
    for artifact_name, amount_columns in SUMMARY_ARTIFACTS:
        summary_path = summaries_dir / f"{artifact_name}.csv"
        if summary_path.exists():
            summary_baselines[artifact_name] = load_summary_baseline(
                summary_path, amount_columns
            )
        else:
            print(
                f"Note: summary baseline not found: {summary_path} (skipping this artifact)",
                file=sys.stderr,
            )

    result = run_pipeline(args.input, args.config, args.category_catalog)
    engine_current = extract_engine_claims(result) if engine_baseline is not None else None
    summary_currents = {
        artifact_name: extract_summary_baseline(result, artifact_name, amount_columns)
        for artifact_name, amount_columns in SUMMARY_ARTIFACTS
    }
    report = compare_transactions(
        baseline,
        extract_baseline(result),
        _engine_versions(result),
        engine_baseline=engine_baseline,
        engine_current=engine_current,
        baseline_meta=baseline_meta,
        current_meta=_run_meta(args.input, args.config, args.category_catalog),
        summary_baselines=summary_baselines,
        summary_currents=summary_currents,
    )
    print_diff(report)

    if report.has_differences:
        print("\nConclusion: differences vs baseline found (exit 1)")
        return 1
    print("\nConclusion: no differences vs baseline (exit 0)")
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
        sub.add_argument("--summaries-dir", default=str(DEFAULT_SUMMARIES_DIR), help="Deterministic summary baselines directory.")
        sub.add_argument("--config", default=str(DEFAULT_PIPELINE_CONFIG), help="Pipeline JSON configuration path.")
        sub.add_argument("--category-catalog", default=str(DEFAULT_CATEGORY_CATALOG), help="Category catalog JSON path.")
        if name == "save":
            sub.add_argument("--replace", action="store_true", help="Explicitly replace existing baseline artifacts.")
            sub.add_argument("--reason", help="Reason for an explicit baseline replacement.")
        sub.set_defaults(func=cmd_save if name == "save" else cmd_diff)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
