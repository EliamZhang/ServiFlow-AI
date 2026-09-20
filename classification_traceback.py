"""交易分类回溯流水线（单文件 CLI，直连仓库引擎代码，无 pkl 注册层）。

合并自 backfill/ 独立子项目（classification_traceback.py / summary_traceback.py /
batch_classification.py / run_model.py / traceback_common.py / prepare_corpus.py），
统一在仓库根目录运行。模型加载 = 直接构造
``classification_core.orchestrator.ClassificationOrchestrator(config=
load_pipeline_config(), category_owners=load_category_owners())``——configs/ 与
引擎代码即活体来源，不存在 engine_classifier.pkl / joblib / sys.path 注入，
改引擎/规则/配置后无需任何"重建模型"步骤。

注意：本文件不能命名为 traceback.py——会遮蔽标准库 traceback 模块（logging 等
标准库内部 import traceback 时会加载本文件，引发循环导入）。

四种模式（--mode）：
    batch       整帧单次 run 全量批量分类（943 平台契约：四个数据文件夹 + manifest.json）
    traceback   逐 application_id 行级回溯（四个数据文件夹 + run_meta/.progress/error_detail）
    summary     逐 application_id 汇总回溯（只三个 summary 数据文件夹）
    app         单申请 JSON → JSON（追加 engineClaims / engineStats；flowTime → 行级
                sample_datetime——liability 流识别依赖该列，缺它少数贷款产品会识别不同，
                实测 app 2534115 的 20 笔 SACC Loans 变 Unknown Loans）

输出契约（batch/traceback/summary 三模式统一，CSV 全部 utf-8-sig）：
    out_dir/
    ├── run_meta.json          run_meta v2：引擎清单/版本 + configs 与规则资源 SHA-256
    ├── manifest.json          仅 batch（mode=batch + 行数统计）
    ├── error_detail.csv       仅回溯两模式（失败 app：application_id+error_message+输入快照）
    ├── .progress.txt          仅回溯两模式（断点续跑：每行一个已完成 app_id，隐藏文件）
    ├── transactions.csv/      块文件 {dataset}_{seq:06d}.csv，按 --chunk-rows 精确分块
    ├── income_summary.csv/    （summary 模式不写 transactions.csv/）
    ├── liability_summary.csv/
    └── category_summary.csv/

运行注意（与根目录 backfill.py / baseline.py 同一惯例）：
- 从仓库根执行（仓库代码优先于 .venv-backfill/site-packages 的快照副本）。
- 多进程复现固定环境变量 PYTHONHASHSEED=42（批量模式受引擎非确定性影响：liability
  少数双流贷款的派生单元格跨进程翻动，逐 app 回溯模式跨进程稳定）。
- Windows 控制台打印业务 JSON 等非 ASCII 输出时设 PYTHONIOENCODING=utf-8。
- 回溯两模式的行序 = 结果到达序（Pool.imap_unordered），非全局排序；stream_id 为
  app 内编号（与 batch 的跨 app 全局编号不同，属设计内差异）。
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from classification_core.config import (
    DEFAULT_CATEGORY_CATALOG,
    DEFAULT_PIPELINE_CONFIG,
    load_category_owners,
    load_pipeline_config,
)
from classification_core.orchestrator import ClassificationOrchestrator
from classification_core.registry import build_engine
from classification_core.service import build_transactions_frame, serialize_result

PROJECT_ROOT = Path(__file__).resolve().parent

logger = logging.getLogger(__name__)

# =====================================================
# 常量
# =====================================================

CHUNK_SIZE = 200_000  # 索引/提取读大 CSV 的 chunksize
POOL_CHUNK = 50  # imap_unordered 的 chunksize
DEFAULT_CHUNK_ROWS = 500_000  # 每个数据文件夹按多少行分一块

OUTPUT_DATASETS = (
    "transactions",
    "income_summary",
    "liability_summary",
    "category_summary",
)
SUMMARY_DATASETS = ("income_summary", "liability_summary", "category_summary")
DATASET_DIR_NAMES = {name: f"{name}.csv" for name in OUTPUT_DATASETS}

RUN_META_PATH = "run_meta.json"
MANIFEST_NAME = "manifest.json"
ERROR_DETAIL_PATH = "error_detail.csv"
PROGRESS_PATH = ".progress.txt"  # 隐藏文件：每行一个已完成 app_id（含失败/未找到）
RUN_META_FORMAT_VERSION = 2  # v1 = pkl 注册时代；v2 = 直构 orchestrator + configs/规则指纹

# 错误明细列（对齐 model_development 的 err_fields）
ERR_FIELDS = ("application_id", "error_message", "input_data_json")

# 单文件多进程模式（--mode traceback --input）的 out_dir 内部临时产物
SINGLE_FILE_CORPUS_NAME = ".corpus_single"  # 语料目录（每 app 一个切分文件）
SINGLE_FILE_SAMPLES_NAME = "samples_auto.csv"  # 自动样本清单（放语料目录外，避免被索引扫到）
AUTO_CLEAN_NAME = ".input_auto_clean.csv"  # 脏 spark 输入自动清洗后的临时干净副本

APP_FILE_PREFIX = "app_"  # 语料目录内每 app 一个文件：app_{application_id}.csv
SAMPLES_COLUMNS = ("user_id", "application_id", "sample_datetime")


# =====================================================
# 语料索引与提取（对应原 traceback_common）
# =====================================================


def build_app_index(txn_dir: str | Path) -> dict[str, set[str]]:
    """扫描语料目录下所有 CSV，建立 application_id(统一转 str) → 文件集合的索引。

    索引只读 key 列，chunked 防大文件；坏文件跳过并告警，不影响整体。
    """
    logger.info("Building corpus index: %s", txn_dir)
    index: dict[str, set[str]] = defaultdict(set)

    for fp in sorted(Path(txn_dir).glob("*.csv")):
        try:
            for chunk in pd.read_csv(
                fp, usecols=["application_id"], chunksize=CHUNK_SIZE
            ):
                uniq = chunk.drop_duplicates(subset=["application_id"])
                for app_id in uniq["application_id"].astype(str).to_numpy():
                    index[app_id].add(str(fp))
        except Exception as e:  # noqa: BLE001 - 单个坏文件不阻断整体
            logger.warning("Index failed, skipped: %s | %s", fp, e)

    logger.info("Index complete: %d keys", len(index))
    return dict(index)


def load_app_transactions(
    index: dict[str, set[str]],
    application_id: str,
) -> pd.DataFrame | None:
    """从索引命中的文件 chunked 读取该申请的全部交易（全部列）。

    不同语料文件列结构可能不同（如 sample.csv 无 bsb/account_no），
    pd.concat 自动取并集补空。key 统一按字符串比较。
    """
    fps = index.get(str(application_id))
    if not fps:
        return None

    matched = []
    for fp in sorted(fps):
        try:
            for chunk in pd.read_csv(fp, chunksize=CHUNK_SIZE):
                m = chunk[
                    chunk["application_id"].astype(str) == str(application_id)
                ]
                if not m.empty:
                    matched.append(m)
        except Exception:  # noqa: BLE001 - 单文件读取失败跳过，与索引期语义一致
            continue

    if matched:
        return pd.concat(matched, ignore_index=True)
    return None


# =====================================================
# 输出契约：四个数据文件夹 + 按行分块（全入口统一）
# =====================================================

# 943 输出契约：数据只有这四类，各自一个文件夹；线上产出很大时按行分块放进对应
# 文件夹。文件夹名带 .csv 后缀（如 transactions.csv/），块文件命名
# {dataset}_{seq:06d}.csv（如 transactions_000000.csv）。


def dataset_dir(out_dir: Path, dataset: str) -> Path:
    """四个数据文件夹之一的路径（out_dir/transactions.csv 等）。"""
    return out_dir / DATASET_DIR_NAMES[dataset]


def _write_block(frame: pd.DataFrame, dataset: str, out_dir: Path, seq: int) -> None:
    """写一块：out_dir/{dataset}.csv/{dataset}_{seq:06d}.csv。"""
    dir_path = dataset_dir(out_dir, dataset)
    dir_path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        dir_path / f"{dataset}_{seq:06d}.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_dataset_chunks(
    frame: pd.DataFrame,
    out_dir: Path,
    dataset: str,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> int:
    """整帧按 chunk_rows 精确切块写入，返回块数（空帧返回 0）。

    batch 模式用：结果一次成型，直接切块落盘，无需跨 app 缓冲。
    """
    if frame is None or frame.empty:
        return 0
    tail = frame
    seq = 0
    while len(tail) > chunk_rows:
        head, tail = tail.iloc[:chunk_rows], tail.iloc[chunk_rows:]
        _write_block(head, dataset, out_dir, seq)
        seq += 1
    _write_block(tail, dataset, out_dir, seq)
    return seq + 1


class ChunkedBuffer:
    """逐 app 到达结果的跨 app 缓冲分块写入（回溯两模式用）。

    按样本顺序累积各 app 的帧；任一数据集缓冲行数达到 chunk_rows 即把**完整块**
    落盘，余数（不足一块）保留为下块开头。精确按行切块（同一 app 的行可跨块），
    与 batch 模式的分块语义一致。

    断点续跑语义：行分块下块边界与 app 无关，已落盘的块**无法增量更新**——
    续跑 = 保留已有块（含已完成 app 的数据），新块序号接续追加。--replace 时
    调用方负责先清空块文件，seq 自然从 0 开始。
    """

    def __init__(self, out_dir: Path, chunk_rows: int = DEFAULT_CHUNK_ROWS):
        self.out_dir = out_dir
        self.chunk_rows = chunk_rows
        self._frames: dict[str, list[pd.DataFrame]] = {
            name: [] for name in OUTPUT_DATASETS
        }
        self._counts = {name: 0 for name in OUTPUT_DATASETS}
        # 续跑：已有块文件数即起始序号（旧块保留，新块接续）
        self._seqs = {
            name: len(list(dataset_dir(out_dir, name).glob("*.csv")))
            if dataset_dir(out_dir, name).is_dir()
            else 0
            for name in OUTPUT_DATASETS
        }

    def add(self, dataset: str, frame: pd.DataFrame | None) -> None:
        if frame is None or frame.empty:
            return
        self._frames[dataset].append(frame)
        self._counts[dataset] += len(frame)
        if self._counts[dataset] >= self.chunk_rows:
            self._flush(dataset, keep_tail=True)

    def _flush(self, dataset: str, keep_tail: bool) -> None:
        frames = self._frames[dataset]
        self._frames[dataset] = []
        self._counts[dataset] = 0
        if not frames:
            return
        frame = pd.concat(frames, ignore_index=True)
        tail = frame
        while len(tail) > self.chunk_rows:
            head, tail = tail.iloc[: self.chunk_rows], tail.iloc[self.chunk_rows :]
            _write_block(head, dataset, self.out_dir, self._seqs[dataset])
            self._seqs[dataset] += 1
        if keep_tail and len(tail) > 0:
            self._frames[dataset] = [tail]
            self._counts[dataset] = len(tail)
        elif not keep_tail and len(tail) > 0:
            _write_block(tail, dataset, self.out_dir, self._seqs[dataset])
            self._seqs[dataset] += 1

    def flush(self) -> None:
        """结束冲刷：残余也写块（不保留）。"""
        for dataset in OUTPUT_DATASETS:
            self._flush(dataset, keep_tail=False)


def load_progress(out_dir: Path) -> set[str]:
    """断点续跑：读已完成 app_id 集合（.progress.txt，每行一个）。"""
    path = out_dir / PROGRESS_PATH
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def mark_done(out_dir: Path, app_id: str) -> None:
    """追加标记一个 app 完成（立即落盘，崩溃后断点续跑可用）。"""
    with (out_dir / PROGRESS_PATH).open("a", encoding="utf-8") as file:
        file.write(f"{app_id}\n")


def clear_dataset_chunks(out_dir: Path) -> None:
    """清空四个数据文件夹里的所有块文件（--replace 全量重跑时用）。

    行分块下块边界与 app 无关，重跑必须清块从头写；续跑（非 replace）则保留
    已有块、序号接续追加（见 ChunkedBuffer）。
    """
    for name in OUTPUT_DATASETS:
        dir_path = dataset_dir(out_dir, name)
        if dir_path.is_dir():
            for f in dir_path.glob("*.csv"):
                f.unlink()


def detect_legacy_conflicts(out_dir: Path) -> list[str]:
    """旧版本单文件结构与新文件夹同名冲突的文件列表。

    旧版本输出 out_dir/transactions.csv 等是**文件**；新版是同名**文件夹**。
    Windows 无法文件-目录同名共存，迁到旧输出目录时给出清晰错误而非 WinError 183。
    """
    return [
        name
        for name in DATASET_DIR_NAMES.values()
        if (out_dir / name).exists() and not (out_dir / name).is_dir()
    ]


def _cleanup_outputs(out_dir: Path) -> None:
    """--replace 全量重跑：清掉进度标记、错误快照、全部块文件与旧版本残留。

    四个数据文件夹的块文件必须清空从头写（行分块下块边界与 app 无关，旧块
    残留会混入输出）；.progress.txt / error_detail.csv 是 append 语义，必须清。
    顺带清旧版本 shard_* 残留目录与旧结构单文件（app_summary.csv / manifest.json /
    各数据集同名单文件），语义为分类与汇总两模式清理逻辑的并集。
    """
    clear_dataset_chunks(out_dir)
    progress = out_dir / PROGRESS_PATH
    if progress.exists():
        progress.unlink()
    err_path = out_dir / ERROR_DETAIL_PATH
    if err_path.exists():
        err_path.unlink()
    for d in out_dir.glob("shard_*"):
        if d.is_dir():
            shutil.rmtree(d)
    for legacy in ("app_summary.csv", MANIFEST_NAME, *DATASET_DIR_NAMES.values()):
        p = out_dir / legacy
        if p.exists() and p.is_file():
            p.unlink()


def _prepare_out_dir(
    out_dir: Path, replace: bool, allow_existing_batch: bool = False
) -> None:
    """输出目录统一前置：建目录 + replace 清理或旧版单文件冲突早退。

    allow_existing_batch=True（batch 模式）：目录已有块文件时只告警，由调用方
    在写盘前清空（batch 无 --replace，全量重跑语义）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if replace:
        _cleanup_outputs(out_dir)
    else:
        legacy_conflicts = detect_legacy_conflicts(out_dir)
        if legacy_conflicts:
            raise FileExistsError(
                f"Output directory {out_dir} has legacy single-file structure "
                f"({', '.join(legacy_conflicts)}): current version writes folders "
                "with the same names. Clean those files or use a new --out-dir "
                "(--replace clears them)."
            )


# =====================================================
# 直构 orchestrator + 引擎元信息（run_meta 用）
# =====================================================


def build_orchestrator(
    pipeline_config_path: str | Path = DEFAULT_PIPELINE_CONFIG,
    category_catalog_path: str | Path = DEFAULT_CATEGORY_CATALOG,
) -> ClassificationOrchestrator:
    """直接构造 orchestrator（同根 backfill.py）：configs/ 与引擎代码即活体来源。

    本文件所有模式统一走这里，不经过任何 pkl 快照/注册表。
    """
    return ClassificationOrchestrator(
        config=load_pipeline_config(pipeline_config_path),
        category_owners=load_category_owners(category_catalog_path),
    )


def engine_specs(config) -> list[dict]:
    return [
        {
            "engine_id": spec.engine_id,
            "priority": spec.priority,
            "enabled": spec.enabled,
        }
        for spec in config.engines
    ]


def engine_versions(config) -> dict[str, str]:
    """逐台 enabled 引擎取活代码的 engine_version（值随仓库代码即时更新）。"""
    return dict(
        sorted(
            {
                spec.engine_id: build_engine(spec.engine_id).engine_version
                for spec in config.enabled_engines
            }.items()
        )
    )


# =====================================================
# run_meta v2（用什么跑的：引擎清单/版本 + configs 与规则资源指纹）
# =====================================================


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """相对仓库根的便携路径；不在根下时退回绝对路径。"""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _rule_files() -> list[Path]:
    """全部规则数据文件：各引擎 resources/**/*.csv + initial_engine/merchant_kb.csv。

    口径与 baseline.py 的规则指纹一致（含 engine resources 递归 glob，追加
    merchant_kb）。configs/ 两个 JSON 不走这里，单独记录在 fingerprints.configs。
    """
    files = list(PROJECT_ROOT.glob("*_engine/resources/**/*.csv"))
    files.append(PROJECT_ROOT / "initial_engine" / "merchant_kb.csv")
    return sorted({path.resolve() for path in files if path.is_file()})


def _fingerprints(
    input_file: str | Path,
    pipeline_config_path: str | Path,
    category_catalog_path: str | Path,
) -> dict:
    return {
        "input": {
            "path": _display_path(Path(input_file)),
            "sha256": _sha256(input_file),
        },
        "configs": {
            "pipeline": {
                "path": _display_path(Path(pipeline_config_path)),
                "sha256": _sha256(pipeline_config_path),
            },
            "category_catalog": {
                "path": _display_path(Path(category_catalog_path)),
                "sha256": _sha256(category_catalog_path),
            },
        },
        "rule_files": {
            _display_path(path): _sha256(path) for path in _rule_files()
        },
    }


def build_run_meta(
    config,
    fingerprint_input: str | Path,
    pipeline_config_path: str | Path = DEFAULT_PIPELINE_CONFIG,
    category_catalog_path: str | Path = DEFAULT_CATEGORY_CATALOG,
    mode: str = "traceback",
) -> dict:
    """run_meta v2：直接使用活 configs（无 pkl 注册）；指纹记录实际用的文件。"""
    return {
        "run_meta_format_version": RUN_META_FORMAT_VERSION,
        "runtime": {
            "entry": "classification_traceback.py",
            "mode": mode,
            "classification_source": "live ClassificationOrchestrator(configs/ + repo engine code)",
        },
        "engines": engine_specs(config),
        "on_engine_error": config.on_engine_error,
        "engine_versions": engine_versions(config),
        "fingerprints": _fingerprints(
            fingerprint_input, pipeline_config_path, category_catalog_path
        ),
    }


def _write_run_meta(
    out_dir: Path,
    config,
    fingerprint_input: str | Path,
    pipeline_config_path: str | Path,
    category_catalog_path: str | Path,
    mode: str,
) -> None:
    (out_dir / RUN_META_PATH).write_text(
        json.dumps(
            build_run_meta(
                config,
                fingerprint_input,
                pipeline_config_path,
                category_catalog_path,
                mode=mode,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


# =====================================================
# 认领层 + 单申请 JSON 序列化（--mode app 专用）
# =====================================================

# 认领层 / 引擎统计输出映射（内部诊断格式，保留 camelCase；service.py 的输出键
# 已改为 snake_case，本层为 traceback 专用、不随其变更）
CLAIM_OUTPUT_MAP = {
    "engine_id": "engineId",
    "application_id": "applicationNo",
    "transaction_id": "transactionId",
    "bscat": "bscat",
    "counterparty": "counterparty",
    "classification_rule_id": "classificationRuleId",
    "classification_reason": "classificationReason",
    "stream_id": "streamId",
    "priority": "priority",
}

_EXECUTION_OUTPUT_MAP = {
    "engine_id": "engineId",
    "engine_version": "engineVersion",
    "priority": "priority",
    "candidate_count": "candidateCount",
    "prediction_count": "predictionCount",
    "accepted_count": "acceptedCount",
    "duration_seconds": "durationSeconds",
}

CLAIM_KEY_COLUMNS = ("application_id", "transaction_id")
ENGINE_CLAIM_COLUMNS = (
    "engine_id",
    *CLAIM_KEY_COLUMNS,
    "bscat",
    "counterparty",
    "classification_rule_id",
    "classification_reason",
    "stream_id",
    "priority",
)


def extract_engine_claims(result) -> pd.DataFrame:
    """把每台引擎的认领快照拼成一张帧（对齐 baseline.extract_engine_claims）。"""
    frames = []
    for execution in result.executions:
        claims = execution.claims.copy()
        for col in ENGINE_CLAIM_COLUMNS:
            if col not in claims.columns:
                claims[col] = pd.NA
        claims["engine_id"] = execution.engine_id
        frames.append(claims[list(ENGINE_CLAIM_COLUMNS)])
    output = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=ENGINE_CLAIM_COLUMNS)
    )
    for col in (*CLAIM_KEY_COLUMNS, "engine_id"):
        output[col] = output[col].astype(str)
    for col in ENGINE_CLAIM_COLUMNS:
        if col not in CLAIM_KEY_COLUMNS and col != "engine_id":
            output[col] = output[col].fillna("").astype(str)
    return output.reset_index(drop=True)


def _to_camel(snake: str) -> str:
    head, *rest = snake.split("_")
    return head + "".join(part.capitalize() for part in rest)


def _serialize_value(value: Any) -> Any:
    # 与 classification_core/service.py 的 _serialize_value 同语义
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str) and value == "":
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (pd.Timedelta, type(pd.NaT))):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (float, int, str, bool)):
        return value
    return str(value)


def _serialize_records(
    frame: pd.DataFrame,
    field_map: dict[str, str] | None = None,
    exclude: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    # 认领层/引擎统计专用的小序列化器（service._serialize_records 是私有实现，
    # 语义一致不跨模块引用；单申请主体序列化复用 service.serialize_result）
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        record: dict[str, Any] = {}
        for column in frame.columns:
            if column in exclude:
                continue
            name = (
                field_map.get(column, _to_camel(column))
                if field_map is not None
                else _to_camel(column)
            )
            record[name] = _serialize_value(row[column])
        records.append(record)
    return records


def serialize_claims(result) -> list[dict[str, Any]]:
    claims = extract_engine_claims(result)
    if claims is None or claims.empty:
        return []
    return _serialize_records(claims, field_map=CLAIM_OUTPUT_MAP)


def serialize_engine_stats(result) -> list[dict[str, Any]]:
    return _serialize_records(
        pd.DataFrame(
            [
                {
                    "engine_id": e.engine_id,
                    "engine_version": e.engine_version,
                    "priority": e.priority,
                    "candidate_count": e.candidate_count,
                    "prediction_count": e.prediction_count,
                    "accepted_count": e.accepted_count,
                    "duration_seconds": e.duration_seconds,
                }
                for e in result.executions
            ]
        ),
        field_map=_EXECUTION_OUTPUT_MAP,
    )


def prepare_transactions(payload: dict) -> pd.DataFrame:
    """build_transactions_frame + 补齐行级 sample_datetime。

    JSON 链路的交易记录不含 sample_datetime（在顶层 flowTime），而 liability
    引擎的流识别/还款匹配依赖该列——缺它会导致少数贷款产品识别与 CSV 链路不同
    （实测 app 2534115 的 20 笔 SACC Loans 变 Unknown Loans）。根 model_main /
    verify_model 的链路不补此列（生产契约保持不动），本模式刻意补上。
    """
    frame = build_transactions_frame(payload)
    if "sample_datetime" not in frame.columns:
        flow_time = payload.get("flowTime")
        if flow_time is not None:
            frame["sample_datetime"] = flow_time
    return frame


def generate_classification(
    input_vars: dict[str, Any],
    pipeline_config_path: str | Path = DEFAULT_PIPELINE_CONFIG,
    category_catalog_path: str | Path = DEFAULT_CATEGORY_CATALOG,
) -> dict[str, Any]:
    """单函数入口：一个申请 JSON 进，分类结果 JSON 出。

    输出 = service.serialize_result 的完整结果（bscat_transactions + bscat_summaries +
    bscat_stats + bank_accounts）＋ engineClaims（认领层，含被覆盖的认领）＋ engineStats。
    失败返回 ``{"status": "failed", "error": ..., "stats": {...}}``（run_model 原语义，
    键名不随 service 输出改名），不抛异常。
    """
    try:
        transactions = prepare_transactions(input_vars)
        orchestrator = build_orchestrator(
            pipeline_config_path, category_catalog_path
        )
        result = orchestrator.run(transactions)
        output = serialize_result(result, input_vars)
    except Exception as exc:  # noqa: BLE001 - 对外接口不抛异常，失败体同 run_model 原语义
        logger.warning("Classification failed | %s", exc)
        return {
            "status": "failed",
            "error": str(exc),
            "stats": {"txnRawInputCnt": 0, "transactionDateMax": None},
        }

    output["engineClaims"] = serialize_claims(result)
    output["engineStats"] = serialize_engine_stats(result)
    return output


# =====================================================
# 多进程 worker 基座（模块级函数才能被 spawn pickle）
# =====================================================

_INDEX = None
_HANDLE = None


def _init_worker(index, pipeline_config_path, category_catalog_path) -> None:
    """每个 worker 进程内各直构一个 orchestrator（轻量，仅读两个 configs JSON）。"""
    global _INDEX, _HANDLE
    _INDEX = index
    _HANDLE = build_orchestrator(pipeline_config_path, category_catalog_path)


def _classify_one(app_id: str, index, orchestrator):
    """单个申请的完整回溯：提取 → 重跑流水线 → 拆出行级 + 汇总结果。"""
    frame = None
    try:
        frame = load_app_transactions(index, app_id)
        if frame is None or frame.empty:
            return {
                "app_id": app_id,
                "status": "not_found",
                "transactions": None,
                "summaries": None,
            }, None

        result = orchestrator.run(frame)
        return {
            "app_id": app_id,
            "status": "ok",
            "transactions": result.transactions,
            "summaries": {
                artifact.name: artifact.data for artifact in result.summaries
            },
        }, None

    except Exception as e:  # noqa: BLE001 - 单 app 失败不中断整体，记快照
        logger.warning("Traceback failed app=%s | %s", app_id, e)
        err_detail = {
            "application_id": app_id,
            "error_message": str(e),
            "input_data_json": json.dumps(
                frame.to_dict("records") if frame is not None else [],
                ensure_ascii=False,
                default=str,
            ),
        }
        return {
            "app_id": app_id,
            "status": "error",
            "transactions": None,
            "summaries": None,
            "error": str(e),
        }, err_detail

    finally:
        gc.collect()


def process_one_app(app_id: str):
    """Pool worker 入口（初值经 _init_worker 注入）。"""
    return _classify_one(app_id, _INDEX, _HANDLE)


# =====================================================
# 语料预处理（原 prepare_corpus.py：按 app 切分 + 脏 spark 导出清洗）
# =====================================================


def split_by_application(
    input_csv: str | Path,
    corpus_dir: str | Path,
    samples_path: str | Path | None = None,
) -> list[dict[str, str]]:
    """单遍流式把 input_csv 按 application_id 切成 corpus_dir 下的 per-app 文件。

    --mode traceback --input 单文件模式内部调用（自动切 app → 多进程回溯）；
    输入须为标准可读 CSV（含逗号的 text 列需引号包裹）。按 app 缓冲**原始行文本**
    （约等于文件体积的一半内存量级），读完后逐 app 落盘。

    返回 samples 行（app 首现的 user_id / application_id / sample_datetime），
    顺序 = 文件内首现顺序——小批试跑取前 N 行即可。
    """
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # app_id -> 该 app 的全部原始行文本（保留原始引号转义，写盘时原样输出）
    lines: dict[str, list[str]] = {}
    samples: list[dict[str, str]] = []
    seen: set[str] = set()

    with Path(input_csv).open(encoding="utf-8-sig", newline="") as fin:
        header = next(fin)
        if header.endswith("\n"):
            header = header[:-1]
        header_cols = header.split(",")
        for col in SAMPLES_COLUMNS:
            if col not in header_cols:
                raise ValueError(f"Input CSV is missing {col} column: {input_csv}")
        app_idx, user_idx, dt_idx = (
            header_cols.index("application_id"),
            header_cols.index("user_id"),
            header_cols.index("sample_datetime"),
        )
        for row in fin:
            # 找第 app_idx 个字段的边界：字段号 = 前面出现的逗号个数
            # （application_id 在前 17 个定界字段内，无逗号歧义）
            start, commas = 0, 0
            while commas < app_idx:
                pos = row.index(",", start)
                start = pos + 1
                commas += 1
            end = row.index(",", start)
            app_id = row[start:end]
            lines.setdefault(app_id, []).append(row)
            if app_id not in seen:
                seen.add(app_id)
                row_cols = row[:-1].split(",", app_idx + 1)
                samples.append(
                    {
                        "user_id": row_cols[user_idx],
                        "application_id": app_id,
                        "sample_datetime": row_cols[dt_idx],
                    }
                )

    for app_id, app_lines in lines.items():
        with (corpus_dir / f"{APP_FILE_PREFIX}{app_id}.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as fout:
            fout.write(header + "\n")
            fout.writelines(app_lines)

    if samples_path is not None:
        samples_path = Path(samples_path)
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        with samples_path.open("w", encoding="utf-8-sig", newline="") as fout:
            w = csv.writer(fout)
            w.writerow(SAMPLES_COLUMNS)
            w.writerows(
                [
                    (s["user_id"], s["application_id"], s["sample_datetime"])
                    for s in samples
                ]
            )
    return samples


def detect_dirty_csv(
    input_csv: str | Path, probe_rows: int = 20_000
) -> tuple[bool, int, int | None]:
    """探测 input_csv 是否脏格式：csv.reader 解析前 probe_rows 行，字段数与表头不符即脏。

    返回 (is_clean, probed_rows, first_bad_row)。干净文件（含引号包裹的逗号字段）
    能正确解析为表头等宽，不会误报。
    """
    with Path(input_csv).open(encoding="utf-8-sig", newline="") as fin:
        reader = csv.reader(fin)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Input is an empty file: {input_csv}")
        ncols = len(header)
        for i, row in enumerate(reader, start=2):
            if i > probe_rows + 1:
                break
            if len(row) != ncols:
                return False, probe_rows, i
    return True, probe_rows, None


def auto_clean_csv(input_csv: str | Path, output_csv: str | Path) -> dict[str, int]:
    """把脏 spark 导出洗成标准可读 CSV（逐行流式，可处理 2M 行级文件）。

    逐原始行处理：
    - 逗号数 == 表头字段数-1：干净行，原样透传（保留其自带引号）；
    - 逗号数 > 表头字段数-1：裸逗号行——无引号时按 split(',', ncols-1) 修复
      （多余逗号须都在末字段），含引号则报错（无法安全判断字段边界）；
    - 逗号数 < 表头字段数-1：畸形行，报错带行号。

    返回 {input_rows, repaired_rows, output_rows}。输出为 utf-8-sig 标准 CSV。
    """
    with Path(input_csv).open(encoding="utf-8-sig", newline="") as fin, Path(
        output_csv
    ).open("w", encoding="utf-8-sig", newline="") as fout:
        header_line = next(fin)
        ncols = len(header_line.rstrip("\r\n").split(","))
        fout.write(header_line)
        writer = csv.writer(fout)
        repaired = 0
        input_rows = 0
        for ln, raw in enumerate(fin, start=2):
            raw = raw.rstrip("\r\n")
            input_rows += 1
            n_commas = raw.count(",")
            if n_commas == ncols - 1:
                fout.write(raw + "\n")  # 干净行透传
            elif n_commas < ncols - 1:
                raise ValueError(
                    f"Cannot auto-clean: row {ln} has {n_commas + 1} fields, fewer "
                    f"than header {ncols} (missing columns). Fix the file first."
                )
            elif '"' in raw:
                raise ValueError(
                    f"Cannot auto-clean: row {ln} contains double quotes with excess "
                    "commas — extra commas may sit inside quoted fields, beyond "
                    "auto-repair. Fix the file manually."
                )
            else:
                writer.writerow(raw.split(",", ncols - 1))  # 末字段保留多余逗号
                repaired += 1
    return {"input_rows": input_rows, "repaired_rows": repaired, "output_rows": input_rows}


# =====================================================
# 回溯流水线核心（classification / summary 共用执行体）
# =====================================================


def _load_samples(
    sample_path: str | Path | None, app_ids: list[str] | None
) -> pd.DataFrame:
    """读入样本清单（sample_path 或 app_ids 二选一），统一 application_id 为 str。"""
    if sample_path is not None:
        samples = pd.read_csv(sample_path, encoding="utf-8-sig")
        if "application_id" not in samples.columns:
            raise ValueError(
                f"Samples file is missing application_id column: {sample_path}"
            )
    elif app_ids is not None:
        samples = pd.DataFrame({"application_id": [str(a) for a in app_ids]})
    else:
        raise ValueError("Must provide sample_path or app_ids")

    samples = samples.copy()
    samples["application_id"] = samples["application_id"].astype(str)
    return samples


def _run_app_pipeline(
    samples: pd.DataFrame,
    txn_dir: str | Path,
    out_dir: Path,
    orchestrator,
    pipeline_config_path: str | Path,
    category_catalog_path: str | Path,
    chunk_rows: int,
    n_workers: int | None,
    replace: bool,
    datasets: tuple[str, ...],
    fingerprint_input: str | Path,
    mode: str,
) -> Path:
    """逐 app 回溯执行体：断点续跑 + 索引 + run_meta + Pool/内联 + 分块落盘。

    datasets 决定写哪些数据文件夹：classification 传 OUTPUT_DATASETS（含
    transactions），summary 传 SUMMARY_DATASETS（只三个汇总）。返回 out_dir。
    """
    _prepare_out_dir(out_dir, replace)
    done_ids = load_progress(out_dir)
    if done_ids:
        logger.info("Resume enabled: %d completed applications skipped", len(done_ids))

    pending = samples[~samples["application_id"].isin(done_ids)]
    if pending.empty:
        logger.info("No pending applications")
        return out_dir

    if txn_dir is None:
        raise ValueError("Must provide txn_dir (transaction corpus directory)")
    index = build_app_index(txn_dir)

    # run_meta（用什么跑的）——app_ids 模式没有真实 samples 文件可指纹时，
    # 先把样本清单落到 out_dir 再指纹
    if fingerprint_input is None:
        fingerprint_input = out_dir / "samples.csv"
        samples.to_csv(fingerprint_input, index=False, encoding="utf-8-sig")
    _write_run_meta(
        out_dir,
        orchestrator.config,
        fingerprint_input,
        pipeline_config_path,
        category_catalog_path,
        mode=mode,
    )

    # ── 执行：逐 app 结果进 ChunkedBuffer，按行分块落盘 ────────
    n_workers = n_workers or max(os.cpu_count() - 2, 1)
    pending_ids = pending["application_id"].tolist()
    sequential = n_workers <= 1

    buffers = ChunkedBuffer(out_dir, chunk_rows)
    error_rows: list[dict] = []

    def handle_result(app_id: str, payload: dict, err_detail: dict | None) -> None:
        if err_detail is not None:
            error_rows.append(err_detail)
        else:
            if "transactions" in datasets:
                buffers.add("transactions", payload["transactions"])
            summaries = payload.get("summaries") or {}
            for dataset in SUMMARY_DATASETS:
                if dataset in datasets:
                    buffers.add(dataset, summaries.get(dataset))
        # 立即标记完成（含 not_found/error），崩溃后断点续跑不重跑
        mark_done(out_dir, app_id)

    logger.info(
        "Starting traceback of %d applications | workers=%s | chunk_rows=%d",
        len(pending_ids),
        "inline" if sequential else n_workers,
        chunk_rows,
    )

    if sequential:
        for app_id in tqdm(pending_ids, desc="Traceback apps", ascii=True):
            payload, err = _classify_one(app_id, index, orchestrator)
            handle_result(app_id, payload, err)
    else:
        from multiprocessing import Pool

        with Pool(
            n_workers,
            initializer=_init_worker,
            initargs=(index, pipeline_config_path, category_catalog_path),
        ) as pool:
            for payload, err in tqdm(
                pool.imap_unordered(
                    process_one_app, pending_ids, chunksize=POOL_CHUNK
                ),
                total=len(pending_ids),
                desc="Traceback apps", ascii=True,
            ):
                handle_result(payload["app_id"], payload, err)

    buffers.flush()
    gc.collect()

    # ── 落盘：error_detail ────────────────────────────────────
    if error_rows:
        err_path = out_dir / ERROR_DETAIL_PATH
        err_frame = pd.DataFrame(error_rows)
        if err_path.exists() and not replace:
            old = pd.read_csv(err_path, encoding="utf-8-sig")
            err_frame = pd.concat([old, err_frame], ignore_index=True)
        err_frame.to_csv(err_path, index=False, encoding="utf-8-sig")

    logger.info("Done: %s", out_dir)
    return out_dir


def run_classification_pipeline(
    sample_path: str | Path | None = None,
    txn_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    app_ids: list[str] | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    n_workers: int | None = None,
    replace: bool = False,
    pipeline_config_path: str | Path = DEFAULT_PIPELINE_CONFIG,
    category_catalog_path: str | Path = DEFAULT_CATEGORY_CATALOG,
) -> Path:
    """按 application_id 回溯分类流水线（四个数据文件夹 + 按行分块）。

    sample_path 与 app_ids 二选一；txn_dir 为交易语料目录。返回实际使用的 out_dir。
    """
    samples = _load_samples(sample_path, app_ids)
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "output" / f"traceback_{timestamp}"
    out_dir = Path(out_dir)
    orchestrator = build_orchestrator(pipeline_config_path, category_catalog_path)
    return _run_app_pipeline(
        samples=samples,
        txn_dir=txn_dir,
        out_dir=out_dir,
        orchestrator=orchestrator,
        pipeline_config_path=pipeline_config_path,
        category_catalog_path=category_catalog_path,
        chunk_rows=chunk_rows,
        n_workers=n_workers,
        replace=replace,
        datasets=OUTPUT_DATASETS,
        fingerprint_input=Path(sample_path) if sample_path is not None else None,
        mode="traceback",
    )


def run_summary_pipeline(
    sample_path: str | Path | None = None,
    txn_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    app_ids: list[str] | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    n_workers: int | None = None,
    replace: bool = False,
    pipeline_config_path: str | Path = DEFAULT_PIPELINE_CONFIG,
    category_catalog_path: str | Path = DEFAULT_CATEGORY_CATALOG,
) -> Path:
    """按 application_id 回溯汇总指标（app 级输出，只三个 summary 数据文件夹）。

    复用与分类链路相同的提取/重跑 worker，只是不写行级 transactions。
    返回实际使用的 out_dir。
    """
    samples = _load_samples(sample_path, app_ids)
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "output" / f"traceback_summary_{timestamp}"
    out_dir = Path(out_dir)
    orchestrator = build_orchestrator(pipeline_config_path, category_catalog_path)
    return _run_app_pipeline(
        samples=samples,
        txn_dir=txn_dir,
        out_dir=out_dir,
        orchestrator=orchestrator,
        pipeline_config_path=pipeline_config_path,
        category_catalog_path=category_catalog_path,
        chunk_rows=chunk_rows,
        n_workers=n_workers,
        replace=replace,
        datasets=SUMMARY_DATASETS,
        fingerprint_input=Path(sample_path) if sample_path is not None else None,
        mode="summary",
    )


def run_single_file_pipeline(
    input_csv: str | Path,
    out_dir: str | Path | None = None,
    n_workers: int | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    replace: bool = False,
    pipeline_config_path: str | Path = DEFAULT_PIPELINE_CONFIG,
    category_catalog_path: str | Path = DEFAULT_CATEGORY_CATALOG,
) -> Path:
    """单文件多进程分类：一份大交易 CSV → 自动按 app 切分 → 逐 app 回溯。

    batch 模式（一次 run）受单进程与整帧内存限制，大文件（几百万行）在单机上
    跑不动；本模式把输入切成语料目录后走逐 app 回溯链路（Pool 多进程 + 断点
    续跑 + 四文件夹分块输出，语义与逐 app 回溯一致——stream_id 为 app 内编号，
    liability 少数双流贷款派生单元格与 batch 范围不同）。

    内部步骤：
    0. 输入格式检测：spark 裸逗号导出（字段数不符）自动清洗到
       out_dir/.input_auto_clean.csv，标准 CSV（含引号字段）直接使用；
    1. 切分：split_by_application 把输入按 app 切成 out_dir/.corpus_single/
       （samples 放 out_dir/samples_auto.csv，避免被语料索引扫到）；
    2. 委托 run_classification_pipeline 执行逐 app 回溯（四文件夹 + run_meta +
       .progress 断点续跑 + error_detail）；
    3. run_meta 追加输入文件 SHA-256 与清洗统计（自动清洗时；分类链路只指纹
       样本清单，本模式把真正输入也留痕）。

    返回实际使用的 out_dir。
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "output" / f"single_file_{timestamp}"
    out_dir = Path(out_dir)
    _prepare_out_dir(out_dir, replace)

    # ── 0.5 输入格式自动检测：spark 裸逗号导出 → 自动清洗（无需手动预清洗） ──
    # 干净标准 CSV（含引号字段）直接切分；脏文件（字段数不符）洗到
    # .input_auto_clean.csv 再切分，run_meta 记录原始文件指纹 + 清洗统计。
    clean_input, cleaning_note = input_csv, None
    is_clean, probed, first_bad = detect_dirty_csv(input_csv)
    if not is_clean:
        clean_path = out_dir / AUTO_CLEAN_NAME
        stats = auto_clean_csv(input_csv, clean_path)
        cleaning_note = {"auto_cleaned": True, "first_bad_row": first_bad, **stats}
        logger.info(
            "Dirty spark format detected (probed %d rows, first bad row %s); "
            "auto-cleaned %d rows (%d repaired) -> %s",
            probed,
            first_bad,
            stats["input_rows"],
            stats["repaired_rows"],
            AUTO_CLEAN_NAME,
        )
        clean_input = clean_path
    else:
        logger.info("Input is standard CSV (no field-count anomalies), split directly")

    # ── 1. 按 app 切分（每 app 一个小文件；samples 放语料目录外） ─────────
    corpus_dir = out_dir / SINGLE_FILE_CORPUS_NAME
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)  # 重新切分保持与输入一致（幂等）
    samples_path = out_dir / SINGLE_FILE_SAMPLES_NAME
    samples = split_by_application(clean_input, corpus_dir, samples_path=samples_path)
    logger.info("Split complete: %d apps -> %s", len(samples), corpus_dir)

    # ── 2. 委托逐 app 回溯（四文件夹 + 断点续跑 + 多进程） ───────────────
    run_classification_pipeline(
        sample_path=samples_path,
        txn_dir=corpus_dir,
        out_dir=out_dir,
        chunk_rows=chunk_rows,
        n_workers=n_workers,
        replace=replace,
        pipeline_config_path=pipeline_config_path,
        category_catalog_path=category_catalog_path,
    )

    # ── 3. run_meta 追加输入文件指纹（单文件模式真正输入留痕） ───────────
    meta_path = out_dir / RUN_META_PATH
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sha = hashlib.sha256()
        with input_csv.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                sha.update(block)
        meta.setdefault("fingerprints", {})["input_file"] = {
            "path": str(input_csv),
            "sha256": sha.hexdigest(),
        }
        if cleaning_note:
            meta["input_cleaning"] = cleaning_note  # 自动清洗统计（原始输入指纹仍指原文件）
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    logger.info("Done: %s", out_dir)
    return out_dir


# =====================================================
# batch 模式（943 平台生产入口）
# =====================================================


def run_batch_pipeline(
    input_csv: str | Path,
    out_dir: str | Path | None = None,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    pipeline_config_path: str | Path = DEFAULT_PIPELINE_CONFIG,
    category_catalog_path: str | Path = DEFAULT_CATEGORY_CATALOG,
) -> Path:
    """单份交易 CSV 全量分类（一次 run 覆盖全量），输出四个数据文件夹。

    与根 backfill.py 完全同语义（同一 orchestrator、同一 batch 范围：stream_id
    为跨 app 全局编号），输出为四文件夹 + 按行分块 + manifest.json。
    返回实际使用的 out_dir。
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "output" / f"batch_{timestamp}"
    out_dir = Path(out_dir)
    _prepare_out_dir(out_dir, replace=False)

    # batch 无 --replace（全量重跑语义）：若目录已有块文件（上次 batch 产物），
    # 必须清空从头写——否则旧块残留会与新块混在一起（块序号从 0 重来会覆盖
    # 前面几块，但多出来的旧块会残留，拼接后数据重复）。
    # 注意：不要把回溯输出目录（含 .progress.txt）传给本入口，清块会破坏它。
    if any(
        (out_dir / name).is_dir() and list((out_dir / name).glob("*.csv"))
        for name in DATASET_DIR_NAMES.values()
    ):
        logger.warning(
            "Output directory %s already has chunk files (previous batch output); "
            "clearing before rewrite",
            out_dir,
        )
        clear_dataset_chunks(out_dir)

    orchestrator = build_orchestrator(pipeline_config_path, category_catalog_path)

    # run_meta（用什么跑的：直构 orchestrator + configs 与规则文件指纹）
    _write_run_meta(
        out_dir,
        orchestrator.config,
        input_csv,
        pipeline_config_path,
        category_catalog_path,
        mode="batch",
    )

    # ── 读取 + 分类（单次 run，batch 范围） ──
    transactions = pd.read_csv(input_csv, encoding="utf-8-sig")
    result = orchestrator.run(transactions)
    logger.info(
        "Classification complete: %d transactions | %d engines | %d summaries",
        len(result.transactions),
        len(result.executions),
        len(result.summaries),
    )

    # ── 行级输出：四个数据文件夹 + 按行分块 ──
    write_dataset_chunks(result.transactions, out_dir, "transactions", chunk_rows)

    summaries = {artifact.name: artifact.data for artifact in result.summaries}
    for dataset in ("income_summary", "liability_summary", "category_summary"):
        data = summaries.get(dataset)
        write_dataset_chunks(data, out_dir, dataset, chunk_rows)

    # ── manifest ──
    (out_dir / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "mode": "batch",
                "input_file": str(input_csv),
                "chunk_rows": chunk_rows,
                "apps": int(result.transactions["application_id"].nunique()),
                "transactions": int(len(result.transactions)),
                "note": "Output = four dataset folders (transactions/income_summary/"
                "liability_summary/category_summary), chunked by chunk_rows",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info("Done: %s", out_dir)
    return out_dir


# =====================================================
# CLI
# =====================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transaction classification traceback pipeline (single file, direct "
            "repo engine code, no pkl registration). Modes:\n"
            "  batch      full-frame batch classification (943 contract: four dataset "
            "folders + manifest.json)\n"
            "  traceback  per-application row-level traceback (four dataset folders + "
            ".progress resume)\n"
            "  summary    per-application summary-only traceback (three summary folders)\n"
            "  app        single-application JSON -> JSON (with engineClaims/engineStats)"
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["batch", "traceback", "summary", "app"],
        help="Run mode (see description).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "batch/app: input file (transaction CSV / application JSON, required). "
            "traceback: a single large transaction CSV (auto-split by app + "
            "multi-process; mutually exclusive with --sample/--txn-dir)."
        ),
    )
    parser.add_argument(
        "--sample", default=None, help="traceback/summary: samples CSV (one application per row)."
    )
    parser.add_argument(
        "--txn-dir", default=None, help="traceback/summary: transaction corpus directory (glob *.csv)."
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="batch/traceback/summary: output directory (default: output/<mode-prefix>_{timestamp}).",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=DEFAULT_CHUNK_ROWS,
        help=f"Rows per chunk file inside each dataset folder (default {DEFAULT_CHUNK_ROWS}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="traceback/summary: worker count; 1 = inline sequential (default: cpu_count-2).",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="traceback/summary: ignore existing .progress and rerun everything from scratch.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="app: output JSON path (default: print to stdout).",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_PIPELINE_CONFIG),
        help=f"Pipeline JSON config path (default {DEFAULT_PIPELINE_CONFIG}).",
    )
    parser.add_argument(
        "--category-catalog",
        default=str(DEFAULT_CATEGORY_CATALOG),
        help=f"Category catalog JSON path (default {DEFAULT_CATEGORY_CATALOG}).",
    )
    return parser.parse_args(argv)


# 各模式允许的 CLI 参数（config/category-catalog 全部模式通用）
_MODE_ALLOWED_FLAGS: dict[str, set[str]] = {
    "batch": {"input", "out_dir", "chunk_rows"},
    "traceback": {"input", "sample", "txn_dir", "out_dir", "chunk_rows", "workers", "replace"},
    "summary": {"sample", "txn_dir", "out_dir", "chunk_rows", "workers", "replace"},
    "app": {"input", "output"},
}


def _validate_mode_args(args: argparse.Namespace) -> None:
    """各模式的参数组合校验（错误即 SystemExit 2）。"""
    given = {
        "input": args.input is not None,
        "sample": args.sample is not None,
        "txn_dir": args.txn_dir is not None,
        "out_dir": args.out_dir is not None,
        "chunk_rows": args.chunk_rows != DEFAULT_CHUNK_ROWS,
        "workers": args.workers is not None,
        "replace": bool(args.replace),
        "output": args.output is not None,
    }
    bad = [
        name
        for name, present in given.items()
        if present and name not in _MODE_ALLOWED_FLAGS[args.mode]
    ]
    if bad:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in bad)
        raise SystemExit(f"Flag(s) {flags} not supported for --mode {args.mode}")

    if args.mode in ("batch", "app") and args.input is None:
        raise SystemExit("--input is required for --mode batch/app")

    if args.mode == "traceback" and args.input is not None and (
        args.sample is not None or args.txn_dir is not None
    ):
        raise SystemExit("--input and --sample/--txn-dir are mutually exclusive")

    if args.mode in ("traceback", "summary") and args.input is None:
        if (args.sample is None) != (args.txn_dir is None):
            raise SystemExit("--sample and --txn-dir must be provided together")
        if args.sample is None:
            raise SystemExit(
                f"--mode {args.mode} needs --input (single large CSV) or "
                "--sample/--txn-dir pair"
            )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    args = parse_args(argv)
    _validate_mode_args(args)

    if args.mode == "batch":
        out_dir = run_batch_pipeline(
            input_csv=args.input,
            out_dir=args.out_dir,
            chunk_rows=args.chunk_rows,
            pipeline_config_path=args.config,
            category_catalog_path=args.category_catalog,
        )
        print(f"Batch classification done: {out_dir}")

    elif args.mode == "traceback":
        if args.input is not None:
            out_dir = run_single_file_pipeline(
                input_csv=args.input,
                out_dir=args.out_dir,
                chunk_rows=args.chunk_rows,
                n_workers=args.workers,
                replace=args.replace,
                pipeline_config_path=args.config,
                category_catalog_path=args.category_catalog,
            )
        else:
            out_dir = run_classification_pipeline(
                sample_path=args.sample,
                txn_dir=args.txn_dir,
                out_dir=args.out_dir,
                chunk_rows=args.chunk_rows,
                n_workers=args.workers,
                replace=args.replace,
                pipeline_config_path=args.config,
                category_catalog_path=args.category_catalog,
            )
        print(f"Traceback done: {out_dir}")

    elif args.mode == "summary":
        out_dir = run_summary_pipeline(
            sample_path=args.sample,
            txn_dir=args.txn_dir,
            out_dir=args.out_dir,
            chunk_rows=args.chunk_rows,
            n_workers=args.workers,
            replace=args.replace,
            pipeline_config_path=args.config,
            category_catalog_path=args.category_catalog,
        )
        print(f"Summary traceback done: {out_dir}")

    else:  # app
        with Path(args.input).open(encoding="utf-8") as file:
            payload = json.load(file)
        result = generate_classification(
            payload,
            pipeline_config_path=args.config,
            category_catalog_path=args.category_catalog,
        )
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"Output written: {args.output}")
        else:
            print(text)


if __name__ == "__main__":
    main()
