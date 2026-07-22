from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE_CONFIG = PROJECT_ROOT / "configs" / "pipeline.json"
DEFAULT_CATEGORY_CATALOG = PROJECT_ROOT / "configs" / "category_catalog.json"


# ── pipeline spec ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EngineSpec:
    engine_id: str
    priority: int
    enabled: bool = True


# ── top-level config ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    engines: tuple[EngineSpec, ...]
    on_engine_error: str = "fail_batch"
    unclassified_category: str = "unclassified"

    @property
    def enabled_engines(self) -> tuple[EngineSpec, ...]:
        return tuple(
            sorted(
                (spec for spec in self.engines if spec.enabled),
                key=lambda spec: spec.priority,
            )
        )


# ── JSON loaders ───────────────────────────────────────────────────────────

def _load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def load_pipeline_config(path: str | Path = DEFAULT_PIPELINE_CONFIG) -> PipelineConfig:
    payload = _load_json(path)
    engines = tuple(
        EngineSpec(
            engine_id=str(item["engine_id"]),
            priority=int(item["priority"]),
            enabled=bool(item.get("enabled", True)),
        )
        for item in payload.get("pipeline", [])
    )
    if not engines:
        raise ValueError("Pipeline configuration must contain at least one engine.")
    if not any(spec.enabled for spec in engines):
        raise ValueError("Pipeline configuration must enable at least one engine.")

    engine_ids = [spec.engine_id for spec in engines]
    if len(engine_ids) != len(set(engine_ids)):
        raise ValueError("Pipeline engine IDs must be unique.")

    enabled_priorities = [spec.priority for spec in engines if spec.enabled]
    if len(enabled_priorities) != len(set(enabled_priorities)):
        raise ValueError("Enabled engine priorities must be unique.")

    execution = payload.get("execution", {})
    on_engine_error = execution.get("on_engine_error", "fail_batch")
    if on_engine_error != "fail_batch":
        raise ValueError("Only on_engine_error='fail_batch' is currently supported.")

    return PipelineConfig(
        engines=engines,
        on_engine_error=on_engine_error,
        unclassified_category=str(
            execution.get("unclassified_category", "unclassified")
        ),
    )


def load_category_owners(
    path: str | Path = DEFAULT_CATEGORY_CATALOG,
) -> dict[str, str]:
    payload = _load_json(path)
    owners: dict[str, str] = {}
    for category, definition in payload.get("categories", {}).items():
        if not definition.get("active", True):
            continue
        owner = str(definition.get("owner_engine_id", "")).strip()
        if not owner:
            raise ValueError(
                f"Category {category!r} does not define owner_engine_id."
            )
        owners[str(category)] = owner
    if not owners:
        raise ValueError("Category catalog must contain at least one active category.")
    return owners
