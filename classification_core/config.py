from __future__ import annotations

import json
from dataclasses import dataclass, field
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


# ── field-level override policy ────────────────────────────────────────────

@dataclass(frozen=True)
class OverrideRule:
    """Allow *by* to overwrite a field previously set by engines in *when_set_by*."""
    by: str
    when_set_by: tuple[str, ...]


@dataclass(frozen=True)
class FieldPolicy:
    """Per-field rules that govern which engine's value survives.

    ``immutable_when_set_by``
        Once an engine in this list sets the field, no later engine may change it.
    ``fill_blank_only``
        If *True*, engines may only write the field when the current value is
        blank/NA.  ``override_rules`` can grant specific exceptions.
    ``override_rules``
        Explicit grants: *by* is allowed to overwrite even when the field already
        has a value, provided the current value was set by an engine in
        *when_set_by*.
    """
    immutable_when_set_by: tuple[str, ...] = ()
    fill_blank_only: bool = False
    override_rules: tuple[OverrideRule, ...] = ()

    # -- queries used by the orchestrator -----------------------------------

    def can_overwrite(
        self,
        engine_id: str,
        *,
        current_set_by: str | None,
        current_is_blank: bool,
    ) -> bool:
        """Return *True* if *engine_id* is allowed to write this field right now."""
        # 1. Immutable guard — nobody overwrites a locked-in value.
        if (
            current_set_by is not None
            and current_set_by in self.immutable_when_set_by
        ):
            return False

        # 2. Fill-blank-only guard — only write when the field is empty …
        if self.fill_blank_only and not current_is_blank:
            # … unless an explicit override rule applies.
            for rule in self.override_rules:
                if (
                    rule.by == engine_id
                    and current_set_by in rule.when_set_by
                ):
                    return True
            return False

        return True


# ── top-level config ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    engines: tuple[EngineSpec, ...]
    on_engine_error: str = "fail_batch"
    unclassified_category: str = "unclassified"
    field_policies: dict[str, FieldPolicy] = field(default_factory=dict)

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
        field_policies=_parse_field_policies(payload.get("field_policy", {})),
    )


def _parse_field_policies(raw: dict) -> dict[str, FieldPolicy]:
    policies: dict[str, FieldPolicy] = {}
    for field_name, raw_policy in raw.items():
        override_rules: list[OverrideRule] = []
        for raw_rule in raw_policy.get("override_rules", []):
            override_rules.append(
                OverrideRule(
                    by=str(raw_rule["by"]),
                    when_set_by=tuple(str(e) for e in raw_rule["when_set_by"]),
                )
            )
        policies[str(field_name)] = FieldPolicy(
            immutable_when_set_by=tuple(
                str(e) for e in raw_policy.get("immutable_when_set_by", [])
            ),
            fill_blank_only=bool(raw_policy.get("fill_blank_only", False)),
            override_rules=tuple(override_rules),
        )
    return policies


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
