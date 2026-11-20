"""Schema versioning and explicit result payload migration for artifact bundles."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from contracts import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS

_SEMVER_RE = re.compile(r"^\s*(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\s*$")
_SUPPORTED_COMMANDS: frozenset[str] = frozenset({"fetch", "analyze", "compare", "portfolio"})
_MAX_DD_TOL = 1e-12
_ZERO_SNAP_TOL = 1e-15


class ArtifactError(ValueError):
    """Base error raised for artifact persistence, migration, or validation failures."""


class UnsupportedSchemaVersionError(ArtifactError):
    """Raised when an artifact envelope carries a missing, malformed, or unsupported schema version."""


class SchemaMigrationError(ArtifactError):
    """Raised when a legacy artifact payload violates its declared schema/command semantics."""


@dataclass(frozen=True)
class SchemaMigrationRecord:
    """Audit record describing whether schema migration ran for a loaded artifact bundle."""

    source_schema_version: str
    target_schema_version: str
    command: str
    applied: bool
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_schema_version": self.source_schema_version,
            "target_schema_version": self.target_schema_version,
            "command": self.command,
            "applied": self.applied,
            "notes": list(self.notes),
        }


def parse_schema_version(raw: Any) -> tuple[int, int, int]:
    """Parse strict ``MAJOR.MINOR.PATCH`` schema strings or raise ``UnsupportedSchemaVersionError``."""
    if not isinstance(raw, str):
        raise UnsupportedSchemaVersionError(
            f"Missing or non-string schema_version in artifact payload: {raw!r}."
        )
    match = _SEMVER_RE.match(raw)
    if match is None:
        raise UnsupportedSchemaVersionError(
            f"Malformed schema_version {raw!r}; expected MAJOR.MINOR.PATCH."
        )
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _coerce_finite_float_or_none(value: Any, *, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SchemaMigrationError(f"{context}.max_drawdown must be numeric, got bool {value!r}.")
    try:
        num = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaMigrationError(
            f"{context}.max_drawdown must be numeric, got {value!r}."
        ) from exc
    if math.isnan(num):
        return None
    if not math.isfinite(num):
        raise SchemaMigrationError(f"{context}.max_drawdown must be finite, got {value!r}.")
    return num


def _migrate_max_drawdown_field(
    mapping: dict[str, Any],
    *,
    expect_non_positive: bool,
    command: str,
    context: str,
    notes: list[str],
) -> None:
    if "max_drawdown" not in mapping:
        return
    num = _coerce_finite_float_or_none(mapping["max_drawdown"], context=context)
    if num is None:
        return

    if expect_non_positive:
        if num > _MAX_DD_TOL:
            raise SchemaMigrationError(
                f"Legacy v0.1.0 compare {context} expects max_drawdown <= 0 (underwater return); "
                f"got unexpected positive value {num!r}. Refusing blind abs() conversion."
            )
        converted = 0.0 if abs(num) <= _ZERO_SNAP_TOL else max(0.0, -num)
        mapping["max_drawdown"] = converted
        notes.append(f"{context}.max_drawdown: converted v0.1.0 <= 0 underwater to >= 0 depth")
        return

    if num < -_MAX_DD_TOL:
        raise SchemaMigrationError(
            f"Legacy v0.1.0 {command} {context} expects max_drawdown >= 0 (depth); got {num!r}."
        )
    normalized = 0.0 if abs(num) <= _ZERO_SNAP_TOL else max(0.0, num)
    mapping["max_drawdown"] = normalized


def _migrate_from_0_1_0(
    migrated: dict[str, Any],
    *,
    command: str,
) -> tuple[str, ...]:
    notes: list[str] = []
    metrics = migrated.get("metrics")

    if command == "compare":
        if isinstance(metrics, dict):
            per_asset = metrics.get("per_asset")
            if isinstance(per_asset, dict):
                for leg_name, leg_metrics in per_asset.items():
                    if isinstance(leg_metrics, dict):
                        _migrate_max_drawdown_field(
                            leg_metrics,
                            expect_non_positive=True,
                            command=command,
                            context=f"metrics.per_asset[{leg_name!r}]",
                            notes=notes,
                        )
        benchmark = migrated.get("benchmark")
        if isinstance(benchmark, dict) and isinstance(benchmark.get("metrics"), dict):
            _migrate_max_drawdown_field(
                benchmark["metrics"],
                expect_non_positive=True,
                command=command,
                context="benchmark.metrics",
                notes=notes,
            )
        methodology = migrated.get("methodology")
        if isinstance(methodology, dict):
            methodology["max_drawdown_convention"] = "non_negative_depth"
        return tuple(notes)

    if command == "analyze":
        if isinstance(metrics, dict):
            _migrate_max_drawdown_field(
                metrics,
                expect_non_positive=False,
                command=command,
                context="metrics",
                notes=notes,
            )
        return tuple(notes)

    if command == "portfolio":
        if isinstance(metrics, dict):
            backtest = metrics.get("backtest")
            if isinstance(backtest, dict):
                for bt_key, entry in backtest.items():
                    if bt_key != "relative_to_equal_weight" and isinstance(entry, dict):
                        _migrate_max_drawdown_field(
                            entry,
                            expect_non_positive=False,
                            command=command,
                            context=f"metrics.backtest[{bt_key!r}]",
                            notes=notes,
                        )
            validation = metrics.get("validation")
            if isinstance(validation, dict):
                for sub_key in (
                    "in_sample_performance",
                    "out_of_sample_performance",
                    "performance",
                    "in_sample_hindsight_reference",
                ):
                    sub = validation.get(sub_key)
                    if isinstance(sub, dict):
                        _migrate_max_drawdown_field(
                            sub,
                            expect_non_positive=False,
                            command=command,
                            context=f"metrics.validation[{sub_key!r}]",
                            notes=notes,
                        )
            sensitivity = metrics.get("sensitivity")
            if isinstance(sensitivity, dict):
                for group_name, rows in sensitivity.items():
                    if isinstance(rows, list):
                        for idx, row in enumerate(rows):
                            if isinstance(row, dict) and row.get("status") == "ok":
                                _migrate_max_drawdown_field(
                                    row,
                                    expect_non_positive=False,
                                    command=command,
                                    context=f"metrics.sensitivity[{group_name!r}][{idx}]",
                                    notes=notes,
                                )
        benchmark = migrated.get("benchmark")
        if isinstance(benchmark, dict) and isinstance(benchmark.get("performance"), dict):
            _migrate_max_drawdown_field(
                benchmark["performance"],
                expect_non_positive=False,
                command=command,
                context="benchmark.performance",
                notes=notes,
            )
        methodology = migrated.get("methodology")
        if isinstance(methodology, dict):
            methodology["max_drawdown_convention"] = "non_negative_depth"
        return tuple(notes)

    notes.append("fetch envelope migrated 0.1.0 -> 0.2.0")
    return tuple(notes)


def migrate_result_payload(
    payload: Mapping[str, Any],
    *,
    target_version: str = SCHEMA_VERSION,
) -> tuple[dict[str, Any], SchemaMigrationRecord]:
    """Validate and migrate a parsed ``result.json`` dictionary up to ``SCHEMA_VERSION``.

    Migrations always dispatch explicitly on ``(schema_version, command)`` rather than guessing
    signs blindly with ``abs()``.
    """
    if target_version != SCHEMA_VERSION:
        raise UnsupportedSchemaVersionError(
            f"Unsupported target schema version {target_version!r}; expected {SCHEMA_VERSION!r}."
        )

    raw_source = payload.get("schema_version")
    source_tuple = parse_schema_version(raw_source)
    canonical_source = f"{source_tuple[0]}.{source_tuple[1]}.{source_tuple[2]}"
    if canonical_source not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(SUPPORTED_SCHEMA_VERSIONS)
        raise UnsupportedSchemaVersionError(
            f"Unsupported artifact schema_version {canonical_source!r}; supported versions: {supported}."
        )

    raw_cmd = payload.get("command")
    command = str(raw_cmd or "").strip().lower()
    if command not in _SUPPORTED_COMMANDS:
        raise ArtifactError(f"Unsupported command in result payload: {raw_cmd!r}.")

    migrated = copy.deepcopy(dict(payload))
    migrated["command"] = command

    if canonical_source == target_version:
        migrated["schema_version"] = target_version
        return migrated, SchemaMigrationRecord(
            source_schema_version=canonical_source,
            target_schema_version=target_version,
            command=command,
            applied=False,
            notes=(),
        )

    if canonical_source == "0.1.0" and target_version == "0.2.0":
        notes = _migrate_from_0_1_0(migrated, command=command)
        migrated["schema_version"] = target_version
        return migrated, SchemaMigrationRecord(
            source_schema_version=canonical_source,
            target_schema_version=target_version,
            command=command,
            applied=True,
            notes=notes,
        )

    raise UnsupportedSchemaVersionError(
        f"No migration path registered from {canonical_source!r} to {target_version!r}."
    )


__all__ = [
    "ArtifactError",
    "SchemaMigrationError",
    "SchemaMigrationRecord",
    "UnsupportedSchemaVersionError",
    "migrate_result_payload",
    "parse_schema_version",
]
