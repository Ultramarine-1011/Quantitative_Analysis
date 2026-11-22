"""Portable, integrity-checked artifact serialization and safe offline loading."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from artifacts.migration import (
    ArtifactError,
    SchemaMigrationRecord,
    migrate_result_payload,
)
from artifacts.transaction import ArtifactIOError, FileTransaction
from contracts import (
    SOFTWARE_VERSION,
    AnalysisResult,
    MultiAssetResult,
    QualityReport,
    SeriesMetadata,
    sync_metadata_to_frame,
    validate_frame_metadata_consistency,
    validate_result_invariants,
)

KNOWN_MULTI_ASSET_FRAMES: tuple[str, ...] = (
    "prices",
    "returns",
    "correlation",
    "covariance",
    "rolling_metrics",
    "weights",
    "frontier",
    "backtest",
    "weight_history",
)

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:(.*)$")


class ArtifactSecurityError(ArtifactError):
    """Raised when an artifact path attempts directory escape, traversal, or unsafe symlinks."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when artifact files are missing, corrupted, substituted, or fail SHA-256 checks."""


@dataclass(frozen=True)
class ArtifactFileDigest:
    """SHA-256 digest and byte size metadata for a single managed file inside an artifact directory."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, key: str = "") -> ArtifactFileDigest:
        path = raw.get("path")
        sha256 = raw.get("sha256")
        size_bytes = raw.get("size_bytes")
        if not isinstance(path, str) or not path.strip():
            raise ArtifactIntegrityError(f"Manifest entry {key!r} has invalid path {path!r}.")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ArtifactIntegrityError(
                f"Manifest entry {key!r} has invalid sha256 digest {sha256!r}."
            )
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ArtifactIntegrityError(
                f"Manifest entry {key!r} has invalid size_bytes {size_bytes!r}."
            )
        return cls(path=path, sha256=sha256.lower(), size_bytes=size_bytes)


@dataclass(frozen=True)
class ArtifactManifest:
    """Integrity manifest covering serialized files produced inside a portable artifact directory."""

    algorithm: str
    files: dict[str, ArtifactFileDigest]

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "files": {key: digest.to_dict() for key, digest in self.files.items()},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ArtifactManifest:
        algorithm = str(raw.get("algorithm") or "").strip().lower()
        if algorithm != "sha256":
            raise ArtifactIntegrityError(
                f"Unsupported artifact_manifest algorithm {raw.get('algorithm')!r}; expected 'sha256'."
            )
        files_raw = raw.get("files")
        if not isinstance(files_raw, Mapping):
            raise ArtifactIntegrityError("artifact_manifest['files'] must be a mapping.")
        parsed: dict[str, ArtifactFileDigest] = {}
        for key, item in files_raw.items():
            if not isinstance(item, Mapping):
                raise ArtifactIntegrityError(
                    f"artifact_manifest['files'][{key!r}] must be a mapping."
                )
            parsed[str(key)] = ArtifactFileDigest.from_dict(item, key=str(key))
        return cls(algorithm=algorithm, files=parsed)


def _is_valid_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        num = float(value)
    except TypeError, ValueError:
        return False
    return math.isfinite(num)


def compute_file_sha256(path: Path) -> tuple[str, int]:
    """Compute lowercase SHA-256 hex digest and total byte size for ``path``."""
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return digest.hexdigest(), total


def build_artifact_manifest(staged_files: Mapping[str, tuple[str, Path]]) -> dict[str, Any]:
    """Build a SHA-256 ``artifact_manifest`` dictionary from staged files."""
    digests: dict[str, ArtifactFileDigest] = {}
    for key in sorted(staged_files):
        rel_path, file_path = staged_files[key]
        sha256, size_bytes = compute_file_sha256(file_path)
        digests[key] = ArtifactFileDigest(path=rel_path, sha256=sha256, size_bytes=size_bytes)
    return ArtifactManifest(algorithm="sha256", files=digests).to_dict()


def _split_any_separators(raw_path: str) -> list[str]:
    return [seg for seg in re.split(r"[/\\]+", raw_path) if seg]


def _is_absolute_or_home_path(raw_path: str) -> bool:
    if raw_path.startswith(("/", "\\", "~")):
        return True
    if _WINDOWS_DRIVE_RE.match(raw_path) is not None:
        return True
    return Path(raw_path).is_absolute()


def _assert_no_symlink_escape(base_resolved: Path, parts: Sequence[str], *, raw_path: str) -> Path:
    current = base_resolved
    for part in parts:
        current = current / part
        if current.is_symlink():
            target = current.resolve()
            if not target.is_relative_to(base_resolved) or target == base_resolved:
                raise ArtifactSecurityError(
                    f"Symlink escape detected while resolving artifact path {raw_path!r}."
                )
    resolved = current.resolve()
    if not resolved.is_relative_to(base_resolved) or resolved == base_resolved:
        raise ArtifactSecurityError(
            f"Resolved artifact path escapes output directory: {raw_path!r}."
        )
    return resolved


def resolve_safe_artifact_path(
    base_dir: Path,
    raw_path: str,
    *,
    allow_legacy_absolute: bool = False,
) -> Path:
    """Resolve an artifact path strictly within ``base_dir``.

    Rejects NUL bytes, ``..`` traversal segments, and symlink escapes unconditionally. Absolute or
    home-prefixed paths are rejected under schema v0.2.0 and mapped strictly to local files inside
    ``base_dir`` (never reading external filesystem targets) only when ``allow_legacy_absolute=True``.
    """
    base_resolved = Path(base_dir).expanduser().resolve()
    if not isinstance(raw_path, str) or "\x00" in raw_path:
        raise ArtifactSecurityError(f"Unsafe or non-string artifact path: {raw_path!r}.")
    cleaned = raw_path.strip()
    if not cleaned:
        raise ArtifactSecurityError("Artifact path cannot be empty.")

    raw_parts = _split_any_separators(cleaned)
    drive_match = _WINDOWS_DRIVE_RE.match(cleaned)
    if drive_match is not None:
        rest_parts = _split_any_separators(drive_match.group(1))
    else:
        rest_parts = raw_parts

    if ".." in raw_parts or ".." in rest_parts:
        raise ArtifactSecurityError(
            f"Parent directory traversal ('..') is forbidden in artifact path: {raw_path!r}."
        )

    if _is_absolute_or_home_path(cleaned):
        if not allow_legacy_absolute:
            raise ArtifactSecurityError(
                f"Absolute artifact path is forbidden in portable v0.2.0 bundle: {raw_path!r}."
            )
        segs = [p for p in rest_parts if p != "."]
        if not segs:
            raise ArtifactSecurityError(f"Malformed legacy absolute artifact path: {raw_path!r}.")
        filename = segs[-1]
        if filename in {"", ".", ".."}:
            raise ArtifactSecurityError(f"Invalid filename in legacy artifact path: {raw_path!r}.")
        if len(segs) >= 2 and segs[-2] == "charts":
            rel_parts = ["charts", filename]
        else:
            rel_parts = [filename]
        return _assert_no_symlink_escape(base_resolved, rel_parts, raw_path=raw_path)

    rel_segs = [p for p in raw_parts if p != "."]
    if not rel_segs:
        raise ArtifactSecurityError(f"Invalid empty relative artifact path: {raw_path!r}.")
    return _assert_no_symlink_escape(base_resolved, rel_segs, raw_path=raw_path)


def verify_artifact_manifest(
    out_dir: Path,
    artifacts: Mapping[str, str],
    manifest_raw: Mapping[str, Any],
) -> None:
    """Verify artifact path consistency, directory containment, file presence, and SHA-256 digests."""
    if not isinstance(manifest_raw, Mapping):
        raise ArtifactIntegrityError("artifact_manifest must be a mapping when present.")
    manifest = ArtifactManifest.from_dict(manifest_raw)

    for key, val in artifacts.items():
        if key in {"output_dir", "result_json"}:
            continue
        if key not in manifest.files:
            raise ArtifactIntegrityError(
                f"Artifact {key!r} ({val!r}) is missing from artifact_manifest['files']."
            )

    for key, digest in manifest.files.items():
        if key in {"output_dir", "result_json"}:
            raise ArtifactIntegrityError(f"Manifest cannot contain envelope metadata key {key!r}.")
        if key in artifacts and artifacts[key] != digest.path:
            raise ArtifactIntegrityError(
                f"Path substitution detected for {key!r}: artifacts={artifacts[key]!r} vs "
                f"manifest={digest.path!r}."
            )
        target = resolve_safe_artifact_path(out_dir, digest.path, allow_legacy_absolute=False)
        if not target.is_file():
            raise ArtifactIntegrityError(
                f"Declared manifest artifact file is missing on disk: {digest.path!r}."
            )
        actual_sha, actual_size = compute_file_sha256(target)
        if actual_size != digest.size_bytes or actual_sha != digest.sha256:
            raise ArtifactIntegrityError(
                f"SHA-256 verification failed for {key!r} ({digest.path}): "
                f"expected ({digest.sha256}, {digest.size_bytes} bytes), "
                f"got ({actual_sha}, {actual_size} bytes)."
            )


def prepare_export_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a clean tabular copy stripped of transient attrs with normalized Date index name."""
    out = df.copy() if df is not None else pd.DataFrame()
    out.attrs = {}
    if isinstance(out.index, pd.DatetimeIndex) and not out.index.name:
        out.index.name = "Date"
    return out


def _parquet_writer(df: pd.DataFrame) -> Callable[[Path], None]:
    write_idx = not isinstance(df.index, pd.RangeIndex)
    return lambda tmp: df.to_parquet(tmp, engine="pyarrow", index=write_idx)


def _csv_writer(df: pd.DataFrame) -> Callable[[Path], None]:
    write_idx = not isinstance(df.index, pd.RangeIndex)
    return lambda tmp: df.to_csv(tmp, index=write_idx, encoding="utf-8-sig")


_MANAGED_OUTPUT_FILENAMES: frozenset[str] = frozenset(
    {
        "result.json",
        "data.parquet",
        "data.csv",
        "report.md",
        "report.html",
        *(f"{name}.parquet" for name in KNOWN_MULTI_ASSET_FRAMES),
        *(f"{name}.csv" for name in KNOWN_MULTI_ASSET_FRAMES),
    }
)

_MANAGED_CHART_TYPES: frozenset[str] = frozenset(
    {
        "price",
        "normalized",
        "drawdown",
        "rolling",
        "correlation",
        "backtest",
        "weights",
        "frontier",
        "weight_history",
    }
)

_MANAGED_CHART_SUFFIXES: frozenset[str] = frozenset({".html", ".png", ".svg"})


def _resolve_non_symlink_managed_file(resolved_dir: Path, candidate: Path) -> Path | None:
    """Return ``candidate`` only when it is a regular non-symlink file inside ``resolved_dir``."""
    if candidate.is_symlink() or not candidate.is_file():
        return None
    resolved_candidate = candidate.resolve()
    if (
        resolved_candidate != candidate
        or resolved_candidate == resolved_dir
        or not resolved_candidate.is_relative_to(resolved_dir)
    ):
        return None
    return resolved_candidate


def collect_stale_managed_artifacts(
    out_dir: Path,
    *,
    new_target_resolved: set[Path],
) -> list[Path]:
    """Identify previously generated quant artifacts in ``out_dir`` not produced by the current run."""
    if not out_dir.exists() or not out_dir.is_dir():
        return []
    resolved_dir = out_dir.resolve()
    charts_dir = resolved_dir / "charts"
    candidates: set[Path] = set()

    for name in _MANAGED_OUTPUT_FILENAMES:
        resolved_item = _resolve_non_symlink_managed_file(resolved_dir, resolved_dir / name)
        if resolved_item is not None:
            candidates.add(resolved_item)

    if not charts_dir.is_symlink() and charts_dir.is_dir() and charts_dir.resolve() == charts_dir:
        for p in charts_dir.iterdir():
            if p.stem in _MANAGED_CHART_TYPES and p.suffix.lower() in _MANAGED_CHART_SUFFIXES:
                resolved_chart = _resolve_non_symlink_managed_file(resolved_dir, p)
                if resolved_chart is not None and resolved_chart.parent == charts_dir:
                    candidates.add(resolved_chart)

    return sorted(
        p
        for p in candidates
        if p not in new_target_resolved
        and not p.is_symlink()
        and p != resolved_dir
        and p.is_relative_to(resolved_dir)
    )


def write_result_outputs(
    result: AnalysisResult | MultiAssetResult,
    *,
    out_dir: Path | None,
    write_csv: bool = False,
    export_csv: Path | None = None,
    include_data: bool = False,
    compact: bool = False,
    attachments: Mapping[str, tuple[str, bytes]] | None = None,
    clean_stale: bool = True,
) -> AnalysisResult | MultiAssetResult:
    """Commit data, optional presentation files and their strict-JSON manifest together."""
    validate_result_invariants(result)
    root = out_dir.resolve() if out_dir is not None else None
    paths: dict[str, str] = {}
    manifest_inputs: dict[str, tuple[str, Path]] = {}

    with FileTransaction() as txn:

        def stage(key: str, name: str, writer: Callable[[Path], object]) -> None:
            assert root is not None
            target = resolve_safe_artifact_path(root, name, allow_legacy_absolute=False)
            if target != root / name:
                raise ArtifactIOError(f"Refusing redirected output: {name}")
            paths[key] = name
            manifest_inputs[key] = (name, txn.stage(target, writer))

        if root is not None:
            paths.update(output_dir=".", result_json="result.json")
            frames = {"data": result.data_frame} if result.data_frame is not None else {}
            if isinstance(result, MultiAssetResult):
                for name, frame in result.frames.items():
                    if not re.fullmatch(r"[\w-]+", name) or name == "data":
                        raise ArtifactIOError(f"Invalid output frame name: {name!r}")
                    frames[name] = frame
            for name, raw in frames.items():
                frame = prepare_export_frame(raw)
                stage(f"{name}_parquet", f"{name}.parquet", _parquet_writer(frame))
                if write_csv:
                    stage(f"{name}_csv", f"{name}.csv", _csv_writer(frame))
            for key, (name, payload) in (attachments or {}).items():
                stage(key, name, lambda tmp, data=payload: tmp.write_bytes(data))

        if export_csv is not None:
            target = export_csv.expanduser().absolute()
            if root and target.is_relative_to(root):
                rel = target.relative_to(root).as_posix()
                if rel == "data.csv" and "data_csv" in manifest_inputs:
                    paths["export_csv"] = rel
                    manifest_inputs["export_csv"] = manifest_inputs["data_csv"]
                elif rel == "result.json" or target in txn.staged:
                    raise ValueError(f"CSV export conflicts with bundle output: {rel}")
                else:
                    stage("export_csv", rel, _csv_writer(prepare_export_frame(result.data_frame)))
            else:
                txn.stage(target, _csv_writer(prepare_export_frame(result.data_frame)))
                if root is None:
                    paths["export_csv"] = str(target)

        manifest = build_artifact_manifest(manifest_inputs) if root is not None else None
        final = result.with_artifacts(paths, artifact_manifest=manifest)
        if root is not None:

            def write_json(tmp: Path) -> None:
                payload = json.dumps(
                    final.to_dict(include_data=include_data),
                    ensure_ascii=False,
                    indent=None if compact else 2,
                    allow_nan=False,
                )
                tmp.write_text(payload + "\n", encoding="utf-8")

            txn.stage(root / "result.json", write_json)
        stale = (
            collect_stale_managed_artifacts(root, new_target_resolved=set(txn.staged))
            if root and clean_stale
            else ()
        )
        txn.commit(stale)
    if root and clean_stale:
        charts = root / "charts"
        if not charts.is_symlink():
            with suppress(OSError):
                charts.rmdir()
    return final


def save_result_bundle(
    result: AnalysisResult | MultiAssetResult,
    output_dir: str | Path,
    *,
    write_csv: bool = False,
    include_data: bool = False,
    compact: bool = False,
) -> AnalysisResult | MultiAssetResult:
    """Serialize a portable bundle, preserving unrelated existing outputs."""
    return write_result_outputs(
        result,
        out_dir=Path(output_dir).expanduser(),
        write_csv=write_csv,
        include_data=include_data,
        compact=compact,
        clean_stale=False,
    )


def _read_frame_from_disk(
    parquet_path: Path | None,
    csv_path: Path | None = None,
) -> pd.DataFrame | None:
    if parquet_path is not None and parquet_path.is_file():
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            raise ArtifactIntegrityError(
                f"Failed reading corrupted Parquet artifact {parquet_path.name}: {exc}"
            ) from exc
    elif csv_path is not None and csv_path.is_file():
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception as exc:
            raise ArtifactIntegrityError(
                f"Failed reading corrupted CSV artifact {csv_path.name}: {exc}"
            ) from exc
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        elif len(df.columns) > 0 and str(df.columns[0]).lower() in {"date", "asset", "index"}:
            idx_col = df.columns[0]
            if str(idx_col).lower() == "date":
                df[idx_col] = pd.to_datetime(df[idx_col])
            df = df.set_index(idx_col)
    else:
        return None

    if "Date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        except Exception:
            pass
    return df


def _normalize_and_validate_artifacts_map(
    out_dir: Path,
    raw_artifacts: Any,
    *,
    allow_legacy_absolute: bool,
    verify_integrity: bool,
) -> tuple[dict[str, str], dict[str, Path]]:
    if raw_artifacts is None:
        return {}, {}
    if not isinstance(raw_artifacts, Mapping):
        raise ArtifactIntegrityError("Artifact envelope field 'artifacts' must be a mapping.")

    normalized: dict[str, str] = {}
    resolved_files: dict[str, Path] = {}

    for raw_key, raw_val in raw_artifacts.items():
        key = str(raw_key)
        if not isinstance(raw_val, str) or "\x00" in raw_val:
            raise ArtifactSecurityError(f"Unsafe artifact value for key {key!r}: {raw_val!r}.")
        val = raw_val.strip()

        if key == "output_dir":
            if ".." in _split_any_separators(val):
                raise ArtifactSecurityError(
                    f"Parent directory traversal ('..') is forbidden in artifacts['output_dir']: {val!r}."
                )
            if not allow_legacy_absolute and val != ".":
                raise ArtifactSecurityError(
                    f"Portable v0.2.0 bundle requires artifacts['output_dir'] == '.'; got {val!r}."
                )
            normalized[key] = "."
            continue

        resolved_path = resolve_safe_artifact_path(
            out_dir, val, allow_legacy_absolute=allow_legacy_absolute
        )
        rel_posix = resolved_path.relative_to(out_dir).as_posix()
        if key == "result_json" and rel_posix != "result.json":
            raise ArtifactSecurityError(
                f"artifacts['result_json'] must refer to 'result.json'; got {rel_posix!r}."
            )
        if verify_integrity and key != "result_json" and not resolved_path.is_file():
            raise ArtifactIntegrityError(
                f"Declared artifact file for {key!r} is missing on disk: {rel_posix!r}."
            )
        normalized[key] = rel_posix
        resolved_files[key] = resolved_path

    return normalized, resolved_files


def _load_declared_or_fallback_frame(
    out_dir: Path,
    resolved_files: Mapping[str, Path],
    *,
    parquet_key: str,
    csv_key: str,
    default_parquet: str,
    default_csv: str,
    allow_undeclared_fallback: bool = False,
) -> pd.DataFrame | None:
    pq_path = resolved_files.get(parquet_key)
    csv_path = resolved_files.get(csv_key)
    if pq_path is None and csv_path is None:
        if not allow_undeclared_fallback:
            return None
        fallback_pq = resolve_safe_artifact_path(out_dir, default_parquet)
        fallback_csv = resolve_safe_artifact_path(out_dir, default_csv)
        pq_path = fallback_pq if fallback_pq.is_file() else None
        csv_path = fallback_csv if fallback_csv.is_file() else None
    elif pq_path is not None and not pq_path.is_file():
        if csv_path is None or not csv_path.is_file():
            raise ArtifactIntegrityError(
                f"Declared Parquet artifact missing on disk: {pq_path.name}."
            )
    elif csv_path is not None and not csv_path.is_file():
        raise ArtifactIntegrityError(f"Declared CSV artifact missing on disk: {csv_path.name}.")

    return _read_frame_from_disk(pq_path, csv_path)


def load_result_with_migration_info(
    output_dir: str | Path,
    *,
    verify_integrity: bool = True,
) -> tuple[AnalysisResult | MultiAssetResult, SchemaMigrationRecord]:
    """Load, migrate, security-check, and integrity-verify an artifact directory offline."""
    out_dir = Path(output_dir).expanduser().resolve()
    result_path = resolve_safe_artifact_path(out_dir, "result.json", allow_legacy_absolute=False)
    if not result_path.is_file():
        raise FileNotFoundError(f"Result envelope not found: {result_path}")

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Failed parsing result.json in {out_dir}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactIntegrityError(f"Invalid result.json object in {result_path}")

    migrated, migration_record = migrate_result_payload(payload)
    is_legacy_v01 = migration_record.source_schema_version == "0.1.0"
    allow_legacy_abs = is_legacy_v01
    allow_undeclared_fallback = is_legacy_v01

    normalized_artifacts, resolved_files = _normalize_and_validate_artifacts_map(
        out_dir,
        migrated.get("artifacts"),
        allow_legacy_absolute=allow_legacy_abs,
        verify_integrity=verify_integrity,
    )

    manifest_raw = migrated.get("artifact_manifest")
    if manifest_raw is not None and not isinstance(manifest_raw, Mapping):
        raise ArtifactIntegrityError("artifact_manifest must be a mapping when present.")
    if verify_integrity:
        if manifest_raw is None:
            if not is_legacy_v01:
                raise ArtifactIntegrityError(
                    "Schema v0.2.0 artifact bundle requires 'artifact_manifest' when verify_integrity=True."
                )
        else:
            verify_artifact_manifest(out_dir, normalized_artifacts, manifest_raw)

    command = str(migrated.get("command", "")).strip().lower()
    if command in {"analyze", "fetch"}:
        asset_raw = migrated.get("asset")
        quality_raw = migrated.get("quality")
        if not isinstance(asset_raw, Mapping) or not isinstance(quality_raw, Mapping):
            raise ArtifactIntegrityError(f"Missing 'asset' or 'quality' section in {result_path}")

        asset_meta = SeriesMetadata.from_dict(asset_raw)
        quality = QualityReport.from_dict(quality_raw)
        df = _load_declared_or_fallback_frame(
            out_dir,
            resolved_files,
            parquet_key="data_parquet",
            csv_key="data_csv",
            default_parquet="data.parquet",
            default_csv="data.csv",
            allow_undeclared_fallback=allow_undeclared_fallback,
        )
        if df is not None:
            validate_frame_metadata_consistency(df, asset_meta)
            params = migrated.get("parameters") or {}
            metrics = migrated.get("metrics") or {}
            sync_metadata_to_frame(df, asset_meta, quality=quality)
            if isinstance(metrics, Mapping) and _is_valid_number(metrics.get("max_drawdown")):
                df.attrs["max_drawdown"] = float(metrics["max_drawdown"])
            vol_cols = [
                c for c in df.columns if str(c).startswith("volatility_") and str(c)[11:].isdigit()
            ]
            inferred_vol_win = int(str(vol_cols[0])[11:]) if vol_cols else 20
            sma_short = int(params.get("sma_short", params.get("sma_window", 20)))
            sma_long = int(params.get("sma_long", 60))
            bb_win = int(params.get("bb_window", 20))
            df.attrs["feature_summary"] = {
                "sma_short_window": sma_short,
                "sma_long_window": sma_long,
                "bollinger_window": bb_win,
                "volatility_window": int(params.get("volatility_window", inferred_vol_win)),
            }

        res: AnalysisResult | MultiAssetResult = AnalysisResult(
            schema_version=str(migrated["schema_version"]),
            status=str(migrated.get("status", "ok")),
            asset=asset_meta,
            quality=quality,
            parameters=dict(migrated.get("parameters") or {}),
            metrics=dict(migrated.get("metrics") or {}),
            metric_reasons={
                str(k): str(v) for k, v in (migrated.get("metric_reasons") or {}).items()
            },
            latest_snapshot=dict(migrated.get("latest_snapshot") or {}),
            data_frame=df,
            warnings=tuple(str(w) for w in (migrated.get("warnings") or ())),
            disclaimers=tuple(str(d) for d in (migrated.get("disclaimers") or ())),
            artifacts=normalized_artifacts,
            command=command,
            software_version=str(migrated.get("software_version") or SOFTWARE_VERSION),
            metric_units={str(k): str(v) for k, v in (migrated.get("metric_units") or {}).items()},
            data_summary=dict(migrated.get("data_summary") or {}),
            artifact_manifest=dict(manifest_raw) if isinstance(manifest_raw, Mapping) else None,
        )
        return res, migration_record

    if command in {"compare", "portfolio"}:
        assets_raw = migrated.get("assets") or {}
        quality_raw = migrated.get("quality") or {}
        assets = {
            str(k): SeriesMetadata.from_dict(v)
            for k, v in assets_raw.items()
            if isinstance(v, Mapping)
        }
        quality_map = {
            str(k): QualityReport.from_dict(v)
            for k, v in quality_raw.items()
            if isinstance(v, Mapping)
        }
        primary_df = _load_declared_or_fallback_frame(
            out_dir,
            resolved_files,
            parquet_key="data_parquet",
            csv_key="data_csv",
            default_parquet="data.parquet",
            default_csv="data.csv",
            allow_undeclared_fallback=allow_undeclared_fallback,
        )

        frames: dict[str, pd.DataFrame] = {}
        for name in KNOWN_MULTI_ASSET_FRAMES:
            loaded = _load_declared_or_fallback_frame(
                out_dir,
                resolved_files,
                parquet_key=f"{name}_parquet",
                csv_key=f"{name}_csv",
                default_parquet=f"{name}.parquet",
                default_csv=f"{name}.csv",
                allow_undeclared_fallback=allow_undeclared_fallback,
            )
            if loaded is not None:
                frames[name] = loaded

        mres = MultiAssetResult(
            schema_version=str(migrated["schema_version"]),
            command=command,
            status=str(migrated.get("status", "ok")),
            asset_order=tuple(str(x) for x in (migrated.get("asset_order") or ())),
            assets=assets,
            quality=quality_map,
            data_summary=dict(migrated.get("data_summary") or {}),
            parameters=dict(migrated.get("parameters") or {}),
            methodology=dict(migrated.get("methodology") or {}),
            metrics=dict(migrated.get("metrics") or {}),
            metric_units={str(k): str(v) for k, v in (migrated.get("metric_units") or {}).items()},
            metric_reasons=dict(migrated.get("metric_reasons") or {}),
            benchmark=migrated.get("benchmark"),
            warnings=tuple(str(w) for w in (migrated.get("warnings") or ())),
            disclaimers=tuple(str(d) for d in (migrated.get("disclaimers") or ())),
            artifacts=normalized_artifacts,
            frames=frames,
            data_frame=primary_df,
            software_version=str(migrated.get("software_version") or SOFTWARE_VERSION),
            artifact_manifest=dict(manifest_raw) if isinstance(manifest_raw, Mapping) else None,
        )
        return mres, migration_record

    raise ArtifactError(f"Unsupported command in {result_path}: {command!r}")


def load_result_from_artifacts(
    output_dir: str | Path,
    *,
    verify_integrity: bool = True,
) -> AnalysisResult | MultiAssetResult:
    """Reconstruct ``AnalysisResult`` or ``MultiAssetResult`` safely from ``output_dir``."""
    result, _ = load_result_with_migration_info(output_dir, verify_integrity=verify_integrity)
    return result


__all__ = [
    "KNOWN_MULTI_ASSET_FRAMES",
    "ArtifactError",
    "ArtifactFileDigest",
    "ArtifactIntegrityError",
    "ArtifactManifest",
    "ArtifactSecurityError",
    "build_artifact_manifest",
    "compute_file_sha256",
    "load_result_from_artifacts",
    "load_result_with_migration_info",
    "prepare_export_frame",
    "resolve_safe_artifact_path",
    "save_result_bundle",
    "verify_artifact_manifest",
]
