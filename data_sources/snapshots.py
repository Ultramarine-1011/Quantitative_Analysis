"""Immutable, portable quote objects and pinned research-input manifests."""

import hashlib
import io
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from contracts import SCHEMA_VERSION
from data_sources.cache import json_safe_attrs
from data_sources.errors import SnapshotError

# Only economic/provenance metadata goes into research inputs. Connection options and
# transient paths do not travel with a snapshot.
_FRAME_ATTRS = frozenset(
    "asset_type symbol canonical_symbol vendor_symbol adjust return_basis series_kind currency "
    "native_currency unit_caption volume_unit data_source fetched_at cache_hit data_quality "
    "source_data_quality fallback_warnings quality_warnings".split()
)


def _json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def request_key(query: Mapping[str, str]) -> str:
    return _sha(_json_bytes(query))


class SnapshotStore:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()

    def _path(self, directory: str, filename: str) -> Path:
        path = self.root / directory / filename
        if path.resolve() != path:
            raise SnapshotError(f"Unexpected symlink in snapshot store: {path}")
        return path

    def _publish(self, path: Path, content: bytes, *, immutable: bool) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                tmp = Path(handle.name)
                try:
                    handle.write(content)
                    handle.close()
                    if immutable:
                        try:
                            os.link(tmp, path)
                        except FileExistsError:
                            if path.read_bytes() != content:
                                raise SnapshotError(
                                    f"Conflicting immutable snapshot: {path.name}"
                                ) from None
                    else:
                        os.replace(tmp, path)
                finally:
                    tmp.unlink(missing_ok=True)
        except OSError as exc:
            raise SnapshotError(f"Failed publishing {path.name}: {exc}") from exc

    def _read(self, directory: str, digest: str, suffix: str) -> bytes:
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise SnapshotError("Invalid SHA-256 snapshot identifier")
        path = self._path(directory, digest + suffix)
        if not path.is_file():
            raise SnapshotError(f"Missing snapshot file {path.name}")
        content = path.read_bytes()
        if _sha(content) != digest:
            raise SnapshotError(f"Snapshot SHA-256 mismatch: {path.name}")
        return content

    def select(self, snapshot_id: str | None) -> tuple[str | None, dict]:
        if snapshot_id is None:
            latest = self._path("", "latest")
            if not latest.exists():
                return None, {}
            snapshot_id = latest.read_text(encoding="ascii").strip()
        payload = json.loads(self._read("manifests", snapshot_id, ".json"))
        if payload.get("version") != 1 or payload.get("schema_version") != SCHEMA_VERSION:
            raise SnapshotError("Unsupported market snapshot schema")
        return snapshot_id, payload["inputs"]

    def save_frame(self, query: dict[str, str], frame: pd.DataFrame) -> dict[str, Any]:
        attrs = json_safe_attrs({k: v for k, v in frame.attrs.items() if k in _FRAME_ATTRS})
        clean = frame.copy(deep=False)
        clean.attrs = {}
        content = clean.to_parquet(engine="pyarrow")
        digest = _sha(content)
        self._publish(self._path("objects", digest + ".parquet"), content, immutable=True)
        frequency = frame.index.freqstr if isinstance(frame.index, pd.DatetimeIndex) else None
        if frequency and pd.tseries.frequencies.to_offset(frequency) != frame.index.freq:
            frequency = None
        return dict(
            frequency=frequency,
            query=query,
            sha256=digest,
            attrs=attrs,
            rows=len(frame),
            actual_start=str(pd.Timestamp(frame.index.min()).date()),
            actual_end=str(pd.Timestamp(frame.index.max()).date()),
        )

    def load_frame(self, entry: Mapping) -> pd.DataFrame:
        try:
            frame = pd.read_parquet(io.BytesIO(self._read("objects", entry["sha256"], ".parquet")))
            if entry.get("frequency"):
                frame.index.freq = entry["frequency"]
            frame.attrs = dict(entry["attrs"])
            return frame
        except SnapshotError:
            raise
        except Exception as exc:
            raise SnapshotError(f"Cannot decode snapshot input: {exc}") from exc

    def publish_run(self, entries: Mapping) -> str:
        content = _json_bytes(dict(version=1, schema_version=SCHEMA_VERSION, inputs=entries))
        digest = _sha(content)
        self._publish(self._path("manifests", digest + ".json"), content, immutable=True)
        self._publish(self._path("", "latest"), (digest + "\n").encode("ascii"), immutable=False)
        return digest
