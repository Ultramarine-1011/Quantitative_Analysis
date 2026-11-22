"""Shared persistent Parquet + JSON sidecar cache for Web UI and CLI."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asset_catalog import SUPPORTED_ADJUSTMENTS, normalize_symbol_alias, resolve_adjustment
from ohlcv import coerce_analysis_date, compute_ohlcv_quality

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX platforms
    msvcrt = None  # type: ignore[assignment]

DEFAULT_CACHE_TTL_SECONDS = 3600.0
CACHE_SCHEMA_VERSION = 1
ENV_CACHE_DIR = "QUANT_CACHE_DIR"
ENV_DISABLE_CACHE = "QUANT_DISABLE_CACHE"
ENV_CACHE_TTL = "QUANT_CACHE_TTL"
_TRANSIENT_OR_SENSITIVE_ATTRS = frozenset({"cache_hit", "cache_path", "proxy_url"})
_KEY_THREAD_LOCKS_GUARD = threading.Lock()
_KEY_THREAD_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_FILE_LOCK_TIMEOUT_SECONDS = 30.0


def is_cache_enabled() -> bool:
    """Return False when disk caching is disabled via environment variable."""
    raw = os.environ.get(ENV_DISABLE_CACHE, "").strip().lower()
    return raw not in {"1", "true", "yes", "on"}


def get_cache_dir() -> Path:
    """Resolve the shared disk cache directory across Web and CLI entrypoints."""
    override = os.environ.get(ENV_CACHE_DIR, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "quantitative_analysis" / "ohlcv"


def resolve_cache_ttl(ttl_seconds: float | None = None) -> float:
    """Resolve effective TTL seconds (argument overrides ``QUANT_CACHE_TTL``)."""
    if ttl_seconds is not None:
        return float(ttl_seconds)
    env_ttl = os.environ.get(ENV_CACHE_TTL, "").strip()
    if env_ttl:
        try:
            return float(env_ttl)
        except ValueError:
            return DEFAULT_CACHE_TTL_SECONDS
    return DEFAULT_CACHE_TTL_SECONDS


def canonical_cache_symbol(asset_type: str, symbol: object) -> str:
    """Normalize symbol identifiers so Web and CLI generate identical cache keys."""
    return normalize_symbol_alias(asset_type, symbol)


@contextmanager
def _msvcrt_file_lock(
    fh: Any, *, timeout_seconds: float = _FILE_LOCK_TIMEOUT_SECONDS
) -> Iterator[None]:
    """Cross-process byte-range lock on Windows via ``msvcrt.locking``."""
    fh.seek(0, os.SEEK_END)
    if fh.tell() == 0:
        fh.write(b"\0")
        fh.flush()
    fh.seek(0)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[union-attr]
            break
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EDEADLK, errno.EAGAIN}:
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for cache entry Windows file lock.") from exc
            time.sleep(0.005)
    try:
        yield
    finally:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[union-attr]


@contextmanager
def _cache_entry_lock(cache_dir: Path, key: str) -> Iterator[None]:
    """Serialize read/write transitions on a single cache key across threads and processes."""
    lock_scope = (os.path.abspath(os.fspath(cache_dir)), key)
    with _KEY_THREAD_LOCKS_GUARD:
        tlock = _KEY_THREAD_LOCKS.setdefault(lock_scope, threading.Lock())
    with tlock:
        if fcntl is None and msvcrt is None:
            yield
            return
        lock_dir = cache_dir / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f"{key}.lock"
        with lock_path.open("a+b") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            else:
                with _msvcrt_file_lock(fh):
                    yield


def canonical_cache_adjust(asset_type: str, adjust: str | None) -> str:
    """Normalize adjustment policy so omitted defaults map to the same entry."""
    asset_key = str(asset_type).strip().lower()
    if asset_key in SUPPORTED_ADJUSTMENTS:
        return resolve_adjustment(asset_key, adjust)
    if adjust:
        return resolve_adjustment(asset_key, adjust)
    return ""


def build_cache_key(
    asset_type: str, symbol: object, adjust: str | None = None, *, namespace: str = ""
) -> str:
    """Build a deterministic SHA-256 hex cache key for ``(asset_type, symbol, adjust)``."""
    asset_key = str(asset_type).strip().lower()
    sym_key = canonical_cache_symbol(asset_key, symbol)
    adj_key = canonical_cache_adjust(asset_key, adjust)
    payload = json.dumps(
        ([namespace] if namespace else []) + [asset_key, sym_key, adj_key],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _entry_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    return cache_dir / f"{key}.parquet", cache_dir / f"{key}.meta.json"


def json_safe_attrs(value: Any) -> Any:
    """Convert DataFrame ``attrs`` recursively into strict JSON-serializable objects."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        fval = float(value)
        return fval if math.isfinite(fval) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [json_safe_attrs(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): json_safe_attrs(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe_attrs(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except TypeError, ValueError:
        pass
    return str(value)


def _write_staged_temp_file(
    parent: Path,
    writer: Any,
    *,
    prefix: str,
    suffix: str,
) -> Path:
    """Create and populate a temporary file in ``parent`` for subsequent ``os.replace``."""
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        writer(tmp_path)
        return tmp_path
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def _atomic_write_bytes_or_parquet(
    target_path: Path,
    writer: Any,
    *,
    prefix: str,
    suffix: str,
) -> None:
    """Write via temp file in ``target_path.parent`` then ``os.replace`` atomically."""
    tmp_path = _write_staged_temp_file(
        target_path.parent,
        writer,
        prefix=prefix,
        suffix=suffix,
    )
    try:
        os.replace(tmp_path, target_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def _backup_existing_file(
    source_path: Path,
    *,
    prefix: str,
    suffix: str,
) -> Path | None:
    """Copy an existing cache file to a temp backup in the same directory before overwrite."""
    if not source_path.exists():
        return None
    return _write_staged_temp_file(
        source_path.parent,
        lambda tmp: shutil.copy2(source_path, tmp),
        prefix=prefix,
        suffix=suffix,
    )


def _restore_or_unlink_file(target_path: Path, backup_path: Path | None) -> Path | None:
    """Restore ``target_path`` from ``backup_path`` or unlink partial newly written output."""
    try:
        if backup_path is not None and backup_path.exists():
            os.replace(backup_path, target_path)
            return None
        if target_path.exists():
            target_path.unlink()
    except OSError:
        pass
    return backup_path


def load_cached_ohlcv(
    asset_type: str,
    symbol: object,
    start_date: object,
    end_date: object,
    *,
    adjust: str | None = None,
    namespace: str = "",
    ttl_seconds: float | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame | None:
    """Return a cached OHLCV slice when present, unexpired, and covering ``[start, end]``."""
    if force_refresh or not is_cache_enabled():
        return None
    effective_ttl = resolve_cache_ttl(ttl_seconds)
    if not math.isfinite(effective_ttl) or effective_ttl <= 0.0:
        return None

    req_start = coerce_analysis_date(start_date, "start_date")
    req_end = coerce_analysis_date(end_date, "end_date")
    if req_start > req_end:
        return None

    cache_dir = get_cache_dir()
    if not cache_dir.is_dir():
        return None
    key = build_cache_key(asset_type, symbol, adjust=adjust, namespace=namespace)
    parquet_path, meta_path = _entry_paths(cache_dir, key)
    if not (meta_path.is_file() and parquet_path.is_file()):
        return None

    with _cache_entry_lock(cache_dir, key):
        if not (meta_path.is_file() and parquet_path.is_file()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict) or int(meta.get("version", 0)) != CACHE_SCHEMA_VERSION:
                return None
            cached_at = float(meta["cached_at"])
            now = time.time()
            age = now - cached_at
            if not math.isfinite(age) or age < -300.0 or age > effective_ttl:
                return None
            cov_start = coerce_analysis_date(meta["covered_start"], "covered_start")
            cov_end = coerce_analysis_date(meta["covered_end"], "covered_end")
        except Exception:
            return None

        if cov_start > req_start or cov_end < req_end:
            return None

        try:
            df = pd.read_parquet(parquet_path, engine="pyarrow")
        except Exception:
            return None

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="raise")
        except Exception:
            return None
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.as_unit("ns")
    df = df.sort_index()
    df.index.name = "Date"

    sliced = df.loc[(df.index >= req_start) & (df.index <= req_end)].copy()
    if sliced.empty:
        return None
    attrs = meta.get("attrs")
    if isinstance(attrs, dict):
        sliced.attrs.update(attrs)
    raw_dq = attrs.get("data_quality") if isinstance(attrs, dict) else None
    prev_dq = dict(raw_dq) if isinstance(raw_dq, dict) else {}
    missing_policy = str(prev_dq.get("missing_policy", "drop") or "drop")
    is_full_slice = len(sliced) == len(df)
    if is_full_slice:
        dropped_rows = int(prev_dq.get("dropped_invalid_rows", prev_dq.get("dropped_rows", 0)) or 0)
        filled_rows = int(prev_dq.get("filled_price_rows", prev_dq.get("ffilled_cells", 0)) or 0)
        missing_vol_rows = int(prev_dq.get("missing_volume_rows", 0) or 0)
    else:
        dropped_rows = 0
        filled_rows = 0
        missing_vol_rows = 0
        if prev_dq:
            sliced.attrs["cache_source_data_quality"] = prev_dq
    slice_dq = compute_ohlcv_quality(
        sliced,
        missing_policy=missing_policy,
        dropped_invalid_rows=dropped_rows,
        filled_price_rows=filled_rows,
        missing_volume_rows=missing_vol_rows,
    )
    slice_dq["quality_scope"] = "full" if is_full_slice else "slice"
    sliced.attrs["data_quality"] = slice_dq
    sliced.attrs.setdefault("canonical_symbol", canonical_cache_symbol(asset_type, symbol))
    sliced.attrs.setdefault("fetched_at", datetime.fromtimestamp(cached_at, tz=UTC).isoformat())
    sliced.attrs["cache_hit"] = True
    sliced.attrs["cached_at"] = cached_at
    sliced.attrs["cache_key"] = key
    sliced.attrs["cache_path"] = str(parquet_path)
    return sliced


def save_cached_ohlcv(
    df: pd.DataFrame,
    asset_type: str,
    symbol: object,
    start_date: object,
    end_date: object,
    *,
    adjust: str | None = None,
    namespace: str = "",
) -> Path | None:
    """Persist a non-empty OHLCV DataFrame atomically as Parquet + JSON sidecar."""
    if not is_cache_enabled() or df is None or df.empty:
        return None

    req_start = coerce_analysis_date(start_date, "start_date")
    req_end = coerce_analysis_date(end_date, "end_date")
    cache_dir = get_cache_dir()
    key = build_cache_key(asset_type, symbol, adjust=adjust, namespace=namespace)
    parquet_path, meta_path = _entry_paths(cache_dir, key)

    to_store = df.copy()
    to_store.attrs = {}
    if not isinstance(to_store.index, pd.DatetimeIndex):
        to_store.index = pd.to_datetime(to_store.index, errors="raise")
    if to_store.index.tz is not None:
        to_store.index = to_store.index.tz_localize(None)
    to_store.index = to_store.index.as_unit("ns")
    to_store = to_store.sort_index()
    to_store.index.name = "Date"

    # Cover at least the requested query boundaries and observed timestamps.
    cov_start = req_start
    cov_end = req_end
    if not to_store.empty:
        first_ts = pd.Timestamp(to_store.index[0]).normalize()
        last_ts = pd.Timestamp(to_store.index[-1]).normalize()
        if first_ts < cov_start:
            cov_start = first_ts
        if last_ts > cov_end:
            cov_end = last_ts

    cached_at_ts = float(time.time())
    clean_attrs = json_safe_attrs(
        {k: v for k, v in dict(df.attrs).items() if k not in _TRANSIENT_OR_SENSITIVE_ATTRS}
    )
    if not clean_attrs.get("fetched_at"):
        clean_attrs["fetched_at"] = datetime.fromtimestamp(cached_at_ts, tz=UTC).isoformat()
    meta_payload = {
        "version": CACHE_SCHEMA_VERSION,
        "cached_at": cached_at_ts,
        "asset_type": str(asset_type).strip().lower(),
        "symbol": canonical_cache_symbol(asset_type, symbol),
        "adjust": canonical_cache_adjust(asset_type, adjust),
        "covered_start": cov_start.strftime("%Y-%m-%d"),
        "covered_end": cov_end.strftime("%Y-%m-%d"),
        "row_count": int(len(to_store)),
        "attrs": clean_attrs,
    }
    meta_bytes = json.dumps(meta_payload, ensure_ascii=False, indent=2).encode("utf-8")

    try:
        with _cache_entry_lock(cache_dir, key):
            tmp_parquet: Path | None = None
            tmp_meta: Path | None = None
            bak_parquet: Path | None = None
            bak_meta: Path | None = None
            try:
                tmp_parquet = _write_staged_temp_file(
                    parquet_path.parent,
                    lambda tmp: to_store.to_parquet(tmp, engine="pyarrow", index=True),
                    prefix=f".{key}.",
                    suffix=".parquet.tmp",
                )
                tmp_meta = _write_staged_temp_file(
                    meta_path.parent,
                    lambda tmp: tmp.write_bytes(meta_bytes),
                    prefix=f".{key}.",
                    suffix=".meta.tmp",
                )
                bak_parquet = _backup_existing_file(
                    parquet_path,
                    prefix=f".{key}.",
                    suffix=".parquet.bak",
                )
                bak_meta = _backup_existing_file(
                    meta_path,
                    prefix=f".{key}.",
                    suffix=".meta.bak",
                )
                parquet_replaced = False
                meta_replaced = False
                try:
                    os.replace(tmp_parquet, parquet_path)
                    tmp_parquet = None
                    parquet_replaced = True
                    os.replace(tmp_meta, meta_path)
                    tmp_meta = None
                    meta_replaced = True
                except Exception:
                    # Restore only targets that were actually overwritten before the failure.
                    if parquet_replaced:
                        bak_parquet = _restore_or_unlink_file(parquet_path, bak_parquet)
                    if meta_replaced:
                        bak_meta = _restore_or_unlink_file(meta_path, bak_meta)
                    raise
            finally:
                for leftover in (tmp_parquet, tmp_meta, bak_parquet, bak_meta):
                    if leftover is not None:
                        try:
                            if leftover.exists():
                                leftover.unlink()
                        except OSError:
                            pass
    except Exception:
        return None
    return parquet_path


def clear_ohlcv_cache() -> int:
    """Remove all cached ``.parquet`` and ``.meta.json`` files in the cache directory."""
    cache_dir = get_cache_dir()
    if not cache_dir.is_dir():
        return 0

    keys_to_clear: set[str] = set()
    for path in cache_dir.glob("*.parquet"):
        keys_to_clear.add(path.name[: -len(".parquet")])
    for path in cache_dir.glob("*.meta.json"):
        keys_to_clear.add(path.name[: -len(".meta.json")])

    removed = 0
    for key in sorted(keys_to_clear):
        with _cache_entry_lock(cache_dir, key):
            for path in _entry_paths(cache_dir, key):
                try:
                    path.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
    return removed


def get_ohlcv_cache_stats() -> dict[str, Any]:
    """Return entry count, disk usage bytes, and active directory path."""
    cache_dir = get_cache_dir()
    if not cache_dir.is_dir():
        return {"enabled": is_cache_enabled(), "dir": str(cache_dir), "entries": 0, "bytes": 0}
    parquet_files = list(cache_dir.glob("*.parquet"))
    total_bytes = 0
    for path in [*parquet_files, *cache_dir.glob("*.meta.json")]:
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass
    return {
        "enabled": is_cache_enabled(),
        "dir": str(cache_dir),
        "entries": len(parquet_files),
        "bytes": int(total_bytes),
    }


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_CACHE_TTL_SECONDS",
    "ENV_CACHE_DIR",
    "ENV_CACHE_TTL",
    "ENV_DISABLE_CACHE",
    "build_cache_key",
    "canonical_cache_adjust",
    "canonical_cache_symbol",
    "clear_ohlcv_cache",
    "get_cache_dir",
    "get_ohlcv_cache_stats",
    "is_cache_enabled",
    "load_cached_ohlcv",
    "resolve_cache_ttl",
    "save_cached_ohlcv",
]
