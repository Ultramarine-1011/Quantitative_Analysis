"""Shared persistent Parquet + JSON sidecar cache for Web UI and CLI."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asset_catalog import SUPPORTED_ADJUSTMENTS, resolve_adjustment
from ohlcv import coerce_analysis_date

DEFAULT_CACHE_TTL_SECONDS = 3600.0
CACHE_SCHEMA_VERSION = 1
ENV_CACHE_DIR = "QUANT_CACHE_DIR"
ENV_DISABLE_CACHE = "QUANT_DISABLE_CACHE"
ENV_CACHE_TTL = "QUANT_CACHE_TTL"
_TRANSIENT_OR_SENSITIVE_ATTRS = frozenset({"cache_hit", "cache_path", "proxy_url"})


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
    asset_key = str(asset_type).strip().lower()
    raw = "" if symbol is None else str(symbol).strip()
    if asset_key == "bond_us_yield":
        if not raw or raw.upper() in {"10Y", "默认", "DEFAULT"}:
            return "10Y"
        return raw
    return raw.upper()


def canonical_cache_adjust(asset_type: str, adjust: str | None) -> str:
    """Normalize adjustment policy so omitted defaults map to the same entry."""
    asset_key = str(asset_type).strip().lower()
    if asset_key in SUPPORTED_ADJUSTMENTS:
        return resolve_adjustment(asset_key, adjust)
    if adjust:
        return resolve_adjustment(asset_key, adjust)
    return ""


def build_cache_key(asset_type: str, symbol: object, adjust: str | None = None) -> str:
    """Build a deterministic SHA-256 hex cache key for ``(asset_type, symbol, adjust)``."""
    asset_key = str(asset_type).strip().lower()
    sym_key = canonical_cache_symbol(asset_key, symbol)
    adj_key = canonical_cache_adjust(asset_key, adjust)
    payload = json.dumps([asset_key, sym_key, adj_key], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _entry_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    return cache_dir / f"{key}.parquet", cache_dir / f"{key}.meta.json"


def _to_json_safe(value: Any) -> Any:
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
        return [_to_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _atomic_write_bytes_or_parquet(
    target_path: Path,
    writer: Any,
    *,
    prefix: str,
    suffix: str,
) -> None:
    """Write via temp file in ``target_path.parent`` then ``os.replace`` atomically."""
    parent = target_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        writer(tmp_path)
        os.replace(tmp_path, target_path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def load_cached_ohlcv(
    asset_type: str,
    symbol: object,
    start_date: object,
    end_date: object,
    *,
    adjust: str | None = None,
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
    key = build_cache_key(asset_type, symbol, adjust=adjust)
    parquet_path, meta_path = _entry_paths(cache_dir, key)
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
) -> Path | None:
    """Persist a non-empty OHLCV DataFrame atomically as Parquet + JSON sidecar."""
    if not is_cache_enabled() or df is None or df.empty:
        return None

    req_start = coerce_analysis_date(start_date, "start_date")
    req_end = coerce_analysis_date(end_date, "end_date")
    cache_dir = get_cache_dir()
    key = build_cache_key(asset_type, symbol, adjust=adjust)
    parquet_path, meta_path = _entry_paths(cache_dir, key)

    to_store = df.copy()
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

    clean_attrs = _to_json_safe(
        {k: v for k, v in dict(df.attrs).items() if k not in _TRANSIENT_OR_SENSITIVE_ATTRS}
    )
    meta_payload = {
        "version": CACHE_SCHEMA_VERSION,
        "cached_at": float(time.time()),
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
        _atomic_write_bytes_or_parquet(
            parquet_path,
            lambda tmp: to_store.to_parquet(tmp, engine="pyarrow", index=True),
            prefix=f".{key}.",
            suffix=".parquet.tmp",
        )
        _atomic_write_bytes_or_parquet(
            meta_path,
            lambda tmp: tmp.write_bytes(meta_bytes),
            prefix=f".{key}.",
            suffix=".meta.tmp",
        )
    except Exception:
        return None
    return parquet_path


def clear_ohlcv_cache() -> int:
    """Remove all cached ``.parquet`` and ``.meta.json`` files in the cache directory."""
    cache_dir = get_cache_dir()
    if not cache_dir.is_dir():
        return 0
    removed = 0
    for pattern in ("*.parquet", "*.meta.json"):
        for path in cache_dir.glob(pattern):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def get_ohlcv_cache_stats() -> dict[str, Any]:
    """Return entry count, disk usage bytes, and active directory path."""
    cache_dir = get_cache_dir()
    if not cache_dir.is_dir():
        return {"enabled": is_cache_enabled(), "dir": str(cache_dir), "entries": 0, "bytes": 0}
    parquet_files = list(cache_dir.glob("*.parquet"))
    total_bytes = sum(
        p.stat().st_size
        for p in list(cache_dir.glob("*.parquet")) + list(cache_dir.glob("*.meta.json"))
    )
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
