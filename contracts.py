"""Unified financial data contracts for M1/M2+ analysis services and CLIs."""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from asset_catalog import (
    SUPPORTED_ASSET_TYPES,
    get_asset_trading_days,
    normalize_symbol_alias,
    resolve_adjustment,
    resolve_return_basis,
    resolve_semantic_series_kind,
)
from ohlcv import compute_ohlcv_quality, validate_analysis_date_range

SCHEMA_VERSION = "0.1.0"
SUPPORTED_RETURN_CONVENTIONS: tuple[str, ...] = ("log", "simple")


def get_software_version() -> str:
    """Return active project version from ``pyproject.toml`` in source trees or package metadata."""
    pyproject_path = Path(__file__).resolve().with_name("pyproject.toml")
    if pyproject_path.is_file():
        try:
            parsed = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            project = parsed.get("project")
            if isinstance(project, Mapping) and isinstance(project.get("version"), str):
                ver = project["version"].strip()
                if ver:
                    return ver
        except Exception:
            pass
    try:
        return importlib_metadata.version("quantitative-analysis")
    except importlib_metadata.PackageNotFoundError:
        return "0.2.0"


SOFTWARE_VERSION = get_software_version()


def _json_safe_value(value: Any) -> Any:
    """Convert floats/timestamps/containers recursively to strict JSON-safe types."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return _json_safe_value(value.item())
    return str(value)


@dataclass(frozen=True)
class AnalysisRequest:
    """Normalized request contract shared across UI, CLI, and analysis services."""

    asset_type: str
    symbol: str
    start_date: str
    end_date: str
    adjust: str | None = None
    base_currency: str = "LOCAL"
    risk_free_rate: float = 0.016
    trading_days: int | None = None
    return_convention: Literal["log", "simple"] = "log"
    sma_short: int = 20
    sma_long: int = 60
    bb_window: int = 20
    include_momentum: bool = False
    allow_usdt_peg: bool = False
    force_refresh: bool = False
    proxy_url: str | None = None

    def __post_init__(self) -> None:
        asset_key = str(self.asset_type).strip().lower()
        if asset_key not in SUPPORTED_ASSET_TYPES:
            raise ValueError(f"不支持的资产类型: {self.asset_type!r}")
        object.__setattr__(self, "asset_type", asset_key)

        sym = str(self.symbol or "").strip()
        if not sym and asset_key != "bond_us_yield":
            raise ValueError("标的代码不能为空。")
        object.__setattr__(self, "symbol", sym or "10Y")

        start_ts, end_ts = validate_analysis_date_range(self.start_date, self.end_date)
        object.__setattr__(self, "start_date", start_ts.strftime("%Y-%m-%d"))
        object.__setattr__(self, "end_date", end_ts.strftime("%Y-%m-%d"))

        adj_input = None if self.adjust is None else str(self.adjust).strip()
        resolved_adj = resolve_adjustment(asset_key, adj_input)
        object.__setattr__(self, "adjust", resolved_adj)

        ccy = str(self.base_currency or "LOCAL").strip().upper()
        if ccy not in {"LOCAL", "CNY", "USD"}:
            raise ValueError(f"不支持的基准货币 {self.base_currency!r}，可选: LOCAL, CNY, USD。")
        object.__setattr__(self, "base_currency", ccy)

        rf = float(self.risk_free_rate)
        if not math.isfinite(rf) or rf <= -1.0:
            raise ValueError("risk_free_rate 必须为大于 -1 的有限数值。")
        object.__setattr__(self, "risk_free_rate", rf)

        if self.trading_days is not None:
            td = int(self.trading_days)
            if td < 1:
                raise ValueError("trading_days 必须为正整数。")
            object.__setattr__(self, "trading_days", td)

        conv = str(self.return_convention or "log").strip().lower()
        if conv not in SUPPORTED_RETURN_CONVENTIONS:
            raise ValueError(f"不支持的收益率口径 {self.return_convention!r}，可选: log, simple。")
        object.__setattr__(self, "return_convention", conv)

        for field_name in ("sma_short", "sma_long", "bb_window"):
            val = int(getattr(self, field_name))
            if val < 2:
                raise ValueError(f"{field_name} 必须 >= 2，当前为 {val}。")
            object.__setattr__(self, field_name, val)

    def to_dict(self) -> dict[str, Any]:
        """Serialize parameters (redacting no proxy URLs unless explicit)."""
        return {
            "asset_type": self.asset_type,
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "adjust": self.adjust,
            "base_currency": self.base_currency,
            "risk_free_rate": self.risk_free_rate,
            "trading_days": self.trading_days or get_asset_trading_days(self.asset_type),
            "return_convention": self.return_convention,
            "sma_short": self.sma_short,
            "sma_long": self.sma_long,
            "bb_window": self.bb_window,
            "include_momentum": self.include_momentum,
            "allow_usdt_peg": self.allow_usdt_peg,
            "force_refresh": self.force_refresh,
        }


@dataclass(frozen=True)
class SeriesMetadata:
    """Provenance and financial semantic metadata for a standardized OHLCV series."""

    asset_type: str
    symbol: str
    canonical_symbol: str
    adjust: str
    return_basis: str
    series_kind: str
    native_currency: str | None
    output_currency: str | None
    fx_converted: bool
    fx_pair: str | None
    trading_days: int
    data_source: str | None
    fetched_at: str | None
    cache_hit: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_type": self.asset_type,
            "symbol": self.symbol,
            "canonical_symbol": self.canonical_symbol,
            "adjust": self.adjust,
            "return_basis": self.return_basis,
            "series_kind": self.series_kind,
            "native_currency": self.native_currency,
            "output_currency": self.output_currency,
            "fx_converted": self.fx_converted,
            "fx_pair": self.fx_pair,
            "trading_days": self.trading_days,
            "data_source": self.data_source,
            "fetched_at": self.fetched_at,
            "cache_hit": self.cache_hit,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class QualityReport:
    """Structured OHLCV data-quality report over a delivered date slice."""

    missing_policy: str
    total_rows: int
    start_date: str
    end_date: str
    dropped_invalid_rows: int = 0
    filled_price_rows: int = 0
    missing_volume_rows: int = 0
    zero_volume_rows: int = 0
    non_positive_price_rows: int = 0
    negative_volume_rows: int = 0
    ohlc_anomaly_rows: int = 0
    true_zero_return_rows: int = 0
    max_calendar_gap_days: int = 0
    quality_scope: str = "slice"

    @property
    def has_anomalies(self) -> bool:
        return any(
            (
                self.dropped_invalid_rows > 0,
                self.filled_price_rows > 0,
                self.missing_volume_rows > 0,
                self.non_positive_price_rows > 0,
                self.negative_volume_rows > 0,
                self.ohlc_anomaly_rows > 0,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_policy": self.missing_policy,
            "total_rows": self.total_rows,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "dropped_invalid_rows": self.dropped_invalid_rows,
            "filled_price_rows": self.filled_price_rows,
            "missing_volume_rows": self.missing_volume_rows,
            "zero_volume_rows": self.zero_volume_rows,
            "non_positive_price_rows": self.non_positive_price_rows,
            "negative_volume_rows": self.negative_volume_rows,
            "ohlc_anomaly_rows": self.ohlc_anomaly_rows,
            "true_zero_return_rows": self.true_zero_return_rows,
            "max_calendar_gap_days": self.max_calendar_gap_days,
            "quality_scope": self.quality_scope,
        }


@dataclass(frozen=True)
class AnalysisResult:
    """Versioned analysis output envelope produced by the service layer."""

    schema_version: str
    status: str
    asset: SeriesMetadata
    quality: QualityReport
    parameters: dict[str, Any]
    metrics: dict[str, Any]
    metric_reasons: dict[str, str]
    latest_snapshot: dict[str, Any]
    data_frame: pd.DataFrame | None
    warnings: tuple[str, ...] = ()
    disclaimers: tuple[str, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
    command: str = "analyze"
    software_version: str = SOFTWARE_VERSION
    metric_units: dict[str, str] = field(default_factory=dict)
    data_summary: dict[str, Any] = field(default_factory=dict)

    def with_artifacts(self, artifacts: Mapping[str, str]) -> AnalysisResult:
        """Return a copy of the result with updated output artifact paths."""
        return replace(self, artifacts={str(k): str(v) for k, v in artifacts.items()})

    def to_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "software_version": self.software_version,
            "command": self.command,
            "status": self.status,
            "asset": self.asset.to_dict(),
            "quality": self.quality.to_dict(),
            "data_summary": _json_safe_value(self.data_summary),
            "parameters": _json_safe_value(self.parameters),
            "metrics": _json_safe_value(self.metrics),
            "metric_units": dict(self.metric_units),
            "metric_reasons": dict(self.metric_reasons),
            "latest_snapshot": _json_safe_value(self.latest_snapshot),
            "warnings": list(self.warnings),
            "disclaimers": list(self.disclaimers),
            "artifacts": dict(self.artifacts),
        }
        if include_data and self.data_frame is not None:
            records = self.data_frame.reset_index().to_dict(orient="records")
            payload["data"] = _json_safe_value(records)
        return payload


def build_metric_units(
    metrics: Mapping[str, Any],
    *,
    series_kind: str = "price",
    output_currency: str | None = None,
) -> dict[str, str]:
    """Return canonical machine-readable units for every emitted metric key."""
    if series_kind == "yield_level":
        price_unit = "percentage_points_yield"
        level_unit = "percentage_points_yield"
        change_unit = "percentage_points_change"
    elif series_kind == "fx_rate":
        price_unit = f"fx_rate:{output_currency}" if output_currency else "fx_rate"
        level_unit = price_unit
        change_unit = "fx_rate_change"
    elif series_kind == "index_points":
        price_unit = "index_points"
        level_unit = "index_points"
        change_unit = "index_points_change"
    else:
        price_unit = f"currency:{output_currency}" if output_currency else "price_units"
        level_unit = price_unit
        change_unit = "price_change"

    unit_map = {
        "cumulative_return": "ratio_decimal",
        "cagr": "ratio_decimal_per_year",
        "max_drawdown": "ratio_decimal",
        "annualized_volatility": "ratio_decimal_per_year",
        "sharpe_ratio": "dimensionless_ratio",
        "sortino_ratio": "dimensionless_ratio",
        "calmar_ratio": "dimensionless_ratio",
        "var_95": "ratio_decimal",
        "cvar_95": "ratio_decimal",
        "latest_close": price_unit,
        "latest_level": level_unit,
        "level_change": change_unit,
        "level_min": level_unit,
        "level_max": level_unit,
        "level_diff_std": level_unit,
        "trading_days": "trading_days_per_year",
        "periods_per_year": "trading_days_per_year",
        "return_convention": "categorical",
        "latest_signal": "categorical",
    }
    return {str(key): unit_map[str(key)] for key in metrics if str(key) in unit_map}


def build_data_summary(
    df: pd.DataFrame | None,
    *,
    requested_start: str = "",
    requested_end: str = "",
) -> dict[str, Any]:
    """Summarize the delivered DataFrame sample bounds and schema for Agent inspection."""
    if df is None or df.empty:
        return {
            "requested_start": requested_start,
            "requested_end": requested_end,
            "actual_start": "",
            "actual_end": "",
            "row_count": 0,
            "index_name": str(df.index.name or "Date") if df is not None else "Date",
            "columns": [str(col) for col in df.columns] if df is not None else [],
        }

    idx = pd.to_datetime(df.index, errors="coerce").dropna()
    actual_start = idx.min().strftime("%Y-%m-%d") if len(idx) else ""
    actual_end = idx.max().strftime("%Y-%m-%d") if len(idx) else ""
    return {
        "requested_start": requested_start,
        "requested_end": requested_end,
        "actual_start": actual_start,
        "actual_end": actual_end,
        "row_count": int(len(df)),
        "index_name": str(df.index.name or "Date"),
        "columns": [str(col) for col in df.columns],
    }


def build_quality_report(df: pd.DataFrame) -> QualityReport:
    """Build a typed QualityReport from a DataFrame, reusing ohlcv quality logic."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Cannot build QualityReport from an empty DataFrame.")

    idx = pd.to_datetime(df.index, errors="coerce")
    valid_idx = idx.dropna()
    start_str = valid_idx.min().strftime("%Y-%m-%d") if len(valid_idx) else ""
    end_str = valid_idx.max().strftime("%Y-%m-%d") if len(valid_idx) else ""

    raw_quality = df.attrs.get("data_quality")
    existing = dict(raw_quality) if isinstance(raw_quality, Mapping) else {}
    policy = str(existing.get("missing_policy", "drop") or "drop")
    dropped = int(existing.get("dropped_invalid_rows", existing.get("dropped_rows", 0)) or 0)
    filled = int(existing.get("filled_price_rows", existing.get("ffilled_cells", 0)) or 0)
    missing_vol = int(existing.get("missing_volume_rows", 0) or 0)

    existing_start = str(existing.get("start_date") or "")
    existing_end = str(existing.get("end_date") or "")
    reused_existing = bool(
        existing
        and int(existing.get("total_rows", -1)) == len(df)
        and "max_calendar_gap_days" in existing
        and (not existing_start or existing_start == start_str)
        and (not existing_end or existing_end == end_str)
    )

    if reused_existing:
        computed = existing
    else:
        computed = compute_ohlcv_quality(
            df,
            missing_policy=policy,
            dropped_invalid_rows=dropped,
            filled_price_rows=filled,
            missing_volume_rows=missing_vol,
        )

    quality_scope = (
        str(existing.get("quality_scope", "slice") or "slice") if reused_existing else "slice"
    )
    return QualityReport(
        missing_policy=str(computed.get("missing_policy", policy)),
        total_rows=int(computed.get("total_rows", len(df))),
        start_date=start_str or existing_start,
        end_date=end_str or existing_end,
        dropped_invalid_rows=int(computed.get("dropped_invalid_rows", dropped)),
        filled_price_rows=int(computed.get("filled_price_rows", filled)),
        missing_volume_rows=int(computed.get("missing_volume_rows", missing_vol)),
        zero_volume_rows=int(computed.get("zero_volume_rows", 0)),
        non_positive_price_rows=int(computed.get("non_positive_price_rows", 0)),
        negative_volume_rows=int(computed.get("negative_volume_rows", 0)),
        ohlc_anomaly_rows=int(computed.get("ohlc_anomaly_rows", 0)),
        true_zero_return_rows=int(computed.get("true_zero_return_rows", 0)),
        max_calendar_gap_days=int(computed.get("max_calendar_gap_days", 0)),
        quality_scope=quality_scope,
    )


def build_series_metadata(
    df: pd.DataFrame,
    *,
    asset_type: str | None = None,
    symbol: str | None = None,
    trading_days: int | None = None,
    warnings: Sequence[str] = (),
) -> SeriesMetadata:
    """Build a typed SeriesMetadata snapshot from normalized OHLCV attrs."""
    attrs = df.attrs if isinstance(df, pd.DataFrame) else {}
    asset_key = str(asset_type or attrs.get("asset_type", "") or "").strip().lower()
    sym = str(attrs.get("symbol") or symbol or "").strip()
    canonical_sym = str(
        attrs.get("canonical_symbol") or normalize_symbol_alias(asset_key, sym or symbol)
    ).strip()
    has_explicit_adjust = "adjust" in attrs and attrs.get("adjust") is not None
    adjust = str(attrs.get("adjust", "") or "").strip()
    return_basis = resolve_return_basis(
        asset_key,
        adjust=adjust if has_explicit_adjust else None,
        attrs=attrs,
    )
    series_kind = resolve_semantic_series_kind(asset_key, return_basis=return_basis, attrs=attrs)

    output_ccy_raw = attrs.get("currency")
    output_ccy = str(output_ccy_raw).strip() if output_ccy_raw else None
    native_ccy_raw = attrs.get("native_currency") or output_ccy
    native_ccy = str(native_ccy_raw).strip() if native_ccy_raw else None
    fx_converted = bool(attrs.get("fx_converted", False))
    if not fx_converted and native_ccy and output_ccy and native_ccy != output_ccy:
        fx_converted = True
    fx_pair_raw = attrs.get("fx_pair")
    fx_pair = str(fx_pair_raw).strip() if fx_pair_raw else None

    eff_trading_days = (
        int(trading_days) if trading_days is not None else get_asset_trading_days(asset_key or None)
    )
    data_source_raw = attrs.get("data_source")
    data_source = str(data_source_raw).strip() if data_source_raw else None
    fetched_at_raw = attrs.get("fetched_at")
    fetched_at = str(fetched_at_raw).strip() if fetched_at_raw else None
    cache_hit = bool(attrs.get("cache_hit", False))

    merged_warnings: list[str] = []
    for item in (
        *(attrs.get("fx_warnings") or ()),
        *(attrs.get("fallback_warnings") or ()),
        *warnings,
    ):
        text = str(item).strip()
        if text and text not in merged_warnings:
            merged_warnings.append(text)

    return SeriesMetadata(
        asset_type=asset_key,
        symbol=sym,
        canonical_symbol=canonical_sym,
        adjust=adjust,
        return_basis=return_basis,
        series_kind=series_kind,
        native_currency=native_ccy,
        output_currency=output_ccy,
        fx_converted=fx_converted,
        fx_pair=fx_pair,
        trading_days=eff_trading_days,
        data_source=data_source,
        fetched_at=fetched_at,
        cache_hit=cache_hit,
        warnings=tuple(merged_warnings),
    )


__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "QualityReport",
    "SCHEMA_VERSION",
    "SOFTWARE_VERSION",
    "SUPPORTED_RETURN_CONVENTIONS",
    "SeriesMetadata",
    "build_data_summary",
    "build_metric_units",
    "build_quality_report",
    "build_series_metadata",
    "get_software_version",
]
