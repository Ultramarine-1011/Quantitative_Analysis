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
SUPPORTED_CALENDAR_MODES: tuple[str, ...] = ("common_trading_dates", "weekly_friday", "mtm_outer")
SUPPORTED_COV_METHODS: tuple[str, ...] = ("sample", "ledoit_wolf", "diagonal")
SUPPORTED_REBALANCE_FREQS: tuple[str, ...] = ("buy_and_hold", "daily", "monthly", "quarterly")
SUPPORTED_PORTFOLIO_STRATEGIES: tuple[str, ...] = (
    "max_sharpe",
    "min_variance",
    "equal_weight",
    "inverse_volatility",
    "risk_parity",
    "custom",
)
SUPPORTED_VALIDATION_MODES: tuple[str, ...] = (
    "train_test",
    "walk_forward_rolling",
    "walk_forward_expanding",
    "in_sample",
)


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
        return "1.0.0"


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


def _validate_multi_asset_common(
    *,
    assets: tuple[AssetSpec, ...],
    start_date: str,
    end_date: str,
    calendar_mode: str,
    base_currency: str,
    risk_free_rate: float,
    trading_days: int | None,
    cov_method: str,
) -> tuple[str, str, str, str, float, int | None, str]:
    if len(assets) < 2:
        raise ValueError("多资产比较与组合研究至少需要 2 个不重复标的。")
    seen: set[tuple[str, str]] = set()
    for spec in assets:
        canon_key = (spec.asset_type, spec.canonical_symbol)
        if canon_key in seen:
            raise ValueError(
                f"多资产列表中包含重复标的（含同义代码折叠）: {spec.asset_type}:{spec.canonical_symbol}"
            )
        seen.add(canon_key)

    start_ts, end_ts = validate_analysis_date_range(start_date, end_date)
    cal = str(calendar_mode or "common_trading_dates").strip().lower()
    if cal not in SUPPORTED_CALENDAR_MODES:
        raise ValueError(
            f"不支持的交易日历对齐模式 {calendar_mode!r}，可选: {', '.join(SUPPORTED_CALENDAR_MODES)}。"
        )
    ccy = str(base_currency or "CNY").strip().upper()
    if ccy not in {"LOCAL", "CNY", "USD"}:
        raise ValueError(f"不支持的基准货币 {base_currency!r}，可选: LOCAL, CNY, USD。")
    rf = float(risk_free_rate)
    if not math.isfinite(rf) or rf <= -1.0:
        raise ValueError("risk_free_rate 必须为大于 -1 的有限数值。")
    td_resolved: int | None = None
    if trading_days is not None:
        if isinstance(trading_days, bool) or int(trading_days) < 1:
            raise ValueError("trading_days 必须为正整数。")
        td_resolved = int(trading_days)
    cov = str(cov_method or "ledoit_wolf").strip().lower()
    if cov not in SUPPORTED_COV_METHODS:
        raise ValueError(
            f"不支持的协方差估计方法 {cov_method!r}，可选: {', '.join(SUPPORTED_COV_METHODS)}。"
        )
    return (
        start_ts.strftime("%Y-%m-%d"),
        end_ts.strftime("%Y-%m-%d"),
        cal,
        ccy,
        rf,
        td_resolved,
        cov,
    )


@dataclass(frozen=True)
class AssetSpec:
    """Normalized specification for one asset leg in multi-asset workflows."""

    asset_type: str
    symbol: str
    adjust: str | None = None

    def __post_init__(self) -> None:
        asset_key = str(self.asset_type or "").strip().lower()
        if asset_key not in SUPPORTED_ASSET_TYPES:
            raise ValueError(f"不支持的资产类型: {self.asset_type!r}")
        object.__setattr__(self, "asset_type", asset_key)

        sym = str(self.symbol or "").strip()
        if not sym:
            raise ValueError(f"标的 {asset_key} 的代码不能为空。")
        object.__setattr__(self, "symbol", sym)

        adj_raw = None if self.adjust is None else str(self.adjust).strip()
        resolved_adj = resolve_adjustment(asset_key, adj_raw)
        object.__setattr__(self, "adjust", resolved_adj)

    @property
    def canonical_symbol(self) -> str:
        return normalize_symbol_alias(self.asset_type, self.symbol)

    @property
    def leg_name(self) -> str:
        return f"{self.asset_type}:{self.symbol}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_type": self.asset_type,
            "symbol": self.symbol,
            "canonical_symbol": self.canonical_symbol,
            "leg_name": self.leg_name,
            "adjust": self.adjust,
        }


@dataclass(frozen=True)
class CompareRequest:
    """Normalized request contract for multi-asset comparison (``quant compare``)."""

    assets: tuple[AssetSpec, ...]
    start_date: str
    end_date: str
    benchmark: AssetSpec | None = None
    calendar_mode: Literal["common_trading_dates", "weekly_friday", "mtm_outer"] = (
        "common_trading_dates"
    )
    base_currency: str = "CNY"
    allow_usdt_peg: bool = False
    risk_free_rate: float = 0.016
    trading_days: int | None = None
    return_convention: Literal["log", "simple"] = "simple"
    rolling_window: int = 20
    cov_method: Literal["sample", "ledoit_wolf", "diagonal"] = "ledoit_wolf"
    force_refresh: bool = False
    proxy_url: str | None = None

    def __post_init__(self) -> None:
        normalized_assets = tuple(self.assets)
        object.__setattr__(self, "assets", normalized_assets)
        start_str, end_str, cal, ccy, rf, td, cov = _validate_multi_asset_common(
            assets=normalized_assets,
            start_date=self.start_date,
            end_date=self.end_date,
            calendar_mode=self.calendar_mode,
            base_currency=self.base_currency,
            risk_free_rate=self.risk_free_rate,
            trading_days=self.trading_days,
            cov_method=self.cov_method,
        )
        object.__setattr__(self, "start_date", start_str)
        object.__setattr__(self, "end_date", end_str)
        object.__setattr__(self, "calendar_mode", cal)
        object.__setattr__(self, "base_currency", ccy)
        object.__setattr__(self, "risk_free_rate", rf)
        object.__setattr__(self, "trading_days", td)
        object.__setattr__(self, "cov_method", cov)

        conv = str(self.return_convention or "simple").strip().lower()
        if conv not in SUPPORTED_RETURN_CONVENTIONS:
            raise ValueError(f"不支持的收益率口径 {self.return_convention!r}，可选: log, simple。")
        object.__setattr__(self, "return_convention", conv)

        rw = int(self.rolling_window)
        if isinstance(self.rolling_window, bool) or rw < 2:
            raise ValueError("rolling_window 必须为 >= 2 的整数。")
        object.__setattr__(self, "rolling_window", rw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assets": [item.to_dict() for item in self.assets],
            "start_date": self.start_date,
            "end_date": self.end_date,
            "benchmark": self.benchmark.to_dict() if self.benchmark is not None else None,
            "calendar_mode": self.calendar_mode,
            "base_currency": self.base_currency,
            "allow_usdt_peg": self.allow_usdt_peg,
            "risk_free_rate": self.risk_free_rate,
            "trading_days": self.trading_days,
            "return_convention": self.return_convention,
            "rolling_window": self.rolling_window,
            "cov_method": self.cov_method,
            "force_refresh": self.force_refresh,
            "proxy_configured": bool(self.proxy_url),
        }


@dataclass(frozen=True)
class PortfolioRequest:
    """Normalized request contract for portfolio optimization & backtest (``quant portfolio``)."""

    assets: tuple[AssetSpec, ...]
    start_date: str
    end_date: str
    strategy: Literal[
        "max_sharpe",
        "min_variance",
        "equal_weight",
        "inverse_volatility",
        "risk_parity",
        "custom",
    ] = "max_sharpe"
    validation_mode: Literal[
        "train_test",
        "walk_forward_rolling",
        "walk_forward_expanding",
        "in_sample",
    ] = "train_test"
    benchmark: AssetSpec | None = None
    calendar_mode: Literal["common_trading_dates", "weekly_friday", "mtm_outer"] = (
        "common_trading_dates"
    )
    base_currency: str = "CNY"
    allow_usdt_peg: bool = False
    risk_free_rate: float = 0.016
    trading_days: int | None = None
    cov_method: Literal["sample", "ledoit_wolf", "diagonal"] = "ledoit_wolf"
    rebalance_freq: Literal["buy_and_hold", "daily", "monthly", "quarterly"] = "monthly"
    transaction_cost_bps: float = 10.0
    min_weight: float = 0.0
    max_weight: float = 1.0
    train_ratio: float = 0.7
    split_date: str | None = None
    train_window: int = 60
    custom_weights: tuple[float, ...] | None = None
    frontier_points: int = 25
    num_random_portfolios: int = 0
    random_seed: int | None = 42
    include_sensitivity: bool = True
    force_refresh: bool = False
    proxy_url: str | None = None

    def __post_init__(self) -> None:
        normalized_assets = tuple(self.assets)
        object.__setattr__(self, "assets", normalized_assets)
        start_str, end_str, cal, ccy, rf, td, cov = _validate_multi_asset_common(
            assets=normalized_assets,
            start_date=self.start_date,
            end_date=self.end_date,
            calendar_mode=self.calendar_mode,
            base_currency=self.base_currency,
            risk_free_rate=self.risk_free_rate,
            trading_days=self.trading_days,
            cov_method=self.cov_method,
        )
        object.__setattr__(self, "start_date", start_str)
        object.__setattr__(self, "end_date", end_str)
        object.__setattr__(self, "calendar_mode", cal)
        object.__setattr__(self, "base_currency", ccy)
        object.__setattr__(self, "risk_free_rate", rf)
        object.__setattr__(self, "trading_days", td)
        object.__setattr__(self, "cov_method", cov)

        strat = str(self.strategy or "max_sharpe").strip().lower()
        if strat not in SUPPORTED_PORTFOLIO_STRATEGIES:
            raise ValueError(
                f"不支持的组合策略 {self.strategy!r}，可选: {', '.join(SUPPORTED_PORTFOLIO_STRATEGIES)}。"
            )
        object.__setattr__(self, "strategy", strat)

        vmode = str(self.validation_mode or "train_test").strip().lower()
        if vmode not in SUPPORTED_VALIDATION_MODES:
            raise ValueError(
                f"不支持的回测验证模式 {self.validation_mode!r}，可选: {', '.join(SUPPORTED_VALIDATION_MODES)}。"
            )
        object.__setattr__(self, "validation_mode", vmode)

        reb = str(self.rebalance_freq or "monthly").strip().lower()
        if reb not in SUPPORTED_REBALANCE_FREQS:
            raise ValueError(
                f"不支持的调仓频率 {self.rebalance_freq!r}，可选: {', '.join(SUPPORTED_REBALANCE_FREQS)}。"
            )
        object.__setattr__(self, "rebalance_freq", reb)

        bps = float(self.transaction_cost_bps)
        if not math.isfinite(bps) or bps < 0.0:
            raise ValueError("transaction_cost_bps 必须为非负有限数值。")
        object.__setattr__(self, "transaction_cost_bps", bps)

        w_min = float(self.min_weight)
        w_max = float(self.max_weight)
        n_assets = len(normalized_assets)
        if (
            not (math.isfinite(w_min) and math.isfinite(w_max))
            or w_min < 0.0
            or w_max > 1.0
            or w_min > w_max
        ):
            raise ValueError("组合权重边界需满足 0.0 <= min_weight <= max_weight <= 1.0。")
        if w_min * n_assets > 1.0 + 1e-9 or w_max * n_assets < 1.0 - 1e-9:
            raise ValueError(
                f"权重边界对于 {n_assets} 个标的不可行: min_weight={w_min}, max_weight={w_max}。"
            )
        object.__setattr__(self, "min_weight", w_min)
        object.__setattr__(self, "max_weight", w_max)

        tr = float(self.train_ratio)
        if not math.isfinite(tr) or tr <= 0.0 or tr >= 1.0:
            raise ValueError("train_ratio 必须严格介于 0 与 1 之间。")
        object.__setattr__(self, "train_ratio", tr)

        split_norm: str | None = None
        if self.split_date is not None and str(self.split_date).strip():
            split_ts = pd.Timestamp(str(self.split_date).strip())
            if pd.isna(split_ts):
                raise ValueError(f"无效的 split_date: {self.split_date!r}")
            split_norm = split_ts.strftime("%Y-%m-%d")
        object.__setattr__(self, "split_date", split_norm)

        tw = int(self.train_window)
        if isinstance(self.train_window, bool) or tw < 5:
            raise ValueError("train_window 必须为 >= 5 的整数。")
        object.__setattr__(self, "train_window", tw)

        fp = int(self.frontier_points)
        nr = int(self.num_random_portfolios)
        if fp < 0 or nr < 0:
            raise ValueError("frontier_points 与 num_random_portfolios 必须为非负整数。")
        object.__setattr__(self, "frontier_points", fp)
        object.__setattr__(self, "num_random_portfolios", nr)

        if strat == "custom":
            if vmode.startswith("walk_forward"):
                raise ValueError("custom 固定权重策略仅支持 train_test 或 in_sample 验证模式。")
            if self.custom_weights is None:
                raise ValueError("使用 custom 策略时必须通过 --weights 提供各资产权重。")
            if len(self.custom_weights) != n_assets:
                raise ValueError(
                    f"custom_weights 数量 ({len(self.custom_weights)}) 必须与标的数量 ({n_assets}) 一致。"
                )
            arr = [float(x) for x in self.custom_weights]
            if any((not math.isfinite(x) or x < 0.0) for x in arr):
                raise ValueError("custom_weights 必须为非负有限数值。")
            total = float(sum(arr))
            if total <= 1e-12:
                raise ValueError("custom_weights 权重之和必须大于 0。")
            normalized_w = tuple(x / total for x in arr)
            tol = 1e-9
            if any(w < w_min - tol or w > w_max + tol for w in normalized_w):
                raise ValueError(
                    f"归一化后的 custom_weights {[round(w, 6) for w in normalized_w]} "
                    f"违反权重边界约束 [{w_min}, {w_max}]。"
                )
            object.__setattr__(self, "custom_weights", normalized_w)
        elif self.custom_weights is not None:
            raise ValueError("仅当 strategy='custom' 时才可指定 custom_weights。")

    def to_dict(self) -> dict[str, Any]:
        return {
            "assets": [item.to_dict() for item in self.assets],
            "start_date": self.start_date,
            "end_date": self.end_date,
            "strategy": self.strategy,
            "validation_mode": self.validation_mode,
            "benchmark": self.benchmark.to_dict() if self.benchmark is not None else None,
            "calendar_mode": self.calendar_mode,
            "base_currency": self.base_currency,
            "allow_usdt_peg": self.allow_usdt_peg,
            "risk_free_rate": self.risk_free_rate,
            "trading_days": self.trading_days,
            "cov_method": self.cov_method,
            "rebalance_freq": self.rebalance_freq,
            "transaction_cost_bps": self.transaction_cost_bps,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "train_ratio": self.train_ratio,
            "split_date": self.split_date,
            "train_window": self.train_window,
            "custom_weights": list(self.custom_weights)
            if self.custom_weights is not None
            else None,
            "frontier_points": self.frontier_points,
            "num_random_portfolios": self.num_random_portfolios,
            "random_seed": self.random_seed,
            "include_sensitivity": self.include_sensitivity,
            "force_refresh": self.force_refresh,
            "proxy_configured": bool(self.proxy_url),
        }


@dataclass(frozen=True)
class MultiAssetResult:
    """Versioned multi-asset analysis/portfolio output envelope."""

    schema_version: str
    command: str
    status: str
    asset_order: tuple[str, ...]
    assets: dict[str, SeriesMetadata]
    quality: dict[str, QualityReport]
    data_summary: dict[str, Any]
    parameters: dict[str, Any]
    methodology: dict[str, Any]
    metrics: dict[str, Any]
    metric_units: dict[str, str] = field(default_factory=dict)
    metric_reasons: dict[str, Any] = field(default_factory=dict)
    benchmark: dict[str, Any] | None = None
    warnings: tuple[str, ...] = ()
    disclaimers: tuple[str, ...] = ()
    artifacts: dict[str, str] = field(default_factory=dict)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    data_frame: pd.DataFrame | None = None
    software_version: str = SOFTWARE_VERSION

    def with_artifacts(self, artifacts: Mapping[str, str]) -> MultiAssetResult:
        """Return a copy of the result with updated output artifact paths."""
        return replace(self, artifacts={str(k): str(v) for k, v in artifacts.items()})

    def to_dict(self, *, include_data: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "software_version": self.software_version,
            "command": self.command,
            "status": self.status,
            "asset_order": list(self.asset_order),
            "assets": {k: v.to_dict() for k, v in self.assets.items()},
            "quality": {k: v.to_dict() for k, v in self.quality.items()},
            "data_summary": _json_safe_value(self.data_summary),
            "parameters": _json_safe_value(self.parameters),
            "methodology": _json_safe_value(self.methodology),
            "metrics": _json_safe_value(self.metrics),
            "metric_units": dict(self.metric_units),
            "metric_reasons": _json_safe_value(self.metric_reasons),
            "benchmark": _json_safe_value(self.benchmark) if self.benchmark is not None else None,
            "warnings": list(self.warnings),
            "disclaimers": list(self.disclaimers),
            "artifacts": dict(self.artifacts),
        }
        if include_data and self.data_frame is not None:
            records = self.data_frame.reset_index().to_dict(orient="records")
            payload["data"] = _json_safe_value(records)
        return payload


__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "AssetSpec",
    "CompareRequest",
    "MultiAssetResult",
    "PortfolioRequest",
    "QualityReport",
    "SCHEMA_VERSION",
    "SOFTWARE_VERSION",
    "SUPPORTED_CALENDAR_MODES",
    "SUPPORTED_COV_METHODS",
    "SUPPORTED_PORTFOLIO_STRATEGIES",
    "SUPPORTED_REBALANCE_FREQS",
    "SUPPORTED_RETURN_CONVENTIONS",
    "SUPPORTED_VALIDATION_MODES",
    "SeriesMetadata",
    "build_data_summary",
    "build_metric_units",
    "build_quality_report",
    "build_series_metadata",
    "get_software_version",
]
