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
    SUPPORTED_ADJUSTMENTS,
    SUPPORTED_ASSET_TYPES,
    get_asset_trading_days,
    normalize_symbol_alias,
    resolve_adjustment,
    resolve_return_basis,
    resolve_semantic_series_kind,
)
from ohlcv import compute_ohlcv_quality, validate_analysis_date_range

SCHEMA_VERSION = "0.2.0"
LEGACY_SCHEMA_VERSIONS: tuple[str, ...] = ("0.1.0",)
SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = ("0.1.0", "0.2.0")
VALID_RESULT_STATUSES: frozenset[str] = frozenset({"ok", "degraded", "error"})
VALID_SERIES_KINDS: frozenset[str] = frozenset(
    {
        "price",
        "yield_level",
        "fx_rate",
        "cumulative_nav",
        "futures_continuous",
        "price_index",
        "exchange_bond_price",
    }
)
YIELD_BLOCKED_METRICS: tuple[str, ...] = (
    "cumulative_return",
    "cagr",
    "max_drawdown",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "var_95",
    "cvar_95",
)
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


def _is_missing_or_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _parse_semver_tuple(version: str) -> tuple[int, int, int]:
    raw = str(version or "").strip()
    parts = raw.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Invalid semantic schema_version: {version!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


@dataclass(frozen=True)
class MetricDefinition:
    """Canonical domain specification for a quantitative metric."""

    name: str
    unit: str
    sign_convention: Literal[
        "signed",
        "non_negative",
        "non_positive",
        "non_negative_depth",
        "underwater_non_positive",
        "positive",
        "categorical",
    ]
    display_convention: str
    annualization_basis: Literal["none", "calendar_365_25", "trading_days", "effective_periods"]
    return_convention_dependent: bool
    yield_level_allowed: bool
    description: str


METRIC_DEFINITIONS: dict[str, MetricDefinition] = {
    "cumulative_return": MetricDefinition(
        name="cumulative_return",
        unit="ratio_decimal",
        sign_convention="signed",
        display_convention="signed_percent",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Total holding-period simple return P_T / P_0 - 1.",
    ),
    "total_return": MetricDefinition(
        name="total_return",
        unit="ratio_decimal",
        sign_convention="signed",
        display_convention="signed_percent",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Total backtest holding-period return (alias of cumulative_return).",
    ),
    "cagr": MetricDefinition(
        name="cagr",
        unit="ratio_decimal_per_year",
        sign_convention="signed",
        display_convention="signed_percent",
        annualization_basis="calendar_365_25",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Compound annual growth rate scaled by calendar span days / 365.25.",
    ),
    "max_drawdown": MetricDefinition(
        name="max_drawdown",
        unit="ratio_decimal",
        sign_convention="non_negative_depth",
        display_convention="underwater_percent",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Maximum peak-to-trough depth stored as non-negative ratio >= 0; rendered underwater <= 0%.",
    ),
    "underwater_drawdown": MetricDefinition(
        name="underwater_drawdown",
        unit="ratio_decimal",
        sign_convention="underwater_non_positive",
        display_convention="underwater_percent",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Time-series underwater drawdown curve <= 0.",
    ),
    "annualized_volatility": MetricDefinition(
        name="annualized_volatility",
        unit="ratio_decimal_per_year",
        sign_convention="non_negative",
        display_convention="percent",
        annualization_basis="trading_days",
        return_convention_dependent=True,
        yield_level_allowed=False,
        description="Sample return standard deviation (ddof=1) scaled by sqrt(periods_per_year).",
    ),
    "sharpe_ratio": MetricDefinition(
        name="sharpe_ratio",
        unit="dimensionless_ratio",
        sign_convention="signed",
        display_convention="ratio",
        annualization_basis="trading_days",
        return_convention_dependent=True,
        yield_level_allowed=False,
        description="Annualized excess return divided by annualized sample volatility.",
    ),
    "sortino_ratio": MetricDefinition(
        name="sortino_ratio",
        unit="dimensionless_ratio",
        sign_convention="signed",
        display_convention="ratio",
        annualization_basis="trading_days",
        return_convention_dependent=True,
        yield_level_allowed=False,
        description="Annualized excess return divided by downside deviation below period risk-free MAR.",
    ),
    "calmar_ratio": MetricDefinition(
        name="calmar_ratio",
        unit="dimensionless_ratio",
        sign_convention="signed",
        display_convention="ratio",
        annualization_basis="calendar_365_25",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="CAGR divided by non-negative maximum drawdown depth.",
    ),
    "var_95": MetricDefinition(
        name="var_95",
        unit="ratio_decimal",
        sign_convention="signed",
        display_convention="signed_percent",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Historical 5th percentile simple period return.",
    ),
    "cvar_95": MetricDefinition(
        name="cvar_95",
        unit="ratio_decimal",
        sign_convention="signed",
        display_convention="signed_percent",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Historical mean simple period return at or below the 5th percentile.",
    ),
    "latest_close": MetricDefinition(
        name="latest_close",
        unit="price_units",
        sign_convention="signed",
        display_convention="price_with_unit",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=True,
        description="Latest delivered Close observation in output currency or series units.",
    ),
    "latest_level": MetricDefinition(
        name="latest_level",
        unit="percentage_points_yield",
        sign_convention="signed",
        display_convention="yield_percent",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=True,
        description="Latest yield level in percentage points.",
    ),
    "level_change": MetricDefinition(
        name="level_change",
        unit="percentage_points_change",
        sign_convention="signed",
        display_convention="yield_pp_signed",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=True,
        description="End-minus-start change in yield percentage points.",
    ),
    "level_min": MetricDefinition(
        name="level_min",
        unit="percentage_points_yield",
        sign_convention="signed",
        display_convention="yield_percent",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=True,
        description="Minimum observed level in delivered window.",
    ),
    "level_max": MetricDefinition(
        name="level_max",
        unit="percentage_points_yield",
        sign_convention="signed",
        display_convention="yield_percent",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=True,
        description="Maximum observed level in delivered window.",
    ),
    "level_diff_std": MetricDefinition(
        name="level_diff_std",
        unit="percentage_points_change",
        sign_convention="non_negative",
        display_convention="yield_pp",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=True,
        description="Sample standard deviation of daily yield level first differences.",
    ),
    "trading_days": MetricDefinition(
        name="trading_days",
        unit="trading_days_per_year",
        sign_convention="positive",
        display_convention="integer",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=True,
        description="Annualization trading days per year.",
    ),
    "periods_per_year": MetricDefinition(
        name="periods_per_year",
        unit="trading_days_per_year",
        sign_convention="positive",
        display_convention="integer",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=True,
        description="Effective annualization periods per year.",
    ),
    "return_convention": MetricDefinition(
        name="return_convention",
        unit="categorical",
        sign_convention="categorical",
        display_convention="categorical",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=True,
        description="Return definition ('log' or 'simple').",
    ),
    "latest_signal": MetricDefinition(
        name="latest_signal",
        unit="categorical",
        sign_convention="categorical",
        display_convention="categorical",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=True,
        description="Latest Bollinger band regime label.",
    ),
    "tracking_error": MetricDefinition(
        name="tracking_error",
        unit="ratio_decimal_per_year",
        sign_convention="non_negative",
        display_convention="percent",
        annualization_basis="effective_periods",
        return_convention_dependent=True,
        yield_level_allowed=False,
        description="Annualized standard deviation of active returns vs benchmark.",
    ),
    "information_ratio": MetricDefinition(
        name="information_ratio",
        unit="dimensionless_ratio",
        sign_convention="signed",
        display_convention="ratio",
        annualization_basis="effective_periods",
        return_convention_dependent=True,
        yield_level_allowed=False,
        description="Annualized active return divided by tracking error.",
    ),
    "beta": MetricDefinition(
        name="beta",
        unit="dimensionless_ratio",
        sign_convention="signed",
        display_convention="ratio",
        annualization_basis="none",
        return_convention_dependent=True,
        yield_level_allowed=False,
        description="CAPM slope of excess asset return on excess benchmark return.",
    ),
    "jensen_alpha": MetricDefinition(
        name="jensen_alpha",
        unit="ratio_decimal_per_year",
        sign_convention="signed",
        display_convention="signed_percent",
        annualization_basis="effective_periods",
        return_convention_dependent=True,
        yield_level_allowed=False,
        description="Annualized CAPM Jensen alpha.",
    ),
    "correlation": MetricDefinition(
        name="correlation",
        unit="dimensionless_ratio",
        sign_convention="signed",
        display_convention="ratio",
        annualization_basis="none",
        return_convention_dependent=True,
        yield_level_allowed=False,
        description="Linear Pearson correlation coefficient in [-1, 1].",
    ),
    "r_squared": MetricDefinition(
        name="r_squared",
        unit="dimensionless_ratio",
        sign_convention="non_negative",
        display_convention="ratio",
        annualization_basis="none",
        return_convention_dependent=True,
        yield_level_allowed=False,
        description="Coefficient of determination in [0, 1].",
    ),
    "up_capture_ratio": MetricDefinition(
        name="up_capture_ratio",
        unit="dimensionless_ratio",
        sign_convention="signed",
        display_convention="percent",
        annualization_basis="effective_periods",
        return_convention_dependent=True,
        yield_level_allowed=False,
        description="Annualized asset return ratio during positive benchmark periods.",
    ),
    "down_capture_ratio": MetricDefinition(
        name="down_capture_ratio",
        unit="dimensionless_ratio",
        sign_convention="signed",
        display_convention="percent",
        annualization_basis="effective_periods",
        return_convention_dependent=True,
        yield_level_allowed=False,
        description="Annualized asset return ratio during negative benchmark periods.",
    ),
    "portfolio_volatility": MetricDefinition(
        name="portfolio_volatility",
        unit="ratio_decimal_per_year",
        sign_convention="non_negative",
        display_convention="percent",
        annualization_basis="effective_periods",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Annualized portfolio standard deviation from covariance matrix.",
    ),
    "expected_return": MetricDefinition(
        name="expected_return",
        unit="ratio_decimal_per_year",
        sign_convention="signed",
        display_convention="signed_percent",
        annualization_basis="effective_periods",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Annualized portfolio expected return from estimation window.",
    ),
    "diversification_ratio": MetricDefinition(
        name="diversification_ratio",
        unit="dimensionless_ratio",
        sign_convention="non_negative",
        display_convention="ratio",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Weighted average asset volatility divided by portfolio volatility.",
    ),
    "effective_n_weights": MetricDefinition(
        name="effective_n_weights",
        unit="effective_asset_count",
        sign_convention="positive",
        display_convention="ratio",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Inverse Herfindahl index of portfolio weights.",
    ),
    "effective_n_risk": MetricDefinition(
        name="effective_n_risk",
        unit="effective_asset_count",
        sign_convention="positive",
        display_convention="ratio",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Inverse Herfindahl index of fractional risk contributions.",
    ),
    "total_turnover": MetricDefinition(
        name="total_turnover",
        unit="ratio_decimal",
        sign_convention="non_negative",
        display_convention="percent",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Cumulative one-way portfolio rebalance turnover.",
    ),
    "total_cost_ratio": MetricDefinition(
        name="total_cost_ratio",
        unit="ratio_decimal",
        sign_convention="non_negative",
        display_convention="percent",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Cumulative transaction cost drag relative to initial NAV.",
    ),
    "excess_cumulative_return": MetricDefinition(
        name="excess_cumulative_return",
        unit="ratio_decimal",
        sign_convention="signed",
        display_convention="signed_percent",
        annualization_basis="none",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Asset cumulative return minus benchmark cumulative return.",
    ),
    "excess_annualized_return": MetricDefinition(
        name="excess_annualized_return",
        unit="ratio_decimal_per_year",
        sign_convention="signed",
        display_convention="signed_percent",
        annualization_basis="effective_periods",
        return_convention_dependent=False,
        yield_level_allowed=False,
        description="Annualized asset return minus annualized benchmark return.",
    ),
}

_MULTI_ASSET_METRIC_UNIT_KEYS: tuple[str, ...] = (
    "cumulative_return",
    "total_return",
    "cagr",
    "max_drawdown",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "var_95",
    "cvar_95",
    "tracking_error",
    "information_ratio",
    "beta",
    "jensen_alpha",
    "correlation",
    "r_squared",
    "up_capture_ratio",
    "down_capture_ratio",
    "portfolio_volatility",
    "expected_return",
    "diversification_ratio",
    "effective_n_weights",
    "effective_n_risk",
    "total_turnover",
    "total_cost_ratio",
)


def get_metric_definition(name: str) -> MetricDefinition:
    """Return canonical domain specification for *name*, raising KeyError if unknown."""
    key = str(name).strip()
    if key not in METRIC_DEFINITIONS:
        raise KeyError(f"Unknown domain metric definition: {name!r}")
    return METRIC_DEFINITIONS[key]


def build_multi_asset_metric_units() -> dict[str, str]:
    """Return canonical machine-readable units for multi-asset compare and portfolio results."""
    return {key: METRIC_DEFINITIONS[key].unit for key in _MULTI_ASSET_METRIC_UNIT_KEYS}


def validate_metric_value(
    name: str,
    value: Any,
    *,
    schema_version: str = SCHEMA_VERSION,
    command: str = "analyze",
) -> None:
    """Validate that a metric scalar respects domain sign, finiteness, and range invariants."""
    if _is_missing_or_nan(value):
        return
    defn = METRIC_DEFINITIONS.get(str(name).strip())
    if defn is None or defn.sign_convention == "categorical":
        return
    if isinstance(value, bool):
        raise ValueError(f"Metric {name!r} must be numeric, got bool {value!r}.")
    try:
        num = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metric {name!r} must be numeric, got {value!r}.") from exc
    if not math.isfinite(num):
        raise ValueError(f"Metric {name!r} must be finite, got {value!r}.")

    tol = 1e-12
    if defn.name == "max_drawdown":
        ver = _parse_semver_tuple(schema_version)
        if ver >= (0, 2, 0):
            if num < -tol:
                raise ValueError(
                    f"Schema v{schema_version} requires max_drawdown >= 0 (non-negative depth); "
                    f"got {num!r} for command={command!r}."
                )
        elif command == "compare":
            if num > tol:
                raise ValueError(
                    f"Legacy schema v0.1.0 compare expects max_drawdown <= 0; got {num!r}."
                )
        else:
            if num < -tol:
                raise ValueError(
                    f"Legacy schema v0.1.0 {command!r} expects max_drawdown >= 0; got {num!r}."
                )
        return

    if defn.sign_convention == "non_negative" and num < -tol:
        raise ValueError(f"Metric {name!r} must be non-negative (>= 0); got {num!r}.")
    if defn.sign_convention in {"non_positive", "underwater_non_positive"} and num > tol:
        raise ValueError(f"Metric {name!r} must be non-positive (<= 0); got {num!r}.")
    if defn.sign_convention == "positive" and num <= 0.0:
        raise ValueError(f"Metric {name!r} must be strictly positive (> 0); got {num!r}.")
    if defn.name == "correlation" and not (-1.0 - 1e-9 <= num <= 1.0 + 1e-9):
        raise ValueError(f"Correlation must lie in [-1, 1]; got {num!r}.")
    if defn.name == "r_squared" and not (-1e-9 <= num <= 1.0 + 1e-9):
        raise ValueError(f"R-squared must lie in [0, 1]; got {num!r}.")


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

    def __post_init__(self) -> None:
        asset_key = str(self.asset_type or "").strip().lower()
        if asset_key and asset_key not in SUPPORTED_ASSET_TYPES:
            raise ValueError(f"不支持的资产类型: {self.asset_type!r}")
        object.__setattr__(self, "asset_type", asset_key)

        sym = str(self.symbol or "").strip()
        canon = str(self.canonical_symbol or "").strip()
        object.__setattr__(self, "symbol", sym)
        object.__setattr__(self, "canonical_symbol", canon)

        adj_raw = "" if self.adjust is None else str(self.adjust).strip()
        if asset_key in SUPPORTED_ADJUSTMENTS:
            supported = SUPPORTED_ADJUSTMENTS[asset_key]
            if adj_raw not in supported:
                labels = ", ".join(repr(x) for x in supported)
                raise ValueError(
                    f"SeriesMetadata 复权口径无效: asset_type={asset_key!r} 不支持 adjust={adj_raw!r}（可选 {labels}）。"
                )
            adj_norm = adj_raw
        else:
            adj_norm = "" if adj_raw == "none" else adj_raw
        object.__setattr__(self, "adjust", adj_norm)

        basis = str(self.return_basis or "").strip()
        if asset_key in {"etf", "ashare", "us_stock", "hk_stock"}:
            if adj_norm in {"hfq", "qfq"} and basis == "unadjusted_price":
                raise ValueError(
                    f"SeriesMetadata 语义矛盾: adjust={adj_norm!r} 与 return_basis={basis!r} 不一致。"
                )
            if adj_norm == "" and basis == "adjusted_price":
                raise ValueError(
                    f"SeriesMetadata 语义矛盾: adjust={adj_norm!r} 与 return_basis={basis!r} 不一致。"
                )
        object.__setattr__(self, "return_basis", basis)

        kind = str(self.series_kind or "").strip().lower()
        if kind and kind not in VALID_SERIES_KINDS:
            raise ValueError(f"不支持的 series_kind: {self.series_kind!r}")
        object.__setattr__(self, "series_kind", kind)

        if isinstance(self.trading_days, bool) or int(self.trading_days) < 1:
            raise ValueError(f"trading_days 必须为正整数，当前为 {self.trading_days!r}。")
        object.__setattr__(self, "trading_days", int(self.trading_days))

        native_ccy = str(self.native_currency).strip() if self.native_currency else None
        output_ccy = str(self.output_currency).strip() if self.output_currency else None
        object.__setattr__(self, "native_currency", native_ccy)
        object.__setattr__(self, "output_currency", output_ccy)

        fx_conv = bool(self.fx_converted)
        if fx_conv and kind in {"yield_level", "fx_rate"}:
            raise ValueError(f"series_kind={kind!r} 不允许标记 fx_converted=True。")
        object.__setattr__(self, "fx_converted", fx_conv)

        fx_pair = str(self.fx_pair).strip() if self.fx_pair else None
        object.__setattr__(self, "fx_pair", fx_pair)
        object.__setattr__(self, "cache_hit", bool(self.cache_hit))
        object.__setattr__(self, "warnings", tuple(str(w) for w in self.warnings))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SeriesMetadata:
        return cls(
            asset_type=str(raw.get("asset_type", "")),
            symbol=str(raw.get("symbol", "")),
            canonical_symbol=str(raw.get("canonical_symbol", "")),
            adjust="" if raw.get("adjust") is None else str(raw.get("adjust")),
            return_basis=str(raw.get("return_basis", "")),
            series_kind=str(raw.get("series_kind", "price")),
            native_currency=raw.get("native_currency"),
            output_currency=raw.get("output_currency"),
            fx_converted=bool(raw.get("fx_converted", False)),
            fx_pair=raw.get("fx_pair"),
            trading_days=int(raw.get("trading_days", 252)),
            data_source=raw.get("data_source"),
            fetched_at=raw.get("fetched_at"),
            cache_hit=bool(raw.get("cache_hit", False)),
            warnings=tuple(str(w) for w in (raw.get("warnings") or ())),
        )

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

    def __post_init__(self) -> None:
        policy = str(self.missing_policy or "").strip()
        if not policy:
            raise ValueError("QualityReport.missing_policy 不能为空。")
        object.__setattr__(self, "missing_policy", policy)

        scope = str(self.quality_scope or "").strip()
        if not scope:
            raise ValueError("QualityReport.quality_scope 不能为空。")
        object.__setattr__(self, "quality_scope", scope)

        for count_field in (
            "total_rows",
            "dropped_invalid_rows",
            "filled_price_rows",
            "missing_volume_rows",
            "zero_volume_rows",
            "non_positive_price_rows",
            "negative_volume_rows",
            "ohlc_anomaly_rows",
            "true_zero_return_rows",
            "max_calendar_gap_days",
        ):
            raw_val = getattr(self, count_field)
            if isinstance(raw_val, bool):
                raise ValueError(f"QualityReport.{count_field} 不能为布尔值: {raw_val!r}")
            val = int(raw_val)
            if val < 0:
                raise ValueError(f"QualityReport.{count_field} 不能为负数: {val}")
            object.__setattr__(self, count_field, val)

        object.__setattr__(self, "start_date", str(self.start_date or "").strip())
        object.__setattr__(self, "end_date", str(self.end_date or "").strip())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> QualityReport:
        return cls(
            missing_policy=str(raw.get("missing_policy", "ffill_drop")),
            total_rows=int(raw.get("total_rows", 0)),
            start_date=str(raw.get("start_date", "")),
            end_date=str(raw.get("end_date", "")),
            dropped_invalid_rows=int(raw.get("dropped_invalid_rows", 0)),
            filled_price_rows=int(raw.get("filled_price_rows", 0)),
            missing_volume_rows=int(raw.get("missing_volume_rows", 0)),
            zero_volume_rows=int(raw.get("zero_volume_rows", 0)),
            non_positive_price_rows=int(raw.get("non_positive_price_rows", 0)),
            negative_volume_rows=int(raw.get("negative_volume_rows", 0)),
            ohlc_anomaly_rows=int(raw.get("ohlc_anomaly_rows", 0)),
            true_zero_return_rows=int(raw.get("true_zero_return_rows", 0)),
            max_calendar_gap_days=int(raw.get("max_calendar_gap_days", 0)),
            quality_scope=str(raw.get("quality_scope", "slice")),
        )

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
    artifact_manifest: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _parse_semver_tuple(self.schema_version)
        cmd = str(self.command or "").strip().lower()
        if cmd not in {"fetch", "analyze"}:
            raise ValueError(f"AnalysisResult 不支持 command={self.command!r}。")
        object.__setattr__(self, "command", cmd)

        status = str(self.status or "").strip().lower()
        if status not in VALID_RESULT_STATUSES:
            raise ValueError(f"AnalysisResult 不支持 status={self.status!r}。")
        object.__setattr__(self, "status", status)
        validate_result_invariants(self)

    def with_artifacts(
        self,
        artifacts: Mapping[str, str],
        *,
        artifact_manifest: Mapping[str, Any] | None = None,
    ) -> AnalysisResult:
        """Return a copy of the result with updated output artifact paths and optional manifest."""
        manifest_copy = (
            dict(artifact_manifest) if artifact_manifest is not None else self.artifact_manifest
        )
        return replace(
            self,
            artifacts={str(k): str(v) for k, v in artifacts.items()},
            artifact_manifest=manifest_copy,
        )

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
        if self.artifact_manifest is not None:
            payload["artifact_manifest"] = _json_safe_value(self.artifact_manifest)
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

    single_asset_keys = (
        "cumulative_return",
        "cagr",
        "max_drawdown",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "var_95",
        "cvar_95",
        "latest_close",
        "latest_level",
        "level_change",
        "level_min",
        "level_max",
        "level_diff_std",
        "trading_days",
        "periods_per_year",
        "return_convention",
        "latest_signal",
    )
    unit_map = {key: METRIC_DEFINITIONS[key].unit for key in single_asset_keys}
    unit_map.update(
        {
            "latest_close": price_unit,
            "latest_level": level_unit,
            "level_change": change_unit,
            "level_min": level_unit,
            "level_max": level_unit,
            "level_diff_std": level_unit,
        }
    )
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


def _normalize_adjust_for_asset(asset_key: str, raw_adjust: Any) -> str:
    adj_str = "" if raw_adjust is None else str(raw_adjust).strip()
    if asset_key not in SUPPORTED_ADJUSTMENTS and adj_str == "none":
        return ""
    return adj_str


def build_series_metadata(
    df: pd.DataFrame,
    *,
    asset_type: str | None = None,
    symbol: str | None = None,
    adjust: str | None = None,
    return_basis: str | None = None,
    series_kind: str | None = None,
    native_currency: str | None = None,
    output_currency: str | None = None,
    fx_converted: bool | None = None,
    fx_pair: str | None = None,
    trading_days: int | None = None,
    data_source: str | None = None,
    fetched_at: str | None = None,
    cache_hit: bool | None = None,
    warnings: Sequence[str] = (),
    strict: bool = False,
) -> SeriesMetadata:
    """Build a typed SeriesMetadata snapshot from explicit parameters validated against df.attrs."""
    attrs = df.attrs if isinstance(df, pd.DataFrame) else {}

    arg_asset = str(asset_type or "").strip().lower()
    attr_asset = str(attrs.get("asset_type", "") or "").strip().lower()
    if arg_asset and attr_asset and arg_asset != attr_asset:
        raise ValueError(
            f"SeriesMetadata asset_type 冲突: 参数={arg_asset!r} vs df.attrs={attr_asset!r}"
        )
    asset_key = arg_asset or attr_asset
    if strict and not asset_key:
        raise ValueError("SeriesMetadata strict 模式要求提供非空 asset_type。")

    arg_sym = str(symbol or "").strip()
    attr_sym = str(attrs.get("symbol") or "").strip()
    if arg_sym and attr_sym and asset_key:
        if normalize_symbol_alias(asset_key, arg_sym) != normalize_symbol_alias(
            asset_key, attr_sym
        ):
            raise ValueError(
                f"SeriesMetadata symbol 冲突: 参数={arg_sym!r} vs df.attrs={attr_sym!r}"
            )
    sym = attr_sym or arg_sym
    if strict and not sym:
        raise ValueError("SeriesMetadata strict 模式要求提供非空 symbol。")

    canon_attr = str(attrs.get("canonical_symbol") or "").strip()
    canonical_sym = str(canon_attr or normalize_symbol_alias(asset_key, sym or arg_sym)).strip()

    has_attr_adjust = "adjust" in attrs and attrs.get("adjust") is not None
    attr_adj_norm = (
        _normalize_adjust_for_asset(asset_key, attrs.get("adjust")) if has_attr_adjust else None
    )
    arg_adj_norm = _normalize_adjust_for_asset(asset_key, adjust) if adjust is not None else None
    if attr_adj_norm is not None and arg_adj_norm is not None and attr_adj_norm != arg_adj_norm:
        raise ValueError(
            f"SeriesMetadata adjust 冲突: 参数={arg_adj_norm!r} vs df.attrs={attr_adj_norm!r}"
        )
    resolved_adj_opt = arg_adj_norm if arg_adj_norm is not None else attr_adj_norm
    if asset_key in SUPPORTED_ADJUSTMENTS:
        if resolved_adj_opt is None:
            raise ValueError(
                f"资产类别 {asset_key!r} 缺少显式复权口径 adjust，禁止静默推断默认值。"
            )
        resolved_adjust = resolve_adjustment(asset_key, resolved_adj_opt)
    else:
        resolved_adjust = resolved_adj_opt if resolved_adj_opt is not None else ""

    attr_basis = str(attrs.get("return_basis") or "").strip()
    arg_basis = str(return_basis or "").strip()
    if arg_basis and attr_basis and attr_basis != "adjusted_or_price" and arg_basis != attr_basis:
        raise ValueError(
            f"SeriesMetadata return_basis 冲突: 参数={arg_basis!r} vs df.attrs={attr_basis!r}"
        )
    resolved_basis = arg_basis or resolve_return_basis(
        asset_key,
        adjust=resolved_adjust if asset_key in SUPPORTED_ADJUSTMENTS else None,
        attrs=attrs,
    )

    resolved_kind = (
        str(series_kind).strip().lower()
        if series_kind
        else resolve_semantic_series_kind(asset_key, return_basis=resolved_basis, attrs=attrs)
    )

    attr_out_ccy = str(attrs.get("currency")).strip() if attrs.get("currency") else None
    arg_out_ccy = str(output_currency).strip() if output_currency else None
    if arg_out_ccy and attr_out_ccy and arg_out_ccy.upper() != attr_out_ccy.upper():
        raise ValueError(
            f"SeriesMetadata output_currency 冲突: 参数={arg_out_ccy!r} vs df.attrs={attr_out_ccy!r}"
        )
    output_ccy = arg_out_ccy or attr_out_ccy
    if strict and not output_ccy:
        raise ValueError("SeriesMetadata strict 模式要求提供非空 output_currency。")

    attr_nat_ccy = (
        str(attrs.get("native_currency")).strip() if attrs.get("native_currency") else None
    )
    arg_nat_ccy = str(native_currency).strip() if native_currency else None
    if arg_nat_ccy and attr_nat_ccy and arg_nat_ccy.upper() != attr_nat_ccy.upper():
        raise ValueError(
            f"SeriesMetadata native_currency 冲突: 参数={arg_nat_ccy!r} vs df.attrs={attr_nat_ccy!r}"
        )
    native_ccy = arg_nat_ccy or attr_nat_ccy or output_ccy

    if fx_converted is not None:
        fx_conv_val = bool(fx_converted)
    else:
        fx_conv_val = bool(attrs.get("fx_converted", False))
        if not fx_conv_val and native_ccy and output_ccy and native_ccy != output_ccy:
            fx_conv_val = True

    attr_fx_pair = str(attrs.get("fx_pair")).strip() if attrs.get("fx_pair") else None
    arg_fx_pair = str(fx_pair).strip() if fx_pair else None
    resolved_fx_pair = arg_fx_pair or attr_fx_pair

    eff_trading_days = (
        int(trading_days) if trading_days is not None else get_asset_trading_days(asset_key or None)
    )
    attr_source = str(attrs.get("data_source")).strip() if attrs.get("data_source") else None
    arg_source = str(data_source).strip() if data_source else None
    resolved_source = arg_source or attr_source

    attr_fetched = str(attrs.get("fetched_at")).strip() if attrs.get("fetched_at") else None
    arg_fetched = str(fetched_at).strip() if fetched_at else None
    resolved_fetched = arg_fetched or attr_fetched

    resolved_cache_hit = (
        bool(cache_hit) if cache_hit is not None else bool(attrs.get("cache_hit", False))
    )

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
        adjust=resolved_adjust,
        return_basis=resolved_basis,
        series_kind=resolved_kind,
        native_currency=native_ccy,
        output_currency=output_ccy,
        fx_converted=fx_conv_val,
        fx_pair=resolved_fx_pair,
        trading_days=eff_trading_days,
        data_source=resolved_source,
        fetched_at=resolved_fetched,
        cache_hit=resolved_cache_hit,
        warnings=tuple(merged_warnings),
    )


def sync_metadata_to_frame(
    df: pd.DataFrame,
    metadata: SeriesMetadata,
    quality: QualityReport | None = None,
) -> pd.DataFrame:
    """Dual-write canonical SeriesMetadata (and optional QualityReport) back onto df.attrs."""
    df.attrs["asset_type"] = metadata.asset_type
    df.attrs["symbol"] = metadata.symbol
    df.attrs["canonical_symbol"] = metadata.canonical_symbol
    df.attrs["adjust"] = metadata.adjust
    df.attrs["return_basis"] = metadata.return_basis
    df.attrs["series_kind"] = metadata.series_kind
    if metadata.native_currency is not None:
        df.attrs["native_currency"] = metadata.native_currency
    if metadata.output_currency is not None:
        df.attrs["currency"] = metadata.output_currency
        df.attrs["output_currency"] = metadata.output_currency
    df.attrs["fx_converted"] = metadata.fx_converted
    df.attrs["fx_pair"] = metadata.fx_pair
    if metadata.data_source is not None:
        df.attrs["data_source"] = metadata.data_source
    if metadata.fetched_at is not None:
        df.attrs["fetched_at"] = metadata.fetched_at
    df.attrs["cache_hit"] = metadata.cache_hit
    if quality is not None:
        df.attrs["data_quality"] = quality.to_dict()
    return df


def validate_frame_metadata_consistency(
    df: pd.DataFrame,
    metadata: SeriesMetadata,
) -> None:
    """Verify df.attrs fields match the canonical SeriesMetadata without silent drift."""
    attrs = df.attrs if isinstance(df, pd.DataFrame) else {}
    checks: list[tuple[str, Any, Any]] = [
        ("asset_type", attrs.get("asset_type"), metadata.asset_type),
        ("canonical_symbol", attrs.get("canonical_symbol"), metadata.canonical_symbol),
        (
            "adjust",
            _normalize_adjust_for_asset(metadata.asset_type, attrs.get("adjust"))
            if "adjust" in attrs and attrs.get("adjust") is not None
            else None,
            metadata.adjust,
        ),
        ("return_basis", attrs.get("return_basis"), metadata.return_basis),
        ("series_kind", attrs.get("series_kind"), metadata.series_kind),
        ("currency", attrs.get("currency"), metadata.output_currency),
        ("native_currency", attrs.get("native_currency"), metadata.native_currency),
    ]
    for field_name, attr_val, meta_val in checks:
        if attr_val is None or meta_val is None or str(attr_val).strip() == "":
            continue
        if str(attr_val).strip().lower() != str(meta_val).strip().lower():
            raise ValueError(
                f"df.attrs 与 SeriesMetadata 字段 {field_name!r} 不一致: "
                f"attrs={attr_val!r} vs metadata={meta_val!r}"
            )
    if "fx_converted" in attrs and bool(attrs["fx_converted"]) != metadata.fx_converted:
        raise ValueError(
            f"df.attrs 与 SeriesMetadata 字段 'fx_converted' 不一致: "
            f"attrs={attrs['fx_converted']!r} vs metadata={metadata.fx_converted!r}"
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
    artifact_manifest: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _parse_semver_tuple(self.schema_version)
        cmd = str(self.command or "").strip().lower()
        if cmd not in {"compare", "portfolio"}:
            raise ValueError(f"MultiAssetResult 不支持 command={self.command!r}。")
        object.__setattr__(self, "command", cmd)

        status = str(self.status or "").strip().lower()
        if status not in VALID_RESULT_STATUSES:
            raise ValueError(f"MultiAssetResult 不支持 status={self.status!r}。")
        object.__setattr__(self, "status", status)

        order = tuple(str(x) for x in self.asset_order)
        object.__setattr__(self, "asset_order", order)
        if status != "error":
            if len(order) < 2:
                raise ValueError("MultiAssetResult.asset_order 至少需要 2 个标的。")
            if len(set(order)) != len(order):
                raise ValueError(f"MultiAssetResult.asset_order 包含重复标的: {order!r}")
            for leg in order:
                if leg not in self.assets:
                    raise ValueError(f"MultiAssetResult.assets 缺少 {leg!r}。")
                if leg not in self.quality:
                    raise ValueError(f"MultiAssetResult.quality 缺少 {leg!r}。")
        validate_result_invariants(self)

    def with_artifacts(
        self,
        artifacts: Mapping[str, str],
        *,
        artifact_manifest: Mapping[str, Any] | None = None,
    ) -> MultiAssetResult:
        """Return a copy of the result with updated output artifact paths and optional manifest."""
        manifest_copy = (
            dict(artifact_manifest) if artifact_manifest is not None else self.artifact_manifest
        )
        return replace(
            self,
            artifacts={str(k): str(v) for k, v in artifacts.items()},
            artifact_manifest=manifest_copy,
        )

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
        if self.artifact_manifest is not None:
            payload["artifact_manifest"] = _json_safe_value(self.artifact_manifest)
        if include_data and self.data_frame is not None:
            records = self.data_frame.reset_index().to_dict(orient="records")
            payload["data"] = _json_safe_value(records)
        return payload


def _validate_metric_mapping(
    metrics_map: Mapping[str, Any],
    *,
    schema_version: str,
    command: str,
) -> None:
    for key, val in metrics_map.items():
        validate_metric_value(
            str(key),
            val,
            schema_version=schema_version,
            command=command,
        )


def validate_result_invariants(result: AnalysisResult | MultiAssetResult) -> None:
    """Validate domain-level invariants across a typed AnalysisResult or MultiAssetResult."""
    schema_ver = result.schema_version
    cmd = result.command
    if isinstance(result, AnalysisResult):
        if (
            cmd == "analyze"
            and result.asset.series_kind == "yield_level"
            and result.status != "error"
        ):
            for key in YIELD_BLOCKED_METRICS:
                if key in result.metrics and not _is_missing_or_nan(result.metrics[key]):
                    raise ValueError(
                        f"yield_level 序列禁止输出 {key!r} 非空数值，当前为 {result.metrics[key]!r}。"
                    )
                if (
                    key in result.metrics
                    and result.metric_reasons.get(key) != "yield_level_not_return"
                ):
                    raise ValueError(
                        f"yield_level 序列必须记录 metric_reasons[{key!r}] == 'yield_level_not_return'。"
                    )
        _validate_metric_mapping(result.metrics, schema_version=schema_ver, command=cmd)
        return

    metrics = result.metrics if isinstance(result.metrics, Mapping) else {}
    if cmd == "compare":
        per_asset = metrics.get("per_asset")
        if isinstance(per_asset, Mapping):
            for leg_metrics in per_asset.values():
                if isinstance(leg_metrics, Mapping):
                    _validate_metric_mapping(
                        leg_metrics, schema_version=schema_ver, command="compare"
                    )
        if isinstance(result.benchmark, Mapping):
            bench_m = result.benchmark.get("metrics")
            if isinstance(bench_m, Mapping):
                _validate_metric_mapping(bench_m, schema_version=schema_ver, command="compare")
        return

    if cmd == "portfolio":
        bt = metrics.get("backtest")
        if isinstance(bt, Mapping):
            for bt_key, entry in bt.items():
                if bt_key != "relative_to_equal_weight" and isinstance(entry, Mapping):
                    _validate_metric_mapping(entry, schema_version=schema_ver, command="portfolio")
        rc = metrics.get("risk_contributions")
        if isinstance(rc, Mapping):
            _validate_metric_mapping(rc, schema_version=schema_ver, command="portfolio")
        val = metrics.get("validation")
        if isinstance(val, Mapping):
            for sub_key in (
                "in_sample_performance",
                "out_of_sample_performance",
                "performance",
                "in_sample_hindsight_reference",
            ):
                sub = val.get(sub_key)
                if isinstance(sub, Mapping):
                    _validate_metric_mapping(sub, schema_version=schema_ver, command="portfolio")
        sens = metrics.get("sensitivity")
        if isinstance(sens, Mapping):
            for rows in sens.values():
                if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
                    for row in rows:
                        if isinstance(row, Mapping) and row.get("status") == "ok":
                            _validate_metric_mapping(
                                row, schema_version=schema_ver, command="portfolio"
                            )
        if isinstance(result.benchmark, Mapping):
            bp = result.benchmark.get("performance")
            if isinstance(bp, Mapping):
                _validate_metric_mapping(bp, schema_version=schema_ver, command="portfolio")


__all__ = [
    "AnalysisRequest",
    "AnalysisResult",
    "AssetSpec",
    "CompareRequest",
    "LEGACY_SCHEMA_VERSIONS",
    "METRIC_DEFINITIONS",
    "MetricDefinition",
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
    "SUPPORTED_SCHEMA_VERSIONS",
    "SUPPORTED_VALIDATION_MODES",
    "SeriesMetadata",
    "VALID_RESULT_STATUSES",
    "VALID_SERIES_KINDS",
    "YIELD_BLOCKED_METRICS",
    "build_data_summary",
    "build_metric_units",
    "build_multi_asset_metric_units",
    "build_quality_report",
    "build_series_metadata",
    "get_metric_definition",
    "get_software_version",
    "sync_metadata_to_frame",
    "validate_frame_metadata_consistency",
    "validate_metric_value",
    "validate_result_invariants",
]
