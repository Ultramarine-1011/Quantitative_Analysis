"""Financial formatting helpers and artifact deserialization for reports and charts."""

from __future__ import annotations

import math
from typing import Any, Literal

from artifacts.store import load_result_from_artifacts
from asset_catalog import ASSET_LABELS

METRIC_REASON_LABELS: dict[str, str] = {
    "yield_level_not_return": "利率/收益率水平序列不作为资产价格计算收益率指标",
    "non_positive_price_series": "价格序列含非正值或有效收盘价不足",
    "insufficient_samples": "有效样本量不足，无法稳健估计统计量",
    "zero_volatility": "区间波动率为零，比率分母为 0",
    "zero_downside_deviation": "区间无有效下行波动，Sortino 比率分母为 0",
    "zero_max_drawdown": "区间最大回撤为 0，Calmar 比率分母为 0",
    "zero_benchmark_variance": "基准方差为 0，无法计算 Beta",
    "zero_tracking_error": "跟踪误差为 0，无法计算信息比率",
    "zero_var": "采样样本中收益率方差为 0，无法计算检验统计量",
}

_UNIT_DESCRIPTIONS: dict[str, str] = {
    "ratio_decimal": "比例 (十进小数，报表展示为 %)",
    "ratio_decimal_per_year": "年化比例 (十进小数，报表展示为 %)",
    "dimensionless_ratio": "无量纲比率",
    "percentage_points_yield": "百分点 (%)",
    "percentage_points_change": "百分点变动 (pp)",
    "fx_rate": "汇率标价",
    "fx_rate_change": "汇率点位变动",
    "index_points": "指数点位",
    "index_points_change": "指数点位变动",
    "price_units": "原始价格单位",
    "price_change": "价格变动",
    "trading_days_per_year": "年化周期数 (天/年)",
    "categorical": "分类标签",
}


def is_valid_number(value: Any) -> bool:
    """Return True iff *value* is a real finite numeric scalar (excluding bool)."""
    if value is None or isinstance(value, bool):
        return False
    try:
        num = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(num)


def describe_unit(unit: str | None) -> str:
    """Translate a machine-readable metric unit into a concise human description."""
    if not unit:
        return "-"
    if unit.startswith("currency:"):
        ccy = unit.split(":", 1)[1].strip().upper()
        return f"货币 ({ccy})" if ccy else "货币"
    if unit.startswith("fx_rate:"):
        ccy = unit.split(":", 1)[1].strip().upper()
        return f"汇率 ({ccy})" if ccy else "汇率"
    return _UNIT_DESCRIPTIONS.get(unit, unit)


def format_reason_label(reason: str | None) -> str:
    """Return a readable explanation for a blocked/missing metric reason code."""
    if not reason:
        return "无可用数值"
    return METRIC_REASON_LABELS.get(reason, reason)


def format_asset_type_label(asset_type: str | None) -> str:
    """Return bilingual display label for an asset_type key."""
    if not asset_type:
        return "未知资产"
    return ASSET_LABELS.get(asset_type, asset_type)


def format_percent(
    value: Any,
    *,
    decimals: int = 2,
    signed: bool = False,
    na_text: str = "N/A",
) -> str:
    """Format a decimal ratio (e.g. ``0.1234`` -> ``12.34%``)."""
    if not is_valid_number(value):
        return na_text
    pct = float(value) * 100.0
    if abs(pct) < 10 ** (-(decimals + 2)):
        pct = 0.0
    fmt = f"{{:{'+' if signed else ''}.{decimals}f}}%"
    return fmt.format(pct)


def format_yield_percent(
    value: Any,
    *,
    decimals: int = 2,
    signed: bool = False,
    is_change: bool = False,
    na_text: str = "N/A",
) -> str:
    """Format a series already denominated in percentage points without multiplying by 100."""
    if not is_valid_number(value):
        return na_text
    num = float(value)
    if abs(num) < 10 ** (-(decimals + 2)):
        num = 0.0
    suffix = " pp" if is_change else "%"
    fmt = f"{{:{'+' if signed else ''}.{decimals}f}}{suffix}"
    return fmt.format(num)


def format_drawdown_percent(
    value: Any,
    *,
    decimals: int = 2,
    na_text: str = "N/A",
    input_convention: Literal["depth", "underwater"] = "depth",
) -> str:
    """Format drawdown as non-positive underwater percentage (``<= 0%``) with explicit sign checks.

    All schema v0.2.0 commands store ``max_drawdown`` as non-negative depth (``>= 0``,
    ``input_convention="depth"``). Time-series underwater drawdown values (``<= 0``) must opt in
    explicitly via ``input_convention="underwater"`` rather than relying on silent ``-abs()``.
    """
    if not is_valid_number(value):
        return na_text
    num = float(value)
    tol = 1e-12
    if input_convention == "depth":
        if num < -tol:
            raise ValueError(
                f"format_drawdown_percent(input_convention='depth') expects non-negative drawdown >= 0; got {num!r}."
            )
        pct = -max(0.0, num) * 100.0
    elif input_convention == "underwater":
        if num > tol:
            raise ValueError(
                f"format_drawdown_percent(input_convention='underwater') expects non-positive drawdown <= 0; got {num!r}."
            )
        pct = min(0.0, num) * 100.0
    else:
        raise ValueError(
            f"Unsupported input_convention {input_convention!r}; expected 'depth' or 'underwater'."
        )
    if abs(pct) < 10 ** (-(decimals + 2)):
        return f"{0.0:.{decimals}f}%"
    return f"{pct:.{decimals}f}%"


def format_ratio(
    value: Any,
    *,
    decimals: int = 2,
    signed: bool = False,
    na_text: str = "N/A",
) -> str:
    """Format a dimensionless ratio or plain floating statistic."""
    if not is_valid_number(value):
        return na_text
    num = float(value)
    if abs(num) < 10 ** (-(decimals + 2)):
        num = 0.0
    fmt = f"{{:{'+' if signed else ''}.{decimals}f}}"
    return fmt.format(num)


def format_integer(value: Any, *, na_text: str = "N/A") -> str:
    """Format an integer count safely."""
    if not is_valid_number(value):
        return na_text
    return f"{int(round(float(value))):,}"


def format_p_value(value: Any, *, na_text: str = "N/A") -> str:
    """Format a statistical test p-value cleanly."""
    if not is_valid_number(value):
        return na_text
    p = float(value)
    if p < 1e-4:
        return "< 0.0001"
    return f"{p:.4f}"


def format_price_with_unit(
    value: Any,
    *,
    series_kind: str = "price",
    output_currency: str | None = None,
    decimals: int | None = None,
    signed: bool = False,
    is_change: bool = False,
    na_text: str = "N/A",
) -> str:
    """Format latest close/level/change value honoring series_kind and currency units."""
    if not is_valid_number(value):
        return na_text
    num = float(value)
    if series_kind == "yield_level":
        return format_yield_percent(
            num,
            decimals=decimals if decimals is not None else 2,
            signed=signed,
            is_change=is_change,
            na_text=na_text,
        )
    if series_kind == "fx_rate":
        dec = decimals if decimals is not None else 4
        fmt = f"{{:{'+' if signed else ''}.{dec}f}}"
        ccy = f" {output_currency}" if output_currency else ""
        return f"{fmt.format(num)}{ccy}"
    if series_kind == "index_points":
        dec = decimals if decimals is not None else 2
        fmt = f"{{:{'+' if signed else ''},.{dec}f}}"
        return f"{fmt.format(num)} pts"
    dec = decimals if decimals is not None else 2
    fmt = f"{{:{'+' if signed else ''},.{dec}f}}"
    ccy = f" {output_currency}" if output_currency else ""
    return f"{fmt.format(num)}{ccy}"


def format_metric_value(
    key: str,
    value: Any,
    *,
    unit: str | None = None,
    reason: str | None = None,
    series_kind: str = "price",
    output_currency: str | None = None,
    include_reason: bool = False,
) -> str:
    """Format any single-asset or portfolio metric respecting key semantics, units, and N/A reasons."""
    if not is_valid_number(value):
        if include_reason and reason:
            return f"N/A ({format_reason_label(reason)})"
        return "N/A"

    key_l = str(key).lower()
    if "drawdown" in key_l:
        return format_drawdown_percent(value)
    if unit == "percentage_points_yield" or (
        series_kind == "yield_level" and key_l in {"latest_level", "level_min", "level_max"}
    ):
        return format_yield_percent(value, is_change=False)
    if unit == "percentage_points_change" or (
        series_kind == "yield_level" and key_l in {"level_change"}
    ):
        return format_yield_percent(value, signed=True, is_change=True)
    if series_kind == "yield_level" and key_l == "level_diff_std":
        return format_yield_percent(value, signed=False, is_change=True)
    if key_l in {"latest_close", "latest_level"}:
        return format_price_with_unit(
            value, series_kind=series_kind, output_currency=output_currency
        )
    if unit in {"ratio_decimal", "ratio_decimal_per_year"} or key_l in {
        "cumulative_return",
        "cagr",
        "annualized_return",
        "annualized_volatility",
        "var_95",
        "cvar_95",
        "excess_return",
        "excess_cagr",
        "tracking_error",
        "alpha_ann",
        "up_capture",
        "down_capture",
        "total_turnover",
        "total_cost",
        "total_cost_ratio",
        "weight",
        "pct_risk_contribution",
    }:
        signed = key_l in {
            "cumulative_return",
            "cagr",
            "annualized_return",
            "var_95",
            "cvar_95",
            "excess_return",
            "excess_cagr",
            "alpha_ann",
        }
        return format_percent(value, signed=signed)
    if key_l in {"trading_days", "periods_per_year", "n_obs", "rebalance_count"}:
        return format_integer(value)
    return format_ratio(value)


__all__ = [
    "METRIC_REASON_LABELS",
    "describe_unit",
    "format_asset_type_label",
    "format_drawdown_percent",
    "format_integer",
    "format_metric_value",
    "format_p_value",
    "format_percent",
    "format_price_with_unit",
    "format_ratio",
    "format_reason_label",
    "format_yield_percent",
    "is_valid_number",
    "load_result_from_artifacts",
]
