"""Financial formatting helpers and artifact deserialization for reports and charts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from asset_catalog import ASSET_LABELS
from contracts import (
    SCHEMA_VERSION,
    AnalysisResult,
    MultiAssetResult,
    QualityReport,
    SeriesMetadata,
    get_software_version,
)

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

_KNOWN_MULTI_ASSET_FRAMES: tuple[str, ...] = (
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
) -> str:
    """Format drawdown consistently as non-positive underwater percentage (``<= 0%``).

    ``analyze`` and ``portfolio`` store ``max_drawdown`` as non-negative depth (e.g. ``0.25``),
    whereas ``compare`` stores it as non-positive underwater return (``-0.25``). Both render as
    ``-25.00%``, and exact zero renders as ``0.00%``.
    """
    if not is_valid_number(value):
        return na_text
    pct = -abs(float(value) * 100.0)
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


def _deserialize_series_metadata(raw: Mapping[str, Any]) -> SeriesMetadata:
    return SeriesMetadata(
        asset_type=str(raw.get("asset_type", "")),
        symbol=str(raw.get("symbol", "")),
        canonical_symbol=str(raw.get("canonical_symbol", "")),
        adjust=str(raw.get("adjust", "none")),
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
        warnings=tuple(str(w) for w in raw.get("warnings", ())),
    )


def _deserialize_quality_report(raw: Mapping[str, Any]) -> QualityReport:
    return QualityReport(
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


def _read_frame_from_disk(parquet_path: Path, csv_path: Path | None = None) -> pd.DataFrame | None:
    if parquet_path.is_file():
        df = pd.read_parquet(parquet_path)
    elif csv_path is not None and csv_path.is_file():
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
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


def _resolve_artifact_path(out_dir: Path, raw_path: str | None, fallback_name: str) -> Path:
    if raw_path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_file():
            return candidate
    return out_dir / fallback_name


def load_result_from_artifacts(output_dir: str | Path) -> AnalysisResult | MultiAssetResult:
    """Reconstruct ``AnalysisResult`` or ``MultiAssetResult`` purely from ``result.json`` & Parquet.

    Allows report and chart rendering to execute completely offline and independently of the
    computation/data-fetching pipeline.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    result_path = out_dir / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Result envelope not found: {result_path}")

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid result.json object in {result_path}")

    command = str(payload.get("command", "")).strip()
    artifacts_raw = payload.get("artifacts")
    artifacts = (
        {str(k): str(v) for k, v in artifacts_raw.items()}
        if isinstance(artifacts_raw, Mapping)
        else {}
    )

    if command in {"analyze", "fetch"}:
        asset_raw = payload.get("asset")
        quality_raw = payload.get("quality")
        if not isinstance(asset_raw, Mapping) or not isinstance(quality_raw, Mapping):
            raise ValueError(f"Missing 'asset' or 'quality' section in {result_path}")

        asset_meta = _deserialize_series_metadata(asset_raw)
        quality = _deserialize_quality_report(quality_raw)
        pq_path = _resolve_artifact_path(out_dir, artifacts.get("data_parquet"), "data.parquet")
        csv_path = _resolve_artifact_path(out_dir, artifacts.get("data_csv"), "data.csv")
        df = _read_frame_from_disk(pq_path, csv_path)
        if df is not None:
            params = payload.get("parameters") or {}
            metrics = payload.get("metrics") or {}
            df.attrs["asset_type"] = asset_meta.asset_type
            df.attrs["series_kind"] = asset_meta.series_kind
            df.attrs["canonical_symbol"] = asset_meta.canonical_symbol
            df.attrs["output_currency"] = asset_meta.output_currency
            if is_valid_number(metrics.get("max_drawdown")):
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

        return AnalysisResult(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            status=str(payload.get("status", "ok")),
            asset=asset_meta,
            quality=quality,
            parameters=dict(payload.get("parameters") or {}),
            metrics=dict(payload.get("metrics") or {}),
            metric_reasons={
                str(k): str(v) for k, v in (payload.get("metric_reasons") or {}).items()
            },
            latest_snapshot=dict(payload.get("latest_snapshot") or {}),
            data_frame=df,
            warnings=tuple(str(w) for w in (payload.get("warnings") or ())),
            disclaimers=tuple(str(d) for d in (payload.get("disclaimers") or ())),
            artifacts=artifacts,
            command=command,
            software_version=str(payload.get("software_version", get_software_version())),
            metric_units={str(k): str(v) for k, v in (payload.get("metric_units") or {}).items()},
            data_summary=dict(payload.get("data_summary") or {}),
        )

    if command in {"compare", "portfolio"}:
        assets_raw = payload.get("assets") or {}
        quality_raw = payload.get("quality") or {}
        assets = {
            str(k): _deserialize_series_metadata(v)
            for k, v in assets_raw.items()
            if isinstance(v, Mapping)
        }
        quality_map = {
            str(k): _deserialize_quality_report(v)
            for k, v in quality_raw.items()
            if isinstance(v, Mapping)
        }
        primary_pq = _resolve_artifact_path(out_dir, artifacts.get("data_parquet"), "data.parquet")
        primary_csv = _resolve_artifact_path(out_dir, artifacts.get("data_csv"), "data.csv")
        primary_df = _read_frame_from_disk(primary_pq, primary_csv)

        frames: dict[str, pd.DataFrame] = {}
        for name in _KNOWN_MULTI_ASSET_FRAMES:
            pq = _resolve_artifact_path(
                out_dir, artifacts.get(f"{name}_parquet"), f"{name}.parquet"
            )
            cs = _resolve_artifact_path(out_dir, artifacts.get(f"{name}_csv"), f"{name}.csv")
            loaded = _read_frame_from_disk(pq, cs)
            if loaded is not None:
                frames[name] = loaded

        return MultiAssetResult(
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            command=command,
            status=str(payload.get("status", "ok")),
            asset_order=tuple(str(x) for x in (payload.get("asset_order") or ())),
            assets=assets,
            quality=quality_map,
            data_summary=dict(payload.get("data_summary") or {}),
            parameters=dict(payload.get("parameters") or {}),
            methodology=dict(payload.get("methodology") or {}),
            metrics=dict(payload.get("metrics") or {}),
            metric_units={str(k): str(v) for k, v in (payload.get("metric_units") or {}).items()},
            metric_reasons=dict(payload.get("metric_reasons") or {}),
            benchmark=payload.get("benchmark"),
            warnings=tuple(str(w) for w in (payload.get("warnings") or ())),
            disclaimers=tuple(str(d) for d in (payload.get("disclaimers") or ())),
            artifacts=artifacts,
            frames=frames,
            data_frame=primary_df,
            software_version=str(payload.get("software_version", get_software_version())),
        )

    raise ValueError(f"Unsupported command in {result_path}: {command!r}")


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
