"""Single-asset quantitative analysis service shared by CLI, Web UI, and future Agent CLI."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from analysis_metrics import get_quant_metrics
from asset_catalog import (
    ASSET_RETURN_BASIS_NOTES,
    SUPPORTED_ADJUSTMENTS,
    get_asset_trading_days,
    normalize_symbol_alias,
    resolve_return_basis,
)
from asset_loader import load_asset_data
from contracts import (
    SCHEMA_VERSION,
    AnalysisRequest,
    AnalysisResult,
    build_quality_report,
    build_series_metadata,
)
from currency import convert_ohlcv_to_base_currency, infer_asset_currency
from data_sources.errors import EmptySymbolDataError
from ohlcv import DateLike

AssetLoaderFn = Callable[..., pd.DataFrame]


def _extract_latest_snapshot(df: pd.DataFrame, metrics: dict[str, Any]) -> dict[str, Any]:
    """Extract the newest bar OHLCV + indicator snapshot for structured output."""
    if df.empty:
        return {}
    last_row = df.iloc[-1]
    snapshot: dict[str, Any] = {
        "date": pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d"),
        "open": float(last_row["Open"]),
        "high": float(last_row["High"]),
        "low": float(last_row["Low"]),
        "close": float(last_row["Close"]),
        "volume": float(last_row["Volume"]),
        "latest_signal": str(metrics.get("latest_signal", "")),
    }
    for col in df.columns:
        if col in {"Open", "High", "Low", "Close", "Volume"}:
            continue
        val = last_row[col]
        snapshot[str(col)] = None if pd.isna(val) else float(val)
    return snapshot


def _build_metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    """Separate scalar/numeric analysis metrics from DataFrame & warning payloads."""
    keys = (
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
    return {k: metrics[k] for k in keys if k in metrics}


def analyze_single_asset(
    request: AnalysisRequest,
    *,
    loader: AssetLoaderFn | None = None,
    fx_loader: Callable[[str, DateLike, DateLike], pd.DataFrame] | None = None,
    usdt_usd_loader: Callable[[DateLike, DateLike, str | None], pd.DataFrame] | None = None,
    raw_df: pd.DataFrame | None = None,
) -> AnalysisResult:
    """Execute load -> currency normalization -> single-pass features -> metrics -> contracts."""
    fetch_fn = loader or load_asset_data
    trading_days = request.trading_days or get_asset_trading_days(request.asset_type)

    if raw_df is None:
        loader_kwargs: dict[str, Any] = {
            "proxy_url": request.proxy_url,
            "adjust": request.adjust if request.asset_type in SUPPORTED_ADJUSTMENTS else None,
        }
        if request.force_refresh:
            loader_kwargs["force_refresh"] = True
        loaded = fetch_fn(
            request.asset_type,
            request.symbol,
            request.start_date,
            request.end_date,
            **loader_kwargs,
        )
    else:
        loaded = raw_df.copy()

    if loaded is None or loaded.empty:
        raise EmptySymbolDataError(
            f"未找到标的 {request.asset_type}:{request.symbol} 在 {request.start_date}~{request.end_date} 的有效数据。"
        )

    work_df = loaded.copy()
    work_df.attrs = dict(loaded.attrs)
    work_df.attrs.setdefault("asset_type", request.asset_type)
    work_df.attrs.setdefault("symbol", request.symbol)
    if request.asset_type in SUPPORTED_ADJUSTMENTS:
        work_df.attrs.setdefault("adjust", request.adjust)
    resolved_sym = str(work_df.attrs.get("symbol", request.symbol) or request.symbol).strip()
    work_df.attrs["return_basis"] = resolve_return_basis(
        request.asset_type,
        adjust=request.adjust if request.asset_type in SUPPORTED_ADJUSTMENTS else None,
        attrs=work_df.attrs,
    )
    work_df.attrs.setdefault(
        "canonical_symbol", normalize_symbol_alias(request.asset_type, resolved_sym)
    )
    work_df.attrs.setdefault(
        "currency", infer_asset_currency(request.asset_type, resolved_sym, work_df.attrs)
    )
    work_df.attrs.setdefault("fetched_at", datetime.now(UTC).isoformat())
    work_df.attrs.setdefault("cache_hit", False)

    if request.base_currency != "LOCAL":
        work_df = convert_ohlcv_to_base_currency(
            work_df,
            request.asset_type,
            resolved_sym,
            request.base_currency,
            proxy_url=request.proxy_url,
            allow_usdt_peg=request.allow_usdt_peg,
            fx_loader=fx_loader,
            usdt_usd_loader=usdt_usd_loader,
        )
    else:
        work_df.attrs.setdefault("native_currency", work_df.attrs.get("currency"))
        work_df.attrs.setdefault("fx_converted", False)
        work_df.attrs.setdefault("fx_pair", None)

    metrics = get_quant_metrics(
        work_df,
        risk_free_rate=request.risk_free_rate,
        trading_days=trading_days,
        return_convention=request.return_convention,
        sma_short_window=request.sma_short,
        sma_long_window=request.sma_long,
        bollinger_window=request.bb_window,
        include_momentum_indicators=request.include_momentum,
    )
    featured_df: pd.DataFrame = metrics["featured_df"]
    featured_df.attrs["analysis_start"] = request.start_date
    featured_df.attrs["analysis_end"] = request.end_date

    quality_report = build_quality_report(featured_df)
    metric_reasons = dict(metrics.get("metric_reasons") or {})
    semantic_warnings = [
        str(w).strip() for w in (metrics.get("semantic_warnings") or ()) if str(w).strip()
    ]

    all_warnings: list[str] = []
    for candidate in (
        *(featured_df.attrs.get("fx_warnings") or ()),
        *(featured_df.attrs.get("fallback_warnings") or ()),
        *semantic_warnings,
    ):
        text = str(candidate).strip()
        if text and text not in all_warnings:
            all_warnings.append(text)
    if quality_report.has_anomalies:
        all_warnings.append(
            f"数据质量提示：dropped={quality_report.dropped_invalid_rows}, "
            f"filled={quality_report.filled_price_rows}, "
            f"missing_vol={quality_report.missing_volume_rows}, "
            f"ohlc_anomaly={quality_report.ohlc_anomaly_rows}, "
            f"non_positive_price={quality_report.non_positive_price_rows} "
            f"(missing_policy={quality_report.missing_policy})。"
        )

    asset_meta = build_series_metadata(
        featured_df,
        asset_type=request.asset_type,
        symbol=request.symbol,
        trading_days=trading_days,
        warnings=all_warnings,
    )

    disclaimers: list[str] = []
    basis_note = ASSET_RETURN_BASIS_NOTES.get(asset_meta.return_basis)
    if basis_note:
        disclaimers.append(basis_note)
    metric_disclaimer = str(metrics.get("metric_disclaimer") or "").strip()
    if metric_disclaimer and metric_disclaimer not in disclaimers:
        disclaimers.append(metric_disclaimer)

    degraded_reasons = {"yield_level_not_return", "insufficient_data"}
    has_degraded_metrics = any(r in degraded_reasons for r in metric_reasons.values())
    status = (
        "degraded"
        if (
            quality_report.has_anomalies
            or has_degraded_metrics
            or bool(featured_df.attrs.get("fx_warnings"))
            or bool(featured_df.attrs.get("fallback_warnings"))
        )
        else "ok"
    )

    return AnalysisResult(
        schema_version=SCHEMA_VERSION,
        status=status,
        asset=asset_meta,
        quality=quality_report,
        parameters=request.to_dict(),
        metrics=_build_metric_payload(metrics),
        metric_reasons=metric_reasons,
        latest_snapshot=_extract_latest_snapshot(featured_df, metrics),
        data_frame=featured_df,
        warnings=tuple(all_warnings),
        disclaimers=tuple(disclaimers),
    )


__all__ = ["analyze_single_asset"]
