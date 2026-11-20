"""Single-asset quantitative analysis service shared by CLI, Web UI, and future Agent CLI."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from analysis_metrics import get_quant_metrics
from asset_catalog import (
    ASSET_RETURN_BASIS_NOTES,
    SUPPORTED_ADJUSTMENTS,
    get_asset_trading_days,
    normalize_symbol_alias,
    resolve_adjustment,
    resolve_return_basis,
    resolve_semantic_series_kind,
)
from asset_loader import load_asset_data
from contracts import (
    SCHEMA_VERSION,
    SOFTWARE_VERSION,
    AnalysisRequest,
    AnalysisResult,
    build_data_summary,
    build_metric_units,
    build_quality_report,
    build_series_metadata,
    sync_metadata_to_frame,
    validate_frame_metadata_consistency,
)
from currency import convert_ohlcv_to_base_currency, infer_asset_currency
from data_sources.errors import EmptySymbolDataError
from ohlcv import (
    REQUIRED_OHLCV_COLUMNS,
    DateLike,
    compute_ohlcv_quality,
    prepare_ohlcv,
)

AssetLoaderFn = Callable[..., pd.DataFrame]


def _extract_latest_snapshot(df: pd.DataFrame, metrics: Mapping[str, Any]) -> dict[str, Any]:
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
    }
    if "latest_signal" in metrics:
        snapshot["latest_signal"] = str(metrics.get("latest_signal", ""))
    for col in df.columns:
        if col in {"Open", "High", "Low", "Close", "Volume"}:
            continue
        val = last_row[col]
        snapshot[str(col)] = None if pd.isna(val) else float(val)
    return snapshot


def _slice_to_requested_window(
    df: pd.DataFrame,
    *,
    asset_type: str,
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Clip a loaded OHLCV frame to ``[start_date, end_date]`` and update slice quality if narrowed."""
    idx = pd.to_datetime(df.index, errors="coerce")
    valid_mask = ~pd.isna(idx)
    if not valid_mask.any():
        raise EmptySymbolDataError(
            f"未找到标的 {asset_type}:{symbol} 在 {start_date}~{end_date} 的有效数据。"
        )
    idx_valid = pd.DatetimeIndex(idx[valid_mask])
    if idx_valid.tz is not None:
        idx_cmp = idx_valid.tz_convert("UTC").tz_localize(None).normalize()
    else:
        idx_cmp = idx_valid.normalize()

    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    range_mask = (idx_cmp >= start_ts) & (idx_cmp <= end_ts)

    full_mask = np.zeros(len(df), dtype=bool)
    full_mask[np.asarray(valid_mask, dtype=bool)] = np.asarray(range_mask, dtype=bool)
    if not full_mask.any():
        raise EmptySymbolDataError(
            f"未找到标的 {asset_type}:{symbol} 在 {start_date}~{end_date} 的有效数据。"
        )
    if full_mask.all():
        return df

    sliced = df.loc[full_mask].copy()
    sliced.attrs = dict(df.attrs)
    raw_quality = sliced.attrs.get("data_quality")
    existing = dict(raw_quality) if isinstance(raw_quality, Mapping) else {}
    if existing and "source_data_quality" not in sliced.attrs:
        sliced.attrs["source_data_quality"] = dict(existing)
    policy = str(existing.get("missing_policy", "drop") or "drop")
    sliced.attrs["data_quality"] = compute_ohlcv_quality(
        sliced,
        missing_policy=policy,
        dropped_invalid_rows=0,
        filled_price_rows=0,
        missing_volume_rows=0,
    )
    sliced.attrs["data_quality"]["quality_scope"] = "slice"
    return sliced


def _prepare_single_asset_frame(
    request: AnalysisRequest,
    *,
    loader: AssetLoaderFn | None = None,
    fx_loader: Callable[[str, DateLike, DateLike], pd.DataFrame] | None = None,
    usdt_usd_loader: Callable[[DateLike, DateLike, str | None], pd.DataFrame] | None = None,
    raw_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, int]:
    """Load, slice to requested date range, attach provenance, and normalize currency once."""
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

    work_df = _slice_to_requested_window(
        loaded,
        asset_type=request.asset_type,
        symbol=request.symbol,
        start_date=request.start_date,
        end_date=request.end_date,
    )
    if work_df is loaded:
        work_df = loaded.copy()
        work_df.attrs = dict(loaded.attrs)

    work_df.attrs["asset_type"] = request.asset_type
    requested_canonical = normalize_symbol_alias(request.asset_type, request.symbol)
    raw_attr_symbol = str(work_df.attrs.get("symbol") or "").strip()
    if raw_attr_symbol and (
        normalize_symbol_alias(request.asset_type, raw_attr_symbol) == requested_canonical
    ):
        resolved_sym = raw_attr_symbol
    else:
        resolved_sym = request.symbol
    work_df.attrs["symbol"] = resolved_sym
    has_attr_adjust = "adjust" in work_df.attrs and work_df.attrs.get("adjust") is not None
    raw_attr_adjust = str(work_df.attrs.get("adjust") or "").strip() if has_attr_adjust else None
    if raw_attr_adjust is not None:
        if request.asset_type not in SUPPORTED_ADJUSTMENTS and raw_attr_adjust == "none":
            effective_adjust = ""
        elif request.asset_type in SUPPORTED_ADJUSTMENTS:
            effective_adjust = resolve_adjustment(request.asset_type, raw_attr_adjust)
        else:
            effective_adjust = raw_attr_adjust
    else:
        effective_adjust = request.adjust
    work_df.attrs["adjust"] = effective_adjust
    work_df.attrs["return_basis"] = resolve_return_basis(
        request.asset_type,
        adjust=effective_adjust if request.asset_type in SUPPORTED_ADJUSTMENTS else None,
        attrs=work_df.attrs,
    )
    work_df.attrs["series_kind"] = resolve_semantic_series_kind(
        request.asset_type,
        return_basis=work_df.attrs["return_basis"],
        attrs=work_df.attrs,
    )
    work_df.attrs["canonical_symbol"] = normalize_symbol_alias(request.asset_type, resolved_sym)
    if not work_df.attrs.get("currency"):
        work_df.attrs["currency"] = infer_asset_currency(
            request.asset_type, resolved_sym, work_df.attrs
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

    work_df.attrs["analysis_start"] = request.start_date
    work_df.attrs["analysis_end"] = request.end_date
    return work_df, trading_days


def _collect_quality_warning(quality_report: Any) -> str | None:
    if not quality_report.has_anomalies:
        return None
    return (
        f"数据质量提示：dropped={quality_report.dropped_invalid_rows}, "
        f"filled={quality_report.filled_price_rows}, "
        f"missing_vol={quality_report.missing_volume_rows}, "
        f"ohlc_anomaly={quality_report.ohlc_anomaly_rows}, "
        f"non_positive_price={quality_report.non_positive_price_rows} "
        f"(missing_policy={quality_report.missing_policy})。"
    )


def fetch_single_asset(
    request: AnalysisRequest,
    *,
    loader: AssetLoaderFn | None = None,
    fx_loader: Callable[[str, DateLike, DateLike], pd.DataFrame] | None = None,
    usdt_usd_loader: Callable[[DateLike, DateLike, str | None], pd.DataFrame] | None = None,
    raw_df: pd.DataFrame | None = None,
) -> AnalysisResult:
    """Fetch, validate, normalize currency, and package OHLCV series + quality metadata."""
    work_df, trading_days = _prepare_single_asset_frame(
        request,
        loader=loader,
        fx_loader=fx_loader,
        usdt_usd_loader=usdt_usd_loader,
        raw_df=raw_df,
    )
    prev_quality = work_df.attrs.get("data_quality")
    prev_scope = (
        str(prev_quality.get("quality_scope", "") or "")
        if isinstance(prev_quality, Mapping)
        else ""
    )
    has_missing_ohlcv = (
        any(col not in work_df.columns for col in REQUIRED_OHLCV_COLUMNS)
        or work_df[list(REQUIRED_OHLCV_COLUMNS)].isna().any().any()
    )
    prepared_df = prepare_ohlcv(
        work_df,
        drop_invalid=has_missing_ohlcv,
        missing_policy="drop" if has_missing_ohlcv else None,
    )
    if prev_scope and isinstance(prepared_df.attrs.get("data_quality"), dict):
        prepared_df.attrs["data_quality"]["quality_scope"] = prev_scope

    quality_report = build_quality_report(prepared_df)
    all_warnings: list[str] = []
    for candidate in (
        *(prepared_df.attrs.get("fx_warnings") or ()),
        *(prepared_df.attrs.get("fallback_warnings") or ()),
    ):
        text = str(candidate).strip()
        if text and text not in all_warnings:
            all_warnings.append(text)
    quality_warn = _collect_quality_warning(quality_report)
    if quality_warn:
        all_warnings.append(quality_warn)

    asset_meta = build_series_metadata(
        prepared_df,
        asset_type=request.asset_type,
        symbol=request.symbol,
        adjust=str(prepared_df.attrs.get("adjust", request.adjust) or ""),
        return_basis=str(prepared_df.attrs.get("return_basis") or ""),
        series_kind=str(prepared_df.attrs.get("series_kind") or ""),
        native_currency=prepared_df.attrs.get("native_currency"),
        output_currency=prepared_df.attrs.get("currency"),
        fx_converted=bool(prepared_df.attrs.get("fx_converted", False)),
        fx_pair=prepared_df.attrs.get("fx_pair"),
        trading_days=trading_days,
        warnings=all_warnings,
        strict=True,
    )
    sync_metadata_to_frame(prepared_df, asset_meta, quality_report)
    validate_frame_metadata_consistency(prepared_df, asset_meta)
    disclaimers: list[str] = []
    basis_note = ASSET_RETURN_BASIS_NOTES.get(asset_meta.return_basis)
    if basis_note:
        disclaimers.append(basis_note)

    status = (
        "degraded"
        if (
            quality_report.has_anomalies
            or bool(prepared_df.attrs.get("fx_warnings"))
            or bool(prepared_df.attrs.get("fallback_warnings"))
        )
        else "ok"
    )

    return AnalysisResult(
        schema_version=SCHEMA_VERSION,
        status=status,
        asset=asset_meta,
        quality=quality_report,
        parameters=request.to_dict(),
        metrics={},
        metric_reasons={},
        latest_snapshot=_extract_latest_snapshot(prepared_df, {}),
        data_frame=prepared_df,
        warnings=tuple(all_warnings),
        disclaimers=tuple(disclaimers),
        command="fetch",
        software_version=SOFTWARE_VERSION,
        metric_units={},
        data_summary=build_data_summary(
            prepared_df,
            requested_start=request.start_date,
            requested_end=request.end_date,
        ),
    )


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
    work_df, trading_days = _prepare_single_asset_frame(
        request,
        loader=loader,
        fx_loader=fx_loader,
        usdt_usd_loader=usdt_usd_loader,
        raw_df=raw_df,
    )

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
    quality_warn = _collect_quality_warning(quality_report)
    if quality_warn:
        all_warnings.append(quality_warn)

    asset_meta = build_series_metadata(
        featured_df,
        asset_type=request.asset_type,
        symbol=request.symbol,
        adjust=str(featured_df.attrs.get("adjust", request.adjust) or ""),
        return_basis=str(featured_df.attrs.get("return_basis") or ""),
        series_kind=str(featured_df.attrs.get("series_kind") or ""),
        native_currency=featured_df.attrs.get("native_currency"),
        output_currency=featured_df.attrs.get("currency"),
        fx_converted=bool(featured_df.attrs.get("fx_converted", False)),
        fx_pair=featured_df.attrs.get("fx_pair"),
        trading_days=trading_days,
        warnings=all_warnings,
        strict=True,
    )
    sync_metadata_to_frame(featured_df, asset_meta, quality_report)
    validate_frame_metadata_consistency(featured_df, asset_meta)

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

    metric_payload = _build_metric_payload(metrics)
    metric_units = build_metric_units(
        metric_payload,
        series_kind=asset_meta.series_kind,
        output_currency=asset_meta.output_currency,
    )

    return AnalysisResult(
        schema_version=SCHEMA_VERSION,
        status=status,
        asset=asset_meta,
        quality=quality_report,
        parameters=request.to_dict(),
        metrics=metric_payload,
        metric_reasons=metric_reasons,
        latest_snapshot=_extract_latest_snapshot(featured_df, metrics),
        data_frame=featured_df,
        warnings=tuple(all_warnings),
        disclaimers=tuple(disclaimers),
        command="analyze",
        software_version=SOFTWARE_VERSION,
        metric_units=metric_units,
        data_summary=build_data_summary(
            featured_df,
            requested_start=request.start_date,
            requested_end=request.end_date,
        ),
    )


__all__ = ["analyze_single_asset", "fetch_single_asset"]
