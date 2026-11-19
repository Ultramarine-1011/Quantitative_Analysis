"""Service layer for multi-asset comparison (``quant compare``) and portfolio research (``quant portfolio``)."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from analysis_metrics import (
    ReturnConvention,
    compute_benchmark_relative_metrics,
    compute_calmar_ratio,
    compute_cumulative_return,
    compute_return_distribution_stats,
    compute_rolling_comparison_stats,
)
from asset_catalog import (
    SUPPORTED_ADJUSTMENTS,
    normalize_symbol_alias,
    resolve_adjustment,
    resolve_return_basis,
    resolve_semantic_series_kind,
)
from asset_loader import (
    MultiAssetAlignmentResult,
    align_multi_asset_frames,
    load_asset_data,
)
from contracts import (
    SCHEMA_VERSION,
    AssetSpec,
    CompareRequest,
    MultiAssetResult,
    PortfolioRequest,
    QualityReport,
    SeriesMetadata,
    build_multi_asset_metric_units,
    build_quality_report,
    build_series_metadata,
    sync_metadata_to_frame,
    validate_frame_metadata_consistency,
)
from currency import convert_ohlcv_to_base_currency, infer_asset_currency
from data_sources.errors import EmptySymbolDataError
from quant_models import (
    CovarianceEstimate,
    PortfolioSolution,
    backtest_portfolio,
    clean_joint_returns,
    compute_backtest_metrics,
    compute_risk_contributions,
    estimate_covariance_and_mean,
    normalize_close_to_base_one,
    run_walk_forward_backtest,
    solve_equal_weight_portfolio,
    solve_exact_efficient_frontier,
    solve_inverse_volatility_portfolio,
    solve_minimum_variance_portfolio,
    solve_risk_parity_portfolio,
    solve_tangency_portfolio,
    split_train_test_returns,
)
from services.analysis_service import AssetLoaderFn, _slice_to_requested_window

DateLike = Any

_COMPARE_METRIC_UNITS: dict[str, str] = build_multi_asset_metric_units()


def _dedupe_messages(*groups: Sequence[str]) -> tuple[str, ...]:
    merged: list[str] = []
    for group in groups:
        for item in group:
            text = str(item).strip()
            if text and text not in merged:
                merged.append(text)
    return tuple(merged)


def _invoke_asset_loader(
    fetch_fn: Callable[..., pd.DataFrame],
    spec: AssetSpec,
    start_date: str,
    end_date: str,
    *,
    proxy_url: str | None,
    force_refresh: bool,
) -> pd.DataFrame:
    """Invoke ``load_asset_data`` or a custom test/mock loader without breaking narrow signatures."""
    kwargs: dict[str, Any] = {
        "asset_type": spec.asset_type,
        "symbol": spec.symbol,
        "start_date": start_date,
        "end_date": end_date,
        "proxy_url": proxy_url,
    }
    try:
        sig = inspect.signature(fetch_fn)
        params = sig.parameters
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if (has_var_kw or "adjust" in params) and spec.asset_type in SUPPORTED_ADJUSTMENTS:
            kwargs["adjust"] = spec.adjust
        if (has_var_kw or "force_refresh" in params) and force_refresh:
            kwargs["force_refresh"] = True
    except (TypeError, ValueError):
        pass
    return fetch_fn(**kwargs)


def _load_and_slice_spec(
    spec: AssetSpec,
    start_date: str,
    end_date: str,
    *,
    fetch_fn: Callable[..., pd.DataFrame],
    proxy_url: str | None,
    force_refresh: bool,
) -> pd.DataFrame:
    loaded = _invoke_asset_loader(
        fetch_fn,
        spec,
        start_date,
        end_date,
        proxy_url=proxy_url,
        force_refresh=force_refresh,
    )
    if loaded is None or loaded.empty:
        raise EmptySymbolDataError(
            f"未找到标的 {spec.leg_name} 在 {start_date}~{end_date} 的有效数据。"
        )
    sliced = _slice_to_requested_window(
        loaded,
        asset_type=spec.asset_type,
        symbol=spec.symbol,
        start_date=start_date,
        end_date=end_date,
    )
    if sliced is loaded:
        sliced = loaded.copy()
        sliced.attrs = dict(loaded.attrs)

    sliced.attrs["asset_type"] = spec.asset_type
    requested_canonical = normalize_symbol_alias(spec.asset_type, spec.symbol)
    raw_attr_symbol = str(sliced.attrs.get("symbol") or "").strip()
    if raw_attr_symbol and (
        normalize_symbol_alias(spec.asset_type, raw_attr_symbol) == requested_canonical
    ):
        resolved_sym = raw_attr_symbol
    else:
        resolved_sym = spec.symbol
    sliced.attrs["symbol"] = resolved_sym
    has_attr_adjust = "adjust" in sliced.attrs and sliced.attrs.get("adjust") is not None
    raw_attr_adjust = str(sliced.attrs.get("adjust") or "").strip() if has_attr_adjust else None
    if raw_attr_adjust is not None:
        if spec.asset_type not in SUPPORTED_ADJUSTMENTS and raw_attr_adjust == "none":
            effective_adjust = ""
        elif spec.asset_type in SUPPORTED_ADJUSTMENTS:
            effective_adjust = resolve_adjustment(spec.asset_type, raw_attr_adjust)
        else:
            effective_adjust = raw_attr_adjust
    else:
        effective_adjust = spec.adjust
    sliced.attrs["adjust"] = effective_adjust
    basis = resolve_return_basis(
        spec.asset_type,
        adjust=effective_adjust if spec.asset_type in SUPPORTED_ADJUSTMENTS else None,
        attrs=sliced.attrs,
    )
    sliced.attrs["return_basis"] = basis
    sliced.attrs["series_kind"] = resolve_semantic_series_kind(
        spec.asset_type, return_basis=basis, attrs=sliced.attrs
    )
    sliced.attrs["canonical_symbol"] = normalize_symbol_alias(spec.asset_type, resolved_sym)
    if not sliced.attrs.get("currency"):
        sliced.attrs["currency"] = infer_asset_currency(spec.asset_type, spec.symbol, sliced.attrs)
    return sliced


def _fetch_multi_asset_frames(
    spec_map: Mapping[str, AssetSpec],
    start_date: str,
    end_date: str,
    *,
    loader: AssetLoaderFn | None,
    proxy_url: str | None,
    force_refresh: bool,
) -> dict[str, pd.DataFrame]:
    fetch_fn = loader or load_asset_data
    items = list(spec_map.items())

    if loader is None and len(items) > 1:
        with ThreadPoolExecutor(max_workers=min(4, len(items))) as pool:
            futs = [
                pool.submit(
                    _load_and_slice_spec,
                    spec,
                    start_date,
                    end_date,
                    fetch_fn=fetch_fn,
                    proxy_url=proxy_url,
                    force_refresh=force_refresh,
                )
                for _, spec in items
            ]
            return {col_key: fut.result() for (col_key, _), fut in zip(items, futs)}

    return {
        col_key: _load_and_slice_spec(
            spec,
            start_date,
            end_date,
            fetch_fn=fetch_fn,
            proxy_url=proxy_url,
            force_refresh=force_refresh,
        )
        for col_key, spec in items
    }


def _collect_metadata_and_quality(
    specs_by_key: Mapping[str, AssetSpec],
    raw_frames: Mapping[str, pd.DataFrame],
    aligned: MultiAssetAlignmentResult,
    *,
    trading_days: int | None,
) -> tuple[dict[str, SeriesMetadata], dict[str, QualityReport], bool, list[str]]:
    """Build per-asset metadata & quality reports and surface FX / calendar / local-currency caveats."""
    ccy_meta_root = dict(aligned.overlap_summary.get("currency_metadata", {}))
    fx_pairs_meta = dict(ccy_meta_root.get("fx_pairs", {}))
    assets_meta: dict[str, SeriesMetadata] = {}
    quality_map: dict[str, QualityReport] = {}
    extra_warnings: list[str] = []
    degraded = False

    for leg, spec in specs_by_key.items():
        frame = raw_frames[leg]
        q = build_quality_report(frame)
        base_meta = build_series_metadata(
            frame,
            asset_type=spec.asset_type,
            symbol=spec.symbol,
            adjust=str(frame.attrs.get("adjust", spec.adjust) or ""),
            return_basis=str(frame.attrs.get("return_basis") or ""),
            series_kind=str(frame.attrs.get("series_kind") or ""),
            output_currency=frame.attrs.get("currency"),
            trading_days=trading_days,
            strict=True,
        )
        sync_metadata_to_frame(frame, base_meta, q)
        validate_frame_metadata_consistency(frame, base_meta)
        leg_ccy_meta = dict(fx_pairs_meta.get(leg, {}))
        native_ccy = (
            str(
                leg_ccy_meta.get("native_currency")
                or base_meta.native_currency
                or aligned.asset_currencies.get(leg, "")
            ).strip()
            or None
        )
        output_ccy = (
            str(
                leg_ccy_meta.get("output_currency")
                or (native_ccy if aligned.base_currency == "LOCAL" else aligned.base_currency)
            ).strip()
            or None
        )
        pairs_used = leg_ccy_meta.get("pairs_used") or ()
        is_identity = leg_ccy_meta.get("source") == "identity" or (
            native_ccy is not None and output_ccy is not None and native_ccy == output_ccy
        )
        inferred_converted = bool(
            aligned.base_currency != "LOCAL"
            and not is_identity
            and (pairs_used or leg_ccy_meta.get("usdt_peg_assumed") or (native_ccy and output_ccy))
        )
        fx_converted = bool(
            leg_ccy_meta.get("fx_converted", base_meta.fx_converted or inferred_converted)
        )
        inferred_pair = (
            " * ".join(str(p) for p in pairs_used)
            if pairs_used
            else (
                f"{native_ccy}/{output_ccy}" if fx_converted and native_ccy and output_ccy else ""
            )
        )
        fx_pair = (
            str(leg_ccy_meta.get("fx_pair") or base_meta.fx_pair or inferred_pair).strip() or None
        )

        assets_meta[leg] = SeriesMetadata(
            asset_type=base_meta.asset_type,
            symbol=base_meta.symbol,
            canonical_symbol=base_meta.canonical_symbol,
            adjust=base_meta.adjust,
            return_basis=base_meta.return_basis,
            series_kind=base_meta.series_kind,
            native_currency=native_ccy,
            output_currency=output_ccy,
            fx_converted=fx_converted,
            fx_pair=fx_pair,
            trading_days=base_meta.trading_days,
            data_source=base_meta.data_source,
            fetched_at=base_meta.fetched_at,
            cache_hit=base_meta.cache_hit,
            warnings=base_meta.warnings,
        )
        quality_map[leg] = q
        if q.dropped_invalid_rows > 0 or q.filled_price_rows > 0 or base_meta.warnings:
            degraded = True

    distinct_local_ccys = {
        str(ccy).strip().upper() for ccy in aligned.asset_currencies.values() if str(ccy).strip()
    }
    mixed_local = aligned.base_currency == "LOCAL" and len(distinct_local_ccys) > 1
    if mixed_local:
        degraded = True
        extra_warnings.append(
            f"当前处于 base_currency='LOCAL' 且标的跨多种计价货币（{sorted(distinct_local_ccys)}）；"
            "跨币种名义收益汇总或相关对比未包含汇率折算（等效于假设零成本完全对冲），请改用 CNY/USD 统一计价口径。"
        )
    if any("fallback" in w.lower() or "peg" in w.lower() or "裁剪" in w for w in aligned.warnings):
        degraded = True

    return assets_meta, quality_map, (degraded or mixed_local), extra_warnings


def _compute_aligned_asset_metrics(
    price_series: pd.Series,
    period_returns: pd.Series,
    simple_returns: pd.Series,
    *,
    periods_per_year: float,
    risk_free_rate: float,
    return_convention: ReturnConvention,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Compute risk/return metrics cleanly from aligned valuation prices and concurrent returns."""
    reasons: dict[str, str] = {}
    clean_p = pd.to_numeric(price_series, errors="coerce").dropna()
    clean_r = pd.to_numeric(period_returns, errors="coerce").dropna()
    clean_s = pd.to_numeric(simple_returns, errors="coerce").dropna()

    cum_ret = compute_cumulative_return(clean_p)
    elapsed_days = float((clean_p.index[-1] - clean_p.index[0]).days) if len(clean_p) >= 2 else 0.0
    if len(clean_p) >= 2 and elapsed_days > 0.0 and (clean_p > 0.0).all():
        cagr = float(
            (float(clean_p.iloc[-1]) / float(clean_p.iloc[0])) ** (365.25 / elapsed_days) - 1.0
        )
    else:
        cagr = float("nan")
        reasons["cagr"] = "insufficient_observations"

    if len(clean_p) >= 2 and (clean_p > 0.0).all():
        running_max = clean_p.cummax()
        max_dd = float((clean_p / running_max - 1.0).min())
    else:
        max_dd = float("nan")
        reasons["max_drawdown"] = "insufficient_observations"

    calmar = compute_calmar_ratio(cagr, max_dd)
    if not np.isfinite(calmar):
        reasons["calmar_ratio"] = (
            "zero_max_drawdown"
            if (np.isfinite(cagr) and abs(max_dd) <= 0.0)
            else "cagr_unavailable"
        )

    ann_vol = float("nan")
    sharpe = float("nan")
    sortino = float("nan")
    if len(clean_r) < 2:
        reasons["annualized_volatility"] = "insufficient_observations"
        reasons["sharpe_ratio"] = "insufficient_observations"
        reasons["sortino_ratio"] = "insufficient_observations"
    else:
        std_r = float(clean_r.std(ddof=1))
        ann_vol = float(std_r * math.sqrt(float(periods_per_year)))
        period_rf = (
            float(risk_free_rate) / float(periods_per_year)
            if return_convention == "simple"
            else float(np.log1p(float(risk_free_rate))) / float(periods_per_year)
        )
        ex = clean_r - period_rf
        if std_r > 1e-15:
            sharpe = float(math.sqrt(float(periods_per_year)) * float(ex.mean()) / std_r)
        else:
            reasons["sharpe_ratio"] = "zero_volatility"
        downside = np.minimum(ex.to_numpy(dtype=float), 0.0)
        downside_dev = float(np.sqrt(np.mean(downside * downside)))
        if np.isfinite(downside_dev) and downside_dev > 1e-15:
            sortino = float(math.sqrt(float(periods_per_year)) * float(ex.mean()) / downside_dev)
        else:
            reasons["sortino_ratio"] = "zero_downside_deviation"

    if clean_s.empty:
        var_95, cvar_95 = float("nan"), float("nan")
        reasons["var_95"] = "insufficient_observations"
        reasons["cvar_95"] = "insufficient_observations"
    else:
        arr_s = clean_s.to_numpy(dtype=float)
        var_95 = float(np.quantile(arr_s, 0.05))
        tail = arr_s[arr_s <= var_95]
        cvar_95 = float(np.mean(tail)) if tail.size > 0 else var_95

    metrics = {
        "cumulative_return": cum_ret,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "start_price_base": float(clean_p.iloc[0]) if not clean_p.empty else float("nan"),
        "latest_price_base": float(clean_p.iloc[-1]) if not clean_p.empty else float("nan"),
        "normalized_end_value": float(clean_p.iloc[-1] / clean_p.iloc[0])
        if len(clean_p) >= 1 and float(clean_p.iloc[0]) > 0.0
        else float("nan"),
        "valuation_points": int(len(clean_p)),
        "return_periods": int(len(clean_r)),
    }
    return metrics, reasons


def compare_assets(
    request: CompareRequest,
    *,
    loader: AssetLoaderFn | None = None,
    fx_loader: Callable[[str, DateLike, DateLike], pd.DataFrame] | None = None,
    usdt_usd_loader: Callable[[DateLike, DateLike, str | None], pd.DataFrame] | None = None,
) -> MultiAssetResult:
    """Execute aligned multi-asset retrieval, statistical comparison, correlation/covariance & rolling analysis."""
    spec_map: dict[str, AssetSpec] = {spec.leg_name: spec for spec in request.assets}
    benchmark_col: str | None = None
    bench_spec = request.benchmark
    if bench_spec is not None:
        matched = next(
            (
                spec
                for spec in request.assets
                if spec.asset_type == bench_spec.asset_type
                and spec.canonical_symbol == bench_spec.canonical_symbol
                and spec.adjust == bench_spec.adjust
            ),
            None,
        )
        if matched is not None:
            benchmark_col = matched.leg_name
        else:
            benchmark_col = (
                bench_spec.leg_name
                if bench_spec.leg_name not in spec_map
                else f"benchmark:{bench_spec.leg_name}"
            )
            spec_map[benchmark_col] = bench_spec

    raw_frames = _fetch_multi_asset_frames(
        spec_map,
        request.start_date,
        request.end_date,
        loader=loader,
        proxy_url=request.proxy_url,
        force_refresh=request.force_refresh,
    )

    aligned = align_multi_asset_frames(
        raw_frames,
        calendar_mode=request.calendar_mode,
        base_currency=request.base_currency,
        fx_loader=fx_loader,
        usdt_usd_loader=usdt_usd_loader,
        proxy_url=request.proxy_url,
        allow_usdt_usd_peg=request.allow_usdt_peg,
    )

    periods_per_year = (
        float(request.trading_days)
        if request.trading_days is not None
        else float(aligned.effective_periods_per_year)
    )
    ann_days = max(int(round(periods_per_year)), 1)

    normalized_prices = pd.DataFrame(
        {
            str(col): normalize_close_to_base_one(aligned.base_prices, price_col=str(col))
            for col in aligned.base_prices.columns
        },
        index=aligned.base_prices.index,
    )
    corr_simple = aligned.correlation_returns_base
    if request.return_convention == "log":
        corr_ret_conv = pd.DataFrame(
            np.log1p(corr_simple.where(corr_simple > -1.0)),
            index=corr_simple.index,
            columns=corr_simple.columns,
        )
        nav_ret_conv = pd.DataFrame(
            np.log1p(aligned.returns_base.where(aligned.returns_base > -1.0)),
            index=aligned.returns_base.index,
            columns=aligned.returns_base.columns,
        )
    else:
        corr_ret_conv = corr_simple
        nav_ret_conv = aligned.returns_base

    assets_meta, quality_map, degraded, extra_warnings = _collect_metadata_and_quality(
        spec_map,
        raw_frames,
        aligned,
        trading_days=request.trading_days,
    )

    per_asset_metrics: dict[str, Any] = {}
    metric_reasons: dict[str, Any] = {}
    dist_stats: dict[str, Any] = {}

    for col in aligned.base_prices.columns:
        col_name = str(col)
        m, r = _compute_aligned_asset_metrics(
            aligned.base_prices[col_name],
            corr_ret_conv[col_name],
            corr_simple[col_name],
            periods_per_year=periods_per_year,
            risk_free_rate=request.risk_free_rate,
            return_convention=request.return_convention,
        )
        d = compute_return_distribution_stats(corr_ret_conv[col_name])
        per_asset_metrics[col_name] = m
        dist_stats[col_name] = d
        if r:
            metric_reasons[col_name] = r

    pearson_corr = corr_ret_conv.corr(method="pearson")
    spearman_corr = corr_ret_conv.corr(method="spearman")

    min_hist_cov = min(30, max(5, len(corr_simple)))
    if len(corr_simple) < 30:
        degraded = True
        extra_warnings.append(
            f"共同有效收益率样本仅 {len(corr_simple)} 期（建议 >= 30 期），相关性/协方差矩阵及正态性检验稳定性有限。"
        )
    cov_est = estimate_covariance_and_mean(
        corr_simple,
        cov_method=request.cov_method,
        trading_days=ann_days,
        min_history=min_hist_cov,
    )
    cov_df = pd.DataFrame(
        cov_est.cov_matrix, index=cov_est.asset_names, columns=cov_est.asset_names
    )

    rolling_df, rolling_summary = compute_rolling_comparison_stats(
        corr_ret_conv,
        window=request.rolling_window,
        periods_per_year=periods_per_year,
        risk_free_rate=request.risk_free_rate,
        benchmark_col=benchmark_col,
        return_convention=request.return_convention,
    )

    benchmark_info: dict[str, Any] | None = None
    relative_to_benchmark: dict[str, Any] = {}
    if benchmark_col is not None and benchmark_col in corr_ret_conv.columns:
        bench_series = corr_ret_conv[benchmark_col]
        for col in [s.leg_name for s in request.assets]:
            if col == benchmark_col:
                continue
            relative_to_benchmark[col] = compute_benchmark_relative_metrics(
                corr_ret_conv[col],
                bench_series,
                periods_per_year=periods_per_year,
                risk_free_rate=request.risk_free_rate,
                return_convention=request.return_convention,
            )
        benchmark_info = {
            "leg_name": benchmark_col,
            "asset": assets_meta[benchmark_col].to_dict(),
            "metrics": per_asset_metrics[benchmark_col],
            "relative_comparison": relative_to_benchmark,
        }

    # Best/worst cross-asset summary for rapid Agent consumption
    asset_cols = [s.leg_name for s in request.assets]
    ranked_by_sharpe = sorted(
        asset_cols,
        key=lambda c: (
            per_asset_metrics[c]["sharpe_ratio"]
            if np.isfinite(per_asset_metrics[c]["sharpe_ratio"])
            else -1e9
        ),
        reverse=True,
    )
    ranked_by_cagr = sorted(
        asset_cols,
        key=lambda c: (
            per_asset_metrics[c]["cagr"] if np.isfinite(per_asset_metrics[c]["cagr"]) else -1e9
        ),
        reverse=True,
    )

    pairwise_corr_pairs: list[dict[str, Any]] = []
    for i in range(len(asset_cols)):
        for j in range(i + 1, len(asset_cols)):
            a, b = asset_cols[i], asset_cols[j]
            pairwise_corr_pairs.append(
                {
                    "asset_a": a,
                    "asset_b": b,
                    "pearson": float(pearson_corr.loc[a, b]),
                    "spearman": float(spearman_corr.loc[a, b]),
                }
            )

    warnings_tuple = _dedupe_messages(aligned.warnings, extra_warnings)
    disclaimers = (
        "所有多资产比较指标已对齐至同一估值日历与计价基准货币；收益率序列仅代表历史统计特征，不构成未来收益保证。",
    )
    distinct_local_ccys = {
        str(v).strip().upper() for v in aligned.asset_currencies.values() if str(v).strip()
    }
    methodology = {
        "calendar_mode": aligned.calendar_mode,
        "base_currency": aligned.base_currency,
        "mixed_local_currencies": bool(
            aligned.base_currency == "LOCAL" and len(distinct_local_ccys) > 1
        ),
        "return_convention": request.return_convention,
        "cov_method": request.cov_method,
        "rolling_window": request.rolling_window,
        "effective_periods_per_year": float(periods_per_year),
        "backtest_periods_per_year": float(aligned.backtest_periods_per_year),
        "mtm_correlation_uses_concurrent_quotes": bool(aligned.calendar_mode == "mtm_outer"),
        "normalized_base": 1.0,
    }
    data_summary = {
        "requested_start": request.start_date,
        "requested_end": request.end_date,
        "effective_start": aligned.overlap_summary.get("effective_start", ""),
        "effective_end": aligned.overlap_summary.get("effective_end", ""),
        "valuation_points": int(len(aligned.base_prices)),
        "return_periods": int(len(aligned.returns_base)),
        "concurrent_return_periods": int(len(corr_simple)),
        "raw_counts": aligned.overlap_summary.get("raw_counts", {}),
        "raw_ranges": aligned.overlap_summary.get("raw_ranges", {}),
        "dropped_non_overlap_days_by_asset": aligned.overlap_summary.get(
            "dropped_non_overlap_days_by_asset", {}
        ),
    }

    metrics: dict[str, Any] = {
        "per_asset": per_asset_metrics,
        "distribution": dist_stats,
        "correlation": {
            "pearson": pearson_corr.to_dict(),
            "spearman": spearman_corr.to_dict(),
            "pairs": pairwise_corr_pairs,
        },
        "covariance": {
            "method": cov_est.cov_method,
            "shrinkage": cov_est.shrinkage,
            "condition_number": cov_est.condition_number,
            "min_eigenvalue": cov_est.min_eigenvalue,
            "ridge_applied": cov_est.ridge_applied,
            "matrix": cov_df.to_dict(),
        },
        "rolling_summary": rolling_summary,
        "rankings": {
            "by_sharpe": ranked_by_sharpe,
            "by_cagr": ranked_by_cagr,
        },
    }
    if benchmark_info is not None:
        metrics["benchmark_comparison"] = benchmark_info["relative_comparison"]

    frames: dict[str, pd.DataFrame] = {
        "prices": aligned.base_prices,
        "returns": nav_ret_conv,
        "correlation": pearson_corr,
        "covariance": cov_df,
        "rolling_metrics": rolling_df,
    }

    return MultiAssetResult(
        schema_version=SCHEMA_VERSION,
        command="compare",
        status="degraded" if degraded else "ok",
        asset_order=tuple(asset_cols),
        assets=assets_meta,
        quality=quality_map,
        data_summary=data_summary,
        parameters=request.to_dict(),
        methodology=methodology,
        metrics=metrics,
        metric_units=_COMPARE_METRIC_UNITS,
        metric_reasons=metric_reasons,
        benchmark=benchmark_info,
        warnings=warnings_tuple,
        disclaimers=disclaimers,
        frames=frames,
        data_frame=normalized_prices,
    )


def _solve_portfolio_snapshot(
    returns_window: pd.DataFrame,
    *,
    strategy: str,
    cov_method: str,
    risk_free_rate: float,
    trading_days: int,
    min_weight: float,
    max_weight: float,
    custom_weights: tuple[float, ...] | None,
) -> tuple[PortfolioSolution, CovarianceEstimate]:
    min_hist = min(30, max(5, len(returns_window)))
    est = estimate_covariance_and_mean(
        returns_window,
        cov_method=cov_method,  # type: ignore[arg-type]
        trading_days=trading_days,
        min_history=min_hist,
    )
    if strategy == "custom":
        assert custom_weights is not None
        w = np.asarray(custom_weights, dtype=float)
        ret = float(est.mean_returns @ w)
        vol = math.sqrt(max(float(w @ est.cov_matrix @ w), 0.0))
        sharpe = (ret - float(risk_free_rate)) / vol if vol > 0.0 else float("nan")
        sol = PortfolioSolution(
            weights=w,
            expected_return=ret,
            volatility=vol,
            sharpe=sharpe,
            converged=True,
            solver_message="custom_exogenous_weights",
        )
        return sol, est
    if strategy == "equal_weight":
        sol = solve_equal_weight_portfolio(
            returns_window,
            cov_method=cov_method,  # type: ignore[arg-type]
            risk_free_rate=risk_free_rate,
            trading_days=trading_days,
            min_history=min_hist,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        return sol, est
    if strategy == "inverse_volatility":
        sol = solve_inverse_volatility_portfolio(
            returns_window,
            cov_method=cov_method,  # type: ignore[arg-type]
            risk_free_rate=risk_free_rate,
            trading_days=trading_days,
            min_history=min_hist,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        return sol, est
    if strategy == "risk_parity":
        sol = solve_risk_parity_portfolio(
            returns_window,
            cov_method=cov_method,  # type: ignore[arg-type]
            risk_free_rate=risk_free_rate,
            trading_days=trading_days,
            min_history=min_hist,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        return sol, est
    if strategy == "min_variance":
        sol = solve_minimum_variance_portfolio(
            returns_window,
            cov_method=cov_method,  # type: ignore[arg-type]
            risk_free_rate=risk_free_rate,
            trading_days=trading_days,
            min_history=min_hist,
            min_weight=min_weight,
            max_weight=max_weight,
        )
        return sol, est
    sol = solve_tangency_portfolio(
        returns_window,
        cov_method=cov_method,  # type: ignore[arg-type]
        risk_free_rate=risk_free_rate,
        trading_days=trading_days,
        min_history=min_hist,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    return sol, est


def _align_benchmark_to_oos_nav(
    bench_spec: AssetSpec,
    bench_raw: pd.DataFrame,
    target_nav_df: pd.DataFrame,
    strategy_col: str,
    *,
    base_currency: str,
    allow_usdt_peg: bool,
    fx_loader: Callable[[str, DateLike, DateLike], pd.DataFrame] | None,
    usdt_usd_loader: Callable[[DateLike, DateLike, str | None], pd.DataFrame] | None,
    proxy_url: str | None,
    periods_per_year: float,
    risk_free_rate: float,
) -> tuple[pd.Series, dict[str, Any], SeriesMetadata, QualityReport, list[str]]:
    """Align an external benchmark onto the exact OOS NAV timeline & currency without lookahead."""
    b_attrs = getattr(bench_raw, "attrs", {}) or {}
    b_kind = str(b_attrs.get("series_kind", "price") or "price").strip().lower()
    b_basis = str(b_attrs.get("return_basis", "") or "").strip().lower()
    if (
        b_kind == "yield_level"
        or b_basis == "yield_level"
        or bench_spec.asset_type == "bond_us_yield"
    ):
        raise ValueError(
            f"基准标的 {bench_spec.leg_name!r} 为收益率水平序列（yield_level），不是总收益/价格序列，不能作为组合净值对照基准。"
        )
    if (b_kind == "fx_rate" or b_basis == "fx_rate" or bench_spec.asset_type == "fx") and str(
        base_currency
    ).strip().upper() != "LOCAL":
        raise ValueError(
            f"基准标的 {bench_spec.leg_name!r} 为汇率比值序列（fx_rate），不支持按 {base_currency} 基准货币折算。"
        )

    converted_bench = convert_ohlcv_to_base_currency(
        bench_raw,
        bench_spec.asset_type,
        bench_spec.symbol,
        base_currency,
        proxy_url=proxy_url,
        allow_usdt_peg=allow_usdt_peg,
        fx_loader=fx_loader,
        usdt_usd_loader=usdt_usd_loader,
    )
    bench_idx = pd.DatetimeIndex(pd.to_datetime(converted_bench.index, errors="raise"))
    if bench_idx.tz is not None:
        bench_idx = bench_idx.tz_localize(None)
    bench_prices = (
        pd.Series(
            pd.to_numeric(converted_bench["Close"], errors="coerce").to_numpy(dtype=float),
            index=bench_idx.as_unit("ns"),
            name=bench_spec.leg_name,
        )
        .sort_index(kind="stable")
        .dropna()
    )
    bench_prices = bench_prices.loc[~bench_prices.index.duplicated(keep="last")]
    if (bench_prices <= 0.0).any():
        raise ValueError(f"基准标的 {bench_spec.leg_name!r} 包含非正收盘价，无法构建对比净值。")

    target_idx = pd.DatetimeIndex(target_nav_df.index)
    first_bench_dt = bench_prices.index[0]
    last_bench_dt = bench_prices.index[-1]
    # Forward-fill benchmark prices strictly within [first_bench_dt, last_bench_dt] (no backfill or tail extrapolation).
    union_idx = bench_prices.index.union(target_idx).sort_values()
    bench_ff = bench_prices.reindex(union_idx).ffill().reindex(target_idx)
    bench_ff = bench_ff.where((target_idx >= first_bench_dt) & (target_idx <= last_bench_dt))
    valid_bench = bench_ff.dropna()
    if len(valid_bench) < 2:
        raise ValueError(
            f"基准标的 {bench_spec.leg_name} 在回测验证区间内有效报价不足 2 条，无法对齐比较。"
        )

    full_bench_nav = pd.Series(float("nan"), index=target_idx, name="Benchmark", dtype="float64")
    anchor_val = float(valid_bench.iloc[0])
    full_bench_nav.loc[valid_bench.index] = valid_bench / anchor_val

    # Slice both benchmark and strategy onto the exact overlapping OOS span for apples-to-apples stats.
    overlap_idx = valid_bench.index
    sub_strat_nav = target_nav_df.loc[overlap_idx, strategy_col].astype("float64")
    sub_strat_nav = sub_strat_nav / float(sub_strat_nav.iloc[0])
    sub_bench_nav = full_bench_nav.loc[overlap_idx]

    strat_ret = sub_strat_nav.pct_change().dropna()
    bench_ret = sub_bench_nav.pct_change().dropna()

    bench_perf = compute_backtest_metrics(
        sub_bench_nav,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        return_convention="simple",
    )
    rel_vs_bench = compute_benchmark_relative_metrics(
        strat_ret,
        bench_ret,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        return_convention="simple",
    )

    extra_warnings = [str(w) for w in (converted_bench.attrs.get("fx_warnings") or ())]
    if len(overlap_idx) < len(target_idx):
        extra_warnings.append(
            f"外部基准 {bench_spec.leg_name} 仅覆盖样本外区间 {overlap_idx[0].strftime('%Y-%m-%d')}~"
            f"{overlap_idx[-1].strftime('%Y-%m-%d')}（共 {len(overlap_idx)}/{len(target_idx)} 个点），基准比较已裁至精确重叠子区间。"
        )

    b_meta = build_series_metadata(
        converted_bench,
        asset_type=bench_spec.asset_type,
        symbol=bench_spec.symbol,
        adjust=str(converted_bench.attrs.get("adjust", bench_spec.adjust) or ""),
        return_basis=str(converted_bench.attrs.get("return_basis") or ""),
        series_kind=str(converted_bench.attrs.get("series_kind") or ""),
        output_currency=converted_bench.attrs.get("currency"),
        strict=True,
    )
    b_quality = build_quality_report(converted_bench)
    sync_metadata_to_frame(converted_bench, b_meta, b_quality)
    validate_frame_metadata_consistency(converted_bench, b_meta)

    bench_summary = {
        "leg_name": bench_spec.leg_name,
        "asset": b_meta.to_dict(),
        "effective_start": overlap_idx[0].strftime("%Y-%m-%d"),
        "effective_end": overlap_idx[-1].strftime("%Y-%m-%d"),
        "valuation_points": int(len(overlap_idx)),
        "performance": bench_perf,
        "relative_to_strategy": rel_vs_bench,
    }
    return full_bench_nav, bench_summary, b_meta, b_quality, extra_warnings


def _run_portfolio_validation_case(
    clean_r: pd.DataFrame,
    tradable_mask: pd.DataFrame,
    *,
    strategy: str,
    validation_mode: str,
    cov_method: str,
    rebalance_freq: str,
    tx_bps: float,
    risk_free_rate: float,
    min_weight: float,
    max_weight: float,
    train_ratio: float,
    split_date: str | None,
    train_window: int,
    custom_weights: tuple[float, ...] | None,
    periods_per_year: float,
    min_train_rows: int,
    min_test_rows: int,
) -> dict[str, Any]:
    """Run one validation/backtest configuration cleanly (shared by primary run and sensitivity grid)."""
    ann_days = max(int(round(periods_per_year)), 1)
    if validation_mode == "in_sample":
        sol, est = _solve_portfolio_snapshot(
            clean_r,
            strategy=strategy,
            cov_method=cov_method,
            risk_free_rate=risk_free_rate,
            trading_days=ann_days,
            min_weight=min_weight,
            max_weight=max_weight,
            custom_weights=custom_weights,
        )
        bt = backtest_portfolio(
            clean_r,
            sol.weights,
            min_history=min(30, len(clean_r)),
            rebalance_freq=rebalance_freq,  # type: ignore[arg-type]
            transaction_cost_bps=tx_bps,
            tradable_mask=tradable_mask,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        )
        perf = dict(bt.attrs["performance"]["Optimal_Portfolio"])
        return {
            "performance": perf,
            "condition_number": est.condition_number,
            "nav": bt["Optimal_Portfolio"],
        }

    if validation_mode == "train_test":
        if strategy == "custom":
            assert custom_weights is not None
            # Reuse exact train/test split boundaries then evaluate fixed weights strictly on OOS.
            eq_split = split_train_test_returns(
                clean_r,
                train_ratio=train_ratio,
                split_date=split_date,
                min_train_rows=min_train_rows,
                min_test_rows=min_test_rows,
                objective="equal_weight",
                cov_method=cov_method,  # type: ignore[arg-type]
                risk_free_rate=risk_free_rate,
                min_weight=0.0,
                max_weight=1.0,
                rebalance_freq=rebalance_freq,  # type: ignore[arg-type]
                transaction_cost_bps=tx_bps,
                tradable_mask=tradable_mask,
                periods_per_year=periods_per_year,
            )
            custom_arr = np.asarray(custom_weights, dtype=float)
            oos_bt = backtest_portfolio(
                eq_split.test_returns,
                custom_arr,
                min_history=min_test_rows,
                rebalance_freq=rebalance_freq,  # type: ignore[arg-type]
                transaction_cost_bps=tx_bps,
                tradable_mask=tradable_mask,
                base_date=eq_split.train_returns.index[-1]
                if isinstance(clean_r.index, pd.DatetimeIndex)
                else None,
                periods_per_year=periods_per_year,
                risk_free_rate=risk_free_rate,
            )
            perf = dict(oos_bt.attrs["performance"]["Optimal_Portfolio"])
            est = estimate_covariance_and_mean(
                eq_split.train_returns,
                cov_method=cov_method,  # type: ignore[arg-type]
                trading_days=ann_days,
                min_history=min(30, max(5, len(eq_split.train_returns))),
            )
            return {
                "performance": perf,
                "condition_number": est.condition_number,
                "nav": oos_bt["Optimal_Portfolio"],
            }

        res = split_train_test_returns(
            clean_r,
            train_ratio=train_ratio,
            split_date=split_date,
            min_train_rows=min_train_rows,
            min_test_rows=min_test_rows,
            objective=strategy,  # type: ignore[arg-type]
            cov_method=cov_method,  # type: ignore[arg-type]
            risk_free_rate=risk_free_rate,
            min_weight=min_weight,
            max_weight=max_weight,
            rebalance_freq=rebalance_freq,  # type: ignore[arg-type]
            transaction_cost_bps=tx_bps,
            tradable_mask=tradable_mask,
            periods_per_year=periods_per_year,
        )
        est = estimate_covariance_and_mean(
            res.train_returns,
            cov_method=cov_method,  # type: ignore[arg-type]
            trading_days=ann_days,
            min_history=min(30, max(5, len(res.train_returns))),
        )
        perf = dict(res.out_of_sample_performance["Optimal_Portfolio"])
        return {
            "performance": perf,
            "condition_number": est.condition_number,
            "nav": res.out_of_sample_backtest["Optimal_Portfolio"],
        }

    wf_mode = "rolling" if validation_mode == "walk_forward_rolling" else "expanding"
    wf = run_walk_forward_backtest(
        clean_r,
        window_mode=wf_mode,
        train_window=train_window,
        rebalance_freq=rebalance_freq,  # type: ignore[arg-type]
        cov_method=cov_method,  # type: ignore[arg-type]
        objective=strategy,  # type: ignore[arg-type]
        risk_free_rate=risk_free_rate,
        min_weight=min_weight,
        max_weight=max_weight,
        transaction_cost_bps=tx_bps,
        periods_per_year=periods_per_year,
        tradable_mask=tradable_mask,
        min_test_rows=min(5, max(2, len(clean_r) - train_window)),
    )
    init_est = estimate_covariance_and_mean(
        clean_r.iloc[:train_window],
        cov_method=cov_method,  # type: ignore[arg-type]
        trading_days=ann_days,
        min_history=min(30, max(5, train_window)),
    )
    perf = dict(wf.performance["OOS_Portfolio"])
    return {
        "performance": perf,
        "condition_number": init_est.condition_number,
        "nav": wf.nav_df["OOS_Portfolio"],
    }


def _compute_sensitivity_tables(
    request: PortfolioRequest,
    clean_r: pd.DataFrame,
    tradable_mask: pd.DataFrame,
    *,
    periods_per_year: float,
    min_train_rows: int,
    min_test_rows: int,
) -> dict[str, list[dict[str, Any]]]:
    cov_rows: list[dict[str, Any]] = []
    cov_methods = (
        ("ledoit_wolf", "sample", "diagonal")
        if request.strategy not in ("equal_weight", "custom")
        else (request.cov_method,)
    )
    for cm in cov_methods:
        try:
            case = _run_portfolio_validation_case(
                clean_r,
                tradable_mask,
                strategy=request.strategy,
                validation_mode=request.validation_mode,
                cov_method=cm,
                rebalance_freq=request.rebalance_freq,
                tx_bps=request.transaction_cost_bps,
                risk_free_rate=request.risk_free_rate,
                min_weight=request.min_weight,
                max_weight=request.max_weight,
                train_ratio=request.train_ratio,
                split_date=request.split_date,
                train_window=request.train_window,
                custom_weights=request.custom_weights,
                periods_per_year=periods_per_year,
                min_train_rows=min_train_rows,
                min_test_rows=min_test_rows,
            )
            p = case["performance"]
            cov_rows.append(
                {
                    "cov_method": cm,
                    "status": "ok",
                    "condition_number": case["condition_number"],
                    "cagr": p.get("cagr"),
                    "annualized_volatility": p.get("annualized_volatility"),
                    "sharpe_ratio": p.get("sharpe_ratio"),
                    "max_drawdown": p.get("max_drawdown"),
                    "total_turnover": p.get("total_turnover"),
                    "total_cost_ratio": p.get("total_cost_ratio"),
                }
            )
        except Exception as exc:
            cov_rows.append({"cov_method": cm, "status": "error", "error": str(exc)})

    bps_grid = sorted({0.0, 5.0, 10.0, 20.0, 50.0, float(request.transaction_cost_bps)})
    tx_rows: list[dict[str, Any]] = []
    for bps in bps_grid:
        try:
            case = _run_portfolio_validation_case(
                clean_r,
                tradable_mask,
                strategy=request.strategy,
                validation_mode=request.validation_mode,
                cov_method=request.cov_method,
                rebalance_freq=request.rebalance_freq,
                tx_bps=bps,
                risk_free_rate=request.risk_free_rate,
                min_weight=request.min_weight,
                max_weight=request.max_weight,
                train_ratio=request.train_ratio,
                split_date=request.split_date,
                train_window=request.train_window,
                custom_weights=request.custom_weights,
                periods_per_year=periods_per_year,
                min_train_rows=min_train_rows,
                min_test_rows=min_test_rows,
            )
            p = case["performance"]
            tx_rows.append(
                {
                    "transaction_cost_bps": float(bps),
                    "status": "ok",
                    "cumulative_return": p.get("cumulative_return"),
                    "cagr": p.get("cagr"),
                    "sharpe_ratio": p.get("sharpe_ratio"),
                    "max_drawdown": p.get("max_drawdown"),
                    "total_turnover": p.get("total_turnover"),
                    "total_cost_ratio": p.get("total_cost_ratio"),
                }
            )
        except Exception as exc:
            tx_rows.append(
                {"transaction_cost_bps": float(bps), "status": "error", "error": str(exc)}
            )

    freqs = ("buy_and_hold", "monthly", "quarterly")
    if request.rebalance_freq == "daily" or request.calendar_mode != "weekly_friday":
        freqs = ("buy_and_hold", "daily", "monthly", "quarterly")
    reb_rows: list[dict[str, Any]] = []
    for rf_freq in freqs:
        try:
            case = _run_portfolio_validation_case(
                clean_r,
                tradable_mask,
                strategy=request.strategy,
                validation_mode=request.validation_mode,
                cov_method=request.cov_method,
                rebalance_freq=rf_freq,
                tx_bps=request.transaction_cost_bps,
                risk_free_rate=request.risk_free_rate,
                min_weight=request.min_weight,
                max_weight=request.max_weight,
                train_ratio=request.train_ratio,
                split_date=request.split_date,
                train_window=request.train_window,
                custom_weights=request.custom_weights,
                periods_per_year=periods_per_year,
                min_train_rows=min_train_rows,
                min_test_rows=min_test_rows,
            )
            p = case["performance"]
            reb_rows.append(
                {
                    "rebalance_freq": rf_freq,
                    "status": "ok",
                    "cagr": p.get("cagr"),
                    "annualized_volatility": p.get("annualized_volatility"),
                    "sharpe_ratio": p.get("sharpe_ratio"),
                    "max_drawdown": p.get("max_drawdown"),
                    "total_turnover": p.get("total_turnover"),
                    "total_cost_ratio": p.get("total_cost_ratio"),
                }
            )
        except Exception as exc:
            reb_rows.append({"rebalance_freq": rf_freq, "status": "error", "error": str(exc)})

    return {
        "covariance_method": cov_rows,
        "transaction_cost_bps": tx_rows,
        "rebalance_freq": reb_rows,
    }


def analyze_portfolio(
    request: PortfolioRequest,
    *,
    loader: AssetLoaderFn | None = None,
    fx_loader: Callable[[str, DateLike, DateLike], pd.DataFrame] | None = None,
    usdt_usd_loader: Callable[[DateLike, DateLike, str | None], pd.DataFrame] | None = None,
) -> MultiAssetResult:
    """Execute causal multi-asset portfolio optimization, risk decomposition, efficient frontier & backtest."""
    universe_spec_map: dict[str, AssetSpec] = {spec.leg_name: spec for spec in request.assets}
    fetch_spec_map: dict[str, AssetSpec] = dict(universe_spec_map)
    bench_fetch_key: str | None = None
    if request.benchmark is not None:
        matched = next(
            (
                spec
                for spec in request.assets
                if spec.asset_type == request.benchmark.asset_type
                and spec.canonical_symbol == request.benchmark.canonical_symbol
                and spec.adjust == request.benchmark.adjust
            ),
            None,
        )
        if matched is not None:
            bench_fetch_key = matched.leg_name
        else:
            bench_fetch_key = (
                request.benchmark.leg_name
                if request.benchmark.leg_name not in fetch_spec_map
                else f"benchmark:{request.benchmark.leg_name}"
            )
            fetch_spec_map[bench_fetch_key] = request.benchmark

    raw_frames = _fetch_multi_asset_frames(
        fetch_spec_map,
        request.start_date,
        request.end_date,
        loader=loader,
        proxy_url=request.proxy_url,
        force_refresh=request.force_refresh,
    )
    universe_frames = {spec.leg_name: raw_frames[spec.leg_name] for spec in request.assets}

    aligned = align_multi_asset_frames(
        universe_frames,
        calendar_mode=request.calendar_mode,
        base_currency=request.base_currency,
        fx_loader=fx_loader,
        usdt_usd_loader=usdt_usd_loader,
        proxy_url=request.proxy_url,
        allow_usdt_usd_peg=request.allow_usdt_peg,
    )

    periods_per_year = (
        float(request.trading_days)
        if request.trading_days is not None
        else float(aligned.backtest_periods_per_year)
    )
    ann_days = max(int(round(periods_per_year)), 1)

    clean_r = clean_joint_returns(aligned.returns_base, min_history=10)
    tradable_mask = aligned.tradable_mask.reindex(clean_r.index).fillna(True)

    assets_meta, quality_map, degraded, extra_warnings = _collect_metadata_and_quality(
        universe_spec_map,
        raw_frames,
        aligned,
        trading_days=request.trading_days,
    )

    n_total = len(clean_r)
    if request.validation_mode == "train_test":
        min_train_rows = (
            30 if n_total >= 40 else max(5, int(math.floor(n_total * request.train_ratio)))
        )
        min_test_rows = 10 if n_total >= 40 else max(2, n_total - min_train_rows)
        if n_total < 40:
            degraded = True
            extra_warnings.append(
                f"联合收益样本仅 {n_total} 期（建议 >= 40 期），Train/Test 协方差与样本外区间统计稳定性有限。"
            )
    else:
        min_train_rows = min(30, max(5, n_total))
        min_test_rows = min(10, max(2, n_total))

    extra_frames: dict[str, pd.DataFrame] = {}
    weight_history_df: pd.DataFrame | None = None
    in_sample_sol_perf: dict[str, Any] | None = None
    train_test_summary: dict[str, Any] | None = None
    walk_forward_summary: dict[str, Any] | None = None
    has_lookahead_bias = False

    if request.validation_mode == "in_sample":
        est_returns = clean_r
        est_window_label = "full_sample_in_sample"
        primary_sol, cov_est = _solve_portfolio_snapshot(
            est_returns,
            strategy=request.strategy,
            cov_method=request.cov_method,
            risk_free_rate=request.risk_free_rate,
            trading_days=ann_days,
            min_weight=request.min_weight,
            max_weight=request.max_weight,
            custom_weights=request.custom_weights,
        )
        bt_df = backtest_portfolio(
            clean_r,
            primary_sol.weights,
            min_history=min(30, len(clean_r)),
            rebalance_freq=request.rebalance_freq,
            transaction_cost_bps=request.transaction_cost_bps,
            tradable_mask=tradable_mask,
            base_date=aligned.base_date,
            periods_per_year=periods_per_year,
            risk_free_rate=request.risk_free_rate,
        )
        backtest_df = bt_df.rename(columns={"Optimal_Portfolio": "Portfolio"})
        primary_col = "Portfolio"
        primary_perf = dict(bt_df.attrs["performance"]["Optimal_Portfolio"])
        eq_perf = dict(bt_df.attrs["performance"]["Equal_Weight_Portfolio"])
        if request.strategy != "custom":
            has_lookahead_bias = True
            extra_warnings.append(
                "当前运行于 in_sample 全样本回测模式：权重与收益共用同一全样本区间，存在显式前视偏差（lookahead bias），仅作事后参考。"
            )
        else:
            extra_warnings.append(
                "当前使用 exogenous custom 固定权重在全样本区间回测；若该权重基于事后观察手工选定，仍可能包含选择偏差。"
            )
    elif request.validation_mode == "train_test":
        if request.strategy == "custom":
            assert request.custom_weights is not None
            eq_split = split_train_test_returns(
                clean_r,
                train_ratio=request.train_ratio,
                split_date=request.split_date,
                min_train_rows=min_train_rows,
                min_test_rows=min_test_rows,
                objective="equal_weight",
                cov_method=request.cov_method,
                risk_free_rate=request.risk_free_rate,
                min_weight=0.0,
                max_weight=1.0,
                rebalance_freq=request.rebalance_freq,
                transaction_cost_bps=request.transaction_cost_bps,
                tradable_mask=tradable_mask,
                periods_per_year=periods_per_year,
            )
            train_r, test_r = eq_split.train_returns, eq_split.test_returns
            split_label = str(eq_split.split_date)
            custom_arr = np.asarray(request.custom_weights, dtype=float)
            is_bt = backtest_portfolio(
                train_r,
                custom_arr,
                min_history=min_train_rows,
                rebalance_freq=request.rebalance_freq,
                transaction_cost_bps=request.transaction_cost_bps,
                tradable_mask=tradable_mask,
                periods_per_year=periods_per_year,
                risk_free_rate=request.risk_free_rate,
            )
            oos_bt = backtest_portfolio(
                test_r,
                custom_arr,
                min_history=min_test_rows,
                rebalance_freq=request.rebalance_freq,
                transaction_cost_bps=request.transaction_cost_bps,
                tradable_mask=tradable_mask,
                base_date=train_r.index[-1]
                if isinstance(clean_r.index, pd.DatetimeIndex)
                else None,
                periods_per_year=periods_per_year,
                risk_free_rate=request.risk_free_rate,
            )
            in_sample_sol_perf = dict(is_bt.attrs["performance"]["Optimal_Portfolio"])
            primary_perf = dict(oos_bt.attrs["performance"]["Optimal_Portfolio"])
            eq_perf = dict(oos_bt.attrs["performance"]["Equal_Weight_Portfolio"])
            backtest_df = oos_bt.rename(columns={"Optimal_Portfolio": "OOS_Portfolio"})
        else:
            split_res = split_train_test_returns(
                clean_r,
                train_ratio=request.train_ratio,
                split_date=request.split_date,
                min_train_rows=min_train_rows,
                min_test_rows=min_test_rows,
                objective=request.strategy,
                cov_method=request.cov_method,
                risk_free_rate=request.risk_free_rate,
                min_weight=request.min_weight,
                max_weight=request.max_weight,
                rebalance_freq=request.rebalance_freq,
                transaction_cost_bps=request.transaction_cost_bps,
                tradable_mask=tradable_mask,
                periods_per_year=periods_per_year,
            )
            train_r, test_r = split_res.train_returns, split_res.test_returns
            split_label = str(split_res.split_date)
            in_sample_sol_perf = dict(split_res.in_sample_performance["Optimal_Portfolio"])
            primary_perf = dict(split_res.out_of_sample_performance["Optimal_Portfolio"])
            eq_perf = dict(split_res.out_of_sample_performance["Equal_Weight_Portfolio"])
            backtest_df = split_res.out_of_sample_backtest.rename(
                columns={"Optimal_Portfolio": "OOS_Portfolio"}
            )

        est_returns = train_r
        est_window_label = "train_split_strictly_pre_test"
        primary_sol, cov_est = _solve_portfolio_snapshot(
            est_returns,
            strategy=request.strategy,
            cov_method=request.cov_method,
            risk_free_rate=request.risk_free_rate,
            trading_days=ann_days,
            min_weight=request.min_weight,
            max_weight=request.max_weight,
            custom_weights=request.custom_weights,
        )
        primary_col = "OOS_Portfolio"
        train_test_summary = {
            "split_date": split_label,
            "train_start": train_r.index[0].strftime("%Y-%m-%d"),
            "train_end": train_r.index[-1].strftime("%Y-%m-%d"),
            "train_rows": int(len(train_r)),
            "test_start": test_r.index[0].strftime("%Y-%m-%d"),
            "test_end": test_r.index[-1].strftime("%Y-%m-%d"),
            "test_rows": int(len(test_r)),
            "in_sample_performance": in_sample_sol_perf,
            "out_of_sample_performance": primary_perf,
        }
    else:
        wf_mode = "rolling" if request.validation_mode == "walk_forward_rolling" else "expanding"
        if len(clean_r) <= request.train_window + 1:
            raise ValueError(
                f"样本总量（{len(clean_r)} 期）必须大于 train_window（{request.train_window}）+ 1 才能开展滚动样本外回测。"
            )
        wf_min_test = min(5, max(2, len(clean_r) - request.train_window))
        if request.train_window < 30:
            degraded = True
            extra_warnings.append(
                f"Walk-Forward 训练窗口仅 {request.train_window} 期（建议 >= 30 期），协方差收缩估计噪声可能较大。"
            )
        wf_res = run_walk_forward_backtest(
            clean_r,
            window_mode=wf_mode,
            train_window=request.train_window,
            rebalance_freq=request.rebalance_freq,
            cov_method=request.cov_method,
            objective=request.strategy,
            risk_free_rate=request.risk_free_rate,
            min_weight=request.min_weight,
            max_weight=request.max_weight,
            transaction_cost_bps=request.transaction_cost_bps,
            periods_per_year=periods_per_year,
            tradable_mask=tradable_mask,
            min_test_rows=wf_min_test,
        )
        weight_history_df = wf_res.weight_history
        extra_frames["weight_history"] = weight_history_df
        backtest_df = wf_res.nav_df.copy()
        primary_col = "OOS_Portfolio"
        primary_perf = dict(wf_res.performance["OOS_Portfolio"])
        eq_perf = dict(wf_res.performance["Equal_Weight_Portfolio"])
        in_sample_ref_perf = dict(wf_res.performance["In_Sample_Portfolio"])
        in_sample_ref_perf["hindsight_only"] = True
        in_sample_ref_perf["has_lookahead_bias"] = True

        # Causal window immediately preceding the latest rebalance date for latest weights & risk snapshot.
        last_reb_ts = pd.Timestamp(wf_res.rebalance_dates[-1])
        pos = int(clean_r.index.get_indexer([last_reb_ts])[0])
        w_start = 0 if wf_mode == "expanding" else max(0, pos - request.train_window)
        est_returns = clean_r.iloc[w_start:pos]
        est_window_label = f"latest_walk_forward_window_{wf_mode}"
        primary_sol, cov_est = _solve_portfolio_snapshot(
            est_returns,
            strategy=request.strategy,
            cov_method=request.cov_method,
            risk_free_rate=request.risk_free_rate,
            trading_days=ann_days,
            min_weight=request.min_weight,
            max_weight=request.max_weight,
            custom_weights=request.custom_weights,
        )
        walk_forward_summary = {
            "window_mode": wf_mode,
            "train_window": request.train_window,
            "rebalance_count": int(wf_res.performance["rebalance_count"]),
            "oos_start": clean_r.index[request.train_window].strftime("%Y-%m-%d"),
            "oos_end": clean_r.index[-1].strftime("%Y-%m-%d"),
            "oos_periods": int(len(clean_r) - request.train_window),
            "initial_weights": {
                str(c): float(v)
                for c, v in zip(clean_r.columns, weight_history_df.iloc[0].to_numpy(dtype=float))
            },
            "latest_weights": {
                str(c): float(v) for c, v in zip(clean_r.columns, primary_sol.weights)
            },
            "average_weights": {
                str(c): float(v)
                for c, v in zip(
                    clean_r.columns, weight_history_df.mean(axis=0).to_numpy(dtype=float)
                )
            },
            "in_sample_hindsight_reference": {
                **in_sample_ref_perf,
                "weights": {
                    str(c): float(v)
                    for c, v in zip(clean_r.columns, wf_res.performance["in_sample_weights"])
                },
            },
        }

    # Risk contributions (Euler decomposition) & comparison weights on causal estimation window
    asset_names = list(clean_r.columns)
    risk_contrib = compute_risk_contributions(primary_sol.weights, cov_est.cov_matrix, asset_names)

    # Reference strategy weights strictly from est_returns (zero lookahead in OOS modes)
    eq_sol, _ = _solve_portfolio_snapshot(
        est_returns,
        strategy="equal_weight",
        cov_method=request.cov_method,
        risk_free_rate=request.risk_free_rate,
        trading_days=ann_days,
        min_weight=0.0,
        max_weight=1.0,
        custom_weights=None,
    )
    gmv_sol, _ = _solve_portfolio_snapshot(
        est_returns,
        strategy="min_variance",
        cov_method=request.cov_method,
        risk_free_rate=request.risk_free_rate,
        trading_days=ann_days,
        min_weight=request.min_weight,
        max_weight=request.max_weight,
        custom_weights=None,
    )
    ms_sol, _ = _solve_portfolio_snapshot(
        est_returns,
        strategy="max_sharpe",
        cov_method=request.cov_method,
        risk_free_rate=request.risk_free_rate,
        trading_days=ann_days,
        min_weight=request.min_weight,
        max_weight=request.max_weight,
        custom_weights=None,
    )

    weights_df = pd.DataFrame(
        {
            "weight": primary_sol.weights,
            "expected_return_ann": cov_est.mean_returns,
            "volatility_ann": np.sqrt(np.maximum(np.diag(cov_est.cov_matrix), 0.0)),
            "marginal_risk_contribution": [
                risk_contrib["marginal_risk_contribution"][name] for name in asset_names
            ],
            "risk_contribution": [risk_contrib["risk_contribution"][name] for name in asset_names],
            "pct_risk_contribution": [
                risk_contrib["pct_risk_contribution"][name] for name in asset_names
            ],
            "equal_weight": eq_sol.weights,
            "min_variance_weight": gmv_sol.weights,
            "max_sharpe_weight": ms_sol.weights,
        },
        index=pd.Index(asset_names, name="asset"),
    )

    # Exact SLSQP frontier (and optional Dirichlet scatter cleanly distinguished) on causal est_returns
    frontier_summary: dict[str, Any] | None = None
    if request.frontier_points > 0:
        min_hist_fr = min(30, max(5, len(est_returns)))
        ef = solve_exact_efficient_frontier(
            est_returns,
            n_points=request.frontier_points,
            cov_method=request.cov_method,
            risk_free_rate=request.risk_free_rate,
            trading_days=ann_days,
            min_history=min_hist_fr,
            min_weight=request.min_weight,
            max_weight=request.max_weight,
            num_random_portfolios=request.num_random_portfolios,
            random_state=request.random_seed,
        )
        exact_rows: list[dict[str, Any]] = []
        for idx in range(len(ef.frontier_volatility)):
            row: dict[str, Any] = {
                "frontier_type": "slsqp_exact",
                "algorithm": "slsqp_exact_frontier",
                "point_index": idx,
                "volatility": float(ef.frontier_volatility[idx]),
                "expected_return": float(ef.frontier_returns[idx]),
                "sharpe": float(ef.frontier_sharpe[idx]),
                "converged": bool(ef.frontier_converged[idx]),
            }
            for a_idx, a_name in enumerate(ef.asset_names):
                row[f"weight__{a_name}"] = float(ef.frontier_weights[idx, a_idx])
            exact_rows.append(row)

        all_frontier_rows = list(exact_rows)
        random_summary: dict[str, Any] | None = None
        if (
            ef.random_volatility is not None
            and ef.random_returns is not None
            and ef.random_sharpe is not None
            and ef.random_weights is not None
        ):
            for idx in range(len(ef.random_volatility)):
                r_row: dict[str, Any] = {
                    "frontier_type": "dirichlet_random",
                    "algorithm": "dirichlet_random_sampling",
                    "point_index": idx,
                    "volatility": float(ef.random_volatility[idx]),
                    "expected_return": float(ef.random_returns[idx]),
                    "sharpe": float(ef.random_sharpe[idx]),
                    "converged": True,
                }
                for a_idx, a_name in enumerate(ef.asset_names):
                    r_row[f"weight__{a_name}"] = float(ef.random_weights[idx, a_idx])
                all_frontier_rows.append(r_row)
            valid_sr = np.where(np.isfinite(ef.random_sharpe), ef.random_sharpe, -np.inf)
            best_rand_idx = int(np.argmax(valid_sr))
            min_rand_idx = int(np.argmin(ef.random_volatility))
            random_summary = {
                "algorithm": "dirichlet_random_sampling",
                "num_portfolios": int(len(ef.random_volatility)),
                "best_sampled_sharpe": {
                    "expected_return": float(ef.random_returns[best_rand_idx]),
                    "volatility": float(ef.random_volatility[best_rand_idx]),
                    "sharpe": float(ef.random_sharpe[best_rand_idx]),
                },
                "min_sampled_volatility": {
                    "expected_return": float(ef.random_returns[min_rand_idx]),
                    "volatility": float(ef.random_volatility[min_rand_idx]),
                    "sharpe": float(ef.random_sharpe[min_rand_idx]),
                },
            }

        extra_frames["frontier"] = pd.DataFrame(all_frontier_rows)
        frontier_summary = {
            "estimation_window": est_window_label,
            "estimation_start": est_returns.index[0].strftime("%Y-%m-%d"),
            "estimation_end": est_returns.index[-1].strftime("%Y-%m-%d"),
            "hindsight_only": bool(has_lookahead_bias),
            "exact_slsqp": {
                "algorithm": "slsqp_exact_frontier",
                "num_points": int(len(ef.frontier_volatility)),
                "converged_points": int(np.sum(ef.frontier_converged)),
                "min_variance": {
                    "expected_return": float(ef.min_variance.expected_return),
                    "volatility": float(ef.min_variance.volatility),
                    "sharpe": float(ef.min_variance.sharpe),
                    "weights": {
                        ef.asset_names[i]: float(ef.min_variance.weights[i])
                        for i in range(len(ef.asset_names))
                    },
                },
                "max_sharpe": {
                    "expected_return": float(ef.max_sharpe.expected_return),
                    "volatility": float(ef.max_sharpe.volatility),
                    "sharpe": float(ef.max_sharpe.sharpe),
                    "weights": {
                        ef.asset_names[i]: float(ef.max_sharpe.weights[i])
                        for i in range(len(ef.asset_names))
                    },
                },
            },
            "random_sampling": random_summary,
        }

    # Compare Primary Portfolio vs Equal-Weight Benchmark over the EXACT same OOS span
    strat_oos_ret = backtest_df[primary_col].pct_change().dropna()
    eq_oos_ret = backtest_df["Equal_Weight_Portfolio"].pct_change().dropna()
    vs_equal_weight = compute_benchmark_relative_metrics(
        strat_oos_ret,
        eq_oos_ret,
        periods_per_year=periods_per_year,
        risk_free_rate=request.risk_free_rate,
        return_convention="simple",
    )

    benchmark_info: dict[str, Any] | None = None
    if request.benchmark is not None and bench_fetch_key is not None:
        bench_nav, bench_summary, b_meta, b_qual, b_warns = _align_benchmark_to_oos_nav(
            request.benchmark,
            raw_frames[bench_fetch_key],
            backtest_df,
            primary_col,
            base_currency=aligned.base_currency,
            allow_usdt_peg=request.allow_usdt_peg,
            fx_loader=fx_loader,
            usdt_usd_loader=usdt_usd_loader,
            proxy_url=request.proxy_url,
            periods_per_year=periods_per_year,
            risk_free_rate=request.risk_free_rate,
        )
        backtest_df["Benchmark"] = bench_nav
        benchmark_info = bench_summary
        bench_key = (
            f"benchmark:{request.benchmark.leg_name}"
            if request.benchmark.leg_name in assets_meta
            else request.benchmark.leg_name
        )
        assets_meta[bench_key] = b_meta
        quality_map[bench_key] = b_qual
        extra_warnings.extend(b_warns)

    sensitivity_tables: dict[str, Any] | None = None
    if request.include_sensitivity:
        sensitivity_tables = _compute_sensitivity_tables(
            request,
            clean_r,
            tradable_mask,
            periods_per_year=periods_per_year,
            min_train_rows=min_train_rows,
            min_test_rows=min_test_rows,
        )

    cov_df = pd.DataFrame(cov_est.cov_matrix, index=asset_names, columns=asset_names)
    corr_df = clean_r.corr(method="pearson")

    frames: dict[str, pd.DataFrame] = {
        "prices": aligned.base_prices,
        "returns": clean_r,
        "weights": weights_df,
        "correlation": corr_df,
        "covariance": cov_df,
        "backtest": backtest_df,
        **extra_frames,
    }

    distinct_local_ccys = {
        str(v).strip().upper() for v in aligned.asset_currencies.values() if str(v).strip()
    }
    if benchmark_info is not None:
        bench_native = (
            str(benchmark_info.get("asset", {}).get("native_currency") or "").strip().upper()
        )
        if bench_native:
            distinct_local_ccys.add(bench_native)
    mixed_local = bool(aligned.base_currency == "LOCAL" and len(distinct_local_ccys) > 1)
    if mixed_local:
        degraded = True
        extra_warnings.append(
            f"当前处于 base_currency='LOCAL' 且组合/基准跨多种计价货币（{sorted(distinct_local_ccys)}）；"
            "跨币种比较未包含汇率折算，请改用 CNY/USD 统一计价口径。"
        )
    methodology = {
        "strategy": request.strategy,
        "validation_mode": request.validation_mode,
        "is_out_of_sample": request.validation_mode != "in_sample",
        "has_lookahead_bias": bool(has_lookahead_bias),
        "calendar_mode": aligned.calendar_mode,
        "base_currency": aligned.base_currency,
        "mixed_local_currencies": mixed_local,
        "cov_method": request.cov_method,
        "rebalance_freq": request.rebalance_freq,
        "transaction_cost_bps": float(request.transaction_cost_bps),
        "min_weight": float(request.min_weight),
        "max_weight": float(request.max_weight),
        "backtest_return_convention": "simple",
        "periods_per_year": float(periods_per_year),
        "estimation_window": est_window_label,
        "estimation_start": est_returns.index[0].strftime("%Y-%m-%d"),
        "estimation_end": est_returns.index[-1].strftime("%Y-%m-%d"),
        "non_tradable_rebalance_skipped": True,
    }
    data_summary = {
        "requested_start": request.start_date,
        "requested_end": request.end_date,
        "effective_start": aligned.overlap_summary.get("effective_start", ""),
        "effective_end": aligned.overlap_summary.get("effective_end", ""),
        "valuation_points": int(len(aligned.base_prices)),
        "return_periods": int(len(clean_r)),
        "backtest_start": backtest_df.index[0].strftime("%Y-%m-%d"),
        "backtest_end": backtest_df.index[-1].strftime("%Y-%m-%d"),
        "backtest_points": int(len(backtest_df)),
        "raw_counts": aligned.overlap_summary.get("raw_counts", {}),
        "raw_ranges": aligned.overlap_summary.get("raw_ranges", {}),
        "dropped_non_overlap_days_by_asset": aligned.overlap_summary.get(
            "dropped_non_overlap_days_by_asset", {}
        ),
    }

    metrics: dict[str, Any] = {
        "portfolio": {
            "strategy": request.strategy,
            "converged": bool(primary_sol.converged),
            "solver_message": primary_sol.solver_message,
            "estimation_expected_return": float(primary_sol.expected_return),
            "estimation_volatility": float(primary_sol.volatility),
            "estimation_sharpe": float(primary_sol.sharpe),
            "weights": {
                asset_names[i]: float(primary_sol.weights[i]) for i in range(len(asset_names))
            },
        },
        "risk_contributions": risk_contrib,
        "covariance": {
            "method": cov_est.cov_method,
            "shrinkage": cov_est.shrinkage,
            "condition_number": cov_est.condition_number,
            "min_eigenvalue": cov_est.min_eigenvalue,
            "ridge_applied": cov_est.ridge_applied,
        },
        "backtest": {
            primary_col: primary_perf,
            "Equal_Weight_Portfolio": eq_perf,
            "relative_to_equal_weight": vs_equal_weight,
        },
    }
    if train_test_summary is not None:
        metrics["validation"] = train_test_summary
    elif walk_forward_summary is not None:
        metrics["validation"] = walk_forward_summary
        metrics["backtest"]["In_Sample_Portfolio"] = walk_forward_summary[
            "in_sample_hindsight_reference"
        ]
    else:
        metrics["validation"] = {
            "mode": "in_sample",
            "has_lookahead_bias": has_lookahead_bias,
            "performance": primary_perf,
        }
    if frontier_summary is not None:
        metrics["efficient_frontier"] = frontier_summary
    if sensitivity_tables is not None:
        metrics["sensitivity"] = sensitivity_tables

    warnings_tuple = _dedupe_messages(aligned.warnings, extra_warnings)
    disclaimers = (
        "组合优化、协方差估计与历史回测仅反映历史样本分布及设定交易假设（调仓频率、双边交易成本与多头权重约束），不构成投资建议。",
    )
    return MultiAssetResult(
        schema_version=SCHEMA_VERSION,
        command="portfolio",
        status="degraded" if degraded else "ok",
        asset_order=tuple(asset_names),
        assets=assets_meta,
        quality=quality_map,
        data_summary=data_summary,
        parameters=request.to_dict(),
        methodology=methodology,
        metrics=metrics,
        metric_units=_COMPARE_METRIC_UNITS,
        metric_reasons={},
        benchmark=benchmark_info,
        warnings=warnings_tuple,
        disclaimers=disclaimers,
        frames=frames,
        data_frame=backtest_df,
    )


__all__ = ["analyze_portfolio", "compare_assets"]
