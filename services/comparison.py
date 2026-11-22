"""Aligned multi-asset comparisons."""

from __future__ import annotations

import math
from collections.abc import Callable
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
from asset_loader import (
    align_multi_asset_frames,
)
from contracts import (
    SCHEMA_VERSION,
    AssetSpec,
    CompareRequest,
    MultiAssetResult,
    build_multi_asset_metric_units,
)
from ohlcv import DateLike
from quant_models import (
    estimate_covariance_and_mean,
    normalize_close_to_base_one,
)
from services.preparation import (
    AssetLoaderFn,
    collect_metadata_and_quality,
    dedupe_messages,
    fetch_multi_asset_frames,
)


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
        raw_mdd = float((1.0 - clean_p / running_max).max())
        max_dd = 0.0 if abs(raw_mdd) <= 1e-15 else max(0.0, raw_mdd)
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

    raw_frames = fetch_multi_asset_frames(
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

    assets_meta, quality_map, degraded, extra_warnings = collect_metadata_and_quality(
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

    warnings_tuple = dedupe_messages(aligned.warnings, extra_warnings)
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
        "max_drawdown_convention": "non_negative_depth",
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
        metric_units=build_multi_asset_metric_units(),
        metric_reasons=metric_reasons,
        benchmark=benchmark_info,
        warnings=warnings_tuple,
        disclaimers=disclaimers,
        frames=frames,
        data_frame=normalized_prices,
    )
