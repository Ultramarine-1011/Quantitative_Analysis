"""Portfolio solutions, benchmark alignment and sensitivity validation."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from analysis_metrics import (
    compute_benchmark_relative_metrics,
)
from contracts import (
    AssetSpec,
    PortfolioRequest,
    QualityReport,
    SeriesMetadata,
    build_quality_report,
    build_series_metadata,
    sync_metadata_to_frame,
    validate_frame_metadata_consistency,
)
from currency import convert_ohlcv_to_base_currency
from ohlcv import DateLike
from quant_models import (
    CovarianceEstimate,
    PortfolioSolution,
    backtest_portfolio,
    compute_backtest_metrics,
    estimate_covariance_and_mean,
    run_walk_forward_backtest,
    solve_equal_weight_portfolio,
    solve_inverse_volatility_portfolio,
    solve_minimum_variance_portfolio,
    solve_risk_parity_portfolio,
    solve_tangency_portfolio,
    split_train_test_returns,
)


def solve_portfolio_snapshot(
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


def align_benchmark_to_oos_nav(
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
    validate_frame_metadata_consistency(converted_bench, b_meta)
    sync_metadata_to_frame(converted_bench, b_meta, b_quality)

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
        sol, est = solve_portfolio_snapshot(
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


def compute_sensitivity_tables(
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
