"""Portfolio research orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from analysis_metrics import (
    compute_benchmark_relative_metrics,
)
from asset_loader import (
    align_multi_asset_frames,
)
from contracts import (
    SCHEMA_VERSION,
    AssetSpec,
    MultiAssetResult,
    PortfolioRequest,
    build_multi_asset_metric_units,
)
from ohlcv import DateLike
from quant_models import (
    backtest_portfolio,
    clean_joint_returns,
    compute_risk_contributions,
    run_walk_forward_backtest,
    solve_exact_efficient_frontier,
    split_train_test_returns,
)
from services.portfolio_validation import (
    align_benchmark_to_oos_nav,
    compute_sensitivity_tables,
    solve_portfolio_snapshot,
)
from services.preparation import (
    AssetLoaderFn,
    collect_metadata_and_quality,
    dedupe_messages,
    fetch_multi_asset_frames,
)


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

    raw_frames = fetch_multi_asset_frames(
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

    assets_meta, quality_map, degraded, extra_warnings = collect_metadata_and_quality(
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
        primary_sol, cov_est = solve_portfolio_snapshot(
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
        primary_sol, cov_est = solve_portfolio_snapshot(
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
        primary_sol, cov_est = solve_portfolio_snapshot(
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
    eq_sol, _ = solve_portfolio_snapshot(
        est_returns,
        strategy="equal_weight",
        cov_method=request.cov_method,
        risk_free_rate=request.risk_free_rate,
        trading_days=ann_days,
        min_weight=0.0,
        max_weight=1.0,
        custom_weights=None,
    )
    gmv_sol, _ = solve_portfolio_snapshot(
        est_returns,
        strategy="min_variance",
        cov_method=request.cov_method,
        risk_free_rate=request.risk_free_rate,
        trading_days=ann_days,
        min_weight=request.min_weight,
        max_weight=request.max_weight,
        custom_weights=None,
    )
    ms_sol, _ = solve_portfolio_snapshot(
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
        bench_nav, bench_summary, b_meta, b_qual, b_warns = align_benchmark_to_oos_nav(
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
        if b_qual.has_anomalies or b_meta.warnings:
            degraded = True

    sensitivity_tables: dict[str, Any] | None = None
    if request.include_sensitivity:
        sensitivity_tables = compute_sensitivity_tables(
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
        "max_drawdown_convention": "non_negative_depth",
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

    warnings_tuple = dedupe_messages(aligned.warnings, extra_warnings)
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
        metric_units=build_multi_asset_metric_units(),
        metric_reasons={},
        benchmark=benchmark_info,
        warnings=warnings_tuple,
        disclaimers=disclaimers,
        frames=frames,
        data_frame=backtest_df,
    )
