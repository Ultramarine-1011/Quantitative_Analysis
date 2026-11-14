"""Stage 3 numerical verification: exact SLSQP frontier, Ledoit-Wolf shrinkage & zero-leakage OOS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis_metrics import RISK_FREE_RATE
from quant_models import (
    _ledoit_wolf_cov_daily,
    estimate_covariance_and_mean,
    generate_efficient_frontier,
    run_walk_forward_backtest,
    solve_exact_efficient_frontier,
    solve_minimum_variance_portfolio,
    solve_tangency_portfolio,
    split_train_test_returns,
)


def _synthetic_multi_asset_returns(n_days: int = 240, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, name="Date")
    factor = rng.normal(0.0004, 0.011, n_days)
    data = {
        "Equity": 0.0002 + 1.15 * factor + rng.normal(0.0, 0.008, n_days),
        "Gold": 0.00025 - 0.15 * factor + rng.normal(0.0, 0.007, n_days),
        "Bond": 0.00012 - 0.10 * factor + rng.normal(0.0, 0.0025, n_days),
        "Crypto": 0.0006 + 0.70 * factor + rng.normal(0.0, 0.024, n_days),
    }
    return pd.DataFrame(data, index=dates)


def test_exact_frontier_beats_random_dirichlet_and_is_monotone():
    returns = _synthetic_multi_asset_returns(200, seed=11)
    mc = generate_efficient_frontier(
        returns, num_portfolios=5000, cov_method="sample", random_state=7
    )
    exact = solve_exact_efficient_frontier(
        returns,
        n_points=25,
        cov_method="sample",
        num_random_portfolios=1000,
        random_state=7,
    )

    assert exact.risk_free_rate == pytest.approx(RISK_FREE_RATE)
    assert exact.min_variance.converged
    assert exact.max_sharpe.converged
    assert bool(np.all(exact.frontier_converged))

    np.testing.assert_allclose(exact.min_variance.weights.sum(), 1.0, atol=1e-9)
    np.testing.assert_allclose(exact.max_sharpe.weights.sum(), 1.0, atol=1e-9)
    assert exact.min_variance.volatility <= float(np.min(mc.volatility)) + 1e-8
    assert exact.max_sharpe.sharpe >= float(np.nanmax(mc.sharpe)) - 1e-8

    # Efficient branch must be non-decreasing in both return and volatility.
    assert np.all(np.diff(exact.frontier_returns) >= -1e-9)
    assert np.all(np.diff(exact.frontier_volatility) >= -1e-7)


def test_weight_bounds_enforced_and_infeasible_bounds_rejected():
    returns = _synthetic_multi_asset_returns(150, seed=19)
    gmv = solve_minimum_variance_portfolio(returns, min_weight=0.10, max_weight=0.40)
    tang = solve_tangency_portfolio(returns, min_weight=0.10, max_weight=0.40)
    for sol in (gmv, tang):
        assert sol.converged
        assert np.sum(sol.weights) == pytest.approx(1.0, abs=1e-9)
        assert float(np.min(sol.weights)) >= 0.10 - 1e-8
        assert float(np.max(sol.weights)) <= 0.40 + 1e-8

    fixed = solve_tangency_portfolio(returns, min_weight=0.25, max_weight=0.25)
    np.testing.assert_allclose(fixed.weights, [0.25, 0.25, 0.25, 0.25], atol=1e-12)

    three = returns.iloc[:, :3]
    with pytest.raises(ValueError, match="Infeasible weight bounds"):
        solve_tangency_portfolio(three, max_weight=0.25)
    with pytest.raises(ValueError, match="Infeasible weight bounds"):
        solve_minimum_variance_portfolio(three, min_weight=0.40)


def test_ledoit_wolf_formula_matches_direct_reference_and_rescues_singular_cov():
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 0.015, size=(45, 6))
    shrunk, intensity = _ledoit_wolf_cov_daily(x)

    # Naive textbook double-sum formula verification (Ledoit & Wolf 2004, Lemma 3.4).
    n_obs, n_assets = x.shape
    xc = x - x.mean(axis=0)
    emp = (xc.T @ xc) / n_obs
    mu = np.trace(emp) / n_assets
    delta = np.sum((emp - mu * np.eye(n_assets)) ** 2) / n_assets
    pi_sum = sum(np.sum((np.outer(xc[k], xc[k]) - emp) ** 2) for k in range(n_obs))
    beta = min((pi_sum / (n_obs * n_obs)) / n_assets, delta)
    expected_intensity = beta / delta
    expected_cov = (1.0 - expected_intensity) * emp + expected_intensity * mu * np.eye(n_assets)

    assert intensity == pytest.approx(expected_intensity, rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(shrunk, expected_cov, rtol=1e-12, atol=1e-14)

    # Collinear & redundant series fail under sample, stabilize under Ledoit-Wolf / diagonal.
    dates = pd.bdate_range("2024-01-02", periods=60)
    a = rng.normal(0.0005, 0.01, 60)
    b = rng.normal(0.0002, 0.012, 60)
    redundant_df = pd.DataFrame(
        {"A": a, "B": b, "A_dup": a + 1e-11 * rng.normal(0, 1, 60)}, index=dates
    )

    with pytest.raises(ValueError, match="singular|ill-conditioned"):
        estimate_covariance_and_mean(redundant_df, cov_method="sample")

    est_lw = estimate_covariance_and_mean(redundant_df, cov_method="ledoit_wolf")
    assert est_lw.min_eigenvalue > 0.0
    assert np.isfinite(est_lw.condition_number) and est_lw.condition_number < 1e6
    ef_lw = solve_exact_efficient_frontier(redundant_df, cov_method="ledoit_wolf", n_points=10)
    assert ef_lw.max_sharpe.converged

    est_diag = estimate_covariance_and_mean(redundant_df, cov_method="diagonal")
    assert est_diag.min_eigenvalue > 0.0

    # Binary +/- returns duplicate yields zero empirical fourth-moment term -> ridge fallback preserves PD.
    rademacher = pd.DataFrame(
        {
            "R1": np.where(np.arange(40) % 2 == 0, 0.01, -0.01),
            "R2": np.where(np.arange(40) % 2 == 0, 0.01, -0.01),
        },
        index=pd.bdate_range("2024-01-02", periods=40),
    )
    est_rad = estimate_covariance_and_mean(rademacher, cov_method="ledoit_wolf")
    assert est_rad.min_eigenvalue > 0.0
    assert est_rad.ridge_applied > 0.0

    # Zero-variance constant series must still fail loudly.
    flat_df = pd.DataFrame({"C1": np.zeros(40), "C2": np.zeros(40)}, index=rademacher.index)
    with pytest.raises(ValueError, match="zero return variance"):
        estimate_covariance_and_mean(flat_df, cov_method="ledoit_wolf")


def test_high_dimension_small_n_stabilized_by_ledoit_wolf():
    rng = np.random.default_rng(99)
    n_obs, n_assets = 15, 20
    small_n_wide = pd.DataFrame(
        rng.normal(0.0003, 0.012, size=(n_obs, n_assets)),
        index=pd.bdate_range("2024-01-02", periods=n_obs),
        columns=[f"Asset_{i}" for i in range(n_assets)],
    )
    with pytest.raises(ValueError, match="singular"):
        estimate_covariance_and_mean(small_n_wide, cov_method="sample", min_history=15)

    est = estimate_covariance_and_mean(small_n_wide, cov_method="ledoit_wolf", min_history=15)
    assert est.min_eigenvalue > 0.0
    assert est.shrinkage is not None and 0.0 < est.shrinkage <= 1.0


def test_train_test_split_has_no_test_leakage():
    returns = _synthetic_multi_asset_returns(160, seed=55)
    split_ts = returns.index[110]
    res1 = split_train_test_returns(
        returns, split_date=split_ts, min_train_rows=30, min_test_rows=10
    )

    mutated = returns.copy()
    mutated.loc[split_ts:, "Equity"] = -0.05
    mutated.loc[split_ts:, "Gold"] = 0.08
    res2 = split_train_test_returns(
        mutated, split_date=split_ts, min_train_rows=30, min_test_rows=10
    )

    np.testing.assert_array_equal(res1.optimal_weights, res2.optimal_weights)
    assert float(res1.out_of_sample_backtest["Optimal_Portfolio"].iloc[0]) == pytest.approx(1.0)


@pytest.mark.parametrize("window_mode", ["rolling", "expanding"])
def test_walk_forward_backtest_is_strictly_causal_no_lookahead(window_mode: str):
    returns = _synthetic_multi_asset_returns(180, seed=77)
    base_res = run_walk_forward_backtest(
        returns,
        window_mode=window_mode,  # type: ignore[arg-type]
        train_window=60,
        rebalance_freq="monthly",
        cov_method="ledoit_wolf",
        transaction_cost_bps=10.0,
    )

    cutoff_idx = 125
    cutoff_date = returns.index[cutoff_idx]
    shocked = returns.copy()
    shocked.iloc[cutoff_idx:] = -0.08
    shocked.iloc[cutoff_idx:, 1] = 0.12

    shocked_res = run_walk_forward_backtest(
        shocked,
        window_mode=window_mode,  # type: ignore[arg-type]
        train_window=60,
        rebalance_freq="monthly",
        cov_method="ledoit_wolf",
        transaction_cost_bps=10.0,
    )

    # All OOS weights and NAV strictly before cutoff_date must match bit-for-bit.
    mask_before_nav = base_res.nav_df.index < cutoff_date
    np.testing.assert_array_equal(
        base_res.nav_df.loc[mask_before_nav, "OOS_Portfolio"].to_numpy(),
        shocked_res.nav_df.loc[mask_before_nav, "OOS_Portfolio"].to_numpy(),
    )
    mask_before_reb = base_res.weight_history.index < cutoff_date
    np.testing.assert_array_equal(
        base_res.weight_history.loc[mask_before_reb].to_numpy(),
        shocked_res.weight_history.loc[mask_before_reb].to_numpy(),
    )
    # Conversely, full-sample hindsight reference MUST differ once future returns change.
    assert not np.allclose(
        base_res.nav_df.loc[mask_before_nav, "In_Sample_Portfolio"].to_numpy(),
        shocked_res.nav_df.loc[mask_before_nav, "In_Sample_Portfolio"].to_numpy(),
    )
    assert float(base_res.nav_df["OOS_Portfolio"].iloc[0]) == pytest.approx(1.0)


def test_walk_forward_transaction_costs_and_holiday_deferral():
    returns = _synthetic_multi_asset_returns(150, seed=88)
    no_fee = run_walk_forward_backtest(
        returns,
        train_window=50,
        rebalance_freq="monthly",
        transaction_cost_bps=0.0,
    )
    with_fee = run_walk_forward_backtest(
        returns,
        train_window=50,
        rebalance_freq="monthly",
        transaction_cost_bps=25.0,
    )
    assert with_fee.nav_df["OOS_Portfolio"].iloc[-1] < no_fee.nav_df["OOS_Portfolio"].iloc[-1]
    assert with_fee.performance["OOS_Portfolio"]["total_cost_ratio"] > 0.0  # type: ignore[index]

    # Mark first trading days of a new month in OOS as non-tradable -> rebalance defers cleanly.
    tradable = pd.DataFrame(True, index=returns.index, columns=returns.columns)
    month_starts = [
        i
        for i in range(51, len(returns.index))
        if returns.index[i].month != returns.index[i - 1].month
    ]
    blocked_dt = returns.index[month_starts[0]]
    tradable.loc[blocked_dt, "Gold"] = False

    deferred = run_walk_forward_backtest(
        returns,
        train_window=50,
        rebalance_freq="monthly",
        tradable_mask=tradable,
        transaction_cost_bps=10.0,
    )
    assert blocked_dt not in deferred.rebalance_dates
    assert returns.index[month_starts[0] + 1] in deferred.rebalance_dates
