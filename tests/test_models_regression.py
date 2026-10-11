"""Deterministic model formulas, finite inputs and weight validation."""

import numpy as np
import pandas as pd
import pytest

from quant_models import (
    backtest_portfolio,
    clean_joint_returns,
    generate_efficient_frontier,
    run_monte_carlo_gbm,
)


def sample_returns():
    return pd.DataFrame(np.random.default_rng(31).normal(0.001, 0.01, (90, 3)))


def test_frontier_matches_original_daily_portfolio_formula():
    returns = sample_returns()
    result = generate_efficient_frontier(returns, num_portfolios=100, random_state=42)
    daily = np.einsum("ij,kj->ik", returns.values, result.weights)
    expected_ret = daily.mean(axis=0) * 252
    expected_vol = daily.std(axis=0, ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(result.returns, expected_ret, rtol=1e-12)
    np.testing.assert_allclose(result.volatility, expected_vol, rtol=1e-12)
    np.testing.assert_allclose(result.sharpe, (expected_ret - 0.015) / expected_vol, rtol=1e-12)
    assert result.best_sharpe_idx == int(np.argmax(result.sharpe))
    np.testing.assert_allclose(result.weights.sum(axis=1), 1)


def test_clean_joint_returns_removes_infinity():
    returns = sample_returns()
    returns.iloc[0, 0] = np.inf
    returns.iloc[1, 1] = -np.inf
    assert len(clean_joint_returns(returns)) == len(returns) - 2
    assert np.isfinite(generate_efficient_frontier(returns, num_portfolios=5).sharpe).all()


@pytest.mark.parametrize("weights", [[0, 0, 0], [1, -1, 0], [np.nan, 1, 1], [np.inf, 1, 1]])
def test_backtest_rejects_invalid_weight_sum(weights):
    with pytest.raises(ValueError, match="finite.*non-zero"):
        backtest_portfolio(sample_returns(), weights)


def test_backtest_rejects_zero_first_day_wealth():
    returns = sample_returns()
    returns.iloc[0, :] = -1
    with pytest.raises(ValueError, match="zero first-day"):
        backtest_portfolio(returns, [1, 1, 1])


def test_gbm_matches_fixed_seed_discrete_formula(sample_ohlcv):
    result = run_monte_carlo_gbm(sample_ohlcv, days=8, num_simulations=20, random_state=3)
    close = sample_ohlcv["Close"]
    log_returns = np.log(close / close.shift()).dropna()
    sigma = log_returns.std(ddof=1)
    z = np.random.default_rng(3).standard_normal((20, 8))
    paths = np.exp(np.log(close.iloc[-1]) + np.cumsum(log_returns.mean() + sigma * z, axis=1))
    expected = np.percentile(paths, [5, 50, 95], axis=0)
    np.testing.assert_allclose(result.sample_paths, paths[:5], rtol=1e-12)
    np.testing.assert_allclose(
        [result.quantiles_p05, result.quantiles_p50, result.quantiles_p95], expected, rtol=1e-12
    )
    assert len(result.future_dates) == 8
    assert result.future_dates[0] > close.index[-1]


@pytest.mark.parametrize("price", [0.0, -1.0, np.inf])
def test_gbm_rejects_nonpositive_or_infinite_history(sample_ohlcv, price):
    sample_ohlcv.iloc[2, 3] = price
    with pytest.raises(ValueError, match="strictly positive"):
        run_monte_carlo_gbm(sample_ohlcv)


@pytest.mark.parametrize("value", [0, True, 2.5, np.nan, np.inf])
def test_gbm_rejects_invalid_simulation_count(sample_ohlcv, value):
    with pytest.raises(ValueError, match="integer"):
        run_monte_carlo_gbm(sample_ohlcv, num_simulations=value)


def test_frontier_rejects_redundant_assets():
    returns = sample_returns()
    returns[1] = returns[0]
    with pytest.raises(ValueError, match="singular"):
        generate_efficient_frontier(returns)
