"""组合分析：净值、风险指标与再平衡建议。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio import compute_holdings_values, normalize_holdings, summarize_assets
from portfolio_analysis import (
    build_position_weights,
    compute_portfolio_curve,
    compute_portfolio_metrics,
    compute_rebalance_suggestions,
)


def test_portfolio_curve_matches_manual_weighted_returns() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.DataFrame({"a": [0.10, 0.00, -0.10], "b": [0.00, 0.10, 0.00]}, index=idx)
    nav = compute_portfolio_curve(returns, pd.Series({"a": 0.6, "b": 0.4}))
    expected_daily = pd.Series([0.06, 0.04, -0.06], index=idx)
    expected = (1 + expected_daily).cumprod()
    expected = expected / expected.iloc[0]
    pd.testing.assert_series_equal(nav, expected.rename("portfolio_nav"))


def test_portfolio_metrics_are_predictable_for_simple_nav() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    nav = pd.Series([1.0, 1.1, 1.0, 1.2], index=idx)
    metrics = compute_portfolio_metrics(nav, trading_days=252, risk_free_rate=0.0)
    assert metrics["cumulative_return"] == pytest.approx(0.2)
    assert metrics["max_drawdown"] == pytest.approx(1.0 - 1.0 / 1.1)
    assert metrics["annualized_volatility"] > 0
    assert np.isfinite(metrics["sharpe_ratio"])


def test_rebalance_suggestions_respect_tolerance() -> None:
    current = pd.Series({"权益": 0.52, "现金": 0.48})
    target = pd.Series({"权益": 0.50, "现金": 0.50})
    quiet = compute_rebalance_suggestions(current, target, tolerance=0.05, total_value=1000)
    assert not quiet["needs_rebalance"].any()
    loud = compute_rebalance_suggestions(current, target, tolerance=0.01, total_value=1000)
    assert loud.loc["权益", "needs_rebalance"]
    assert loud.loc["权益", "suggested_trade_value"] == pytest.approx(-20.0)


def test_cash_and_liability_do_not_enter_position_weights() -> None:
    holdings = pd.DataFrame(
        {
            "item_type": ["asset", "asset", "liability"],
            "asset_class": ["权益", "现金", "负债"],
            "asset_type": ["etf", "cash", "loan"],
            "symbol": ["510300", "", ""],
            "quantity": [10, 1, 1],
            "current_price": [10, 100, 50],
        }
    )
    valued = compute_holdings_values(normalize_holdings(holdings))
    summary = summarize_assets(valued)
    weights = build_position_weights(valued)
    assert list(weights.index) == ["510300"]
    assert weights.iloc[0] == pytest.approx(1.0)
    assert summary["net_asset_value"] == pytest.approx(150.0)

