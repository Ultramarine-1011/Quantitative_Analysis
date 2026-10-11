"""CLI/Web numerical conventions and diagnostic labels share one implementation."""

import numpy as np
import pandas as pd
import pytest

import app
import main
from analysis_metrics import get_quant_metrics
from feature_engineering import OHLCVFeatureEngineer


def test_metric_formulas_and_cli_report_match(sample_ohlcv):
    metrics = get_quant_metrics(sample_ohlcv)
    close = sample_ohlcv["Close"]
    years = (close.index[-1] - close.index[0]).days / 365.25
    log_returns = np.log(close / close.shift()).dropna()
    excess = log_returns - np.log1p(0.016) / 252
    expected_sharpe = np.sqrt(252) * excess.mean() / excess.std(ddof=1)
    assert metrics["cagr"] == pytest.approx((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1)
    assert metrics["cumulative_return"] == pytest.approx(close.iloc[-1] / close.iloc[0] - 1)
    assert metrics["sharpe_ratio"] == pytest.approx(expected_sharpe)
    assert app._compute_cagr is main._compute_cagr
    report = main.build_diagnostic_report("510300", metrics["featured_df"], OHLCVFeatureEngineer())
    assert f"{100 * metrics['cagr']:.2f}%" in report
    assert f"{metrics['sharpe_ratio']:.3f}" in report


def test_metrics_recompute_stale_metadata_and_accept_features(sample_ohlcv):
    sample_ohlcv.attrs["max_drawdown"] = 999.0
    old = OHLCVFeatureEngineer().transform(sample_ohlcv, bollinger_window=5)
    metrics = get_quant_metrics(old, trading_days=365)
    featured = metrics["featured_df"]
    assert metrics["max_drawdown"] < 1
    assert featured.attrs["feature_summary"]["bollinger_window"] == 20
    assert featured.attrs["feature_summary"]["trading_days"] == 365
    assert featured.attrs["asset_type"] == "etf"


def test_single_day_and_constant_series_metrics(sample_ohlcv):
    assert pd.isna(get_quant_metrics(sample_ohlcv.iloc[:1])["cagr"])
    assert pd.isna(get_quant_metrics(sample_ohlcv.iloc[:1])["sharpe_ratio"])
    flat = sample_ohlcv.assign(Close=1.0)
    assert pd.isna(get_quant_metrics(flat)["sharpe_ratio"])
    assert get_quant_metrics(flat)["max_drawdown"] == 0.0


def test_report_uses_requested_risk_free_rate(sample_ohlcv):
    featured = OHLCVFeatureEngineer().transform(sample_ohlcv)
    report = main.build_diagnostic_report(
        "510300", featured, OHLCVFeatureEngineer(), risk_free_rate=0.03
    )
    assert "Rf=3.0%" in report
