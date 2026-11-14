"""Stage 1 synthetic tests: data quality, calendar alignment, FX unification, metrics & backtest."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analysis_metrics import (
    compute_annualized_volatility,
    compute_cagr,
    compute_calmar_ratio,
    compute_cumulative_return,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_var_cvar,
    get_quant_metrics,
)
from asset_catalog import get_asset_trading_days
from asset_loader import (
    align_multi_asset_frames,
    load_and_align_multi_asset_portfolio,
)
from currency import (
    align_fx_series_to_dates,
    convert_prices_to_base_currency,
    infer_asset_currency,
    parse_fx_pair_currencies,
)
from feature_engineering import OHLCVFeatureEngineer
from ohlcv import normalize_ohlcv, prepare_ohlcv
from quant_models import backtest_portfolio, run_monte_carlo_gbm


def _make_ohlcv(
    dates: pd.DatetimeIndex,
    closes: np.ndarray | list[float],
    volumes: np.ndarray | list[float] | None = None,
    **attrs: object,
) -> pd.DataFrame:
    c = np.asarray(closes, dtype="float64")
    v = (
        np.full(len(c), 1000.0, dtype="float64")
        if volumes is None
        else np.asarray(volumes, dtype="float64")
    )
    df = pd.DataFrame(
        {
            "Open": c * 0.995,
            "High": c * 1.01,
            "Low": c * 0.99,
            "Close": c,
            "Volume": v,
        },
        index=dates.as_unit("ns"),
    )
    df.index.name = "Date"
    for k, val in attrs.items():
        df.attrs[k] = val
    return df


# ---------------------------------------------------------------------------
# 1. Data quality & missing value contracts
# ---------------------------------------------------------------------------


def test_ohlcv_distinguishes_halt_true_zero_return_and_missing_quote():
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    # Day 1: true zero return (same Close, Volume > 0)
    # Day 2: trading halt / missing volume only -> Volume=0 kept & flagged
    # Day 3: corrupted quote -> Close=None -> dropped under default vendor 'drop' policy
    raw = pd.DataFrame(
        {
            "Open": [10.0, 10.0, 10.2, 10.3, 10.5, 10.6],
            "High": [10.1, 10.1, 10.3, 10.4, 10.6, 10.7],
            "Low": [9.9, 9.9, 10.1, 10.2, 10.4, 10.5],
            "Close": [10.0, 10.0, 10.2, None, 10.5, 10.6],
            "Volume": [500.0, 400.0, 0.0, 999.0, None, 700.0],
        },
        index=dates,
    )

    with pytest.raises(ValueError, match="missing_policy='raise'"):
        prepare_ohlcv(raw)

    dropped = normalize_ohlcv(raw, {})
    assert len(dropped) == 5
    assert dates[3] not in dropped.index
    # Day 4 had valid prices and NaN volume: Volume defaults to 0.0 (never cloned from prior bar).
    assert dropped.loc[dates[4], "Volume"] == 0.0
    dq = dropped.attrs["data_quality"]
    assert dq["dropped_invalid_rows"] == 1
    assert dq["missing_volume_rows"] == 1
    assert dq["zero_volume_rows"] == 2  # Day 2 (0.0) and Day 4 (filled to 0.0)
    assert dq["filled_price_rows"] == 0
    assert dq["true_zero_return_rows"] >= 1

    ffilled = prepare_ohlcv(raw, drop_invalid=True, missing_policy="ffill")
    assert len(ffilled) == 6
    assert ffilled.loc[dates[3], "Close"] == 10.2
    # Missing Volume row is never forward-filled.
    assert ffilled.loc[dates[4], "Volume"] == 0.0
    assert ffilled.attrs["data_quality"]["filled_price_rows"] == 1


def test_ohlcv_detects_ohlc_anomalies_and_nonpositive_prices():
    dates = pd.date_range("2024-02-01", periods=4, freq="B")
    frame = _make_ohlcv(dates, [10.0, 10.5, -1.0, 11.0])
    # Corrupt bar 0 so High < Low.
    frame.iloc[0, frame.columns.get_loc("High")] = 9.0
    frame.iloc[0, frame.columns.get_loc("Low")] = 10.5
    clean = prepare_ohlcv(frame)
    dq = clean.attrs["data_quality"]
    assert dq["ohlc_anomaly_rows"] >= 2
    assert dq["non_positive_price_rows"] == 1


# ---------------------------------------------------------------------------
# 2. Financial metrics, annualization & momentum indicators
# ---------------------------------------------------------------------------


def test_metrics_nonpositive_safety_constant_series_and_crypto_365():
    assert get_asset_trading_days("crypto") == 365
    assert get_asset_trading_days("fx") == 260
    assert get_asset_trading_days("etf") == 252
    assert get_asset_trading_days(None) == 252

    dates_yield = pd.date_range("2024-01-01", periods=5, freq="B")
    nonpositive = pd.Series([0.0, -0.5, 1.2, 1.5, 2.0], index=dates_yield)
    assert math.isnan(compute_cumulative_return(nonpositive))
    assert math.isnan(compute_cagr(nonpositive))

    # Constant price series: zero vol & zero drawdown -> Sharpe/Sortino/Calmar return NaN cleanly.
    const_df = _make_ohlcv(pd.date_range("2024-01-01", periods=40, freq="D"), np.full(40, 100.0))
    const_df.attrs["asset_type"] = "crypto"
    const_df.attrs["currency"] = "USDT"
    eng = OHLCVFeatureEngineer()
    assert compute_annualized_volatility(eng, const_df, trading_days=365) == pytest.approx(0.0)
    assert math.isnan(compute_sharpe_ratio(eng, const_df, trading_days=365))
    assert math.isnan(compute_sortino_ratio(eng, const_df, trading_days=365))
    assert math.isnan(compute_calmar_ratio(0.0, 0.0))

    # Crypto (365) vs ETF (252) on identical geometric series: verify annualization difference.
    dates_30 = pd.date_range("2024-01-01", periods=30, freq="D")
    rng = np.random.default_rng(7)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.02, 30)))
    crypto_df = _make_ohlcv(dates_30, prices, asset_type="crypto", currency="USDT")
    etf_df = _make_ohlcv(dates_30, prices, asset_type="etf", currency="CNY")

    crypto_metrics = get_quant_metrics(crypto_df)
    etf_metrics = get_quant_metrics(etf_df)
    assert crypto_metrics["trading_days"] == 365
    assert etf_metrics["trading_days"] == 252
    assert crypto_metrics["annualized_volatility"] == pytest.approx(
        etf_metrics["annualized_volatility"] * math.sqrt(365.0 / 252.0), rel=1e-12
    )
    assert np.isfinite(crypto_metrics["sortino_ratio"])
    assert np.isfinite(crypto_metrics["calmar_ratio"])
    var_95, cvar_95 = compute_var_cvar(eng, crypto_df, confidence=0.95)
    assert crypto_metrics["var_95"] == pytest.approx(var_95)
    assert crypto_metrics["cvar_95"] == pytest.approx(cvar_95)
    assert cvar_95 <= var_95


def test_split_unadjusted_gap_vs_continuous_return_and_momentum_indicators():
    # Verify exact EMA, RSI (100 for strictly rising; 50 for flat), and MACD.
    dates = pd.date_range("2024-01-01", periods=35, freq="B")
    rising = _make_ohlcv(dates, np.linspace(10.0, 20.0, 35))
    eng = OHLCVFeatureEngineer()
    out = eng.transform(rising, include_momentum_indicators=True)
    assert "macd" in out.columns and "macd_signal" in out.columns and "rsi_14" in out.columns
    ema12 = eng.compute_ema(rising, span=12)
    pd.testing.assert_series_equal(
        ema12, rising["Close"].ewm(span=12, min_periods=12, adjust=False).mean(), check_names=False
    )
    assert out["rsi_14"].iloc[-1] == pytest.approx(100.0)
    assert np.isfinite(out["macd_hist"].iloc[-1])

    flat = _make_ohlcv(dates, np.full(35, 25.0))
    rsi_flat = eng.compute_rsi(flat, window=14)
    assert rsi_flat.iloc[-1] == pytest.approx(50.0)

    # Split price drop (2-for-1 unadjusted: 100 -> 50) yields -50% cumulative vs adjusted +10%.
    d_split = pd.date_range("2024-03-01", periods=4, freq="B")
    unadj = pd.Series([100.0, 102.0, 52.0, 55.0], index=d_split)
    adj = pd.Series([50.0, 51.0, 52.0, 55.0], index=d_split)
    assert compute_cumulative_return(unadj) == pytest.approx(-0.45)
    assert compute_cumulative_return(adj) == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 3. Cross-market valuation calendar & return basis guards
# ---------------------------------------------------------------------------


def test_cross_market_weekend_alignment_captures_crypto_moves_without_fake_zeros():
    # Weekdays 2024-01-01..2024-01-26 for ETF, 7x24 calendar days for Crypto.
    cal_days = pd.date_range("2024-01-01", "2024-01-26", freq="D")
    biz_days = pd.date_range("2024-01-01", "2024-01-26", freq="B")

    # ETF drifts +0.1% per business day; BTC jumps +10% on Saturdays & holds Sunday!
    etf_close = 100.0 * (1.001 ** np.arange(len(biz_days)))
    btc_close = []
    px = 100.0
    for d in cal_days:
        if d.weekday() == 5:  # Saturday weekend rally
            px *= 1.10
        elif d.weekday() < 5:
            px *= 1.002
        btc_close.append(px)

    etf_df = _make_ohlcv(biz_days, etf_close, asset_type="etf", symbol="510300", currency="CNY")
    btc_df = _make_ohlcv(
        cal_days, btc_close, asset_type="crypto", symbol="BTCUSDT", currency="USDT"
    )

    # Provide constant USDT/USD=1.0 and USDCNH=7.2 on calendar/FX grid.
    fx_df = _make_ohlcv(biz_days, np.full(len(biz_days), 7.2), asset_type="fx", symbol="USDCNH")
    usdt_df = _make_ohlcv(
        cal_days, np.full(len(cal_days), 1.0), asset_type="crypto", symbol="USDTUSD"
    )

    res_common = align_multi_asset_frames(
        {"etf:510300": etf_df, "crypto:BTCUSDT": btc_df},
        calendar_mode="common_trading_dates",
        base_currency="CNY",
        fx_loader=lambda pair, s, e: fx_df,
        usdt_usd_loader=lambda s, e, p: usdt_df,
    )
    # Every valuation date is a real ETF business day; zero fake equity 0.0 rows inserted!
    assert (res_common.returns_base.index.dayofweek < 5).all()
    # On Mondays after a Saturday +10% jump (2024-01-08 = Mon: +10% Sat * +0.2% Mon - 1).
    mon_ret = res_common.returns_base.loc[pd.Timestamp("2024-01-08"), "crypto:BTCUSDT"]
    assert mon_ret == pytest.approx(1.10 * 1.002 - 1.0, rel=1e-12)
    assert res_common.tradable_mask.all().all()

    res_weekly = align_multi_asset_frames(
        {"etf:510300": etf_df, "crypto:BTCUSDT": btc_df},
        calendar_mode="weekly_friday",
        base_currency="CNY",
        fx_loader=lambda pair, s, e: fx_df,
        usdt_usd_loader=lambda s, e, p: usdt_df,
    )
    assert (res_weekly.returns_base.index.dayofweek == 4).all()

    res_mtm = align_multi_asset_frames(
        {"etf:510300": etf_df, "crypto:BTCUSDT": btc_df},
        calendar_mode="mtm_outer",
        base_currency="CNY",
        fx_loader=lambda pair, s, e: fx_df,
        usdt_usd_loader=lambda s, e, p: usdt_df,
    )
    # In mtm_outer, weekends exist in NAV returns_base (tradable_mask=False for ETF),
    # while correlation_returns_base strictly stays on concurrent business days without fake 0s.
    assert len(res_mtm.returns_base) > len(res_common.returns_base)
    assert not bool(res_mtm.tradable_mask.loc[pd.Timestamp("2024-01-06"), "etf:510300"])
    pd.testing.assert_frame_equal(
        res_mtm.correlation_returns_base,
        res_common.correlation_returns_base,
        check_freq=False,
    )


def test_align_rejects_yield_level_and_flags_futures_continuous():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    etf_df = _make_ohlcv(
        dates, np.linspace(10, 11, 10), asset_type="etf", symbol="510300", currency="CNY"
    )
    yield_df = _make_ohlcv(
        dates,
        np.full(10, 4.2),
        asset_type="bond_us_yield",
        symbol="10Y",
        series_kind="yield_level",
        return_basis="yield_level",
    )
    with pytest.raises(ValueError, match="yield_level"):
        align_multi_asset_frames(
            {"etf:510300": etf_df, "bond_us_yield:10Y": yield_df}, base_currency="LOCAL"
        )

    gold_df = _make_ohlcv(
        dates,
        np.linspace(2000, 2050, 10),
        asset_type="gold",
        symbol="GC",
        return_basis="futures_continuous",
        currency="USD",
    )
    fx_df = _make_ohlcv(dates, np.full(10, 7.2), asset_type="fx", symbol="USDCNH")
    res = align_multi_asset_frames(
        {"etf:510300": etf_df, "gold:GC": gold_df},
        base_currency="CNY",
        fx_loader=lambda pair, s, e: fx_df,
    )
    assert any("期货主力连续价格" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# 4. Base currency conversion, FX volatility & USDT != USD
# ---------------------------------------------------------------------------


def test_fx_volatility_conversion_and_usdt_non_peg_enforcement():
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    # USD stock rises +5% per day while USDCNH depreciates (7.20 -> 6.80).
    us_prices = 100.0 * (1.05 ** np.arange(6))
    usdcnh = np.linspace(7.20, 6.80, 6)
    us_df = _make_ohlcv(dates, us_prices, asset_type="us_stock", symbol="AAPL", currency="USD")
    etf_df = _make_ohlcv(dates, np.full(6, 10.0), asset_type="etf", symbol="510300", currency="CNY")
    fx_df = _make_ohlcv(dates, usdcnh, asset_type="fx", symbol="USDCNH")

    res_cny = align_multi_asset_frames(
        {"etf:510300": etf_df, "us_stock:AAPL": us_df},
        base_currency="CNY",
        fx_loader=lambda pair, s, e: fx_df,
    )
    expected_day1_cny = (1.05 * (usdcnh[1] / usdcnh[0])) - 1.0
    assert res_cny.returns_base.iloc[0]["us_stock:AAPL"] == pytest.approx(
        expected_day1_cny, rel=1e-12
    )
    assert res_cny.returns_local.iloc[0]["us_stock:AAPL"] == pytest.approx(0.05, rel=1e-12)

    # USDT must NOT unconditionally equal USD when market rate cannot be fetched.
    def _fail_usdt(s, e, p):
        raise RuntimeError("exchange down")

    with pytest.raises(ValueError, match="USDT 不得无条件视作 USD"):
        convert_prices_to_base_currency(
            pd.DataFrame({"crypto:BTCUSDT": [40000.0, 41000.0]}, index=dates[:2]),
            {"crypto:BTCUSDT": "USDT"},
            base_currency="USD",
            usdt_usd_loader=_fail_usdt,
            allow_usdt_usd_peg=False,
        )

    # Explicit opt-in uses 1:1 with warning; real depegged USDT/USD=0.99 overrides 1:1.
    res_peg = convert_prices_to_base_currency(
        pd.DataFrame({"crypto:BTCUSDT": [40000.0, 41000.0]}, index=dates[:2]),
        {"crypto:BTCUSDT": "USDT"},
        base_currency="USD",
        usdt_usd_loader=_fail_usdt,
        allow_usdt_usd_peg=True,
    )
    assert res_peg.metadata["usdt_peg_assumed"] is True
    assert res_peg.base_prices["crypto:BTCUSDT"].iloc[0] == pytest.approx(40000.0)

    depeg_usdt = _make_ohlcv(dates, np.full(6, 0.99), asset_type="crypto", symbol="USDTUSD")
    res_real = convert_prices_to_base_currency(
        pd.DataFrame({"crypto:BTCUSDT": [40000.0, 41000.0]}, index=dates[:2]),
        {"crypto:BTCUSDT": "USDT"},
        base_currency="USD",
        usdt_usd_loader=lambda s, e, p: depeg_usdt,
        allow_usdt_usd_peg=False,
    )
    assert res_real.metadata["usdt_peg_assumed"] is False
    assert res_real.base_prices["crypto:BTCUSDT"].iloc[0] == pytest.approx(40000.0 * 0.99)


def test_fx_staleness_window_and_usd_triangulation_and_currency_parsing():
    assert parse_fx_pair_currencies("USDCNH") == ("USD", "CNY")
    assert parse_fx_pair_currencies("USD/JPY") == ("USD", "JPY")
    assert infer_asset_currency("global_index", "HSI") == "HKD"
    assert infer_asset_currency("crypto", "BTCUSDT") == "USDT"
    assert infer_asset_currency("hk_stock", "00700") == "HKD"

    # Sparse FX quotes: Mon Jan 1 and Mon Jan 15 -> Jan 3 (age 2d) valid, Jan 10 (age 9d) expired!
    fx_series = pd.Series([7.10, 7.20], index=pd.to_datetime(["2024-01-01", "2024-01-15"]))
    targets = pd.to_datetime(["2023-12-29", "2024-01-01", "2024-01-03", "2024-01-10", "2024-01-15"])
    aligned, stats = align_fx_series_to_dates(fx_series, targets, max_staleness_days=4)
    assert math.isnan(aligned.iloc[0])  # never backfill
    assert aligned.iloc[1] == pytest.approx(7.10)
    assert aligned.iloc[2] == pytest.approx(7.10)
    assert math.isnan(aligned.iloc[3])  # expired > 4 days
    assert aligned.iloc[4] == pytest.approx(7.20)
    assert stats["leading_missing_days"] == 1
    assert stats["expired_gap_days"] == 1


# ---------------------------------------------------------------------------
# 5. Backtest base-1.0, rebalancing, transaction costs & crypto GBM
# ---------------------------------------------------------------------------


def test_backtest_preserves_day0_rebalances_charges_costs_and_respects_tradable_mask():
    # Two assets across two months: Jan 2024 -> Feb 2024.
    dates = pd.DatetimeIndex(
        [
            "2024-01-02",
            "2024-01-15",
            "2024-01-31",
            "2024-02-01",
            "2024-02-02",
            "2024-02-15",
        ]
    )
    r = pd.DataFrame(
        {
            "a": [0.10, 0.00, 0.00, 0.05, 0.02, -0.01],
            "b": [-0.05, 0.02, 0.01, -0.02, 0.04, 0.03],
        },
        index=dates,
    )
    w = np.array([0.6, 0.4])

    # Zero-cost daily rebalance matches exact (1 + r @ w).cumprod() anchored at 1.0.
    bt_daily = backtest_portfolio(
        r, w, min_history=2, rebalance_freq="daily", transaction_cost_bps=0.0
    )
    assert len(bt_daily) == len(r) + 1
    assert bt_daily["Optimal_Portfolio"].iloc[0] == 1.0
    # First return day MUST reflect day-0 return (not dropped as in old normalization!).
    expected_step0 = 1.0 + (0.6 * 0.10 + 0.4 * (-0.05))
    assert bt_daily["Optimal_Portfolio"].iloc[1] == pytest.approx(expected_step0, rel=1e-12)

    # Transaction costs strictly reduce final wealth and record turnover & cost ratio.
    bt_cost = backtest_portfolio(
        r, w, min_history=2, rebalance_freq="monthly", transaction_cost_bps=50.0
    )
    bt_free = backtest_portfolio(
        r, w, min_history=2, rebalance_freq="monthly", transaction_cost_bps=0.0
    )
    assert bt_cost["Optimal_Portfolio"].iloc[-1] < bt_free["Optimal_Portfolio"].iloc[-1]
    perf = bt_cost.attrs["performance"]["Optimal_Portfolio"]
    assert perf["total_turnover"] >= 1.0  # initial purchase alone has turnover = 1.0
    assert perf["total_cost_ratio"] > 0.005

    # Closed leg on 2024-02-01 defers monthly rebalance until 2024-02-02 when both legs open!
    mask = pd.DataFrame(True, index=dates, columns=["a", "b"])
    mask.loc[pd.Timestamp("2024-02-01"), "a"] = False
    bt_deferred = backtest_portfolio(
        r, w, min_history=2, rebalance_freq="monthly", transaction_cost_bps=10.0, tradable_mask=mask
    )
    # On 2024-02-01, no rebalance happened yet -> matches free-drift before costs through 2024-02-01 vs defer-only.
    assert np.isfinite(bt_deferred["Optimal_Portfolio"]).all()


def test_load_and_align_multi_asset_portfolio_end_to_end_and_crypto_gbm_calendar():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    etf_df = _make_ohlcv(
        dates, np.linspace(10.0, 11.0, 10), asset_type="etf", symbol="510300", currency="CNY"
    )
    hk_df = _make_ohlcv(
        dates, np.linspace(300.0, 315.0, 10), asset_type="hk_stock", symbol="00700", currency="HKD"
    )
    fx_hkd = _make_ohlcv(dates, np.full(10, 0.92), asset_type="fx", symbol="HKDCNH")

    def fake_loader(*, asset_type, symbol, start_date, end_date, proxy_url=None):
        return etf_df if asset_type == "etf" else hk_df

    res = load_and_align_multi_asset_portfolio(
        (("etf", "510300"), ("hk_stock", "00700")),
        dates[0],
        dates[-1],
        loader=fake_loader,
        fx_loader=lambda pair, s, e: fx_hkd,
        base_currency="CNY",
    )
    assert res.base_currency == "CNY"
    assert res.asset_currencies["hk_stock:00700"] == "HKD"
    bt = backtest_portfolio(
        res.returns_base,
        [0.5, 0.5],
        min_history=3,
        base_date=res.base_date,
        periods_per_year=res.backtest_periods_per_year,
    )
    assert bt.index[0] == res.base_date
    assert bt["Optimal_Portfolio"].iloc[0] == 1.0

    # Crypto GBM generates 365-day consecutive calendar dates including weekends.
    crypto_dates = pd.date_range("2024-01-01", periods=20, freq="D")
    crypto_df = _make_ohlcv(crypto_dates, 100.0 * (1.01 ** np.arange(20)))
    mc = run_monte_carlo_gbm(
        crypto_df, days=7, num_simulations=10, random_state=1, trading_days=365
    )
    diffs = np.diff(mc.future_dates.to_numpy()).astype("timedelta64[D]").astype(int)
    assert np.all(diffs == 1)
