"""M1 V0.1 regression, mathematical alignment, contract, cache, and decoupling tests."""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import data_sources.cache as ohlcv_cache
import main as cli_main
from analysis_metrics import (
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    get_quant_metrics,
)
from asset_catalog import US_TREASURY_YIELD_10Y_CN_COL, normalize_symbol_alias
from asset_loader import align_multi_asset_frames
from contracts import (
    SCHEMA_VERSION,
    AnalysisRequest,
    AnalysisResult,
    build_quality_report,
    build_series_metadata,
)
from currency import convert_ohlcv_to_base_currency
from data_sources.china_equity import AKShareFetcher
from feature_engineering import OHLCVFeatureEngineer
from ohlcv import prepare_ohlcv, validate_ohlcv
from quant_models import backtest_portfolio, compute_backtest_metrics, run_walk_forward_backtest
from services.analysis_service import analyze_single_asset


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
# 1. Unified Contracts & JSON Serialization
# ---------------------------------------------------------------------------


def test_analysis_request_defaults_and_validation() -> None:
    req = AnalysisRequest(
        asset_type="ashare",
        symbol="600519",
        start_date="2024-01-01",
        end_date="2024-03-01",
    )
    assert req.adjust == "hfq"
    assert req.base_currency == "LOCAL"
    assert req.return_convention == "log"

    raw_req = AnalysisRequest(
        asset_type="ashare",
        symbol="600519",
        start_date="2024-01-01",
        end_date="2024-03-01",
        adjust="",
    )
    assert raw_req.adjust == ""

    yield_req = AnalysisRequest(
        asset_type="bond_us_yield",
        symbol="",
        start_date="2024-01-01",
        end_date="2024-03-01",
    )
    assert yield_req.symbol == "10Y"
    assert yield_req.adjust == ""

    with pytest.raises(ValueError):
        AnalysisRequest("us_stock", "AAPL", "2024-01-01", "2024-03-01", adjust="hfq")
    with pytest.raises(ValueError):
        AnalysisRequest("crypto", "BTC/USDT", "2024-01-01", "2024-03-01", adjust="hfq")
    with pytest.raises(ValueError):
        AnalysisRequest("etf", "", "2024-01-01", "2024-03-01")
    with pytest.raises(ValueError):
        AnalysisRequest("etf", "510300", "2024-01-01", "2024-03-01", base_currency="EUR")
    with pytest.raises(ValueError):
        AnalysisRequest("etf", "510300", "2024-01-01", "2024-03-01", return_convention="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AnalysisRequest("etf", "510300", "2024-01-01", "2024-03-01", trading_days=0)
    with pytest.raises(ValueError):
        AnalysisRequest("etf", "510300", "2024-01-01", "2024-03-01", sma_short=1)
    with pytest.raises(ValueError):
        AnalysisRequest("etf", "510300", "2024-01-01", "2024-03-01", risk_free_rate=-1.5)


def test_build_series_metadata_and_json_safe_result() -> None:
    dates = pd.bdate_range("2024-01-02", periods=30)
    closes = np.linspace(4.1, 4.4, len(dates))
    df = _make_ohlcv(
        dates,
        closes,
        asset_type="bond_us_yield",
        symbol="10Y",
        currency="USD",
        data_source="akshare:bond_zh_us_rate",
        adjust="",
        return_basis="yield_level",
        series_kind="yield_level",
    )

    req = AnalysisRequest(
        asset_type="bond_us_yield",
        symbol="10Y",
        start_date="2024-01-02",
        end_date="2024-02-12",
    )
    res = analyze_single_asset(req, raw_df=df)

    assert isinstance(res, AnalysisResult)
    assert res.schema_version == SCHEMA_VERSION
    assert res.status == "degraded"
    assert res.asset.series_kind == "yield_level"
    assert res.asset.return_basis == "yield_level"
    assert res.asset.canonical_symbol == "10Y"

    payload_no_data = res.to_dict(include_data=False)
    assert "data" not in payload_no_data
    dumped = json.dumps(payload_no_data, allow_nan=False)
    parsed = json.loads(dumped)
    assert parsed["metrics"]["cumulative_return"] is None
    assert parsed["metric_reasons"]["sharpe_ratio"] == "yield_level_not_return"

    payload_with_data = res.to_dict(include_data=True)
    assert len(payload_with_data["data"]) == len(dates)
    json.dumps(payload_with_data, allow_nan=False)


def test_series_metadata_adjust_basis_and_quality_anomalies() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10)
    raw_stock = _make_ohlcv(
        dates,
        np.linspace(100.0, 110.0, 10),
        asset_type="ashare",
        symbol="600519",
        adjust="",
        fx_warnings=["warn_a", "warn_a"],
        fallback_warnings=["warn_b"],
    )
    meta = build_series_metadata(raw_stock)
    assert meta.return_basis == "unadjusted_price"
    assert meta.warnings == ("warn_a", "warn_b")

    clean_q = build_quality_report(raw_stock)
    assert not clean_q.has_anomalies

    dirty = raw_stock.copy()
    dirty.attrs["data_quality"] = {
        "missing_policy": "ffill",
        "total_rows": 10,
        "dropped_invalid_rows": 0,
        "filled_price_rows": 2,
        "missing_volume_rows": 0,
        "zero_volume_rows": 0,
        "non_positive_price_rows": 0,
        "negative_volume_rows": 0,
        "ohlc_anomaly_rows": 0,
        "true_zero_return_rows": 0,
        "max_calendar_gap_days": 3,
        "start_date": "2024-01-02",
        "end_date": "2024-01-15",
    }
    dirty_q = build_quality_report(dirty)
    assert dirty_q.has_anomalies
    assert dirty_q.missing_policy == "ffill"


# ---------------------------------------------------------------------------
# 2. OHLCV Quality Preservation & Cache Slice Hardening
# ---------------------------------------------------------------------------


def test_prepare_ohlcv_ffill_preserved_across_validation_and_features() -> None:
    dates = pd.bdate_range("2024-01-02", periods=12)
    closes = [100.0, 101.0, np.nan, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0]
    df = _make_ohlcv(dates, closes)
    prepared = prepare_ohlcv(df, missing_policy="ffill")
    assert prepared.attrs["data_quality"]["missing_policy"] == "ffill"
    assert prepared.attrs["data_quality"]["filled_price_rows"] >= 1

    revalidated = validate_ohlcv(prepared)
    assert revalidated.attrs["data_quality"]["missing_policy"] == "ffill"
    assert (
        revalidated.attrs["data_quality"]["filled_price_rows"]
        == prepared.attrs["data_quality"]["filled_price_rows"]
    )

    featured = OHLCVFeatureEngineer().transform(prepared, sma_short_window=3, sma_long_window=5)
    assert featured.attrs["data_quality"]["missing_policy"] == "ffill"
    assert (
        featured.attrs["data_quality"]["filled_price_rows"]
        == prepared.attrs["data_quality"]["filled_price_rows"]
    )


def test_symbol_alias_normalization_and_cache_slice_quality(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(ohlcv_cache.ENV_DISABLE_CACHE, raising=False)
    monkeypatch.setenv(ohlcv_cache.ENV_CACHE_DIR, str(tmp_path))

    assert normalize_symbol_alias("hk_stock", "700") == "00700"
    assert normalize_symbol_alias("hk_stock", " 00700 ") == "00700"
    assert normalize_symbol_alias("fx", "USD/CNY") == "USDCNH"
    assert normalize_symbol_alias("fx", "USDCNY") == "USDCNH"
    assert normalize_symbol_alias("bond_us_yield", US_TREASURY_YIELD_10Y_CN_COL) == "10Y"
    assert normalize_symbol_alias("gold", "Au99.99") == "GC"

    assert ohlcv_cache.canonical_cache_symbol(
        "hk_stock", "700"
    ) == ohlcv_cache.canonical_cache_symbol("hk_stock", "00700")
    assert ohlcv_cache.canonical_cache_symbol(
        "fx", "USD/CNY"
    ) == ohlcv_cache.canonical_cache_symbol("fx", "USDCNH")

    # January is clean (no zero volume, <=3d weekend gaps); February introduces zero volume + 8d gap
    jan_dates = pd.bdate_range("2024-01-02", "2024-01-19")
    feb_dates = pd.DatetimeIndex(["2024-02-01", "2024-02-12", "2024-02-13"])
    all_dates = jan_dates.append(feb_dates)
    vols = np.full(len(all_dates), 5000.0)
    vols[-1] = 0.0
    full_df = _make_ohlcv(
        all_dates,
        np.linspace(300.0, 320.0, len(all_dates)),
        volumes=vols,
        asset_type="hk_stock",
        symbol="700",
        adjust="hfq",
        currency="HKD",
        data_source="test_vendor",
    )
    full_df = prepare_ohlcv(full_df, missing_policy="ffill")

    ohlcv_cache.save_cached_ohlcv(
        full_df, "hk_stock", "700", "2024-01-02", "2024-02-13", adjust="hfq"
    )

    # Query via zero-padded alias "00700" on clean January slice
    jan_slice = ohlcv_cache.load_cached_ohlcv(
        "hk_stock", "00700", "2024-01-02", "2024-01-19", adjust="hfq"
    )
    assert jan_slice is not None
    assert jan_slice.attrs["cache_hit"] is True
    assert jan_slice.attrs["canonical_symbol"] == "00700"
    assert jan_slice.attrs["data_quality"]["quality_scope"] == "slice"
    assert jan_slice.attrs["data_quality"]["missing_policy"] == "ffill"
    assert jan_slice.attrs["data_quality"]["total_rows"] == len(jan_dates)
    assert jan_slice.attrs["data_quality"]["zero_volume_rows"] == 0
    assert jan_slice.attrs["data_quality"]["max_calendar_gap_days"] <= 3
    assert jan_slice.attrs["cache_source_data_quality"]["zero_volume_rows"] == 1
    assert jan_slice.attrs["fetched_at"]

    # Query full range sets quality_scope == "full" and captures full zero volume & 13-day gap
    full_hit = ohlcv_cache.load_cached_ohlcv(
        "hk_stock", "00700", "2024-01-02", "2024-02-13", adjust="hfq"
    )
    assert full_hit is not None
    assert full_hit.attrs["data_quality"]["quality_scope"] == "full"
    assert full_hit.attrs["data_quality"]["zero_volume_rows"] == 1
    assert full_hit.attrs["data_quality"]["max_calendar_gap_days"] >= 10

    assert ohlcv_cache.clear_ohlcv_cache() == 2  # .parquet + .meta.json for the single entry


# ---------------------------------------------------------------------------
# 3. Financial Semantics: yield_level, fx_rate, and Single-Asset vs Portfolio
# ---------------------------------------------------------------------------


def test_yield_level_hard_blocks_invalid_metrics_and_fx_conversion() -> None:
    dates = pd.bdate_range("2024-01-02", periods=35)
    yields = np.linspace(3.9, 4.25, len(dates))
    df = _make_ohlcv(
        dates,
        yields,
        asset_type="bond_us_yield",
        symbol="10Y",
        currency="USD",
        series_kind="yield_level",
        return_basis="yield_level",
    )

    metrics = get_quant_metrics(df)
    blocked_keys = (
        "cumulative_return",
        "cagr",
        "max_drawdown",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "var_95",
        "cvar_95",
    )
    for k in blocked_keys:
        assert math.isnan(metrics[k]), f"{k} must be NaN for yield_level"
        assert metrics["metric_reasons"][k] == "yield_level_not_return"

    assert metrics["latest_level"] == pytest.approx(yields[-1])
    assert metrics["level_change"] == pytest.approx(yields[-1] - yields[0])
    assert metrics["level_min"] == pytest.approx(yields.min())
    assert metrics["level_max"] == pytest.approx(yields.max())
    assert math.isfinite(metrics["level_diff_std"])
    assert "水平" in metrics["cumulative_return_label"]
    assert "水平" in metrics["cagr_label"]
    assert metrics["metric_disclaimer"]

    with pytest.raises(ValueError, match="yield_level"):
        convert_ohlcv_to_base_currency(df, "bond_us_yield", "10Y", "CNY")


def test_fx_rate_rejects_erroneous_conversion_and_converts_equity_with_warnings() -> None:
    dates = pd.bdate_range("2024-01-02", periods=25)
    fx_series_df = _make_ohlcv(
        dates,
        np.linspace(7.15, 7.25, len(dates)),
        asset_type="fx",
        symbol="USDCNY",
        currency="CNY",
        series_kind="fx_rate",
        return_basis="fx_rate",
    )

    for kwargs in (
        {"asset_key": "fx", "symbol": "USDCNY"},
        {"asset_key": "etf", "symbol": "MOCK"},  # Still blocked via attrs series_kind/return_basis
    ):
        with pytest.raises(ValueError, match="fx_rate"):
            convert_ohlcv_to_base_currency(fx_series_df, target_currency="USD", **kwargs)

    local_same = convert_ohlcv_to_base_currency(fx_series_df, "fx", "USDCNY", "LOCAL")
    assert local_same.attrs["fx_converted"] is False

    fx_metrics = get_quant_metrics(fx_series_df, trading_days=260)
    assert fx_metrics["semantic_warnings"]
    assert any("Carry" in w or "汇率序列" in w for w in fx_metrics["semantic_warnings"])
    assert "汇率序列" in fx_metrics["metric_disclaimer"]
    assert fx_metrics["cumulative_return_label"] == "累计汇率变动"
    assert fx_metrics["cagr_label"] == "年化汇率变动"

    eq_df = _make_ohlcv(
        dates,
        np.full(len(dates), 100.0),
        asset_type="us_stock",
        symbol="AAPL",
        currency="USD",
    )
    with pytest.raises(ValueError, match="fx_rate"):
        align_multi_asset_frames(
            {"us_stock:AAPL": eq_df, "fx:USDCNY": fx_series_df},
            base_currency="USD",
        )
    local_aligned = align_multi_asset_frames(
        {"us_stock:AAPL": eq_df, "fx:USDCNY": fx_series_df},
        base_currency="LOCAL",
    )
    assert any("fx_rate" in w or "即期汇率" in w for w in local_aligned.warnings)

    # Valid US equity converted to CNY with a sparse FX series triggering staleness warning
    sparse_fx_dates = pd.DatetimeIndex([dates[0], dates[12], dates[-1]])
    sparse_fx = _make_ohlcv(
        sparse_fx_dates,
        [7.0, 7.2, 7.25],
        asset_type="fx",
        symbol="USDCNH",
        currency="CNY",
    )
    converted = convert_ohlcv_to_base_currency(
        eq_df,
        "us_stock",
        "AAPL",
        "CNY",
        fx_loader=lambda pair, s, e: sparse_fx,
    )
    assert converted.attrs["fx_converted"] is True
    assert converted.attrs["native_currency"] == "USD"
    assert converted.attrs["currency"] == "CNY"
    assert converted.attrs["fx_pair"] == "USDCNH"
    assert len(converted.attrs["fx_warnings"]) == len(set(converted.attrs["fx_warnings"]))
    assert converted.attrs["fx_warnings"]  # Stale >4d gap warning propagated
    assert converted["Close"].iloc[0] == pytest.approx(700.0)


def test_single_asset_simple_convention_matches_portfolio_backtest_math() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=60)
    daily_simple = rng.normal(0.001, 0.015, size=len(dates) - 1)
    closes = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + daily_simple)])
    df = _make_ohlcv(dates, closes, asset_type="etf", symbol="510300", currency="CNY")

    rf = 0.02
    ppy = 252

    single_simple = get_quant_metrics(
        df, risk_free_rate=rf, trading_days=ppy, return_convention="simple"
    )
    nav = pd.Series(closes / closes[0], index=dates)
    bt = compute_backtest_metrics(
        nav, periods_per_year=ppy, risk_free_rate=rf, return_convention="simple"
    )

    assert single_simple["return_convention"] == "simple"
    assert single_simple["periods_per_year"] == ppy
    assert single_simple["cagr"] == pytest.approx(bt["cagr"], rel=1e-10)
    assert single_simple["annualized_volatility"] == pytest.approx(
        bt["annualized_volatility"], rel=1e-10
    )
    assert single_simple["sharpe_ratio"] == pytest.approx(bt["sharpe_ratio"], rel=1e-10)
    assert single_simple["sortino_ratio"] == pytest.approx(bt["sortino_ratio"], rel=1e-10)
    assert abs(single_simple["max_drawdown"]) == pytest.approx(bt["max_drawdown"], rel=1e-10)

    # Log convention stays the single-asset default and differs slightly under volatility
    eng = OHLCVFeatureEngineer()
    single_log = get_quant_metrics(df, risk_free_rate=rf, trading_days=ppy)
    assert single_log["return_convention"] == "log"
    assert compute_annualized_volatility(
        eng, df, trading_days=ppy, return_convention="simple"
    ) == pytest.approx(bt["annualized_volatility"], rel=1e-10)
    assert compute_sharpe_ratio(
        eng, df, risk_free_rate=rf, trading_days=ppy, return_convention="simple"
    ) == pytest.approx(bt["sharpe_ratio"], rel=1e-10)
    assert compute_sortino_ratio(
        eng, df, risk_free_rate=rf, trading_days=ppy, return_convention="simple"
    ) == pytest.approx(bt["sortino_ratio"], rel=1e-10)
    assert single_log["annualized_volatility"] != pytest.approx(
        bt["annualized_volatility"], rel=1e-4
    )

    # Portfolio backtest & walk-forward explicitly declare simple return_convention in metadata
    ret_df = pd.DataFrame(
        {"A": daily_simple, "B": daily_simple * 0.5 + 0.0002},
        index=dates[1:],
    )
    pf_bt = backtest_portfolio(ret_df, [0.5, 0.5], periods_per_year=ppy, risk_free_rate=rf)
    assert pf_bt.attrs["performance"]["return_convention"] == "simple"
    wf = run_walk_forward_backtest(
        ret_df,
        window_mode="rolling",
        train_window=20,
        periods_per_year=ppy,
        risk_free_rate=rf,
    )
    assert wf.performance["return_convention"] == "simple"


# ---------------------------------------------------------------------------
# 4. Module Decoupling, Service Layer, Logging & CLI Integration
# ---------------------------------------------------------------------------


def test_fresh_import_order_has_no_circular_import() -> None:
    code = (
        "import currency, asset_loader, data_sources.crypto, contracts, services.analysis_service; "
        "assert contracts.SCHEMA_VERSION == '0.1.0'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert cli_main._convert_ohlcv_to_base_currency is convert_ohlcv_to_base_currency


def test_analysis_service_single_pass_parameters_and_china_fallback_warning(
    monkeypatch, caplog
) -> None:
    # Verify custom window & momentum flags take effect in one pass without overwrite
    dates = pd.bdate_range("2024-01-02", periods=40)
    wave_closes = 10.0 + np.linspace(0.0, 4.0, len(dates)) + 0.3 * np.sin(np.arange(len(dates)))
    raw = _make_ohlcv(
        dates,
        wave_closes,
        asset_type="etf",
        symbol="510300",
        currency="CNY",
        adjust="hfq",
        data_source="mock",
    )
    req = AnalysisRequest(
        asset_type="etf",
        symbol="510300",
        start_date="2024-01-02",
        end_date="2024-02-26",
        adjust="hfq",
        sma_short=3,
        sma_long=6,
        bb_window=5,
        include_momentum=True,
        return_convention="simple",
    )
    result = analyze_single_asset(req, loader=lambda *a, **kw: raw)
    assert result.status == "ok"
    assert result.asset.return_basis == "adjusted_price"
    assert result.metrics["return_convention"] == "simple"
    assert result.data_frame is not None
    # With sma_short=3, index 2 must already be finite (default 20 would be NaN at index 2)
    assert pd.notna(result.data_frame["sma_3"].iloc[2])
    assert pd.notna(result.data_frame["sma_6"].iloc[5])
    assert {"bollinger_upper_5", "bollinger_lower_5", "rsi_14", "macd_hist"}.issubset(
        result.data_frame.columns
    )

    # Verify AKShareFetcher logs & records fallback warnings instead of silent swallowing
    fetcher = AKShareFetcher(max_retries=1, retry_delay=0.0)

    def _boom(*args, **kwargs):
        raise RuntimeError("em down")

    def _ok_alt(route, symbol, start_s, end_s, adjust):
        fetcher._last_alt_warnings.append("sina failed: fallback to tx")
        out = pd.DataFrame(
            {
                "日期": ["2024-01-02", "2024-01-03", "2024-01-04"],
                "开盘": [10.0, 10.1, 10.2],
                "最高": [10.2, 10.3, 10.4],
                "最低": [9.9, 10.0, 10.1],
                "收盘": [10.1, 10.2, 10.3],
                "成交量": [1000.0, 1100.0, 1200.0],
            }
        )
        out.attrs["data_source"] = "akshare.stock_zh_a_hist_tx"
        return out

    monkeypatch.setattr(fetcher, "_load_akshare", lambda: object())
    monkeypatch.setattr(fetcher._primary, "fetch_stock_hist", _boom)
    monkeypatch.setattr(fetcher._eastmoney, "fetch_daily_klines", _boom)
    monkeypatch.setattr(fetcher, "_fetch_sina_tencent_china_equity_raw", _ok_alt)

    with caplog.at_level(logging.WARNING, logger="data_sources.china_equity"):
        df_fallback = fetcher.fetch_stock("600519", "2024-01-02", "2024-01-04")
    assert any("sina failed" in w for w in df_fallback.attrs.get("fallback_warnings", []))
    assert df_fallback.attrs.get("data_source") == "akshare.stock_zh_a_hist_tx"

    # Also test real _fetch_sina_tencent_china_equity_raw warning emission on sina -> tx fallback
    import types

    def _boom_sina(**kw):
        raise RuntimeError("sina down")

    def _ok_tx(**kw):
        return pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-03"],
                "open": [1.0, 1.1],
                "high": [1.2, 1.2],
                "low": [0.9, 1.0],
                "close": [1.1, 1.15],
                "volume": [100.0, 120.0],
            }
        )

    real_alt_fetcher = AKShareFetcher(max_retries=1, retry_delay=0.0)
    monkeypatch.setattr(
        real_alt_fetcher,
        "_load_akshare",
        lambda: types.SimpleNamespace(fund_etf_hist_sina=_boom_sina, stock_zh_a_hist_tx=_ok_tx),
    )
    with caplog.at_level(logging.WARNING, logger="data_sources.china_equity"):
        alt_df = real_alt_fetcher._fetch_sina_tencent_china_equity_raw(
            "etf",
            "510300",
            pd.Timestamp("2024-01-02"),
            pd.Timestamp("2024-01-03"),
            "",
        )
    assert not alt_df.empty
    assert any("sina down" in w for w in real_alt_fetcher._last_alt_warnings)
    assert any("sina down" in rec.message for rec in caplog.records)


def test_cli_run_analysis_delegates_to_service_and_prints_quality_and_fx_warnings(
    monkeypatch, capsys, tmp_path
) -> None:
    dates = pd.bdate_range("2024-01-02", periods=30)
    raw = _make_ohlcv(
        dates,
        np.linspace(150.0, 165.0, len(dates)),
        asset_type="us_stock",
        symbol="AAPL",
        currency="USD",
        adjust="qfq",
        data_source="mock_us",
    )
    sparse_fx = _make_ohlcv(
        pd.DatetimeIndex([dates[0], dates[-1]]),
        [7.1, 7.2],
        asset_type="fx",
        symbol="USDCNH",
        currency="CNY",
    )

    monkeypatch.setattr(cli_main, "load_asset_data", lambda *a, **kw: raw.copy())
    monkeypatch.setattr("currency.load_fx_data", lambda pair, s, e: sparse_fx.copy())

    chart_path = tmp_path / "chart.png"
    res = cli_main.run_analysis(
        asset_type="us_stock",
        symbol="AAPL",
        start_date="2024-01-02",
        end_date="2024-02-12",
        base_currency="CNY",
        save_chart=chart_path,
        show=False,
    )
    out = capsys.readouterr().out
    assert isinstance(res["analysis_result"], AnalysisResult)
    assert res["analysis_result"].asset.fx_converted is True
    assert "数据质量" in out
    assert "计算口径" in out
    assert "return_convention=log" in out
    assert "FX 提示" in out or "汇率提示" in out
    assert chart_path.exists()
