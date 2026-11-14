"""M0 baseline characterization and P0 regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import analysis_metrics
import app
import asset_loader
import main
from asset_catalog import (
    ASSET_DEFAULT_CURRENCY,
    ASSET_RETURN_BASIS,
    ASSET_TRADING_DAYS,
    SUPPORTED_ASSET_TYPES,
)
from data_sources.cache import build_cache_key, load_cached_ohlcv, save_cached_ohlcv
from data_sources.errors import EmptySymbolDataError
from feature_engineering import OHLCVFeatureEngineer
from ohlcv import prepare_ohlcv


def _enable_tmp_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    cache_dir = tmp_path / "ohlcv_cache"
    monkeypatch.delenv("QUANT_DISABLE_CACHE", raising=False)
    monkeypatch.setenv("QUANT_CACHE_DIR", str(cache_dir))
    return cache_dir


def test_app_renders_ohlcv_data_quality_warning(
    monkeypatch: pytest.MonkeyPatch, sample_ohlcv: pd.DataFrame
) -> None:
    from streamlit.testing.v1 import AppTest

    dirty = sample_ohlcv.copy()
    dirty.iloc[2, dirty.columns.get_loc("Close")] = float("nan")
    dirty.iloc[3, dirty.columns.get_loc("Open")] = float("nan")
    dirty.iloc[5, dirty.columns.get_loc("Volume")] = float("nan")
    cleaned = prepare_ohlcv(dirty, drop_invalid=True, missing_policy="drop")
    assert cleaned.attrs["data_quality"]["dropped_invalid_rows"] == 2
    assert cleaned.attrs["data_quality"]["missing_volume_rows"] == 1

    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()
    monkeypatch.setattr(asset_loader, "_fetch_etf_data", lambda **kwargs: cleaned.copy())

    page = AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()
    page.sidebar.button[0].click().run()
    assert not page.exception
    warnings = [w.value for w in page.warning]
    assert any(
        "数据质量提示" in msg and "清理异常/缺失 2 行" in msg and "缺失成交量补零 1 行" in msg
        for msg in warnings
    )
    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()


def test_app_renders_ffill_quality_warning_and_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch, sample_ohlcv: pd.DataFrame
) -> None:
    from streamlit.testing.v1 import AppTest

    dirty = sample_ohlcv.copy()
    dirty.iloc[4, dirty.columns.get_loc("Close")] = float("nan")
    ffilled = prepare_ohlcv(dirty, drop_invalid=True, missing_policy="ffill")
    assert ffilled.attrs["data_quality"]["filled_price_rows"] == 1

    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()
    monkeypatch.setattr(asset_loader, "_fetch_etf_data", lambda **kwargs: ffilled.copy())

    page = AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()
    page.sidebar.button[0].click().run()
    assert not page.exception
    warnings = [w.value for w in page.warning]
    assert any("数据质量提示" in msg and "前向填补价格 1 行" in msg for msg in warnings)

    # Also verify fallback rendering if upstream caller injects legacy metric quality keys directly.
    monkeypatch.setattr(
        analysis_metrics,
        "get_quant_metrics",
        lambda df, **kwargs: {
            "cagr": 0.05,
            "cagr_label": "CAGR",
            "max_drawdown": 0.1,
            "sharpe_ratio": 1.0,
            "latest_signal": "均值回归中",
            "cumulative_return": 0.12,
            "cumulative_return_label": "累计收益率",
            "annualized_volatility": 0.15,
            "sortino_ratio": 1.2,
            "calmar_ratio": 0.5,
            "var_95": -0.02,
            "cvar_95": -0.03,
            "featured_df": OHLCVFeatureEngineer().transform(df),
            "data_source": "mock",
            "currency": "CNY",
            "return_basis": "adjusted_price",
            "metric_disclaimer": "",
            "data_quality": {
                "dropped_rows": 3,
                "ffilled_cells": 2,
                "missing_policy": "ffill",
            },
        },
    )
    app.load_asset_data.clear()
    page_legacy = AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()
    page_legacy.sidebar.button[0].click().run()
    assert not page_legacy.exception
    legacy_warnings = [w.value for w in page_legacy.warning]
    assert any(
        "数据质量提示" in msg and "清理异常/缺失 3 行" in msg and "前向填补价格 2 行" in msg
        for msg in legacy_warnings
    )
    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()


def test_cache_strips_proxy_url_from_meta_and_redacts_in_memory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_ohlcv: pd.DataFrame
) -> None:
    import ccxt

    cache_dir = _enable_tmp_cache(monkeypatch, tmp_path)
    ts_ms = [
        int(pd.Timestamp(ts).tz_localize("UTC").timestamp() * 1000) for ts in sample_ohlcv.index
    ]
    ohlcv_rows = [
        [t, float(o), float(h), float(low), float(c), float(v)]
        for t, o, h, low, c, v in zip(
            ts_ms,
            sample_ohlcv["Open"],
            sample_ohlcv["High"],
            sample_ohlcv["Low"],
            sample_ohlcv["Close"],
            sample_ohlcv["Volume"],
        )
    ]

    class FakeExchange:
        def __init__(self, config: dict[str, object]) -> None:
            self.config = config
            self._called = False

        def fetch_ohlcv(
            self, symbol: str, timeframe: str, since: int, limit: int
        ) -> list[list[object]]:
            if self._called:
                return []
            self._called = True
            return ohlcv_rows

    monkeypatch.setattr(ccxt, "binance", FakeExchange)
    start_s = sample_ohlcv.index[0].strftime("%Y-%m-%d")
    end_s = sample_ohlcv.index[-1].strftime("%Y-%m-%d")

    df = asset_loader.load_asset_data(
        "crypto",
        "BTC/USDT",
        start_s,
        end_s,
        proxy_url="http://alice:topsecret@127.0.0.1:7890",
    )
    assert "topsecret" not in str(df.attrs.get("proxy_url", ""))
    assert "alice" not in str(df.attrs.get("proxy_url", ""))

    key = build_cache_key("crypto", "BTC/USDT", adjust=None)
    meta_text = (cache_dir / f"{key}.meta.json").read_text(encoding="utf-8")
    meta = json.loads(meta_text)
    assert "proxy_url" not in meta.get("attrs", {})
    assert "topsecret" not in meta_text
    assert "alice" not in meta_text
    assert "127.0.0.1:7890" not in meta_text

    # Direct save_cached_ohlcv must also strip proxy_url from attrs.
    direct = sample_ohlcv.copy()
    direct.attrs["proxy_url"] = "http://bob:hunter2@proxy.internal:8080"
    saved = save_cached_ohlcv(direct, "etf", "510300", start_s, end_s, adjust="hfq")
    assert saved is not None
    direct_meta_text = saved.with_suffix(".meta.json").read_text(encoding="utf-8")
    assert "proxy_url" not in json.loads(direct_meta_text).get("attrs", {})
    assert "hunter2" not in direct_meta_text


def test_cache_empty_slice_returns_miss_and_triggers_refetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_ohlcv: pd.DataFrame
) -> None:
    _enable_tmp_cache(monkeypatch, tmp_path)

    head = sample_ohlcv.iloc[:5]
    tail = sample_ohlcv.iloc[25:35]
    gapped = pd.concat([head, tail])
    cov_start = gapped.index[0].strftime("%Y-%m-%d")
    cov_end = gapped.index[-1].strftime("%Y-%m-%d")
    gap_start = (sample_ohlcv.index[6]).strftime("%Y-%m-%d")
    gap_end = (sample_ohlcv.index[20]).strftime("%Y-%m-%d")

    assert save_cached_ohlcv(gapped, "etf", "510300", cov_start, cov_end, adjust="hfq") is not None
    assert load_cached_ohlcv("etf", "510300", gap_start, gap_end, adjust="hfq") is None

    # Upstream backfill returns rows for the gap instead of failing on empty cache slice.
    backfill = sample_ohlcv.iloc[6:21].copy()
    backfill.attrs["data_source"] = "upstream.backfill"
    calls = 0

    def fake_fetch(**kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return backfill.copy()

    monkeypatch.setattr(asset_loader, "_fetch_etf_data", fake_fetch)
    loaded = asset_loader.load_asset_data("etf", "510300", gap_start, gap_end, adjust="hfq")
    assert calls == 1
    assert loaded.attrs["cache_hit"] is False
    assert len(loaded) == len(backfill)

    # Re-test when upstream legitimately raises EmptySymbolDataError.
    assert save_cached_ohlcv(gapped, "etf", "510300", cov_start, cov_end, adjust="hfq") is not None

    def fake_empty_fetch(**kwargs: object) -> pd.DataFrame:
        raise EmptySymbolDataError("区间无交易日数据")

    monkeypatch.setattr(asset_loader, "_fetch_etf_data", fake_empty_fetch)
    with pytest.raises(EmptySymbolDataError):
        asset_loader.load_asset_data("etf", "510300", gap_start, gap_end, adjust="hfq")


def test_m0_baseline_contracts_and_cli_compat(
    monkeypatch: pytest.MonkeyPatch, sample_ohlcv: pd.DataFrame, capsys: pytest.CaptureFixture[str]
) -> None:
    assert len(SUPPORTED_ASSET_TYPES) == 13
    for asset_key in SUPPORTED_ASSET_TYPES:
        assert ASSET_TRADING_DAYS[asset_key] > 0
        assert (
            isinstance(ASSET_DEFAULT_CURRENCY[asset_key], str) and ASSET_DEFAULT_CURRENCY[asset_key]
        )
        assert isinstance(ASSET_RETURN_BASIS[asset_key], str) and ASSET_RETURN_BASIS[asset_key]

    monkeypatch.setattr(main, "load_asset_data", lambda *args, **kwargs: sample_ohlcv.copy())
    monkeypatch.setattr(main.MplfinanceVisualizer, "plot", lambda self, df, **kwargs: None)
    ok_code = main.main(["--asset-type", "etf", "--symbol", "510300", "--no-show"])
    assert ok_code == 0
    assert "资产数学诊断报告" in capsys.readouterr().out

    def fail_empty(*args: object, **kwargs: object) -> pd.DataFrame:
        raise EmptySymbolDataError("无数据")

    monkeypatch.setattr(main, "load_asset_data", fail_empty)
    err_code = main.main(["--asset-type", "etf", "--symbol", "510300", "--no-show"])
    assert err_code == 1
