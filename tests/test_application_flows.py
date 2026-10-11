"""Offline CLI, Streamlit, CSV, chart and crypto-pagination smoke tests."""

import io
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import app
import asset_loader
import main
from charts import build_price_volume_figure
from feature_engineering import OHLCVFeatureEngineer
from visualizer import MplfinanceVisualizer


def test_cli_workflow_uses_shared_data_and_report(monkeypatch, sample_ohlcv, capsys):
    seen = []

    def load(asset_type, symbol, start_date, end_date, **kwargs):
        seen.append((asset_type, symbol, kwargs["adjust"]))
        return sample_ohlcv.copy()

    monkeypatch.setattr(main, "load_china_equity_ohlcv", load)
    monkeypatch.setattr(main.MplfinanceVisualizer, "plot", lambda self, df, **kwargs: "figure")
    result = main.run_analysis("etf", "510300", "2024-01-02", "2024-06-30")
    assert seen == [("etf", "510300", "hfq")]
    assert result["figure"] == "figure"
    assert "资产数学诊断报告" in capsys.readouterr().out
    assert result["featured_df"].attrs["symbol"] == "510300"


def test_csv_keeps_contract_bom_and_safe_filename(sample_ohlcv):
    content, filename = app._cleaned_ohlcv_to_csv_bytes_and_filename(
        sample_ohlcv, "crypto", "BTC/USDT", "2024-01-02", "2024-06-30"
    )
    assert content.startswith(b"\xef\xbb\xbf")
    assert filename == "crypto_BTC_USDT_2024-01-02_2024-06-30_ohlcv.csv"
    parsed = pd.read_csv(io.BytesIO(content))
    assert parsed.columns.tolist() == ["Date", "Open", "High", "Low", "Close", "Volume"]
    np.testing.assert_allclose(parsed["Close"], sample_ohlcv["Close"])


@pytest.mark.parametrize(
    "asset,kind,volume,trace",
    [
        ("etf", "price", 100, "candlestick"),
        ("etf", "price", 0, "candlestick"),
        ("mutual_fund", "price", 0, "scatter"),
        ("bond_us_yield", "yield_level", 0, "scatter"),
    ],
)
def test_price_chart_preserves_asset_display_rules(sample_ohlcv, asset, kind, volume, trace):
    frame = OHLCVFeatureEngineer().transform(sample_ohlcv.assign(Volume=float(volume)))
    frame.attrs.update(asset_type=asset, series_kind=kind)
    figure = build_price_volume_figure(frame)
    assert figure.data[0].type == trace
    assert any(t.type == "bar" for t in figure.data) == (asset == "etf" and volume > 0)


@pytest.mark.parametrize("nrows", [3, 120])
@pytest.mark.parametrize("volume", [False, True])
def test_cli_chart_handles_short_history(sample_ohlcv, tmp_path, nrows, volume):
    import matplotlib.pyplot as plt

    featured = OHLCVFeatureEngineer().transform(sample_ohlcv.iloc[:nrows])
    output = tmp_path / "chart.png"
    figure = MplfinanceVisualizer().plot(featured, show=False, volume=volume, savepath=str(output))
    try:
        assert output.stat().st_size > 1000
        assert len(figure.axes) == (8 if volume else 6)
    finally:
        plt.close(figure)


def test_crypto_pagination_includes_end_date_and_normalizes_utc():
    day_ms = 86400000
    start = pd.Timestamp("2024-01-01").value // 1000000
    calls = []

    class Exchange:
        def fetch_ohlcv(self, symbol, **kwargs):
            since = kwargs["since"]
            calls.append(since)
            return [[since + i * day_ms, 10, 12, 9, 11, 100] for i in range(2)]

    rows = asset_loader._fetch_crypto_ohlcv_rows(
        Exchange(), "BTC/USDT", "2024-01-01", "2024-01-03", batch_limit=2
    )
    frame = asset_loader._normalize_crypto_data(rows, "2024-01-01", "2024-01-03")
    assert calls == [start, start + 2 * day_ms]
    assert len(frame) == 3
    assert frame.index[-1] == pd.Timestamp("2024-01-03")
    assert frame.index.tz is None


@pytest.mark.parametrize("custom", [False, True])
def test_cli_chart_supports_custom_or_no_indicators(sample_ohlcv, custom):
    import matplotlib.pyplot as plt

    frame = sample_ohlcv.assign(trend=sample_ohlcv["Close"] * 0.99)
    figure = MplfinanceVisualizer().plot(
        frame,
        show=False,
        main_panel_cols=["trend"] if custom else [],
        lower_panel_cols=[],
    )
    plt.close(figure)


def test_crypto_pagination_stops_on_repeated_batch():
    row = [pd.Timestamp("2024-01-01").value // 1000000, 1, 1, 1, 1, 0]
    calls = []

    class Exchange:
        def fetch_ohlcv(self, *args, **kwargs):
            calls.append(1)
            return [row]

    asset_loader._fetch_crypto_ohlcv_rows(Exchange(), "BTC/USDT", "2024-01-01", "2024-01-03")
    assert len(calls) == 2


def test_multi_asset_alignment_only_forward_fills(monkeypatch, sample_ohlcv):
    def load(asset_type, symbol, **kwargs):
        return sample_ohlcv.iloc[:3] if symbol == "a" else sample_ohlcv.iloc[1:4]

    monkeypatch.setattr(asset_loader, "load_asset_data", load)
    wide = asset_loader.load_multi_leg_close_wide(
        (("etf", "a"), ("us_stock", "b")), "2024-01-02", "2024-01-05"
    )
    assert wide.columns.tolist() == ["etf:a", "us_stock:b"]
    assert pd.isna(wide.iloc[0, 1])
    assert wide.iloc[-1, 0] == sample_ohlcv.iloc[2]["Close"]


def test_library_imports_do_not_require_optional_ui_or_vendor_packages():
    root = str(Path(__file__).resolve().parents[1])
    script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'streamlit', 'plotly', 'ccxt', 'akshare', 'matplotlib', 'mplfinance', 'scipy'}:
            raise ImportError('intentionally unavailable: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import data_fetcher, feature_engineering, quant_models, asset_loader, charts, app, main
assert app.st is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_portfolio_and_single_asset_reuse_the_same_cache(monkeypatch, sample_ohlcv):
    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs["symbol"])
        return sample_ohlcv.copy()

    monkeypatch.setattr(asset_loader, "_fetch_etf_data", fetch)
    app.load_multi_leg_close_wide(
        (("etf", "510300"), ("etf", "510500")), "2024-01-02", "2024-06-30"
    )
    app.load_asset_data(
        asset_type="etf",
        symbol="510300",
        start_date="2024-01-02",
        end_date="2024-06-30",
        proxy_url=None,
    )
    assert calls == ["510300", "510500"]
    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()


def test_streamlit_app_renders_and_diagnoses_offline(monkeypatch, sample_ohlcv):
    from streamlit.testing.v1 import AppTest

    app.load_asset_data.clear()
    monkeypatch.setattr(asset_loader, "_fetch_etf_data", lambda **kwargs: sample_ohlcv.copy())
    page = AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()
    assert not page.exception
    assert len(page.tabs) == 3
    page.sidebar.button[0].click().run()
    assert not page.exception
    assert not page.error
    assert len(page.metric) == 3
    assert page.metric[0].label == "CAGR"
    app.load_asset_data.clear()


def test_streamlit_advanced_algorithms_run_offline(monkeypatch, sample_ohlcv):
    from streamlit.testing.v1 import AppTest

    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()

    def fetch(**kwargs):
        seed = sum(ord(c) for c in kwargs["symbol"])
        close = 100 * np.exp(
            np.cumsum(np.random.default_rng(seed).normal(0.001, 0.01, len(sample_ohlcv)))
        )
        return sample_ohlcv.assign(Open=close, High=close * 1.01, Low=close * 0.99, Close=close)

    monkeypatch.setattr(asset_loader, "_fetch_etf_data", fetch)
    monkeypatch.setattr(
        asset_loader,
        "load_global_market_data",
        lambda route, symbol, start, end: fetch(symbol=symbol),
    )
    page = AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()
    for key in ("tab3_btn_normalize", "tab3_btn_gbm", "tab3_btn_ef"):
        page.button(key).click().run()
        assert not page.exception
        assert not page.error
    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()
