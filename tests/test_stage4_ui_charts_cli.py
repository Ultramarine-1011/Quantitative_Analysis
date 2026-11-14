"""Stage 4 tests: Web UI state/widgets, Plotly builders, Visualizer line mode & expanded CLI."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app
import asset_loader
import charts
import main
from feature_engineering import OHLCVFeatureEngineer
from quant_models import run_walk_forward_backtest, solve_exact_efficient_frontier
from visualizer import MplfinanceVisualizer


def _synthetic_returns(n_days: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2024-01-02", periods=n_days, name="Date")
    return pd.DataFrame(
        {
            "A": rng.normal(0.0006, 0.012, n_days),
            "B": rng.normal(0.0003, 0.009, n_days),
            "C": rng.normal(0.0004, 0.015, n_days),
        },
        index=dates,
    )


def test_plotly_charts_theme_adaptive_and_new_builders(sample_ohlcv):
    featured = OHLCVFeatureEngineer().transform(sample_ohlcv, include_momentum_indicators=True)
    fig_main = charts.build_price_volume_figure(featured)
    assert fig_main.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig_main.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert any("回撤" in str(trace.name) for trace in fig_main.data)
    assert any("波动" in str(trace.name) for trace in fig_main.data)

    fig_risk = charts.build_tab1_risk_figure(featured)
    assert any("RSI" in str(trace.name) for trace in fig_risk.data)

    returns = _synthetic_returns(140)
    prices = (1.0 + returns).cumprod() * 100.0
    fig_multi_norm = charts.build_tab3_multi_normalize_figure(prices, base_currency="USD")
    for trace in fig_multi_norm.data:
        assert float(trace.y[0]) == pytest.approx(1.0)

    exact = solve_exact_efficient_frontier(
        returns, n_points=12, num_random_portfolios=80, random_state=5
    )
    fig_ef = charts.build_tab3_efficient_frontier_figure(exact)
    names = [str(trace.name) for trace in fig_ef.data]
    assert any("精确有效前沿" in n for n in names)
    assert any("GMV" in n for n in names)
    assert any("Tangency" in n for n in names)
    assert any("随机可行组合" in n for n in names)

    wf = run_walk_forward_backtest(returns, train_window=40, rebalance_freq="monthly")
    fig_wf = charts.build_tab3_walk_forward_figure(wf, base_currency="CNY")
    assert len(fig_wf.data) == 5
    fig_w = charts.build_tab3_weight_history_figure(wf.weight_history)
    assert len(fig_w.data) == 3
    assert fig_w.layout.barmode == "stack"


def test_visualizer_removed_mav_and_supports_line_mode(sample_ohlcv, tmp_path):
    import matplotlib.pyplot as plt

    assert "mav" not in inspect.signature(MplfinanceVisualizer.plot).parameters
    featured = OHLCVFeatureEngineer().transform(sample_ohlcv.assign(Volume=0.0))
    featured.attrs.update(asset_type="mutual_fund", series_kind="price", symbol="009691")
    out_png = tmp_path / "fund_line.png"
    fig = MplfinanceVisualizer().plot(featured, show=False, savepath=str(out_png))
    try:
        assert out_png.stat().st_size > 1000
    finally:
        plt.close(fig)


def test_cli_supports_global_assets_currency_and_exports(
    monkeypatch, sample_ohlcv, tmp_path, capsys
):
    def fake_load(asset_type, symbol, start_date, end_date, proxy_url=None, adjust=None):
        df = sample_ohlcv.copy()
        df.attrs.update(
            asset_type=asset_type,
            symbol=symbol,
            series_kind="price",
            data_source="akshare.mock",
            adjust=adjust or "qfq",
            currency="USD" if asset_type == "us_stock" else "CNY",
        )
        return df

    def fake_fx(symbol, start, end):
        fx = sample_ohlcv.copy()
        for col in ("Open", "High", "Low", "Close"):
            fx[col] = 7.25
        fx.attrs.update(asset_type="fx", symbol=symbol, series_kind="price")
        return fx

    monkeypatch.setattr(main, "load_asset_data", fake_load)
    monkeypatch.setattr("currency.load_fx_data", fake_fx)

    chart_path = tmp_path / "charts" / "aapl.png"
    csv_path = tmp_path / "csv" / "aapl.csv"
    exit_code = main.main(
        [
            "--asset-type",
            "us_stock",
            "--symbol",
            "AAPL",
            "--adjust",
            "qfq",
            "--base-currency",
            "CNY",
            "--save-chart",
            str(chart_path),
            "--export-csv",
            str(csv_path),
            "--no-show",
        ]
    )
    assert exit_code == 0
    out_text = capsys.readouterr().out
    assert "资产数学诊断报告" in out_text
    assert "Sortino" in out_text
    assert "Calmar" in out_text
    assert "95% 日 VaR / CVaR" in out_text
    assert chart_path.stat().st_size > 1000
    assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf")

    # Unsupported hfq on us_stock fails explicitly with non-zero exit code.
    bad_code = main.main(
        ["--asset-type", "us_stock", "--symbol", "AAPL", "--adjust", "hfq", "--no-show"]
    )
    assert bad_code == 1


def _build_offline_apptest(monkeypatch, sample_ohlcv):
    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()

    def fetch(**kwargs):
        sym = str(kwargs.get("symbol", "510300"))
        seed = sum(map(ord, sym))
        close = 100.0 * np.exp(
            np.cumsum(np.random.default_rng(seed).normal(0.0008, 0.011, len(sample_ohlcv)))
        )
        df = sample_ohlcv.assign(Open=close, High=close * 1.01, Low=close * 0.99, Close=close)
        df.attrs["symbol"] = sym
        if kwargs.get("adjust") is not None:
            df.attrs["adjust"] = kwargs["adjust"]
        return df

    def fake_global(route, symbol, start, end, **kwargs):
        return fetch(symbol=symbol, **kwargs)

    def fake_fx(symbol, start, end):
        fx = sample_ohlcv.copy()
        for col in ("Open", "High", "Low", "Close"):
            fx[col] = 7.20
        fx.attrs.update(asset_type="fx", symbol=symbol, series_kind="price")
        return fx

    monkeypatch.setattr(asset_loader, "_fetch_etf_data", fetch)
    monkeypatch.setattr(asset_loader, "load_global_market_data", fake_global)
    monkeypatch.setattr(asset_loader, "load_fx_data", fake_fx)
    monkeypatch.setattr(
        asset_loader,
        "load_bond_data",
        lambda route, sym, s, e, **kw: sample_ohlcv.assign(
            Open=4.2, High=4.2, Low=4.2, Close=4.2, Volume=0.0
        ).pipe(
            lambda d: (
                d.assign()
                if not d.attrs.update(asset_type="bond_us_yield", series_kind="yield_level")
                else d
            )
        ),
    )
    return AppTest.from_file(str(Path(app.__file__)), default_timeout=20).run()


def test_app_proxy_persists_and_adjust_matches_capability(monkeypatch, sample_ohlcv):
    page = _build_offline_apptest(monkeypatch, sample_ohlcv)
    page.text_input("proxy_url_input").set_value("http://127.0.0.1:7890").run()
    page.selectbox("asset_type").set_value("us_stock").run()
    assert page.text_input("proxy_url_input").value == "http://127.0.0.1:7890"
    assert tuple(page.selectbox("adjust_input").options) == ("前复权 (qfq)", "不复权 (raw)")
    assert page.selectbox("adjust_input").value == "qfq"

    page.selectbox("asset_type").set_value("etf").run()
    assert page.text_input("proxy_url_input").value == "http://127.0.0.1:7890"
    assert tuple(page.selectbox("adjust_input").options) == (
        "后复权 (hfq)",
        "前复权 (qfq)",
        "不复权 (raw)",
    )

    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()


def test_app_tab3_walk_forward_extra_charts_and_yield_rejection(monkeypatch, sample_ohlcv):
    page = _build_offline_apptest(monkeypatch, sample_ohlcv)
    page.selectbox("tab3_val_mode").set_value(app._VAL_MODE_ROLLING).run()
    page.checkbox("tab3_show_extra_charts").set_value(True).run()
    page.button("tab3_btn_ef").click().run()
    assert not page.exception
    assert not page.error
    # Expect heatmap + exact frontier + walk-forward + weight history + multi normalize = 5 charts
    assert len(page.get("plotly_chart")) == 5

    # Switching preset template updates legs cleanly.
    page.selectbox("tab3_preset_template").set_value("境内大类股债均衡").run()
    assert "etf,511010" in page.text_area("tab3_ef_legs_input").value

    # Yield-level leg must produce user-readable error instead of silent math corruption.
    page.text_area("tab3_ef_legs_input").set_value("etf,510300\nbond_us_yield,10Y").run()
    page.button("tab3_btn_ef").click().run()
    assert not page.exception
    assert any("收益率水平" in err.value for err in page.error)

    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()
