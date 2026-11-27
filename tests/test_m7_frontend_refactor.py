"""M7 frontend refactoring tests: architecture, theme, error taxonomy, state & cache invalidation."""

from __future__ import annotations

import ast
import datetime
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app
import asset_loader
import charts
from data_sources.errors import (
    DataFetchError,
    EmptySymbolDataError,
    NetworkFetchError,
    VendorUpstreamError,
)
from ohlcv import compute_ohlcv_quality
from quant_models import run_walk_forward_backtest, solve_exact_efficient_frontier
from ui.errors import classify_exception, format_ui_error_markdown, make_validation_error
from ui.services_bridge import (
    get_or_build_figures,
    get_or_build_report_exports,
    preview_custom_weights,
    preview_multi_asset_specs,
)
from ui.state import (
    WORKSPACE_COMPARE_KEY,
    WORKSPACE_PORTFOLIO_KEY,
    WORKSPACE_SINGLE_KEY,
    invalidate_if_signature_changed,
    store_workspace_result,
    workspace_state,
)
from ui.theme import CLAUDE_DARK_TOKENS, apply_ui_plotly_theme, build_theme_css


def _install_offline_market(
    monkeypatch: pytest.MonkeyPatch,
    *,
    drift: float = 0.0006,
) -> list[str]:
    dates = pd.bdate_range("2024-01-02", periods=95)
    calls: list[str] = []

    def _series(
        symbol: str,
        seed: int,
        d: float = drift,
        *,
        asset_type: str = "etf",
        adjust: str = "hfq",
        currency: str = "CNY",
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        close = 100.0 * np.exp(np.cumsum(rng.normal(d, 0.011, len(dates))))
        df = pd.DataFrame(
            {
                "Open": close * 0.996,
                "High": close * 1.008,
                "Low": close * 0.992,
                "Close": close,
                "Volume": np.full(len(dates), 10000.0, dtype=float),
            },
            index=dates,
        )
        df.index = pd.DatetimeIndex(df.index).as_unit("ns")
        df.index.name = "Date"
        df.attrs.update(
            {
                "asset_type": asset_type,
                "symbol": symbol,
                "canonical_symbol": symbol,
                "adjust": adjust,
                "currency": currency,
                "data_source": "mock:m7",
            }
        )
        df.attrs["data_quality"] = compute_ohlcv_quality(df, missing_policy="drop")
        return df

    fx_df = _series("USDCNY", seed=9, asset_type="fx", adjust="", currency="CNY")
    for col in ("Open", "High", "Low", "Close"):
        fx_df[col] = 7.2
    fx_df.attrs.update(asset_type="fx", series_kind="fx_rate", currency="CNY")

    def _fetch(
        symbol: str,
        *,
        asset_type: str = "etf",
        adjust: str = "hfq",
        currency: str = "CNY",
        **kwargs: Any,
    ) -> pd.DataFrame:
        calls.append(symbol)
        return _series(
            symbol,
            seed=sum(map(ord, symbol)),
            asset_type=asset_type,
            adjust=adjust,
            currency=currency,
        )

    monkeypatch.setattr(asset_loader, "_fetch_etf_data", _fetch)
    monkeypatch.setattr(
        asset_loader,
        "load_global_market_data",
        lambda route, symbol, start, end, **kw: _fetch(
            symbol,
            asset_type=route,
            adjust="qfq" if route == "us_stock" else "",
            currency="USD",
        ),
    )
    monkeypatch.setattr(
        asset_loader,
        "_fetch_gold_data",
        lambda symbol, start, end, **kw: _fetch(
            symbol, asset_type="gold", adjust="", currency="USD"
        ),
    )
    monkeypatch.setattr(
        asset_loader,
        "load_commodity_data",
        lambda symbol, start, end, **kw: _fetch(
            symbol, asset_type="commodity", adjust="", currency="USD"
        ),
    )
    monkeypatch.setattr(
        asset_loader,
        "_fetch_crypto_data",
        lambda symbol, start, end, **kw: _fetch(
            symbol, asset_type="crypto", adjust="", currency="USD"
        ),
    )
    monkeypatch.setattr(asset_loader, "load_fx_data", lambda symbol, start, end, **kw: fx_df.copy())

    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()
    return calls


# ---------------------------------------------------------------------------
# 1. Architecture & Boundaries
# ---------------------------------------------------------------------------


def test_app_split_is_thin_and_backend_does_not_import_ui() -> None:
    root = Path(__file__).resolve().parents[1]
    app_lines = (root / "app.py").read_text(encoding="utf-8").splitlines()
    assert len(app_lines) < 220

    for mod_name in (
        "ui.labels",
        "ui.theme",
        "ui.errors",
        "ui.state",
        "ui.asset_directory",
        "ui.services_bridge",
        "ui.components.asset_picker",
        "ui.components.header",
        "ui.components.sidebar",
        "ui.components.alerts",
        "ui.components.metrics",
        "ui.components.charts",
        "ui.components.tables",
        "ui.components.exports",
        "ui.workspaces.home",
        "ui.workspaces.single_asset",
        "ui.workspaces.compare",
        "ui.workspaces.portfolio",
    ):
        assert importlib.import_module(mod_name) is not None

    forbidden = {"ui", "app", "streamlit"}
    for pkg in ("models", "services", "data_sources", "reporting", "artifacts"):
        for path in (root / pkg).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module.split(".")[0]]
                else:
                    continue
                assert not (set(mods) & forbidden), f"{path} unexpectedly imports {mods}"


# ---------------------------------------------------------------------------
# 2. Theme System & Plotly Harmonization
# ---------------------------------------------------------------------------


def test_theme_css_and_plotly_harmonization_preserve_data_and_heatmap_contrast() -> None:
    css = build_theme_css()
    assert "color-scheme: dark;" in css
    assert CLAUDE_DARK_TOKENS["bg_primary"] in css and "#262624" in css
    assert "#30302e" in css and "#1f1e1d" in css
    assert "#faf9f5" in css and "#e8e6dc" in css and "#d97757" in css
    assert "Lora" in css and "Source Sans 3" in css and "JetBrains Mono" in css
    assert "filter: invert(" not in css

    config_text = (Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml").read_text(
        encoding="utf-8"
    )
    assert 'base = "dark"' in config_text
    assert 'backgroundColor = "#262624"' in config_text
    assert 'primaryColor = "#D97757"' in config_text

    dates = pd.bdate_range("2024-01-02", periods=30)
    df = pd.DataFrame({"A": np.linspace(100, 120, 30)}, index=dates)
    raw_dd = charts.build_drawdown_figure(df, from_nav=True)
    themed = apply_ui_plotly_theme(raw_dd)

    assert themed.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert themed.layout.plot_bgcolor == "rgba(0,0,0,0)"
    assert isinstance(themed.data[0].y, np.ndarray)
    assert themed.layout.font.color == "#e8e6dc"
    assert themed.layout.title.font.color == "#faf9f5"
    assert themed.layout.hoverlabel.bgcolor == "#30302e"

    # Heatmap cell contrast annotations must retain explicit #111111 / #fafafa colors.
    corr = pd.DataFrame([[1.0, -0.9], [-0.9, 1.0]], columns=["A", "B"], index=["A", "B"])
    hm = charts.build_tab3_correlation_heatmap(corr)
    hm_dark = apply_ui_plotly_theme(hm)
    ann_colors = {ann.font.color for ann in hm_dark.layout.annotations}
    assert ann_colors <= {"#111111", "#fafafa"}


def _open_windowed_page() -> AppTest:
    page = AppTest.from_file(str(Path(app.__file__)), default_timeout=25).run()
    page.date_input("start_date_input").set_value(datetime.date(2024, 1, 2))
    page.date_input("end_date_input").set_value(datetime.date(2024, 5, 13))
    return page.run()


def test_single_claude_dark_theme_has_no_switcher_and_styles_figures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_offline_market(monkeypatch)
    page = _open_windowed_page()
    assert not list(page.get("segmented_control"))
    assert "ui_theme_mode" not in page.session_state

    page.button("nav_btn_single").click().run()
    page.button("single_run_btn").click().run()
    assert not page.exception
    assert not page.error
    assert len(calls) == 1
    ws_single = page.session_state[WORKSPACE_SINGLE_KEY]
    assert len(ws_single["figures_cache_key"]) == 2
    assert ws_single["figures"]["price"][2].layout.font.color == "#e8e6dc"

    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()


# ---------------------------------------------------------------------------
# 3. Error Taxonomy & Semantic Distinction
# ---------------------------------------------------------------------------


def test_error_taxonomy_classifies_all_failure_modes() -> None:
    err_empty = classify_exception(EmptySymbolDataError("no rows"))
    assert err_empty.category == "empty_data"
    assert "空数据" in format_ui_error_markdown(err_empty)

    err_net = classify_exception(NetworkFetchError("timeout"))
    assert err_net.category == "market_data"
    assert "网络传输" in format_ui_error_markdown(err_net)

    err_vendor = classify_exception(VendorUpstreamError("bad json"))
    assert err_vendor.category == "market_data"
    err_df = classify_exception(DataFetchError("upstream error"))
    assert err_df.category == "market_data"

    err_val = classify_exception(ValueError("yield_level 不可纳入组合优化"))
    assert err_val.category == "validation"

    err_comp = classify_exception(ValueError("协方差阵奇异无法求逆"))
    assert err_comp.category == "computation"

    err_dep = classify_exception(ImportError("缺少运行依赖 `plotly`"))
    assert err_dep.category == "dependency"

    v_err = make_validation_error("开始日期不能晚于结束日期")
    assert v_err.category == "validation"


def test_apptest_distinguishes_market_empty_validation_and_negative_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1) Normal negative-drift market returns should NOT trigger error banners.
    _install_offline_market(monkeypatch, drift=-0.004)
    page = _open_windowed_page()
    page.button("nav_btn_single").click().run()
    page.button("single_run_btn").click().run()
    assert not page.exception
    assert not page.error
    assert page.session_state[WORKSPACE_SINGLE_KEY]["result"] is not None

    # 2) Validation error (SMA short >= SMA long) clearly labeled as validation.
    page.button("single_toggle_editor_btn").click().run()
    page.number_input("sma_short_input").set_value(80).run()
    page.number_input("sma_long_input").set_value(60).run()
    page.button("single_run_btn").click().run()
    assert len(page.error) == 1
    assert "输入校验" in page.error[0].value
    assert page.session_state[WORKSPACE_SINGLE_KEY]["result"] is None

    # Restore valid SMA windows, then simulate EmptySymbolDataError.
    page.number_input("sma_short_input").set_value(20).run()
    app.load_asset_data.clear()
    monkeypatch.setattr(
        asset_loader,
        "_fetch_etf_data",
        lambda **kwargs: (_ for _ in ()).throw(EmptySymbolDataError("empty slice")),
    )
    page.button("single_run_btn").click().run()
    assert len(page.error) == 1
    assert "空数据" in page.error[0].value

    # 3) Simulate DataFetchError / NetworkFetchError.
    app.load_asset_data.clear()
    monkeypatch.setattr(
        asset_loader,
        "_fetch_etf_data",
        lambda **kwargs: (_ for _ in ()).throw(NetworkFetchError("connection reset")),
    )
    page.button("single_run_btn").click().run()
    assert len(page.error) == 1
    assert "行情网络传输失败" in page.error[0].value

    app.load_asset_data.clear()


# ---------------------------------------------------------------------------
# 4. State & Export Cache Invalidation
# ---------------------------------------------------------------------------


def test_state_and_export_cache_invalidate_on_skipped_charts_and_workspace_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_offline_market(monkeypatch)
    page = _open_windowed_page()

    # Run all 3 workspaces sequentially via navigation rail.
    page.button("nav_btn_single").click().run()
    page.button("single_run_btn").click().run()
    assert len(page.get("plotly_chart")) == 4

    page.button("nav_btn_compare").click().run()
    page.text_area("compare_assets_input").set_value("etf:510300\netf:510500\nus_stock:AAPL").run()
    page.button("compare_run_btn").click().run()
    assert len(page.get("plotly_chart")) == 4

    page.button("nav_btn_portfolio").click().run()
    page.text_area("tab3_ef_legs_input").set_value("etf:510300\netf:510500\nus_stock:AAPL").run()
    page.button("tab3_btn_ef").click().run()
    assert not page.exception
    assert not page.error
    calls_after_run = len(calls)

    ws_port = page.session_state[WORKSPACE_PORTFOLIO_KEY]
    assert ws_port["exports_cache_key"] is not None
    ver_initial = ws_port["result_version"]
    assert len(page.get("plotly_chart")) == 4

    # Toggling extra charts in Portfolio rebuilds figures without refetching or recomputing.
    page.button("portfolio_toggle_editor_btn").click().run()
    page.checkbox("tab3_show_extra_charts").set_value(True).run()
    assert len(calls) == calls_after_run
    ws_port_after = page.session_state[WORKSPACE_PORTFOLIO_KEY]
    assert ws_port_after["result_version"] == ver_initial
    assert len(page.get("plotly_chart")) == 6  # train_test adds normalized + drawdown
    assert ws_port_after["figures_cache_key"][1] is not None
    assert len(ws_port_after["figures_cache_key"][1]) == 6

    # Walk-forward adds weight_history chart cleanly.
    page.selectbox("tab3_val_mode").set_value("walk_forward_rolling").run()
    page.number_input("portfolio_train_window").set_value(35).run()
    page.button("tab3_btn_ef").click().run()
    assert not page.error
    assert len(page.get("plotly_chart")) == 7

    # Edit compare cov_method: only compare invalidates; single & portfolio remain intact.
    page.button("nav_btn_compare").click().run()
    page.button("compare_toggle_editor_btn").click().run()
    page.selectbox("compare_cov_method").set_value("sample").run()
    assert page.session_state[WORKSPACE_COMPARE_KEY]["result"] is None
    assert page.session_state[WORKSPACE_SINGLE_KEY]["result"] is not None
    assert page.session_state[WORKSPACE_PORTFOLIO_KEY]["result"] is not None

    # Edit proxy_url: invalidates computed single & portfolio cleanly.
    page.text_input("proxy_url_input").set_value("http://127.0.0.1:7890").run()
    assert page.session_state[WORKSPACE_SINGLE_KEY]["result"] is None
    assert page.session_state[WORKSPACE_PORTFOLIO_KEY]["result"] is None

    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()


def test_unit_export_cache_invalidation_when_skipped_charts_or_version_changes(
    sample_ohlcv: pd.DataFrame,
) -> None:
    from contracts import AnalysisRequest
    from services.analysis_service import analyze_single_asset

    res = analyze_single_asset(
        AnalysisRequest(
            asset_type="etf",
            symbol="510300",
            start_date=sample_ohlcv.index[0].strftime("%Y-%m-%d"),
            end_date=sample_ohlcv.index[-1].strftime("%Y-%m-%d"),
        ),
        loader=lambda *a, **kw: sample_ohlcv.copy(),
    )
    session: dict[str, Any] = {}
    state = workspace_state(session, WORKSPACE_SINGLE_KEY)
    store_workspace_result(state, signature=("sig1",), result=res)

    exp1 = get_or_build_report_exports(state, res, skipped_charts={})
    exp2 = get_or_build_report_exports(state, res, skipped_charts={})
    assert exp1 is exp2  # Memoized hit on identical (result_version, skipped_charts).

    # Changing skipped_charts must invalidate export cache and rebuild report text.
    exp3 = get_or_build_report_exports(state, res, skipped_charts={"normalized": "口径限制"})
    assert exp3 is not exp1
    assert "口径限制" in exp3["md"]

    # Figures cache reuses result_version and applies Claude Dark palette.
    figs1, _ = get_or_build_figures(state, res)
    figs2, _ = get_or_build_figures(state, res)
    assert figs1 is not None and figs1 is figs2
    assert figs1["price"][2].layout.font.color == "#e8e6dc"

    assert invalidate_if_signature_changed(state, ("sig2",)) is True
    assert state["result"] is None
    assert state["exports"] is None


def test_input_preview_helpers_handle_valid_and_invalid_inputs() -> None:
    specs, err = preview_multi_asset_specs("etf,sh510300\nus_stock:AAPL")
    assert err is None
    assert [(s.asset_type, s.symbol) for s in specs] == [("etf", "510300"), ("us_stock", "AAPL")]

    empty_specs, empty_err = preview_multi_asset_specs("   ")
    assert empty_specs == () and empty_err is not None

    w, w_err = preview_custom_weights("0.6, 0.4", expected_count=2)
    assert w == (0.6, 0.4) and w_err is None

    _, mismatch_err = preview_custom_weights("0.5, 0.5", expected_count=3)
    assert mismatch_err is not None and "不一致" in mismatch_err

    _, zero_err = preview_custom_weights("0, 0", expected_count=2)
    assert zero_err is not None and "大于 0" in zero_err


# ---------------------------------------------------------------------------
# 5. Plotly Chart Upgrades
# ---------------------------------------------------------------------------


def test_plotly_charts_include_rangeselector_and_hovertemplates(sample_ohlcv: pd.DataFrame) -> None:
    from feature_engineering import OHLCVFeatureEngineer

    featured = OHLCVFeatureEngineer().transform(sample_ohlcv, include_momentum_indicators=True)
    fig_price = charts.build_price_volume_figure(featured)
    assert len(fig_price.layout.xaxis.rangeselector.buttons) == 6
    assert any("%{y" in str(tr.hovertemplate) for tr in fig_price.data if tr.type == "scatter")

    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-02", periods=100, name="Date")
    rets = pd.DataFrame(
        {
            "A": rng.normal(0.0005, 0.01, 100),
            "B": rng.normal(0.0004, 0.012, 100),
        },
        index=dates,
    )
    exact = solve_exact_efficient_frontier(rets, n_points=10, num_random_portfolios=20)
    fig_ef = charts.build_tab3_efficient_frontier_figure(exact)
    assert all(tr.customdata is not None for tr in fig_ef.data)
    assert all("夏普" in str(tr.hovertemplate) for tr in fig_ef.data)

    wf = run_walk_forward_backtest(rets, train_window=35, rebalance_freq="monthly")
    fig_wf = charts.build_tab3_walk_forward_figure(wf, base_currency="CNY")
    assert len(fig_wf.data) == 5
    assert len(fig_wf.layout.xaxis.rangeselector.buttons) == 6
