"""End-to-end UI redesign flow tests: Home launcher, continuity, results-first, deltas & sync."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app
import asset_loader
from ohlcv import compute_ohlcv_quality
from ui.asset_directory import (
    format_asset_display,
    parse_legs_loose,
    resolve_quick_query,
    serialize_legs,
)
from ui.state import (
    RESEARCH_BASKET_KEY,
    WORKSPACE_COMPARE_KEY,
    WORKSPACE_PORTFOLIO_KEY,
    WORKSPACE_SINGLE_KEY,
)


def _install_offline_market(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    dates = pd.bdate_range("2024-01-02", periods=260)
    calls: list[str] = []

    def _series(
        symbol: str,
        seed: int,
        drift: float = 0.0005,
        *,
        asset_type: str = "etf",
        adjust: str = "hfq",
        currency: str = "CNY",
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.011, len(dates))))
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
                "data_source": "mock:redesign",
            }
        )
        df.attrs["data_quality"] = compute_ohlcv_quality(df, missing_policy="drop")
        return df

    fx_df = _series("USDCNY", seed=9, drift=0.0, asset_type="fx", adjust="", currency="CNY")
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
        calls.append(f"{asset_type}:{symbol}")
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


def _open_windowed_home() -> AppTest:
    page = AppTest.from_file(str(Path(app.__file__)), default_timeout=25).run()
    page.date_input("start_date_input").set_value(datetime.date(2024, 1, 2))
    page.date_input("end_date_input").set_value(datetime.date(2024, 12, 30))
    return page.run()


def test_asset_directory_search_and_leg_helpers() -> None:
    assert resolve_quick_query("沪深300") == ("etf", "510300")
    assert resolve_quick_query("中证500") == ("etf", "510500")
    assert resolve_quick_query("苹果") == ("us_stock", "AAPL")
    assert resolve_quick_query("黄金") == ("gold", "GC")
    assert resolve_quick_query("etf,sh510300") == ("etf", "510300")
    assert resolve_quick_query("   ") is None

    assert "沪深300 ETF" in format_asset_display("etf", "510300")
    assert parse_legs_loose("etf:510300") == [("etf", "510300")]
    assert serialize_legs([("etf", "510300"), ("etf", "sh510300"), ("etf", "510500")]) == (
        "etf:510300\netf:510500"
    )


def test_home_quickstart_to_compare_to_portfolio_continuity_and_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_offline_market(monkeypatch)
    page = _open_windowed_home()

    # 1) First-time user clicks 沪深300 ETF quick-start on Home.
    page.button("home_quick_0").click().run()
    assert not page.exception
    assert not page.error
    assert page.session_state["active_view"] == "single"
    ws_single = page.session_state[WORKSPACE_SINGLE_KEY]
    assert ws_single["result"] is not None
    assert ws_single["result"].asset.canonical_symbol == "510300"
    assert ("etf", "510300") in page.session_state[RESEARCH_BASKET_KEY]
    # Results-first: editor collapses automatically after run.
    assert not any(b.key == "single_run_btn" for b in page.button)
    assert any("沪深300 ETF" in m.value for m in page.markdown)

    # Toggling editor open and closed must preserve cached result without refetch or invalidation.
    ver_single = ws_single["result_version"]
    page.button("single_toggle_editor_btn").click().run()
    assert any(b.key == "single_run_btn" for b in page.button)
    page.button("single_collapse_editor_btn").click().run()
    assert page.session_state[WORKSPACE_SINGLE_KEY]["result_version"] == ver_single

    # 2) Handoff Single -> Compare preserves 沪深300 ETF and prompts for 2nd asset.
    page.button("single_to_compare_btn").click().run()
    assert not page.exception
    assert page.session_state["active_view"] == "compare"
    assert page.button("compare_run_btn").disabled
    assert any("输入未完成" in info.value for info in page.info)

    # Quick-add 中证500 ETF and run comparison.
    page.button("compare_quick_etf_510500").click().run()
    assert not page.button("compare_run_btn").disabled
    page.button("compare_run_btn").click().run()
    assert not page.exception
    assert not page.error
    ws_cmp = page.session_state[WORKSPACE_COMPARE_KEY]
    assert ws_cmp["result"] is not None
    assert len(ws_cmp["result"].asset_order) == 2

    # 3) Handoff Compare -> Portfolio inherits compared assets and runs OOS backtest.
    page.button("compare_to_portfolio_btn").click().run()
    assert not page.exception
    assert page.session_state["active_view"] == "portfolio"
    legs_after_handoff = parse_legs_loose(page.text_area("tab3_ef_legs_input").value)
    assert legs_after_handoff == [("etf", "510300"), ("etf", "510500")]
    page.button("tab3_btn_ef").click().run()
    assert not page.exception
    assert not page.error
    ws_port = page.session_state[WORKSPACE_PORTFOLIO_KEY]
    assert ws_port["result"] is not None

    # 4) Modify analysis window via 6M preset -> invalidates & shows change notice -> recompute shows delta strip.
    page.button("date_preset_6M").click().run()
    assert page.session_state[WORKSPACE_PORTFOLIO_KEY]["result"] is None
    assert page.session_state[WORKSPACE_SINGLE_KEY]["result"] is None
    assert page.session_state[WORKSPACE_COMPARE_KEY]["result"] is None
    assert page.session_state[WORKSPACE_PORTFOLIO_KEY]["change_notice"]
    assert any(b.key == "portfolio_recompute_btn" for b in page.button)

    page.button("portfolio_recompute_btn").click().run()
    assert not page.exception
    assert not page.error
    assert page.session_state[WORKSPACE_PORTFOLIO_KEY]["result"] is not None
    assert page.session_state[WORKSPACE_PORTFOLIO_KEY]["prev_summary"] is not None
    assert any("前后结果对比" in m.value for m in page.markdown)

    # 5) Open editor, adjust cov_method, Rf (synced with sidebar), and rebalance_freq -> recompute.
    page.button("portfolio_toggle_editor_btn").click().run()
    page.selectbox("portfolio_cov_method").set_value("sample").run()
    page.number_input("portfolio_rf_pct_input").set_value(2.5).run()
    page.selectbox("portfolio_rebalance_freq").set_value("quarterly").run()
    assert page.number_input("rf_pct_input").value == pytest.approx(2.5)
    assert page.session_state[WORKSPACE_PORTFOLIO_KEY]["result"] is None
    assert page.session_state[WORKSPACE_PORTFOLIO_KEY]["change_notice"]

    page.button("tab3_btn_ef").click().run()
    assert not page.exception
    assert not page.error
    res_updated = page.session_state[WORKSPACE_PORTFOLIO_KEY]["result"]
    assert res_updated is not None
    assert res_updated.methodology["cov_method"] == "sample"
    assert res_updated.parameters["rebalance_freq"] == "quarterly"
    assert res_updated.parameters["risk_free_rate"] == pytest.approx(0.025)

    # 6) Export artifacts exist & navigating back to Single/Compare preserves state without loss.
    assert page.session_state[WORKSPACE_PORTFOLIO_KEY]["exports"] is not None
    assert len(page.get("download_button")) >= 4
    page.button("nav_btn_single").click().run()
    assert page.session_state["active_view"] == "single"
    assert page.session_state[WORKSPACE_SINGLE_KEY]["change_notice"]
    page.button("single_recompute_btn").click().run()
    assert page.session_state[WORKSPACE_SINGLE_KEY]["result"] is not None
    assert page.session_state[WORKSPACE_PORTFOLIO_KEY]["result"] is not None

    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()


def test_delta_polarity_drawdown_and_plotly_legend_separation(sample_ohlcv: pd.DataFrame) -> None:
    import charts
    from feature_engineering import OHLCVFeatureEngineer
    from ui.components.alerts import _format_metric_delta
    from ui.theme import apply_ui_plotly_theme, build_theme_css

    # Deeper drawdown (0.1878 -> 0.2499) must render negative delta tone, shallower renders positive.
    worse_txt, worse_tone = _format_metric_delta(0.1878, 0.2499, "dd")
    assert worse_txt == "-6.21 pp" and worse_tone == "neg"
    better_txt, better_tone = _format_metric_delta(0.2499, 0.1878, "dd")
    assert better_txt == "+6.21 pp" and better_tone == "pos"
    turn_txt, turn_tone = _format_metric_delta(1.073, 1.000, "neutral_ratio")
    assert turn_txt == "-0.073" and turn_tone == "neutral"

    # Claude Dark theme CSS must cover expander summary and popover triggers.
    dark_css = build_theme_css()
    assert '[data-testid="stExpander"] summary' in dark_css
    assert '[data-testid="stPopover"] button' in dark_css

    # Multi-trace price chart moves legend cleanly below plot so rangeselector never overlaps.
    featured = OHLCVFeatureEngineer().transform(sample_ohlcv)
    fig = apply_ui_plotly_theme(charts.build_price_volume_figure(featured))
    assert float(fig.layout.legend.y) < 0.0
