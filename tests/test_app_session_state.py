"""Regression tests for results surviving normal Streamlit reruns and input edits."""

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import app
import asset_loader


@pytest.fixture
def page(monkeypatch, sample_ohlcv):
    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()

    def fetch(**kwargs):
        seed = sum(map(ord, kwargs["symbol"]))
        close = 100 * np.exp(
            np.cumsum(np.random.default_rng(seed).normal(0.001, 0.01, len(sample_ohlcv)))
        )
        return sample_ohlcv.assign(Open=close, High=close * 1.01, Low=close * 0.99, Close=close)

    def fake_fx(symbol, start, end):
        fx = sample_ohlcv.copy()
        for col in ("Open", "High", "Low", "Close"):
            fx[col] = 7.20
        fx.attrs.update(asset_type="fx", symbol=symbol, series_kind="price")
        return fx

    monkeypatch.setattr(asset_loader, "_fetch_etf_data", fetch)
    monkeypatch.setattr(
        asset_loader,
        "load_global_market_data",
        lambda route, symbol, start, end, **kwargs: fetch(symbol=symbol),
    )
    monkeypatch.setattr(asset_loader, "load_fx_data", fake_fx)
    yield AppTest.from_file(app.__file__, default_timeout=20).run()
    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()


def test_dashboard_survives_reruns_and_other_actions(page):
    page.button("nav_btn_single").click().run()
    page.button("single_run_btn").click().run()
    values = [metric.value for metric in page.metric]
    page.run()  # Downloads/reruns preserve computed metrics & charts.
    assert [metric.value for metric in page.metric] == values
    assert len(page.get("plotly_chart")) == 4  # price, rolling, drawdown, normalized
    page.button("single_btn_normalize").click().run()
    assert [metric.value for metric in page.metric] == values
    assert len(page.get("plotly_chart")) == 5
    assert not page.exception


def test_advanced_results_accumulate_and_only_related_edits_invalidate(page):
    page.button("nav_btn_single").click().run()
    for key in ("single_btn_normalize", "single_btn_gbm"):
        page.button(key).click().run()
        assert not page.error
        assert not page.exception
    assert len(page.get("plotly_chart")) == 2  # 1 norm + 1 gbm

    page.button("nav_btn_compare").click().run()
    page.button("compare_run_btn").click().run()
    assert not page.error
    assert len(page.get("plotly_chart")) == 4

    page.button("nav_btn_portfolio").click().run()
    page.button("tab3_btn_ef").click().run()
    assert not page.error
    assert len(page.get("plotly_chart")) == 4
    page.run()
    assert len(page.get("plotly_chart")) == 4

    page.button("portfolio_toggle_editor_btn").click().run()
    page.text_area("tab3_ef_legs_input").set_value("etf,510300\netf,159915").run()
    assert not page.get("plotly_chart")
    assert set(page.session_state["_analysis_results"]) == {"normalize", "gbm", "compare"}

    page.button("tab3_btn_ef").click().run()
    page.button("nav_btn_single").click().run()
    page.text_input("symbol_input").set_value("510500").run()
    assert set(page.session_state["_analysis_results"]) == {"compare", "frontier"}

    page.date_input("start_date_input").set_value(page.date_input("end_date_input").value).run()
    assert not page.get("plotly_chart")
    assert not page.session_state["_analysis_results"]


def test_invalid_dates_disable_all_network_actions(page):
    import datetime

    end = page.date_input("end_date_input").value
    page.date_input("start_date_input").set_value(end + datetime.timedelta(days=1)).run()
    assert any("开始日期不能晚于结束日期" in error.value for error in page.error)
    assert page.button("home_analyze_btn").disabled
    assert not page.session_state["_analysis_results"]

    page.button("nav_btn_single").click().run()
    assert page.button("single_run_btn").disabled
    assert page.button("single_btn_normalize").disabled
    assert page.button("single_btn_gbm").disabled

    page.button("nav_btn_compare").click().run()
    assert page.button("compare_run_btn").disabled

    page.button("nav_btn_portfolio").click().run()
    assert page.button("tab3_btn_ef").disabled

    page.date_input("start_date_input").set_value(end).run()
    assert not page.error
    assert not page.button("tab3_btn_ef").disabled
    page.button("nav_btn_single").click().run()
    assert not page.button("single_run_btn").disabled
    assert not page.get("plotly_chart")


def test_blank_symbol_disables_only_single_asset_actions(page):
    page.button("nav_btn_single").click().run()
    page.text_input("symbol_input").set_value("").run()
    assert page.button("single_run_btn").disabled
    assert page.button("single_btn_normalize").disabled
    assert page.button("single_btn_gbm").disabled

    page.button("nav_btn_compare").click().run()
    assert not page.button("compare_run_btn").disabled

    page.button("nav_btn_portfolio").click().run()
    assert not page.button("tab3_btn_ef").disabled

    page.button("nav_btn_single").click().run()
    page.selectbox("asset_type").set_value("bond_us_yield").run()
    page.text_input("symbol_input").set_value("").run()
    assert not page.error
    assert not page.button("single_run_btn").disabled


@pytest.mark.parametrize("key", ["start_date_input", "end_date_input"])
def test_cleared_date_does_not_crash_or_allow_network_actions(page, key):
    page.date_input(key).set_value(None).run()
    assert not page.exception
    assert any("请选择开始和结束日期" in error.value for error in page.error)
    assert page.button("home_analyze_btn").disabled
    page.button("nav_btn_single").click().run()
    assert page.button("single_run_btn").disabled
    page.button("nav_btn_compare").click().run()
    assert page.button("compare_run_btn").disabled
    page.button("nav_btn_portfolio").click().run()
    assert page.button("tab3_btn_ef").disabled


def test_failed_retry_discards_previous_success_and_recovers(page, monkeypatch, sample_ohlcv):
    page.button("nav_btn_single").click().run()
    page.button("single_run_btn").click().run()
    assert len(page.metric) == 3
    app.load_asset_data.clear()

    def fail(**kwargs):
        raise app.DataFetchError("temporary test failure")

    monkeypatch.setattr(asset_loader, "_fetch_etf_data", fail)
    page.button("single_toggle_editor_btn").click().run()
    page.button("single_run_btn").click().run()
    assert not page.metric
    assert len(page.error) == 1
    page.run()
    assert len(page.error) == 1
    assert not page.metric
    monkeypatch.setattr(asset_loader, "_fetch_etf_data", lambda **kwargs: sample_ohlcv.copy())
    page.button("single_run_btn").click().run()
    assert not page.error
    assert len(page.metric) == 3


def test_chart_dependency_errors_remain_user_readable(page, monkeypatch):
    import charts

    def missing():
        raise ImportError("缺少运行依赖 `plotly`，请先安装：`pip install plotly`。")

    monkeypatch.setattr(charts, "_load_plotly_modules", missing)
    page.button("nav_btn_single").click().run()
    page.button("single_run_btn").click().run()
    assert not page.exception
    assert len(page.metric) == 3
    assert "pip install plotly" in page.error[0].value
