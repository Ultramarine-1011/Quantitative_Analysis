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

    monkeypatch.setattr(asset_loader, "_fetch_etf_data", fetch)
    monkeypatch.setattr(
        asset_loader,
        "load_global_market_data",
        lambda route, symbol, start, end: fetch(symbol=symbol),
    )
    yield AppTest.from_file(app.__file__, default_timeout=20).run()
    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()


def test_dashboard_survives_reruns_and_other_actions(page):
    page.sidebar.button[0].click().run()
    values = [metric.value for metric in page.metric]
    page.run()  # CSV downloads trigger the same rerun in supported Streamlit versions.
    assert [metric.value for metric in page.metric] == values
    page.button("tab3_btn_normalize").click().run()
    assert [metric.value for metric in page.metric] == values
    assert len(page.get("plotly_chart")) == 2
    assert not page.exception


def test_advanced_results_accumulate_and_only_related_edits_invalidate(page):
    for key in ("tab3_btn_normalize", "tab3_btn_gbm", "tab3_btn_ef"):
        page.button(key).click().run()
        assert not page.error
        assert not page.exception
    assert len(page.get("plotly_chart")) == 5
    page.run()
    assert len(page.get("plotly_chart")) == 5
    page.text_area("tab3_ef_legs_input").set_value("etf,510300\netf,159915").run()
    assert len(page.get("plotly_chart")) == 2  # Single-asset results remain.
    page.button("tab3_btn_ef").click().run()
    page.text_input("symbol_input").set_value("510500").run()
    assert len(page.get("plotly_chart")) == 3  # Portfolio inputs did not change.
    assert set(page.session_state["_analysis_results"]) == {"frontier"}
    page.date_input("start_date_input").set_value(page.date_input("end_date_input").value).run()
    assert not page.get("plotly_chart")
    assert not page.session_state["_analysis_results"]


def test_invalid_dates_disable_all_network_actions(page):
    import datetime

    end = page.date_input("end_date_input").value
    page.date_input("start_date_input").set_value(end + datetime.timedelta(days=1)).run()
    assert any("开始日期不能晚于结束日期" in error.value for error in page.error)
    assert all(button.disabled for button in page.button)
    assert not page.session_state["_analysis_results"]
    page.date_input("start_date_input").set_value(end).run()
    assert not page.error
    assert not page.sidebar.button[0].disabled
    assert not page.get("plotly_chart")


def test_blank_symbol_disables_only_single_asset_actions(page):
    page.text_input("symbol_input").set_value("").run()
    assert page.sidebar.button[0].disabled
    assert page.button("tab3_btn_normalize").disabled
    assert page.button("tab3_btn_gbm").disabled
    assert not page.button("tab3_btn_ef").disabled
    page.selectbox("asset_type").set_value("bond_us_yield").run()
    page.text_input("symbol_input").set_value("").run()
    assert not page.error
    assert not page.sidebar.button[0].disabled


@pytest.mark.parametrize("key", ["start_date_input", "end_date_input"])
def test_cleared_date_does_not_crash_or_allow_network_actions(page, key):
    page.date_input(key).set_value(None).run()
    assert not page.exception
    assert any("请选择开始和结束日期" in error.value for error in page.error)
    assert all(button.disabled for button in page.button)


def test_failed_retry_discards_previous_success_and_recovers(page, monkeypatch, sample_ohlcv):
    page.sidebar.button[0].click().run()
    assert len(page.metric) == 3
    app.load_asset_data.clear()

    def fail(**kwargs):
        raise app.DataFetchError("temporary test failure")

    monkeypatch.setattr(asset_loader, "_fetch_etf_data", fail)
    page.sidebar.button[0].click().run()
    assert not page.metric
    assert len(page.error) == 1
    page.run()
    assert len(page.error) == 1
    assert not page.metric
    monkeypatch.setattr(asset_loader, "_fetch_etf_data", lambda **kwargs: sample_ohlcv.copy())
    page.sidebar.button[0].click().run()
    assert not page.error
    assert len(page.metric) == 3


def test_chart_dependency_errors_remain_user_readable(page, monkeypatch):
    import charts

    def missing():
        raise ImportError("缺少运行依赖 `plotly`，请先安装：`pip install plotly`。")

    monkeypatch.setattr(charts, "_load_plotly_modules", missing)
    page.sidebar.button[0].click().run()
    assert not page.exception
    assert len(page.metric) == 3
    assert "pip install plotly" in page.error[0].value
