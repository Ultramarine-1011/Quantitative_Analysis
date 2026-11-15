"""M5 (V1.0) tests: Web/CLI parity, unified UI workflows, semantic boundaries, and exports."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app
import asset_loader
import quant_cli
from contracts import SOFTWARE_VERSION
from ohlcv import compute_ohlcv_quality


def _make_ohlcv_series(
    dates: pd.DatetimeIndex,
    seed: int,
    *,
    asset_type: str = "etf",
    symbol: str = "510300",
    adjust: str = "hfq",
    currency: str | None = "CNY",
    drift: float = 0.0007,
    vol: float = 0.012,
    start_price: float = 100.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(drift, vol, len(dates))
    close = start_price * np.exp(np.cumsum(shocks))
    df = pd.DataFrame(
        {
            "Open": close * 0.996,
            "High": close * 1.008,
            "Low": close * 0.992,
            "Close": close,
            "Volume": np.full(len(dates), 12000.0, dtype=float),
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
            "data_source": "mock:market",
            "fetched_at": "2026-06-01T00:00:00+00:00",
            "cache_hit": False,
        }
    )
    df.attrs["data_quality"] = compute_ohlcv_quality(df, missing_policy="drop")
    return df


def _install_deterministic_market(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, list[str]]:
    """Patch asset_loader fetchers for AppTest AND return matching service loader + fx_loader."""
    dates = pd.bdate_range("2024-01-02", periods=95)
    fx_dates = dates.copy()
    fx_close = np.linspace(7.10, 7.28, len(fx_dates))
    fx_df = pd.DataFrame(
        {
            "Open": fx_close,
            "High": fx_close * 1.001,
            "Low": fx_close * 0.999,
            "Close": fx_close,
            "Volume": np.zeros(len(fx_dates), dtype=float),
        },
        index=fx_dates,
    )
    fx_df.index = pd.DatetimeIndex(fx_df.index).as_unit("ns")
    fx_df.index.name = "Date"
    fx_df.attrs.update(
        {
            "asset_type": "fx",
            "symbol": "USDCNY",
            "canonical_symbol": "USDCNY",
            "adjust": "",
            "currency": "CNY",
            "series_kind": "fx_rate",
            "data_source": "mock:fx",
        }
    )
    fx_df.attrs["data_quality"] = compute_ohlcv_quality(fx_df, missing_policy="drop")
    inv_fx_close = 1.0 / fx_close
    inv_fx_df = pd.DataFrame(
        {
            "Open": inv_fx_close,
            "High": inv_fx_close * 1.001,
            "Low": inv_fx_close * 0.999,
            "Close": inv_fx_close,
            "Volume": np.zeros(len(fx_dates), dtype=float),
        },
        index=fx_dates,
    )
    inv_fx_df.index = pd.DatetimeIndex(inv_fx_df.index).as_unit("ns")
    inv_fx_df.index.name = "Date"
    inv_fx_df.attrs.update(
        {
            "asset_type": "fx",
            "symbol": "CNHUSD",
            "canonical_symbol": "CNHUSD",
            "adjust": "",
            "currency": "USD",
            "series_kind": "fx_rate",
            "data_source": "mock:fx",
        }
    )
    inv_fx_df.attrs["data_quality"] = compute_ohlcv_quality(inv_fx_df, missing_policy="drop")

    us_aapl = _make_ohlcv_series(
        dates,
        seed=11,
        asset_type="us_stock",
        symbol="AAPL",
        adjust="qfq",
        currency="USD",
        start_price=180.0,
    )
    etf_510300 = _make_ohlcv_series(
        dates,
        seed=22,
        asset_type="etf",
        symbol="510300",
        adjust="hfq",
        currency="CNY",
        start_price=3.8,
    )
    etf_510500 = _make_ohlcv_series(
        dates,
        seed=33,
        asset_type="etf",
        symbol="510500",
        adjust="hfq",
        currency="CNY",
        start_price=5.5,
    )
    gold_gc = _make_ohlcv_series(
        dates,
        seed=44,
        asset_type="gold",
        symbol="GC",
        adjust="",
        currency="USD",
        drift=0.0005,
        vol=0.009,
        start_price=2150.0,
    )
    crypto_btc = _make_ohlcv_series(
        dates,
        seed=55,
        asset_type="crypto",
        symbol="BTC/USDT",
        adjust="",
        currency="USD",
        vol=0.025,
        start_price=62000.0,
    )
    yield_vals = np.linspace(3.95, 4.35, len(dates))
    bond_10y = _make_ohlcv_series(
        dates,
        seed=66,
        asset_type="bond_us_yield",
        symbol="10Y",
        adjust="",
        currency="PERCENT",
        start_price=4.1,
    )
    for col in ("Open", "High", "Low", "Close"):
        bond_10y[col] = yield_vals
    bond_10y.attrs["series_kind"] = "yield_level"
    bond_10y.attrs["data_quality"] = compute_ohlcv_quality(bond_10y, missing_policy="drop")

    calls: list[str] = []

    def _dispatch(asset_type: str, symbol: str, start_date: Any, end_date: Any, **kwargs: Any):
        calls.append(f"{asset_type}:{symbol}")
        key = (str(asset_type).lower(), str(symbol).upper())
        mapping = {
            ("us_stock", "AAPL"): us_aapl,
            ("etf", "510300"): etf_510300,
            ("etf", "510500"): etf_510500,
            ("gold", "GC"): gold_gc,
            ("crypto", "BTC/USDT"): crypto_btc,
            ("bond_us_yield", "10Y"): bond_10y,
            ("fx", "USDCNY"): fx_df,
            ("fx", "USDCNH"): fx_df,
            ("fx", "CNHUSD"): inv_fx_df,
            ("fx", "CNYUSD"): inv_fx_df,
        }
        if key not in mapping:
            raise AssertionError(f"Unexpected test asset lookup: {key}")
        return mapping[key].copy()

    monkeypatch.setattr(
        asset_loader,
        "_fetch_etf_data",
        lambda symbol, start_date, end_date, adjust="hfq", **kw: _dispatch(
            "etf", symbol, start_date, end_date, adjust=adjust
        ),
    )
    monkeypatch.setattr(
        asset_loader,
        "load_global_market_data",
        lambda route, symbol, start_date, end_date, **kw: _dispatch(
            route, symbol, start_date, end_date, **kw
        ),
    )
    monkeypatch.setattr(
        asset_loader,
        "_fetch_gold_data",
        lambda symbol, start_date, end_date, **kw: _dispatch("gold", symbol, start_date, end_date),
    )
    monkeypatch.setattr(
        asset_loader,
        "load_commodity_data",
        lambda symbol, start_date, end_date, **kw: _dispatch("gold", symbol, start_date, end_date),
    )
    monkeypatch.setattr(
        asset_loader,
        "_fetch_crypto_data",
        lambda symbol, start_date, end_date, **kw: _dispatch(
            "crypto", symbol, start_date, end_date
        ),
    )
    monkeypatch.setattr(
        asset_loader,
        "load_bond_data",
        lambda route, symbol, start_date, end_date, **kw: _dispatch(
            "bond_us_yield", symbol or "10Y", start_date, end_date
        ),
    )
    monkeypatch.setattr(
        asset_loader,
        "load_fx_data",
        lambda symbol, start_date, end_date, **kw: _dispatch("fx", symbol, start_date, end_date),
    )

    def service_loader(
        asset_type: str,
        symbol: str,
        start_date: Any,
        end_date: Any,
        adjust: str | None = None,
        force_refresh: bool = False,
        proxy_url: str | None = None,
        allow_usdt_peg: bool = False,
    ) -> pd.DataFrame:
        return _dispatch(asset_type, symbol, start_date, end_date, adjust=adjust)

    def service_fx_loader(pair: str, start_date: Any, end_date: Any) -> pd.DataFrame:
        return _dispatch("fx", pair, start_date, end_date)

    app.load_asset_data.clear()
    app.load_multi_leg_close_wide.clear()
    return service_loader, service_fx_loader, calls


def _open_page() -> AppTest:
    page = AppTest.from_file(str(Path(app.__file__)), default_timeout=25).run()
    page.date_input("start_date_input").set_value(datetime.date(2024, 1, 2))
    page.date_input("end_date_input").set_value(datetime.date(2024, 5, 13))
    return page.run()


def test_web_cli_parity_single_asset_us_stock_cny_conversion(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader_fn, fx_fn, _ = _install_deterministic_market(monkeypatch)
    page = _open_page()
    page.selectbox("asset_type").set_value("us_stock").run()
    page.text_input("symbol_input").set_value("AAPL").run()
    page.selectbox("single_base_currency_input").set_value("CNY").run()
    page.number_input("rf_pct_input").set_value(2.0).run()
    page.number_input("sma_short_input").set_value(10).run()
    page.number_input("sma_long_input").set_value(30).run()
    page.number_input("bb_window_input").set_value(15).run()
    page.checkbox("include_momentum_input").set_value(True).run()
    page.button("single_run_btn").click().run()

    assert not page.exception
    assert not page.error
    ws = page.session_state["_workspace_single"]
    web_res = ws["result"]
    assert web_res is not None
    assert web_res.asset.fx_converted is True
    assert web_res.asset.native_currency == "USD"
    assert web_res.asset.output_currency == "CNY"

    code, cli_res = quant_cli.run_cli_command(
        [
            "analyze",
            "us_stock:AAPL",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-13",
            "--base-currency",
            "CNY",
            "--rf",
            "0.02",
            "--sma-short",
            "10",
            "--sma-long",
            "30",
            "--bb-window",
            "15",
            "--include-momentum",
        ],
        loader=loader_fn,
        fx_loader=fx_fn,
    )
    stdout = capsys.readouterr().out
    assert code == quant_cli.EXIT_OK
    assert cli_res is not None
    cli_payload = json.loads(stdout)
    web_payload = json.loads(ws["exports"]["json"])

    for key in (
        "cumulative_return",
        "cagr",
        "annualized_volatility",
        "max_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "var_95",
        "cvar_95",
        "latest_close",
    ):
        assert web_payload["metrics"][key] == pytest.approx(
            cli_payload["metrics"][key], rel=1e-12, abs=1e-12
        )

    assert web_payload["quality"] == cli_payload["quality"]
    assert web_payload["asset"]["canonical_symbol"] == cli_payload["asset"]["canonical_symbol"]
    assert web_payload["asset"]["output_currency"] == cli_payload["asset"]["output_currency"]
    assert web_payload["asset"]["trading_days"] == cli_payload["asset"]["trading_days"]
    assert web_payload["software_version"] == SOFTWARE_VERSION
    assert len(page.get("plotly_chart")) == 4
    assert len(page.get("download_button")) == 4
    assert ws["exports"]["md"] and ws["exports"]["html"].startswith("<!DOCTYPE html>")


def test_web_cli_parity_multi_asset_compare_with_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader_fn, fx_fn, _ = _install_deterministic_market(monkeypatch)
    page = _open_page()
    page.text_area("compare_assets_input").set_value("etf:510300\nus_stock:AAPL\ngold:GC").run()
    page.text_input("compare_benchmark_input").set_value("etf:510500").run()
    page.selectbox("compare_calendar_mode").set_value("common_trading_dates").run()
    page.selectbox("compare_base_currency").set_value("CNY").run()
    page.selectbox("compare_return_convention").set_value("simple").run()
    page.selectbox("compare_cov_method").set_value("ledoit_wolf").run()
    page.number_input("compare_rolling_window").set_value(20).run()
    page.button("compare_run_btn").click().run()

    assert not page.exception
    assert not page.error
    ws = page.session_state["_workspace_compare"]
    web_res = ws["result"]
    assert web_res is not None

    code, cli_res = quant_cli.run_cli_command(
        [
            "compare",
            "etf:510300",
            "us_stock:AAPL",
            "gold:GC",
            "--benchmark",
            "etf:510500",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-13",
            "--calendar-mode",
            "common_trading_dates",
            "--base-currency",
            "CNY",
            "--return-convention",
            "simple",
            "--cov-method",
            "ledoit_wolf",
            "--rolling-window",
            "20",
        ],
        loader=loader_fn,
        fx_loader=fx_fn,
    )
    cli_payload = json.loads(capsys.readouterr().out)
    assert code == quant_cli.EXIT_OK
    assert cli_res is not None
    web_payload = json.loads(ws["exports"]["json"])

    assert web_payload["asset_order"] == cli_payload["asset_order"]
    for leg in web_payload["asset_order"]:
        for metric_key in (
            "cumulative_return",
            "cagr",
            "annualized_volatility",
            "max_drawdown",
            "sharpe_ratio",
        ):
            assert web_payload["metrics"]["per_asset"][leg][metric_key] == pytest.approx(
                cli_payload["metrics"]["per_asset"][leg][metric_key], rel=1e-12, abs=1e-12
            )
        assert web_payload["metrics"]["distribution"][leg]["jarque_bera_p_value"] == pytest.approx(
            cli_payload["metrics"]["distribution"][leg]["jarque_bera_p_value"],
            rel=1e-12,
            abs=1e-12,
        )
        assert web_payload["metrics"]["benchmark_comparison"][leg]["jensen_alpha"] == pytest.approx(
            cli_payload["metrics"]["benchmark_comparison"][leg]["jensen_alpha"],
            rel=1e-12,
            abs=1e-12,
        )
        assert web_payload["metrics"]["benchmark_comparison"][leg]["memmel_sharpe_test"][
            "p_value"
        ] == pytest.approx(
            cli_payload["metrics"]["benchmark_comparison"][leg]["memmel_sharpe_test"]["p_value"],
            rel=1e-12,
            abs=1e-12,
        )

    assert len(page.get("plotly_chart")) == 4
    # Drawdown chart must display non-positive (<= 0) underwater percentages.
    drawdown_fig = ws["figures"]["drawdown"][2]
    for trace in drawdown_fig.data:
        ys = np.asarray(trace.y, dtype=float)
        assert np.all(ys[~np.isnan(ys)] <= 1e-12)


def test_web_cli_parity_portfolio_walk_forward_custom_and_lookahead(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader_fn, fx_fn, _ = _install_deterministic_market(monkeypatch)
    page = _open_page()
    page.text_area("tab3_ef_legs_input").set_value("etf:510300\nus_stock:AAPL\ngold:GC").run()
    page.text_input("portfolio_benchmark_input").set_value("etf:510500").run()
    page.selectbox("portfolio_strategy_input").set_value("max_sharpe").run()
    page.selectbox("tab3_val_mode").set_value("walk_forward_rolling").run()
    page.number_input("portfolio_train_window").set_value(40).run()
    page.number_input("portfolio_frontier_points").set_value(12).run()
    page.number_input("portfolio_num_random").set_value(8).run()
    page.checkbox("portfolio_include_sensitivity").set_value(True).run()
    page.button("tab3_btn_ef").click().run()

    assert not page.exception
    assert not page.error
    ws = page.session_state["_workspace_portfolio"]
    web_payload = json.loads(ws["exports"]["json"])
    assert web_payload["methodology"]["has_lookahead_bias"] is False

    code, _ = quant_cli.run_cli_command(
        [
            "portfolio",
            "etf:510300",
            "us_stock:AAPL",
            "gold:GC",
            "--benchmark",
            "etf:510500",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-13",
            "--strategy",
            "max_sharpe",
            "--validation-mode",
            "walk_forward_rolling",
            "--train-window",
            "40",
            "--frontier-points",
            "12",
            "--num-random-portfolios",
            "8",
        ],
        loader=loader_fn,
        fx_loader=fx_fn,
    )
    cli_payload = json.loads(capsys.readouterr().out)
    assert code == quant_cli.EXIT_OK

    for leg, w in web_payload["metrics"]["portfolio"]["weights"].items():
        assert w == pytest.approx(
            cli_payload["metrics"]["portfolio"]["weights"][leg], rel=1e-10, abs=1e-10
        )
    for k in ("cagr", "max_drawdown", "sharpe_ratio", "total_turnover"):
        assert web_payload["metrics"]["backtest"]["OOS_Portfolio"][k] == pytest.approx(
            cli_payload["metrics"]["backtest"]["OOS_Portfolio"][k], rel=1e-10, abs=1e-10
        )
    assert "sensitivity" in web_payload["metrics"]
    assert "In_Sample_Portfolio" in web_payload["metrics"]["backtest"]

    # Custom weights parity
    page.selectbox("portfolio_strategy_input").set_value("custom").run()
    page.selectbox("tab3_val_mode").set_value("train_test").run()
    page.text_input("portfolio_custom_weights_input").set_value("0.5, 0.3, 0.2").run()
    page.button("tab3_btn_ef").click().run()
    assert not page.error
    custom_web = json.loads(page.session_state["_workspace_portfolio"]["exports"]["json"])

    code2, _ = quant_cli.run_cli_command(
        [
            "portfolio",
            "etf:510300",
            "us_stock:AAPL",
            "gold:GC",
            "--benchmark",
            "etf:510500",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-13",
            "--strategy",
            "custom",
            "--weights",
            "0.5,0.3,0.2",
            "--validation-mode",
            "train_test",
            "--frontier-points",
            "12",
            "--num-random-portfolios",
            "8",
        ],
        loader=loader_fn,
        fx_loader=fx_fn,
    )
    custom_cli = json.loads(capsys.readouterr().out)
    assert code2 == quant_cli.EXIT_OK
    assert custom_web["metrics"]["backtest"]["OOS_Portfolio"]["cagr"] == pytest.approx(
        custom_cli["metrics"]["backtest"]["OOS_Portfolio"]["cagr"], rel=1e-12, abs=1e-12
    )

    # In-sample lookahead disclosure warning (fitted optimizer triggers has_lookahead_bias)
    page.selectbox("portfolio_strategy_input").set_value("max_sharpe").run()
    page.selectbox("tab3_val_mode").set_value("in_sample").run()
    page.button("tab3_btn_ef").click().run()
    assert not page.error
    assert page.session_state["_workspace_portfolio"]["result"].methodology["has_lookahead_bias"]
    assert any("前视偏差" in w.value for w in page.warning)


def test_yield_level_semantic_blocking_across_all_three_workspaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_market(monkeypatch)
    page = _open_page()
    page.selectbox("asset_type").set_value("bond_us_yield").run()
    page.button("single_run_btn").click().run()

    assert not page.exception
    assert not page.error
    ws = page.session_state["_workspace_single"]
    assert ws["result"].asset.series_kind == "yield_level"
    assert any("N/A（口径禁用）" in str(df.value) for df in page.dataframe)
    assert any("跳过" in info.value for info in page.info)
    assert "normalized" in ws["skipped"] and "drawdown" in ws["skipped"]

    # Normalize & GBM reject yield_level cleanly without crashing.
    page.button("single_btn_normalize").click().run()
    assert not page.exception
    assert any("利率水平" in err.value for err in page.error)
    page.button("single_btn_gbm").click().run()
    assert not page.exception
    assert any("利率水平" in err.value for err in page.error)

    # Compare & Portfolio also reject yield_level inputs.
    page.text_area("compare_assets_input").set_value("etf,510300\nbond_us_yield,10Y").run()
    page.button("compare_run_btn").click().run()
    assert not page.exception
    assert any("利率水平" in err.value or "yield_level" in err.value for err in page.error)

    page.text_area("tab3_ef_legs_input").set_value("etf,510300\nbond_us_yield,10Y").run()
    page.button("tab3_btn_ef").click().run()
    assert not page.exception
    assert any("利率水平" in err.value or "yield_level" in err.value for err in page.error)


def test_crypto_adaptive_trading_days_and_gbm_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_deterministic_market(monkeypatch)
    page = _open_page()
    page.selectbox("asset_type").set_value("crypto").run()
    assert page.text_input("symbol_input").value == "BTC/USDT"
    assert int(page.number_input("single_trading_days_input").value) == 365

    page.button("single_run_btn").click().run()
    assert not page.exception
    assert not page.error
    assert page.session_state["_workspace_single"]["result"].asset.trading_days == 365

    page.button("single_btn_gbm").click().run()
    assert not page.exception
    assert not page.error
    ws = page.session_state["_workspace_single"]
    assert ws["gbm_result"] is not None and ws["gbm_fig"] is not None
    assert any("P50" in str(df.value) for df in page.dataframe)


def test_cache_reuse_force_refresh_custom_weight_errors_and_exports_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, calls = _install_deterministic_market(monkeypatch)
    page = _open_page()
    page.button("single_run_btn").click().run()
    assert len(calls) == 1
    page.button("single_run_btn").click().run()
    assert len(calls) == 1  # Hit Streamlit @st.cache_data

    page.checkbox("force_refresh_input").set_value(True).run()
    page.button("single_run_btn").click().run()
    assert len(calls) == 2

    # Invalid custom weights fail gracefully in Portfolio while Single-Asset output stays intact.
    page.selectbox("portfolio_strategy_input").set_value("custom").run()
    page.text_input("portfolio_custom_weights_input").set_value("0.5, 0.5").run()
    page.button("tab3_btn_ef").click().run()
    assert not page.exception
    assert page.error
    assert page.session_state["_workspace_single"]["result"] is not None

    # Validate exports UTF-8, UTF-8-SIG BOM CSV, HTML & strict JSON.
    exports = page.session_state["_workspace_single"]["exports"]
    parsed_json = json.loads(exports["json"])
    assert parsed_json["software_version"] == SOFTWARE_VERSION
    assert parsed_json["schema_version"] == "0.1.0"
    assert exports["csv"].startswith(b"\xef\xbb\xbf")
    assert "<!DOCTYPE html>" in exports["html"]
    assert "510300" in exports["md"]


def test_ui_helpers_leg_parsing_weights_and_edge_cases() -> None:
    assert app._split_ui_leg_token("etf,510300") == ("etf", "510300")
    assert app._split_ui_leg_token("us_stock:AAPL") == ("us_stock", "AAPL")
    with pytest.raises(ValueError):
        app._split_ui_leg_token("510300")
    with pytest.raises(ValueError):
        app._split_ui_leg_token("unknown_type:ABC")

    specs = app._parse_ui_multi_asset_specs("etf,sh510300; etf,510300\nus_stock:AAPL\n")
    assert len(specs) == 2  # Deduplicated canonical ETF symbol + US Stock
    assert app._parse_ui_benchmark_spec("") is None
    bench = app._parse_ui_benchmark_spec("etf:510500")
    assert bench is not None and bench.asset_type == "etf" and bench.symbol == "510500"

    assert app._parse_ui_custom_weights("0.5, 0.3; 0.2") == (0.5, 0.3, 0.2)
    with pytest.raises(ValueError):
        app._parse_ui_custom_weights(" ")
    with pytest.raises(ValueError):
        app._parse_ui_custom_weights("0.5, bad")
    with pytest.raises(ValueError):
        app._parse_ui_custom_weights("nan, 1.0")
