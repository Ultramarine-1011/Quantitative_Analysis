"""Vendor fallback, adjustment, date filtering and NAV alignment regressions."""

from types import SimpleNamespace

import pandas as pd
import pytest

import data_fetcher
from data_fetcher import AKShareFetcher, EmptySymbolDataError, NetworkFetchError
from data_sources import china_equity, funds
from data_sources.eastmoney import EastMoneyDailyKlineClient, parse_daily_kline_payload
from data_sources.forex import _forex_kline_json_to_dataframe
from data_sources.base import BaseDataFetcher


def raw_equity(sample_ohlcv):
    return sample_ohlcv.reset_index().rename(
        columns={
            "Date": "日期",
            "Open": "开盘",
            "High": "最高",
            "Low": "最低",
            "Close": "收盘",
            "Volume": "成交量",
        }
    )


def test_equity_fallback_order_and_date_window(monkeypatch, sample_ohlcv):
    client = AKShareFetcher(max_retries=1, use_system_proxy=True)
    calls = []

    def alternate(*args):
        calls.append("alternate")
        return pd.DataFrame()

    def primary(*args):
        calls.append("primary")
        raise NetworkFetchError("primary down")

    def direct(*args, **kwargs):
        calls.append("direct")
        return raw_equity(sample_ohlcv)

    monkeypatch.setattr(client, "_fetch_sina_tencent_china_equity_raw", alternate)
    monkeypatch.setattr(client._primary, "fetch_etf_hist_em", primary)
    monkeypatch.setattr(client._eastmoney, "fetch_daily_klines", direct)
    out = client.fetch_etf("510300", start="2024-01-03", end="2024-01-05", adjust="hfq")
    assert calls == ["alternate", "primary", "direct"]
    assert len(out) == 3
    assert out.index.min() == pd.Timestamp("2024-01-03")


def test_adjusted_etf_skips_unadjusted_sina(monkeypatch, sample_ohlcv):
    calls = []

    def load_module(name):
        calls.append(name)
        assert name != "akshare.fund.fund_etf_sina"
        return SimpleNamespace(
            stock_zh_a_hist_tx=lambda **kwargs: sample_ohlcv.reset_index().rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "amount",
                }
            )
        )

    monkeypatch.setattr(china_equity.importlib, "import_module", load_module)
    client = AKShareFetcher(max_retries=1, use_system_proxy=True)
    raw = client._fetch_sina_tencent_china_equity_raw(
        "etf", "510300", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-01"), "hfq"
    )
    assert not raw.empty
    assert calls == ["akshare.stock_feature.stock_hist_tx"]


def test_explicit_unadjusted_request_is_preserved(monkeypatch, sample_ohlcv):
    client = AKShareFetcher(max_retries=1, use_system_proxy=True)
    seen = []

    def fetch(symbol, **kwargs):
        seen.append(kwargs["adjust"])
        return raw_equity(sample_ohlcv)

    monkeypatch.setattr(client, "_fetch_etf_raw", fetch)
    client.fetch_etf("510300", start="2024-01-02", end="2024-02-01", adjust="")
    assert seen == [""]


@pytest.mark.parametrize("adjust", ["invalid", "HFQ"])
def test_bad_adjustment_rejected_before_vendor_calls(adjust):
    client = AKShareFetcher(max_retries=1, use_system_proxy=True)
    with pytest.raises(ValueError, match="adjust"):
        client.fetch_etf("510300", start="2024-01-02", end="2024-02-01", adjust=adjust)


def test_retry_preserves_error_and_does_not_retry_empty_data(sample_ohlcv):
    class Stub(BaseDataFetcher):
        def _fetch_raw(self, symbol, **kwargs):
            return sample_ohlcv

    client = Stub(max_retries=2, retry_delay=0)
    attempts = []

    def fail():
        attempts.append(1)
        raise NetworkFetchError("transport detail")

    with pytest.raises(NetworkFetchError, match="transport detail"):
        client._run_with_retry(fail, "stub")
    assert len(attempts) == 2
    attempts.clear()

    def empty():
        attempts.append(1)
        return pd.DataFrame()

    with pytest.raises(EmptySymbolDataError):
        client._run_with_retry(empty, "stub")
    assert len(attempts) == 1


@pytest.mark.parametrize("fields", [6, 11, 12])
def test_eastmoney_payload_handles_extra_or_optional_fields(fields):
    values = ["2024-01-02", "10", "11", "12", "9", "100"] + ["0"] * (fields - 6)
    out = parse_daily_kline_payload({"data": {"klines": [",".join(values)]}})
    assert out.iloc[0]["收盘"] == 11
    assert out.iloc[0]["成交量"] == 100


def test_eastmoney_transport_does_not_swallow_keyboard_interrupt(monkeypatch):
    client = EastMoneyDailyKlineClient()

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(client, "_fetch_daily_klines_one_host", interrupt)
    with pytest.raises(KeyboardInterrupt):
        client.fetch_daily_klines(
            "510300", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), "hfq", trust_env=False
        )


@pytest.mark.parametrize("fields", [8, 13, 14, 15])
def test_forex_parser_handles_appended_vendor_fields(fields):
    row = ["2024-01-02", "1", "2", "3", "0.5", "0", "0", "4"] + ["0"] * (fields - 8)
    out = _forex_kline_json_to_dataframe({"data": {"klines": [",".join(row)]}})
    assert out.iloc[0]["今开"] == 1
    assert out.iloc[0]["振幅"] == 4


def test_fund_cumulative_nav_never_backfills_from_future(monkeypatch):
    def fund_info(symbol, indicator):
        if indicator == "单位净值走势":
            return pd.DataFrame(
                {"净值日期": ["2024-01-01", "2024-01-02", "2024-01-03"], "单位净值": [1, 1.1, 1.2]}
            )
        return pd.DataFrame({"净值日期": ["2024-01-02", "2024-01-03"], "累计净值": [2.1, 2.2]})

    monkeypatch.setattr(
        funds.importlib,
        "import_module",
        lambda name: SimpleNamespace(fund_open_fund_info_em=fund_info),
    )
    out = funds.get_fund_data("009691", "2024-01-01", "2024-01-03")
    assert out.index.min() == pd.Timestamp("2024-01-02")
    assert out["Close"].tolist() == [2.1, 2.2]
    assert out.attrs["asset_type"] == "mutual_fund"


def test_fund_legacy_signature_and_existing_cumulative_column(monkeypatch):
    calls = []

    def fund_info(fund, indicator):
        calls.append((fund, indicator))
        return pd.DataFrame({"净值日期": ["2024-01-02"], "累计净值": [2.0], "单位净值": [1.0]})

    monkeypatch.setattr(
        funds.importlib,
        "import_module",
        lambda name: SimpleNamespace(fund_open_fund_info_em=fund_info),
    )
    out = funds.get_fund_data("009691", "2024-01-02", "2024-01-03")
    assert out.iloc[0]["Close"] == 2.0
    assert calls == [("009691", "单位净值走势")]


@pytest.mark.parametrize(
    "loader,args",
    [
        (data_fetcher.load_global_market_data, ("us_stock", "AAPL")),
        (data_fetcher.load_bond_data, ("us_yield", "")),
        (data_fetcher.load_commodity_data, ("CL",)),
        (data_fetcher.load_reit_data, ("508097",)),
        (data_fetcher.load_fx_data, ("EURUSD",)),
    ],
)
def test_loaders_reject_invalid_dates_without_network(monkeypatch, loader, args):
    def fail():
        pytest.fail("Invalid dates must be rejected before importing/calling a vendor")

    monkeypatch.setattr(data_fetcher, "_import_akshare", fail)
    with pytest.raises(ValueError, match="start_date"):
        loader(*args, "2024-12-31", "2024-01-01")


@pytest.mark.parametrize(
    "loader,args,asset,symbol",
    [
        (data_fetcher.load_global_market_data, ("hk_stock", "700"), "hk_stock", "00700"),
        (data_fetcher.load_global_market_data, ("global_index", ".INX"), "global_index", ".INX"),
        (data_fetcher.load_global_market_data, ("global_index", "HSI"), "global_index", "恒生指数"),
        (data_fetcher.load_bond_data, ("cn_hist", "sh010107"), "bond_cn", "sh010107"),
        (data_fetcher.load_commodity_data, ("CL",), "commodity", "CL"),
        (data_fetcher.load_reit_data, ("508097",), "creit", "508097"),
        (data_fetcher.load_fx_data, ("USD/JPY",), "fx", "USDJPY"),
    ],
)
def test_remaining_market_routes_preserve_contract(
    monkeypatch, sample_ohlcv, loader, args, asset, symbol
):
    english = sample_ohlcv.iloc[:3].reset_index().rename(columns=str.lower)
    chinese = raw_equity(sample_ohlcv.iloc[:3])
    api = SimpleNamespace(
        stock_hk_daily=lambda **kwargs: english,
        index_us_stock_sina=lambda **kwargs: english,
        index_global_hist_em=lambda **kwargs: chinese.drop(columns="成交量"),
        bond_zh_hs_daily=lambda **kwargs: english.drop(columns="volume"),
        futures_foreign_hist=lambda **kwargs: english,
        reits_hist_em=lambda **kwargs: chinese,
    )
    monkeypatch.setattr(data_fetcher, "_import_akshare", lambda: api)
    monkeypatch.setattr(data_fetcher, "_fetch_forex_hist_em_robust", lambda pair: chinese)
    out = loader(*args, "2024-01-02", "2024-01-04")
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert (out.dtypes == "float64").all()
    assert out.attrs["asset_type"] == asset
    assert out.attrs["symbol"] == symbol
    assert out.attrs["series_kind"] == "price"
    assert len(out) == 3
