"""Vendor fallback, adjustment, date filtering and NAV alignment regressions."""

from types import SimpleNamespace

import pandas as pd
import pytest

import data_fetcher
from data_fetcher import AKShareFetcher, DataFetchError, EmptySymbolDataError, NetworkFetchError
from data_sources import china_equity, forex, funds, network
from data_sources.base import BaseDataFetcher
from data_sources.eastmoney import EastMoneyDailyKlineClient, parse_daily_kline_payload
from data_sources.errors import VendorUpstreamError
from data_sources.forex import _forex_kline_json_to_dataframe


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

    def fail_sina(**kwargs):
        raise AssertionError("adjusted ETF must never call fund_etf_hist_sina")

    def ok_tx(**kwargs):
        calls.append("akshare.stock_zh_a_hist_tx")
        return sample_ohlcv.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

    client = AKShareFetcher(max_retries=1, use_system_proxy=True)
    monkeypatch.setattr(
        client,
        "_load_akshare",
        lambda: SimpleNamespace(fund_etf_hist_sina=fail_sina, stock_zh_a_hist_tx=ok_tx),
    )
    raw = client._fetch_sina_tencent_china_equity_raw(
        "etf", "510300", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-01"), "hfq"
    )
    assert not raw.empty
    assert calls == ["akshare.stock_zh_a_hist_tx"]
    assert raw.attrs["data_source"] == "akshare.stock_zh_a_hist_tx"


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


def test_akshare_v120_missing_legacy_attributes_route_to_new_apis_and_fail_cleanly(
    monkeypatch, sample_ohlcv
):
    english = sample_ohlcv.iloc[:3].reset_index().rename(columns=str.lower)
    chinese = pd.DataFrame(
        {
            "日期": english["date"],
            "开盘": english["open"],
            "收盘": english["close"],
            "最高": english["high"],
            "最低": english["low"],
            "成交量": english["volume"],
        }
    )

    seen_calls: list[tuple[str, str | None]] = []

    def _rec_us(symbol: str, adjust: str = ""):
        seen_calls.append(("stock_us_daily_sina", symbol))
        return english

    def _rec_idx_us(symbol: str):
        seen_calls.append(("index_us_daily_sina", symbol))
        return english

    # Only new AKShare >= 1.20.46 public attribute names exist (no legacy names).
    api = SimpleNamespace(
        stock_zh_a_hist_em=lambda **kwargs: chinese,
        futures_foreign_hist_sina=lambda **kwargs: english,
        stock_us_daily_sina=_rec_us,
        stock_hk_daily_sina=lambda **kwargs: english,
        index_us_daily_sina=_rec_idx_us,
        bond_zh_hs_daily_sina=lambda **kwargs: english.drop(columns="volume"),
        bond_zh_us_yield_em=lambda **kwargs: pd.DataFrame(
            {
                "日期": english["date"],
                data_fetcher.US_TREASURY_YIELD_10Y_CN_COL: [4.1, 4.2, 4.3],
            }
        ),
    )

    primary = china_equity.AKShareChinaPrimarySource(lambda: api)
    raw_stock = primary.fetch_stock_hist(
        "600519", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"), "qfq"
    )
    assert len(raw_stock) == 3

    monkeypatch.setattr(data_fetcher, "_import_akshare", lambda: api)
    fut = data_fetcher.load_futures_foreign_ohlcv(
        "GC", "2024-01-02", "2024-01-04", asset_type="gold"
    )
    us = data_fetcher.load_global_market_data("us_stock", "aapl", "2024-01-02", "2024-01-04")
    hk = data_fetcher.load_global_market_data("hk_stock", "700", "2024-01-02", "2024-01-04")
    idx = data_fetcher.load_global_market_data("global_index", ".inx", "2024-01-02", "2024-01-04")
    bond_cn = data_fetcher.load_bond_data("cn_hist", "sh019547", "2024-01-02", "2024-01-04")
    bond_us = data_fetcher.load_bond_data("us_yield", "10Y", "2024-01-02", "2024-01-04")

    assert fut.attrs["data_source"] == "akshare.futures_foreign_hist_sina"
    assert us.attrs["data_source"] == "akshare.stock_us_daily_sina"
    assert us.attrs["symbol"] == "AAPL"
    assert ("stock_us_daily_sina", "AAPL") in seen_calls
    assert hk.attrs["data_source"] == "akshare.stock_hk_daily_sina"
    assert idx.attrs["data_source"] == "akshare.index_us_daily_sina"
    assert idx.attrs["symbol"] == ".INX"
    assert ("index_us_daily_sina", ".INX") in seen_calls
    assert bond_cn.attrs["data_source"] == "akshare.bond_zh_hs_daily_sina"
    assert (bond_cn["Volume"] == 0.0).all()
    assert bond_us.attrs["data_source"] == "akshare.bond_zh_us_yield_em"

    # Empty AKShare namespace must raise VendorUpstreamError cleanly (no fake data/empty fallback).
    empty_api = SimpleNamespace()
    monkeypatch.setattr(data_fetcher, "_import_akshare", lambda: empty_api)
    empty_primary = china_equity.AKShareChinaPrimarySource(lambda: empty_api)
    with pytest.raises(VendorUpstreamError):
        empty_primary.fetch_stock_hist(
            "600519", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"), "qfq"
        )
    with pytest.raises(VendorUpstreamError):
        data_fetcher.load_global_market_data("us_stock", "AAPL", "2024-01-02", "2024-01-04")
    with pytest.raises(VendorUpstreamError):
        data_fetcher.load_bond_data("cn_hist", "sh019547", "2024-01-02", "2024-01-04")


def test_china_equity_tx_prefers_volume_over_amount_and_falls_back_to_sina_daily(sample_ohlcv):
    fetcher = AKShareFetcher(max_retries=1, retry_delay=0.0)
    dates = pd.to_datetime(sample_ohlcv.index[:3]).strftime("%Y-%m-%d")
    tx_with_both = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 11.0, 12.0],
            "close": [10.5, 11.5, 12.5],
            "high": [11.0, 12.0, 13.0],
            "low": [9.5, 10.5, 11.5],
            "volume": [1000.0, 2000.0, 3000.0],
            "turnover": [0.5, 0.6, 0.7],
            "amount": [999999.0, 888888.0, 777777.0],
        }
    )

    api_both = SimpleNamespace(stock_zh_a_hist_tx=lambda **kwargs: tx_with_both)
    fetcher._load_akshare = lambda: api_both  # type: ignore[method-assign]
    conv_both = fetcher._fetch_sina_tencent_china_equity_raw(
        "stock", "600519", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"), "qfq"
    )
    assert conv_both.attrs["data_source"] == "akshare.stock_zh_a_hist_tx"
    assert list(conv_both["成交量"]) == [1000.0, 2000.0, 3000.0]

    # When only legacy 'amount' column exists, still works cleanly.
    api_legacy = SimpleNamespace(stock_zh_a_hist_tx=lambda **kwargs: tx_with_both.drop(columns="volume"))
    fetcher._load_akshare = lambda: api_legacy  # type: ignore[method-assign]
    conv_legacy = fetcher._fetch_sina_tencent_china_equity_raw(
        "stock", "600519", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"), "qfq"
    )
    assert list(conv_legacy["成交量"]) == [999999.0, 888888.0, 777777.0]

    # When stock_zh_a_hist_tx raises, A-share falls back to stock_zh_a_daily_sina and records warning.
    sina_df = pd.DataFrame(
        {
            "date": dates,
            "open": [20.0, 21.0, 22.0],
            "high": [21.0, 22.0, 23.0],
            "low": [19.0, 20.0, 21.0],
            "close": [20.5, 21.5, 22.5],
            "volume": [500.0, 600.0, 700.0],
        }
    )
    api_sina = SimpleNamespace(
        stock_zh_a_hist_tx=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("tx down")),
        stock_zh_a_daily_sina=lambda **kwargs: sina_df,
    )
    fetcher._load_akshare = lambda: api_sina  # type: ignore[method-assign]
    conv_sina = fetcher._fetch_sina_tencent_china_equity_raw(
        "stock", "600519", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"), "qfq"
    )
    assert conv_sina.attrs["data_source"] == "akshare.stock_zh_a_daily_sina"
    assert any("stock_zh_a_hist_tx" in w for w in fetcher._last_alt_warnings)

    # ETF must only use fund_etf_hist_sina when adjust == ""; never for qfq.
    called_etf: list[str] = []

    def _sina_etf(**kwargs):
        called_etf.append("fund_etf_hist_sina")
        return sina_df

    def _tx_etf(**kwargs):
        called_etf.append("stock_zh_a_hist_tx")
        return tx_with_both

    fetcher._load_akshare = lambda: SimpleNamespace(
        fund_etf_hist_sina=_sina_etf, stock_zh_a_hist_tx=_tx_etf
    )  # type: ignore[method-assign]
    conv_etf_qfq = fetcher._fetch_sina_tencent_china_equity_raw(
        "etf", "510300", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"), "qfq"
    )
    assert called_etf == ["stock_zh_a_hist_tx"]
    assert conv_etf_qfq.attrs["data_source"] == "akshare.stock_zh_a_hist_tx"

    called_etf.clear()
    conv_etf_raw = fetcher._fetch_sina_tencent_china_equity_raw(
        "etf", "510300", pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"), ""
    )
    assert called_etf == ["fund_etf_hist_sina"]
    assert conv_etf_raw.attrs["data_source"] == "akshare.fund_etf_hist_sina"


def test_rate_limit_exception_and_call_akshare_short_circuit():
    class Push2RateLimitError(RuntimeError):
        pass

    wrapped = RuntimeError("outer failure")
    wrapped.__cause__ = Push2RateLimitError("push2 blocked")
    assert network.is_rate_limit_exception(wrapped) is True
    assert network.classify_network_exception(wrapped) == "upstream"

    attempts = 0

    def _boom():
        nonlocal attempts
        attempts += 1
        raise wrapped

    with pytest.raises(VendorUpstreamError, match="上游限流/拦截"):
        network._call_akshare("test_push2", _boom, max_retries=3, retry_delay=0.0)
    assert attempts == 1


def test_load_reit_data_fallback_chain_and_all_fail_error(monkeypatch, sample_ohlcv):
    english = sample_ohlcv.iloc[:3].reset_index().rename(columns=str.lower)
    tx_df = english.assign(amount=english["volume"] * 4.0)
    em_raw = pd.DataFrame(
        {
            "日期": english["date"],
            "开盘": english["open"],
            "收盘": english["close"],
            "最高": english["high"],
            "最低": english["low"],
            "成交量": english["volume"],
        }
    )

    # 1) Tencent preferred (volume preferred over amount)
    api_tx = SimpleNamespace(stock_zh_a_hist_tx=lambda **kwargs: tx_df)
    monkeypatch.setattr(data_fetcher, "_import_akshare", lambda: api_tx)
    out_tx = data_fetcher.load_reit_data("sh508097", "2024-01-02", "2024-01-04")
    assert out_tx.attrs["symbol"] == "508097"
    assert out_tx.attrs["data_source"] == "akshare.stock_zh_a_hist_tx"
    assert list(out_tx["Volume"]) == list(sample_ohlcv.iloc[:3]["Volume"])

    # 2) Tencent fails -> fund_reits_hist_em fallback
    api_em = SimpleNamespace(
        stock_zh_a_hist_tx=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("tx fail")),
        fund_reits_hist_em=lambda **kwargs: em_raw,
    )
    monkeypatch.setattr(data_fetcher, "_import_akshare", lambda: api_em)
    out_em = data_fetcher.load_reit_data("508097", "2024-01-02", "2024-01-04")
    assert out_em.attrs["data_source"] == "akshare.fund_reits_hist_em"
    assert any("stock_zh_a_hist_tx" in w for w in out_em.attrs["fallback_warnings"])

    # 3) Tencent & AKShare EM fail -> EastMoneyDirect fallback
    monkeypatch.setattr(
        EastMoneyDailyKlineClient,
        "fetch_daily_klines",
        lambda self, *args, **kwargs: em_raw,
    )
    api_no_em = SimpleNamespace(
        stock_zh_a_hist_tx=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("tx fail")),
    )
    monkeypatch.setattr(data_fetcher, "_import_akshare", lambda: api_no_em)
    out_direct = data_fetcher.load_reit_data("508097", "2024-01-02", "2024-01-04")
    assert out_direct.attrs["data_source"] == "eastmoney_direct"
    assert len(out_direct.attrs["fallback_warnings"]) >= 2

    # 4) EastMoneyDirect also fails -> fund_etf_hist_sina unadjusted fallback
    monkeypatch.setattr(
        EastMoneyDailyKlineClient,
        "fetch_daily_klines",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("em direct fail")),
    )
    api_sina = SimpleNamespace(
        fund_etf_hist_sina=lambda **kwargs: english,
    )
    monkeypatch.setattr(data_fetcher, "_import_akshare", lambda: api_sina)
    out_sina = data_fetcher.load_reit_data("508097", "2024-01-02", "2024-01-04")
    assert out_sina.attrs["data_source"] == "akshare.fund_etf_hist_sina"
    assert any("不复权" in w for w in out_sina.attrs["fallback_warnings"])

    # 5) All sources raise -> explicit DataFetchError (never silent empty frame)
    monkeypatch.setattr(data_fetcher, "_import_akshare", lambda: SimpleNamespace())
    with pytest.raises(DataFetchError, match="508097"):
        data_fetcher.load_reit_data("508097", "2024-01-02", "2024-01-04")

    # 6) Empty payloads without errors -> EmptySymbolDataError
    monkeypatch.setattr(
        EastMoneyDailyKlineClient, "fetch_daily_klines", lambda self, *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(
        data_fetcher,
        "_import_akshare",
        lambda: SimpleNamespace(stock_zh_a_hist_tx=lambda **kwargs: pd.DataFrame()),
    )
    with pytest.raises(EmptySymbolDataError):
        data_fetcher.load_reit_data("508097", "2024-01-02", "2024-01-04")


def test_load_global_market_data_non_dot_fallbacks_and_dxy_sina_fallback(monkeypatch, sample_ohlcv):
    english = sample_ohlcv.iloc[:3].reset_index().rename(columns=str.lower)

    class Push2RateLimitError(RuntimeError):
        pass

    api = SimpleNamespace(
        index_hk_daily_sina=lambda **kwargs: english.drop(columns="volume"),
        index_us_daily_sina=lambda **kwargs: english,
        index_global_sina_symbol_list=lambda: pd.DataFrame(
            {"代码": ["N225"], "名称": ["日经225指数"]}
        ),
        index_global_hist_sina=lambda **kwargs: english.drop(columns="volume"),
        index_global_hist_em=lambda **kwargs: (_ for _ in ()).throw(
            Push2RateLimitError("push2 blocked")
        ),
    )
    monkeypatch.setattr(data_fetcher, "_import_akshare", lambda: api)
    monkeypatch.setattr(
        data_fetcher,
        "fetch_sina_kline_daily",
        lambda sina_sym, **kwargs: pd.DataFrame(
            {
                "日期": english["date"],
                "今开": english["open"],
                "最新价": english["close"],
                "最高": english["high"],
                "最低": english["low"],
            }
        ),
    )

    hsi = data_fetcher.load_global_market_data("global_index", "HSI", "2024-01-02", "2024-01-04")
    spx = data_fetcher.load_global_market_data("global_index", "SPX", "2024-01-02", "2024-01-04")
    n225 = data_fetcher.load_global_market_data("global_index", "N225", "2024-01-02", "2024-01-04")
    dxy = data_fetcher.load_global_market_data("global_index", "DXY", "2024-01-02", "2024-01-04")

    assert hsi.attrs["data_source"] == "akshare.index_hk_daily_sina"
    assert spx.attrs["data_source"] == "akshare.index_us_daily_sina"
    assert spx.attrs["symbol"] == ".INX"
    assert n225.attrs["data_source"] == "akshare.index_global_hist_sina"
    assert n225.attrs["symbol"] == "日经225指数"
    assert dxy.attrs["data_source"] == "sina_forex_direct"
    assert dxy.attrs["symbol"] == "DXY"
    assert any("index_global_hist_em" in w for w in dxy.attrs["fallback_warnings"])

    monkeypatch.setattr(
        data_fetcher,
        "fetch_sina_kline_daily",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sina dxy failed")),
    )
    with pytest.raises(DataFetchError, match="DXY"):
        data_fetcher.load_global_market_data("global_index", "DXY", "2024-01-02", "2024-01-04")


def test_forex_sina_jsonp_mapping_and_fallback_chain(monkeypatch):
    # Sina JSONP field order is [date, open, low, high, close].
    raw_jsonp = (
        '/* comment */\nvar _fx=("2024-01-02,7.1000,7.0800,7.1500,7.1200|'
        '2024-01-03,7.1200,7.0900,7.1600,7.1400,");'
    )
    parsed = forex._sina_forex_jsonp_to_dataframe(raw_jsonp, "USDCNH")
    assert len(parsed) == 2
    assert parsed.iloc[0]["今开"] == pytest.approx(7.1000)
    assert parsed.iloc[0]["最低"] == pytest.approx(7.0800)
    assert parsed.iloc[0]["最高"] == pytest.approx(7.1500)
    assert parsed.iloc[0]["最新价"] == pytest.approx(7.1200)
    assert (parsed["最低"] <= parsed["最高"]).all()

    class RemoteProtocolError(RuntimeError):
        pass

    get_calls = 0

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, *args, **kwargs):
            nonlocal get_calls
            get_calls += 1
            raise RemoteProtocolError("Server disconnected without sending a response.")

    monkeypatch.setattr(forex, "_resolve_forex_market_map", lambda: {"USDCNH": 133})
    monkeypatch.setattr(forex, "_load_httpx_module", lambda: SimpleNamespace(Client=FakeClient))
    monkeypatch.setattr(
        forex,
        "_import_akshare",
        lambda: SimpleNamespace(
            fx_hist_em=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fx_hist_em blocked"))
        ),
    )
    monkeypatch.setattr(
        forex,
        "fetch_sina_kline_daily",
        lambda sina_sym, **kwargs: parsed.assign(代码=sina_sym),
    )

    out = forex._fetch_forex_hist_em_robust("USDCNH")
    assert get_calls == 2  # Short-circuited after 2 RemoteProtocolErrors instead of retrying all mirrors
    assert out.attrs["data_source"] == "sina_forex_direct"
    assert any("eastmoney_forex_direct" in w for w in out.attrs["fallback_warnings"])
    assert any("fx_hist_em" in w for w in out.attrs["fallback_warnings"])

    monkeypatch.setattr(
        forex,
        "fetch_sina_kline_daily",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("sina fx down")),
    )
    with pytest.raises(DataFetchError, match="新浪外汇回退"):
        forex._fetch_forex_hist_em_robust("USDCNH")

