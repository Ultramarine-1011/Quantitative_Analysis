"""Stage 2 tests: non-blocking proxy concurrency, request budgets, forex decoupling & US/HK adjust."""

from __future__ import annotations

import importlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests.sessions
import requests.utils

import asset_loader
import data_fetcher
from data_sources import forex, network
from data_sources.china_equity import AKShareFetcher
from data_sources.eastmoney import EastMoneyDailyKlineClient
from data_sources.errors import EmptySymbolDataError, NetworkFetchError
from data_sources.network import RequestBudget


def _sample_daily_frame(dates: pd.DatetimeIndex, base: float = 100.0) -> pd.DataFrame:
    c = np.linspace(base, base + len(dates) - 1, len(dates))
    df = pd.DataFrame(
        {
            "Open": c * 0.99,
            "High": c * 1.01,
            "Low": c * 0.98,
            "Close": c,
            "Volume": np.full(len(dates), 1000.0),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# 1. Thread-local refcounted proxy bypass concurrency (no global I/O lock)
# ---------------------------------------------------------------------------


def test_without_http_proxy_env_runs_concurrently_without_serializing_io(monkeypatch):
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:8888")

    orig_sessions = lambda url, no_proxy=None: {"http": "http://127.0.0.1:8888"}
    orig_utils = lambda url, no_proxy=None: {"http": "http://127.0.0.1:8888"}
    monkeypatch.setattr(requests.sessions, "get_environ_proxies", orig_sessions)
    monkeypatch.setattr(requests.utils, "get_environ_proxies", orig_utils)

    barrier = threading.Barrier(4)
    results: list[dict[str, str]] = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        with network._without_http_proxy_env():
            barrier.wait(timeout=2.0)
            time.sleep(0.15)
            got = requests.sessions.get_environ_proxies("https://example.org")
            with lock:
                results.append(got)
            if i == 2:
                raise RuntimeError("boom in worker 2")

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = [pool.submit(worker, i) for i in range(4)]
        # Outside thread must keep resolving env proxy even while workers bypass it!
        outside = pool.submit(
            lambda: (
                time.sleep(0.05),
                requests.sessions.get_environ_proxies("https://example.org"),
            )[1]
        )
        assert outside.result(timeout=2.0) == {"http": "http://127.0.0.1:8888"}
        for idx, fut in enumerate(futs):
            if idx == 2:
                with pytest.raises(RuntimeError, match="boom"):
                    fut.result(timeout=2.0)
            else:
                fut.result(timeout=2.0)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.45
    assert results == [{}, {}, {}, {}]
    # All threads exited (including one raising): exact originals restored.
    assert requests.sessions.get_environ_proxies is orig_sessions
    assert requests.utils.get_environ_proxies is orig_utils


# ---------------------------------------------------------------------------
# 2. RequestBudget & EastMoney client pool reuse + short-circuit empty payload
# ---------------------------------------------------------------------------


def test_request_budget_limits_attempts_and_timeouts():
    b = RequestBudget(total_timeout_seconds=5.0, per_request_timeout_seconds=1.2, max_attempts=2)
    assert b.consume_attempt("step1") <= 1.2
    assert b.consume_attempt("step2") <= 1.2
    with pytest.raises(NetworkFetchError, match="最大尝试次数上限"):
        b.consume_attempt("step3")

    expired = RequestBudget(
        total_timeout_seconds=0.0, per_request_timeout_seconds=1.0, max_attempts=5
    )
    time.sleep(0.01)
    with pytest.raises(NetworkFetchError, match="总耗时上限"):
        expired.consume_attempt("expired")


def test_eastmoney_valid_empty_payload_does_not_amplify_retries(monkeypatch):
    client_inits = []
    http_gets = []

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"klines": []}}

    class FakeHttpxClient:
        def __init__(self, **kwargs):
            client_inits.append(kwargs.get("trust_env"))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, params=None, timeout=None):
            http_gets.append((url, params.get("secid")))
            return FakeResp()

    fake_httpx = SimpleNamespace(Client=FakeHttpxClient)
    em = EastMoneyDailyKlineClient()
    monkeypatch.setattr(em, "_load_httpx", lambda: fake_httpx)

    out = em.fetch_daily_klines(
        "510300",
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-10"),
        "hfq",
        trust_env=False,
    )
    assert out.empty
    # Only the first host's 2 market prefixes (1.510300, 0.510300) should be queried—never 4 hosts x 2 libs x 2 proxy modes!
    assert len(http_gets) == 2
    assert len(client_inits) == 2


def test_eastmoney_reuses_single_client_across_hosts_and_tracks_provenance(monkeypatch):
    client_inits = []
    calls = []

    class FakeConnError(Exception):
        pass

    FakeConnError.__module__ = "httpx"
    FakeConnError.__name__ = "ConnectError"

    class FakeOkResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"klines": ["2024-01-02,10,11,12,9,1000,10000,1,1,0.1,1"]}}

    class FakeHttpxClient:
        def __init__(self, **kwargs):
            client_inits.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, params=None, timeout=None):
            calls.append((url, params.get("secid")))
            if "push2his.eastmoney.com" in url and not url.startswith("https://33."):
                raise FakeConnError("host 1 down")
            if url.startswith("https://push2.eastmoney.com"):
                return FakeOkResp()
            return FakeOkResp()

    class DeadReqSession:
        def __init__(self):
            self.headers = {}
            self.trust_env = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            raise requests.ConnectionError("fallback req fail")

    class DeadHTTPSConnection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            raise OSError("fallback stdlib fail")

        def getresponse(self):
            raise OSError("fallback stdlib fail")

        def close(self):
            pass

    em = EastMoneyDailyKlineClient()
    monkeypatch.setattr(em, "_load_httpx", lambda: SimpleNamespace(Client=FakeHttpxClient))
    monkeypatch.setattr("data_sources.eastmoney.requests.Session", DeadReqSession)
    monkeypatch.setattr("http.client.HTTPSConnection", DeadHTTPSConnection)

    out = em.fetch_daily_klines(
        "510300",
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-10"),
        "hfq",
        trust_env=False,
    )
    assert len(out) == 1
    assert out.attrs["data_source"] == "eastmoney_direct"
    assert out.attrs["eastmoney_host"] == "push2.eastmoney.com"
    # Opened only primary+alt httpx clients ONCE for the entire multi-host call.
    assert len(client_inits) == 2


# ---------------------------------------------------------------------------
# 3. China equity error classification & provenance (no double run on empty)
# ---------------------------------------------------------------------------


def test_china_equity_empty_symbol_short_circuits_without_proxy_rerun(monkeypatch):
    fetcher = AKShareFetcher(max_retries=2, retry_delay=0.0, use_system_proxy=False)
    pipeline_runs = []

    def empty_alt(*args):
        pipeline_runs.append("alt")
        return pd.DataFrame()

    def empty_primary(*args):
        pipeline_runs.append("primary")
        return pd.DataFrame()

    def empty_direct(*args, **kwargs):
        pipeline_runs.append("direct")
        return pd.DataFrame()

    monkeypatch.setattr(fetcher, "_fetch_sina_tencent_china_equity_raw", empty_alt)
    monkeypatch.setattr(fetcher._primary, "fetch_etf_hist_em", empty_primary)
    monkeypatch.setattr(fetcher._eastmoney, "fetch_daily_klines", empty_direct)

    with pytest.raises(EmptySymbolDataError):
        fetcher.fetch_etf("999999", start="2024-01-01", end="2024-01-10", adjust="hfq")
    # Must run once per tier cleanly instead of proxy-fallback doubling or retrying EmptySymbolDataError.
    assert pipeline_runs == ["alt", "primary", "direct"]


# ---------------------------------------------------------------------------
# 4. Forex decoupled from akshare.forex.cons
# ---------------------------------------------------------------------------


def test_forex_works_when_akshare_forex_cons_is_unavailable(monkeypatch):
    orig_import = importlib.import_module

    def fake_import(name: str, *args, **kwargs):
        if name == "akshare.forex.cons":
            raise ModuleNotFoundError("akshare removed private forex.cons")
        if name == "httpx":

            class FakeResp:
                def raise_for_status(self):
                    return None

                def json(self):
                    return {
                        "data": {
                            "code": "USDCNH",
                            "name": "离岸人民币",
                            "klines": ["2024-01-02,7.15,7.16,7.18,7.14,0,0,0.5"],
                        }
                    }

            class FakeClient:
                def __init__(self, **kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def get(self, url, params=None, timeout=None):
                    assert params["secid"] == "133.USDCNH"
                    return FakeResp()

            return SimpleNamespace(Client=FakeClient)
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(forex.importlib, "import_module", fake_import)
    out = forex._fetch_forex_hist_em_robust("USDCNH")
    assert len(out) == 1
    assert out.attrs["data_source"] == "eastmoney_forex_direct"


# ---------------------------------------------------------------------------
# 5. US & HK stock adjust support matching API capabilities
# ---------------------------------------------------------------------------


def test_us_hk_adjust_capabilities_and_return_basis(monkeypatch):
    called_adjusts: list[tuple[str, str]] = []
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    mock_raw = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0, 104.0],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )

    class FakeAk:
        @staticmethod
        def stock_us_daily(symbol: str = "", adjust: str = "") -> pd.DataFrame:
            called_adjusts.append(("us", adjust))
            return mock_raw.copy()

        @staticmethod
        def stock_hk_daily(symbol: str = "", adjust: str = "") -> pd.DataFrame:
            called_adjusts.append(("hk", adjust))
            return mock_raw.copy()

    def fail_import():
        raise AssertionError("Unsupported adjust must fail before importing akshare")

    monkeypatch.setattr(data_fetcher, "_import_akshare", fail_import)
    with pytest.raises(ValueError, match="不支持复权方式 'hfq'"):
        data_fetcher.load_global_market_data("us_stock", "AAPL", dates[0], dates[-1], adjust="hfq")

    monkeypatch.setattr(data_fetcher, "_import_akshare", lambda: FakeAk())
    us_default = asset_loader.load_asset_data("us_stock", "AAPL", dates[0], dates[-1])
    us_raw = asset_loader.load_asset_data("us_stock", "AAPL", dates[0], dates[-1], adjust="")
    hk_hfq = asset_loader.load_asset_data("hk_stock", "700", dates[0], dates[-1], adjust="hfq")
    hk_qfq = asset_loader.load_asset_data("hk_stock", "700", dates[0], dates[-1], adjust="qfq")

    assert called_adjusts == [("us", "qfq"), ("us", ""), ("hk", "hfq"), ("hk", "qfq")]
    assert us_default.attrs["adjust"] == "qfq"
    assert us_default.attrs["return_basis"] == "adjusted_price"
    assert us_default.attrs["data_source"] == "akshare.stock_us_daily"
    assert us_raw.attrs["adjust"] == ""
    assert us_raw.attrs["return_basis"] == "unadjusted_price"
    assert hk_hfq.attrs["symbol"] == "00700"
    assert hk_hfq.attrs["adjust"] == "hfq"
    assert hk_qfq.attrs["data_source"] == "akshare.stock_hk_daily"


# ---------------------------------------------------------------------------
# 6. Concurrent multi-asset portfolio loading & provenance aggregation
# ---------------------------------------------------------------------------


def test_load_and_align_multi_asset_portfolio_runs_legs_concurrently(monkeypatch):
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    fx_df = _sample_daily_frame(dates, base=7.2)

    def slow_load_asset_data(*, asset_type, symbol, start_date, end_date, proxy_url=None):
        time.sleep(0.15)
        df = _sample_daily_frame(dates, base=100.0 if asset_type == "etf" else 200.0)
        df.attrs["asset_type"] = asset_type
        df.attrs["symbol"] = symbol
        df.attrs["data_source"] = f"mock.{asset_type}"
        df.attrs["adjust"] = "hfq" if asset_type == "etf" else "qfq"
        return df

    monkeypatch.setattr(asset_loader, "load_asset_data", slow_load_asset_data)

    t0 = time.monotonic()
    res = asset_loader.load_and_align_multi_asset_portfolio(
        (("etf", "510300"), ("us_stock", "AAPL"), ("gold", "GC")),
        dates[0],
        dates[-1],
        base_currency="CNY",
        fx_loader=lambda pair, s, e: fx_df,
    )
    elapsed = time.monotonic() - t0
    assert elapsed < 0.35
    assert list(res.base_prices.columns) == ["etf:510300", "us_stock:AAPL", "gold:GC"]
    assert res.overlap_summary["data_sources"] == {
        "etf:510300": "mock.etf",
        "us_stock:AAPL": "mock.us_stock",
        "gold:GC": "mock.gold",
    }
    assert res.overlap_summary["adjustments"]["us_stock:AAPL"] == "qfq"
