"""China equity routes, including vendor fallback order and adjustment policy."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from typing import Any, Literal

import pandas as pd

from ohlcv import A_SHARE_COLUMN_MAPPING, DateLike, filter_by_date_range
from .base import BaseDataFetcher
from .eastmoney import EastMoneyDailyKlineClient, _eastmoney_sec_market_id
from .errors import ChinaEquityPipelineError, NetworkFetchError, VendorUpstreamError
from .network import _configure_windows_proxy_behavior, _without_http_proxy_env


class AKShareChinaPrimarySource:
    """AKShare 沪深股票 / ETF 主路径，与东方财富直连回退解耦。"""

    def __init__(self, ak_loader: Callable[[], Any]) -> None:
        self._ak_loader = ak_loader

    def fetch_stock_hist(
        self,
        symbol: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        adjust: str,
    ) -> pd.DataFrame:
        ak = self._ak_loader()
        return ak.stock_zh_a_hist(
            symbol=str(symbol).strip(),
            period="daily",
            start_date=start_ts.strftime("%Y%m%d"),
            end_date=end_ts.strftime("%Y%m%d"),
            adjust=adjust,
        )

    def fetch_etf_hist_em(
        self,
        symbol: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        adjust: str,
    ) -> pd.DataFrame:
        ak = self._ak_loader()
        return ak.fund_etf_hist_em(
            symbol=str(symbol).strip(),
            period="daily",
            start_date=start_ts.strftime("%Y%m%d"),
            end_date=end_ts.strftime("%Y%m%d"),
            adjust=adjust,
        )


class AKShareFetcher(BaseDataFetcher):
    """基于 AKShare 的 A 股与 ETF 数据获取器。"""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        *,
        eastmoney_headers: Mapping[str, str] | None = None,
        use_system_proxy: bool | None = None,
    ) -> None:
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        _configure_windows_proxy_behavior(use_system_proxy=use_system_proxy)
        # 默认绕开系统代理；若用户显式允许系统代理，则保留 AKShare 原始网络行为。
        if use_system_proxy is True:
            self._neutralize_env_for_ak_primary = False
        elif use_system_proxy is False:
            self._neutralize_env_for_ak_primary = True
        else:
            self._neutralize_env_for_ak_primary = not (
                os.environ.get("QUANT_USE_SYSTEM_PROXY", "").lower() in {"1", "true", "yes"}
            )
        self._eastmoney = EastMoneyDailyKlineClient(
            None if eastmoney_headers is None else dict(eastmoney_headers)
        )
        self._primary = AKShareChinaPrimarySource(self._load_akshare)

    def _load_akshare(self) -> Any:
        """按需导入 AKShare，避免模块导入阶段产生硬依赖。"""
        try:
            return importlib.import_module("akshare")
        except ImportError as exc:
            raise ImportError(
                "AKShare is required for AKShareFetcher. Install it with `pip install akshare`."
            ) from exc

    def _fetch_sina_tencent_china_equity_raw(
        self,
        equity_route: str,
        sym: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        adjust: str,
    ) -> pd.DataFrame:
        """新浪 / 腾讯日线（非东方财富域名，作为境内权益首选数据源）。"""
        code = str(sym).strip()
        start_s = start_ts.strftime("%Y%m%d")
        end_s = end_ts.strftime("%Y%m%d")
        try:
            prefix = "sh" if _eastmoney_sec_market_id(code) == 1 else "sz"
        except ValueError:
            return pd.DataFrame()
        qq = "%s%s" % (prefix, code)
        adj = adjust

        def _to_em_style(df: pd.DataFrame, vol_col: str) -> pd.DataFrame:
            if df.empty or "date" not in df.columns:
                return pd.DataFrame()
            out = pd.DataFrame(
                {
                    "日期": pd.to_datetime(df["date"], errors="coerce"),
                    "开盘": pd.to_numeric(df["open"], errors="coerce"),
                    "收盘": pd.to_numeric(df["close"], errors="coerce"),
                    "最高": pd.to_numeric(df["high"], errors="coerce"),
                    "最低": pd.to_numeric(df["low"], errors="coerce"),
                    "成交量": pd.to_numeric(df[vol_col], errors="coerce"),
                }
            )
            out = out.dropna(subset=["日期"])
            lo = start_ts.normalize()
            hi = end_ts.normalize()
            mask = (out["日期"] >= lo) & (out["日期"] <= hi)
            return out.loc[mask].copy()

        try:
            if equity_route == "etf" and adjust == "":
                # Sina ETF history is unadjusted; never use it for qfq/hfq.
                try:
                    # ETF 先尝试新浪源，避免东方财富域名不可达时整个链路失败。
                    sina = importlib.import_module("akshare.fund.fund_etf_sina")
                    raw_s = sina.fund_etf_hist_sina(symbol=qq)
                    if not raw_s.empty and "volume" in raw_s.columns:
                        conv = _to_em_style(raw_s, "volume")
                        if not conv.empty:
                            return conv
                except Exception:  # noqa: BLE001
                    pass
            mod_tx = importlib.import_module("akshare.stock_feature.stock_hist_tx")
            # 腾讯源覆盖股票/部分 ETF，作为进入 AKShare 东财主源前的轻量备用路径。
            raw_tx = mod_tx.stock_zh_a_hist_tx(
                symbol=qq,
                start_date=start_s,
                end_date=end_s,
                adjust=adj,
            )
            return _to_em_style(raw_tx, "amount")
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def _fetch_via_primary_then_eastmoney(
        self,
        *,
        sym: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        adjust: str,
        label_zh: str,
        equity_route: str,
        fetch_primary: Callable[[], pd.DataFrame],
        empty_primary_message: str,
    ) -> pd.DataFrame:
        def _pipeline() -> pd.DataFrame:
            # 1) 新浪 / 腾讯（不依赖东方财富域名）
            alt = self._fetch_sina_tencent_china_equity_raw(
                equity_route, sym, start_ts, end_ts, adjust
            )
            if not alt.empty:
                return alt

            # 2) AKShare 主路径。若上游返回空表，后面仍会尝试东财直连。
            first_error: BaseException | None = None
            try:
                via_ak = fetch_primary()
                if not via_ak.empty:
                    return via_ak
                first_error = ValueError(empty_primary_message)
            except ImportError:
                raise
            except Exception as exc:  # noqa: BLE001
                first_error = exc

            # 3) 东方财富直连接口作为最终行情兜底，失败时汇总前面每一步的错误。
            em_fail: BaseException | None = None
            try:
                direct = self._eastmoney.fetch_daily_klines(
                    sym,
                    start_ts,
                    end_ts,
                    adjust,
                    trust_env=False,
                )
                if not direct.empty:
                    return direct
            except (NetworkFetchError, VendorUpstreamError) as exc:
                em_fail = exc
            except Exception as exc:  # noqa: BLE001
                em_fail = exc

            if em_fail is not None:
                raise ChinaEquityPipelineError(
                    "%s：新浪/腾讯无有效数据；AKShare 主源未成功 (%r)；东方财富直连失败 (%s)。"
                    % (label_zh, first_error, em_fail)
                ) from em_fail
            raise ChinaEquityPipelineError(
                "%s：新浪/腾讯无有效数据；AKShare 未返回有效数据 (%r)；东方财富直连返回空 K 线。"
                % (label_zh, first_error)
            ) from first_error

        if self._neutralize_env_for_ak_primary:
            try:
                with _without_http_proxy_env():
                    return _pipeline()
            except ChinaEquityPipelineError:
                # 直连不可达时（例如必须经本机 Clash / 公司代理才能访问外网），再按系统与环境变量走代理重试一轮
                return _pipeline()
        return _pipeline()

    def _fetch_equity_raw(self, route: str, symbol: str, **kwargs: Any) -> pd.DataFrame:
        if not symbol or not str(symbol).strip():
            raise ValueError("symbol must be a non-empty security code.")
        start_date, end_date = kwargs.get("start_date"), kwargs.get("end_date")
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required for equity fetching.")
        start_ts = self._coerce_date(start_date, "start_date")
        end_ts = self._coerce_date(end_date, "end_date")
        if start_ts > end_ts:
            raise ValueError("start_date must be earlier than or equal to end_date.")
        adjust = kwargs.get("adjust", "hfq")
        if adjust not in {"", "qfq", "hfq"}:
            raise ValueError("adjust must be one of {'qfq', 'hfq', ''}.")
        sym = str(symbol).strip()
        is_etf = route == "etf"
        primary = self._primary.fetch_etf_hist_em if is_etf else self._primary.fetch_stock_hist
        label = "ETF" if is_etf else "A-share"
        return self._run_with_retry(
            lambda: self._fetch_via_primary_then_eastmoney(
                sym=sym,
                start_ts=start_ts,
                end_ts=end_ts,
                adjust=adjust,
                label_zh=f"{label} 日线抓取",
                equity_route=route,
                fetch_primary=lambda: primary(sym, start_ts, end_ts, adjust),
                empty_primary_message=f"AKShare {label} returned an empty DataFrame.",
            ),
            f"AKShare {label} historical data fetch",
        )

    def _fetch_raw(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        return self._fetch_equity_raw("stock", symbol, **kwargs)

    def _fetch_etf_raw(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        return self._fetch_equity_raw("etf", symbol, **kwargs)

    def fetch_ashare_data(
        self, symbol: str, start_date: DateLike, end_date: DateLike
    ) -> pd.DataFrame:
        """Legacy entry point for adjusted stock OHLCV."""
        return self.fetch_stock(symbol, start=start_date, end=end_date, adjust="hfq")

    def fetch_china_equity(
        self,
        route: Literal["etf", "stock"],
        symbol: str,
        start: DateLike,
        end: DateLike,
        adjust: Literal["qfq", "hfq"] | None = None,
    ) -> pd.DataFrame:
        route_key = str(route).strip().lower()
        if route_key == "etf":
            return self.fetch_etf(symbol, start=start, end=end, adjust=adjust)
        if route_key == "stock":
            return self.fetch_stock(symbol, start=start, end=end, adjust=adjust)
        raise ValueError(f"fetch_china_equity route must be 'etf' or 'stock', got {route!r}.")

    def _fetch_normalized(
        self,
        route: str,
        symbol: str,
        start: DateLike | None,
        end: DateLike | None,
        adjust: str | None,
    ) -> pd.DataFrame:
        if start is None or end is None:
            raise ValueError(f"fetch_{route} currently requires both start and end.")
        # None defaults to hfq; an explicit empty string selects unadjusted data.
        selected_adjust = "hfq" if adjust is None else adjust
        fetch_raw = self._fetch_etf_raw if route == "etf" else self._fetch_raw
        raw = fetch_raw(symbol=symbol, start_date=start, end_date=end, adjust=selected_adjust)
        normalized = self._normalize_ohlcv(raw, A_SHARE_COLUMN_MAPPING)
        return filter_by_date_range(normalized, start, end)

    def fetch_stock(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        adjust: Literal["qfq", "hfq"] | None = None,
    ) -> pd.DataFrame:
        return self._fetch_normalized("stock", symbol, start, end, adjust)

    def fetch_etf(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        adjust: Literal["qfq", "hfq"] | None = None,
    ) -> pd.DataFrame:
        return self._fetch_normalized("etf", symbol, start, end, adjust)
