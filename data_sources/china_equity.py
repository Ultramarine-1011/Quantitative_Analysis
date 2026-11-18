"""China equity routes, including vendor fallback order and adjustment policy."""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable, Mapping
from typing import Any, Literal

import pandas as pd

from asset_catalog import normalize_symbol_alias
from ohlcv import A_SHARE_COLUMN_MAPPING, DateLike, filter_by_date_range

from .base import BaseDataFetcher
from .eastmoney import EastMoneyDailyKlineClient, eastmoney_sec_market_id
from .errors import (
    ChinaEquityPipelineError,
    EmptySymbolDataError,
    NetworkFetchError,
    VendorUpstreamError,
)
from .network import (
    RequestBudget,
    classify_network_exception,
    configure_windows_proxy_behavior,
    resolve_akshare_callable,
    without_http_proxy_env,
)

_LOGGER = logging.getLogger(__name__)
_eastmoney_sec_market_id = eastmoney_sec_market_id
_configure_windows_proxy_behavior = configure_windows_proxy_behavior
_without_http_proxy_env = without_http_proxy_env


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
        _, fn = resolve_akshare_callable(ak, "stock_zh_a_hist_em", "stock_zh_a_hist")
        return fn(
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
        _, fn = resolve_akshare_callable(ak, "fund_etf_hist_em")
        return fn(
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
        self._last_alt_warnings: list[str] = []
        code = str(sym).strip()
        start_s = start_ts.strftime("%Y%m%d")
        end_s = end_ts.strftime("%Y%m%d")
        try:
            prefix = "sh" if eastmoney_sec_market_id(code) == 1 else "sz"
        except ValueError as exc:
            self._last_alt_warnings.append(f"china_equity market resolution ({code}): {exc!r}")
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

        def _adapt_alt_frame(raw_df: Any, source_name: str) -> pd.DataFrame | None:
            if not isinstance(raw_df, pd.DataFrame) or raw_df.empty:
                self._last_alt_warnings.append(f"{source_name}({qq}) returned empty data")
                return None
            vol_col = next(
                (col for col in ("volume", "Volume", "成交量") if col in raw_df.columns),
                None,
            )
            if vol_col is None:
                # 腾讯等接口的 amount 为成交金额（元），禁止在缺少真实成交量时误映射为 Volume。
                has_amount = "amount" in raw_df.columns or "成交额" in raw_df.columns
                detail = (
                    "missing 'volume' column "
                    "(found 'amount', refusing to map turnover amount to Volume)"
                    if has_amount
                    else "missing 'volume' column"
                )
                msg = f"{source_name}({qq}) {detail}"
                _LOGGER.warning("%s", msg)
                self._last_alt_warnings.append(msg)
                return None
            conv = _to_em_style(raw_df, vol_col)
            if conv.empty:
                self._last_alt_warnings.append(f"{source_name}({qq}) returned empty data")
                return None
            if not conv[["开盘", "收盘", "最高", "最低"]].notna().all(axis=1).any():
                msg = f"{source_name}({qq}) returned no valid OHLC price rows"
                _LOGGER.warning("%s", msg)
                self._last_alt_warnings.append(msg)
                return None
            if not conv["成交量"].notna().any():
                msg = f"{source_name}({qq}) returned no valid 'volume' values"
                _LOGGER.warning("%s", msg)
                self._last_alt_warnings.append(msg)
                return None
            conv.attrs["data_source"] = source_name
            return conv

        ak = self._load_akshare()
        if equity_route == "etf" and adjust == "":
            # Sina ETF history is unadjusted; never use it for qfq/hfq.
            try:
                sina_fn = getattr(ak, "fund_etf_hist_sina", None)
                if not callable(sina_fn):
                    raise AttributeError("akshare has no callable 'fund_etf_hist_sina'")
                raw_s = sina_fn(symbol=qq)
                conv = _adapt_alt_frame(raw_s, "akshare.fund_etf_hist_sina")
                if conv is not None:
                    return conv
            except Exception as exc:  # noqa: BLE001
                msg = f"akshare.fund_etf_hist_sina({qq}) failed: {exc!r}"
                _LOGGER.warning("%s", msg)
                self._last_alt_warnings.append(msg)

        try:
            tx_fn = getattr(ak, "stock_zh_a_hist_tx", None)
            if not callable(tx_fn):
                raise AttributeError("akshare has no callable 'stock_zh_a_hist_tx'")
            # 腾讯源覆盖股票/ETF，优先使用公开接口 ak.stock_zh_a_hist_tx。
            raw_tx = tx_fn(
                symbol=qq,
                start_date=start_s,
                end_date=end_s,
                adjust=adj,
            )
            conv_tx = _adapt_alt_frame(raw_tx, "akshare.stock_zh_a_hist_tx")
            if conv_tx is not None:
                return conv_tx
        except Exception as exc:  # noqa: BLE001
            msg = f"akshare.stock_zh_a_hist_tx({qq}) failed: {exc!r}"
            _LOGGER.warning("%s", msg)
            self._last_alt_warnings.append(msg)

        if equity_route == "stock":
            any_sina_callable = False
            for sina_name in ("stock_zh_a_daily_sina", "stock_zh_a_daily"):
                sina_stock_fn = getattr(ak, sina_name, None)
                if not callable(sina_stock_fn):
                    continue
                any_sina_callable = True
                try:
                    raw_sina = sina_stock_fn(
                        symbol=qq,
                        start_date=start_s,
                        end_date=end_s,
                        adjust=adj,
                    )
                    conv_sina = _adapt_alt_frame(raw_sina, f"akshare.{sina_name}")
                    if conv_sina is not None:
                        return conv_sina
                except Exception as exc:  # noqa: BLE001
                    msg = f"akshare.{sina_name}({qq}) failed: {exc!r}"
                    _LOGGER.warning("%s", msg)
                    self._last_alt_warnings.append(msg)
            if not any_sina_callable:
                self._last_alt_warnings.append(
                    "akshare missing stock_zh_a_daily_sina/stock_zh_a_daily"
                )

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
        primary_source_name: str,
        empty_primary_message: str,
    ) -> pd.DataFrame:
        def _pipeline(budget: RequestBudget) -> pd.DataFrame:
            # 1) 新浪 / 腾讯（不依赖东方财富域名）
            self._last_alt_warnings = []
            alt = self._fetch_sina_tencent_china_equity_raw(
                equity_route, sym, start_ts, end_ts, adjust
            )
            alt_warnings = list(getattr(self, "_last_alt_warnings", ()))
            if not alt.empty:
                alt.attrs.setdefault("data_source", "sina_tencent_fallback")
                if alt_warnings:
                    alt.attrs["fallback_warnings"] = alt_warnings
                return alt

            # 2) AKShare 主路径。若上游返回空表，后面仍会尝试东财直连。
            first_error: BaseException | None = None
            primary_empty = False
            try:
                via_ak = fetch_primary()
                if not via_ak.empty:
                    via_ak.attrs.setdefault("data_source", primary_source_name)
                    if alt_warnings:
                        via_ak.attrs["fallback_warnings"] = alt_warnings
                    return via_ak
                primary_empty = True
                first_error = ValueError(empty_primary_message)
            except ImportError:
                raise
            except Exception as exc:  # noqa: BLE001
                first_error = exc

            # 3) 东方财富直连接口作为最终行情兜底，受限于总预算并记录故障类别。
            em_fail: BaseException | None = None
            try:
                direct = self._eastmoney.fetch_daily_klines(
                    sym,
                    start_ts,
                    end_ts,
                    adjust,
                    trust_env=False,
                    budget=budget,
                )
                if not direct.empty:
                    direct.attrs.setdefault("data_source", "eastmoney_direct")
                    if alt_warnings:
                        direct.attrs["fallback_warnings"] = alt_warnings
                    return direct
                if primary_empty:
                    raise EmptySymbolDataError(
                        "%s（%s）未返回任何有效 K 线数据，请检查代码是否有效。" % (label_zh, sym)
                    )
            except EmptySymbolDataError:
                raise
            except (NetworkFetchError, VendorUpstreamError) as exc:
                em_fail = exc
            except Exception as exc:  # noqa: BLE001
                em_fail = exc

            net_flag = any(
                e is not None and classify_network_exception(e) == "transport"
                for e in (first_error, em_fail)
            )
            if em_fail is not None:
                raise ChinaEquityPipelineError(
                    "%s：新浪/腾讯无有效数据；AKShare 主源未成功 (%r)；东方财富直连失败 (%s)。"
                    % (label_zh, first_error, em_fail),
                    network_related=net_flag,
                ) from em_fail
            raise ChinaEquityPipelineError(
                "%s：新浪/腾讯无有效数据；AKShare 未返回有效数据 (%r)；东方财富直连返回空 K 线。"
                % (label_zh, first_error),
                network_related=net_flag,
            ) from first_error

        budget = RequestBudget(
            total_timeout_seconds=45.0, per_request_timeout_seconds=12.0, max_attempts=10
        )
        if self._neutralize_env_for_ak_primary:
            try:
                with _without_http_proxy_env():
                    return _pipeline(budget)
            except ChinaEquityPipelineError as exc:
                # 仅在传输/代理相关失败且仍有时间预算时，允许再走系统代理试一轮。
                if not exc.network_related or budget.remaining_seconds() <= 1.0:
                    raise
                return _pipeline(budget)
        return _pipeline(budget)

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
        is_etf = route == "etf"
        sym = normalize_symbol_alias("etf" if is_etf else "ashare", symbol)
        primary = self._primary.fetch_etf_hist_em if is_etf else self._primary.fetch_stock_hist
        primary_name = "akshare.fund_etf_hist_em" if is_etf else "akshare.stock_zh_a_hist"
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
                primary_source_name=primary_name,
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
        adjust: Literal["qfq", "hfq", ""] | None = None,
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
        src = raw.attrs.get("data_source")
        host = raw.attrs.get("eastmoney_host")
        fallback_warnings = list(raw.attrs.get("fallback_warnings") or ())
        normalized = self._normalize_ohlcv(raw, A_SHARE_COLUMN_MAPPING)
        out = filter_by_date_range(normalized, start, end)
        if src:
            out.attrs["data_source"] = src
        if host:
            out.attrs["eastmoney_host"] = host
        if fallback_warnings:
            out.attrs["fallback_warnings"] = fallback_warnings
        out.attrs["adjust"] = selected_adjust
        out.attrs["return_basis"] = (
            "unadjusted_price" if selected_adjust == "" else "adjusted_price"
        )
        return out

    def fetch_stock(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        adjust: Literal["qfq", "hfq", ""] | None = None,
    ) -> pd.DataFrame:
        return self._fetch_normalized("stock", symbol, start, end, adjust)

    def fetch_etf(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        adjust: Literal["qfq", "hfq", ""] | None = None,
    ) -> pd.DataFrame:
        return self._fetch_normalized("etf", symbol, start, end, adjust)
