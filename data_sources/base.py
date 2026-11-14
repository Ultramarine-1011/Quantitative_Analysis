"""Small retrying fetcher API retained for custom vendor implementations."""

from __future__ import annotations

import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from ohlcv import DateLike, coerce_analysis_date, normalize_ohlcv, validate_ohlcv

from .errors import DataFetchError, EmptySymbolDataError


class BaseDataFetcher(ABC):
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0) -> None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 1:
            raise ValueError("max_retries must be a positive integer.")
        if not 0 <= retry_delay < float("inf"):
            raise ValueError("retry_delay must be finite and non-negative.")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _run_with_retry(
        self, operation: Callable[[], pd.DataFrame], operation_name: str
    ) -> pd.DataFrame:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                data = operation()
                if not isinstance(data, pd.DataFrame):
                    raise TypeError(f"{operation_name} must return a pandas DataFrame.")
                if data.empty:
                    raise EmptySymbolDataError(f"{operation_name} returned an empty DataFrame.")
                return data
            except (EmptySymbolDataError, ImportError, ValueError, TypeError):
                raise
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_delay)
        # Preserve typed errors and the reason so UI messages remain actionable.
        if isinstance(last_error, DataFetchError):
            raise last_error
        raise DataFetchError(
            f"{operation_name} failed after {self.max_retries} attempts: {last_error}"
        ) from last_error

    def _coerce_date(self, value: DateLike, field_name: str) -> pd.Timestamp:
        return coerce_analysis_date(value, field_name)

    def _format_vendor_date(self, value: DateLike) -> str:
        return self._coerce_date(value, "date").strftime("%Y%m%d")

    def _normalize_ohlcv(self, df: pd.DataFrame, column_mapping: Mapping[str, str]) -> pd.DataFrame:
        return normalize_ohlcv(df, column_mapping)

    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        return validate_ohlcv(df)

    @abstractmethod
    def _fetch_raw(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """Fetch a vendor's raw frame."""


class YFinanceFetcher(BaseDataFetcher):
    """已弃用的 yfinance 占位类；请改用 ``load_asset_data('us_stock', ...)``。"""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0) -> None:
        warnings.warn(
            "YFinanceFetcher is deprecated and unimplemented; use "
            "asset_loader.load_asset_data('us_stock', ...) or "
            "data_fetcher.load_global_market_data('us_stock', ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)

    def _fetch_raw(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        """抓取 yfinance 原始行情数据（已弃用，未实现）。"""
        raise NotImplementedError("YFinanceFetcher is deprecated and unimplemented.")

    def fetch_us_stock(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        interval: str = "1d",
        auto_adjust: bool = False,
    ) -> pd.DataFrame:
        """抓取美股行情（已弃用，请改用 ``load_global_market_data('us_stock', ...)``）。"""
        raise NotImplementedError("YFinanceFetcher is deprecated and unimplemented.")
