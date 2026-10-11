"""Vectorized OHLCV indicators with one validation pass for batch transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd

from ohlcv import REQUIRED_OHLCV_COLUMNS, prepare_ohlcv


class BaseFeatureEngineer(ABC):
    """Compatibility base for custom engineers sharing the OHLCV contract."""

    def validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        return prepare_ohlcv(df)

    @abstractmethod
    def transform(self, df: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """Return a new frame with additional features."""


class OHLCVFeatureEngineer(BaseFeatureEngineer):
    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value < 1
            or int(value) != value
        ):
            raise ValueError(f"{field_name} must be a positive integer.")
        return int(value)

    @staticmethod
    def _validate_positive_float(value: float, field_name: str) -> float:
        numeric = float(value)
        if not np.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"{field_name} must be finite and positive.")
        return numeric

    def _get_price_series(self, df: pd.DataFrame, price_col: str) -> pd.Series:
        if price_col not in df.columns:
            raise ValueError(f"price_col `{price_col}` does not exist in the input DataFrame.")
        price = pd.to_numeric(df[price_col], errors="coerce").astype("float64")
        price = price.replace([np.inf, -np.inf], np.nan).ffill()
        if price.isna().any():
            raise ValueError(
                f"price_col `{price_col}` still contains NaN values after forward-fill."
            )
        return price

    @staticmethod
    def _returns(price: pd.Series, periods: int = 1, log: bool = False) -> pd.Series:
        if log:
            if (price <= 0).any():
                raise ValueError("Log returns require strictly positive prices.")
            result = np.log(price.div(price.shift(periods)))
            result.name = f"log_return_{periods}"
        else:
            result = price.pct_change(periods=periods, fill_method=None)
            result.name = f"return_{periods}"
        return result.astype("float64")

    @staticmethod
    def _bands(price: pd.Series, window: int, num_std: float) -> pd.DataFrame:
        rolling = price.rolling(window=window, min_periods=window)
        mean, std = rolling.mean(), rolling.std()
        return pd.DataFrame(
            {
                f"bollinger_upper_{window}": mean + num_std * std,
                f"bollinger_lower_{window}": mean - num_std * std,
            }
        )

    @staticmethod
    def _drawdown(price: pd.Series) -> pd.Series:
        return (1.0 - price.div(price.cummax())).rename("drawdown").astype("float64")

    def compute_returns(
        self, df: pd.DataFrame, price_col: str = "Close", periods: int = 1, log: bool = False
    ) -> pd.Series:
        """Simple or logarithmic k-period returns; first k observations are NaN."""
        periods = self._validate_positive_int(periods, "periods")
        price = self._get_price_series(self.validate_input(df), price_col)
        return self._returns(price, periods, log)

    def compute_sma(
        self, df: pd.DataFrame, price_col: str = "Close", window: int = 20
    ) -> pd.Series:
        window = self._validate_positive_int(window, "window")
        price = self._get_price_series(self.validate_input(df), price_col)
        return price.rolling(window, min_periods=window).mean().rename(f"sma_{window}")

    def compute_volatility(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
        window: int = 20,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> pd.Series:
        """Rolling sample std of daily log returns, optionally scaled by sqrt(252)."""
        window = self._validate_positive_int(window, "window")
        trading_days = self._validate_positive_int(trading_days, "trading_days")
        price = self._get_price_series(self.validate_input(df), price_col)
        volatility = self._returns(price, log=True).rolling(window, min_periods=window).std()
        if annualize:
            volatility *= np.sqrt(float(trading_days))
        return volatility.rename(f"volatility_{window}")

    def compute_momentum(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
        window: int = 20,
        method: Literal["pct", "diff"] = "pct",
    ) -> pd.Series:
        window = self._validate_positive_int(window, "window")
        price = self._get_price_series(self.validate_input(df), price_col)
        if method == "pct":
            momentum = self._returns(price, periods=window)
        elif method == "diff":
            momentum = price.diff(periods=window)
        else:
            raise ValueError("method must be either 'pct' or 'diff'.")
        return momentum.rename(f"momentum_{window}").astype("float64")

    def compute_bollinger_bands(
        self, df: pd.DataFrame, price_col: str = "Close", window: int = 20, num_std: float = 2.0
    ) -> pd.DataFrame:
        window = self._validate_positive_int(window, "window")
        num_std = self._validate_positive_float(num_std, "num_std")
        price = self._get_price_series(self.validate_input(df), price_col)
        return self._bands(price, window, num_std)

    def compute_drawdown(self, df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
        """Nonnegative depth: 1 - price / running peak."""
        price = self._get_price_series(self.validate_input(df), price_col)
        return self._drawdown(price)

    def compute_max_drawdown(self, df: pd.DataFrame, price_col: str = "Close") -> float:
        return float(self.compute_drawdown(df, price_col).max())

    def transform(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
        sma_short_window: int = 20,
        sma_long_window: int = 60,
        volatility_window: int = 20,
        bollinger_window: int = 20,
        bollinger_num_std: float = 2.0,
        annualize: bool = True,
        trading_days: int = 252,
        include_drawdown: bool = True,
    ) -> pd.DataFrame:
        """Add SMA, log volatility, Bollinger bands and drawdown; preserve attrs.

        Rolling windows and ddof=1 match the standalone methods. Reapplying a
        transform replaces its feature columns instead of failing on a join.
        """
        sma_short_window = self._validate_positive_int(sma_short_window, "sma_short_window")
        sma_long_window = self._validate_positive_int(sma_long_window, "sma_long_window")
        volatility_window = self._validate_positive_int(volatility_window, "volatility_window")
        bollinger_window = self._validate_positive_int(bollinger_window, "bollinger_window")
        bollinger_num_std = self._validate_positive_float(bollinger_num_std, "bollinger_num_std")
        trading_days = self._validate_positive_int(trading_days, "trading_days")
        if sma_short_window >= sma_long_window:
            raise ValueError("sma_short_window must be smaller than sma_long_window.")

        out = self.validate_input(df)
        price = self._get_price_series(out, price_col)
        for window in (sma_short_window, sma_long_window):
            out[f"sma_{window}"] = price.rolling(window, min_periods=window).mean()
        volatility = (
            self._returns(price, log=True)
            .rolling(volatility_window, min_periods=volatility_window)
            .std()
        )
        out[f"volatility_{volatility_window}"] = (
            volatility * np.sqrt(float(trading_days)) if annualize else volatility
        )
        bands = self._bands(price, bollinger_window, bollinger_num_std)
        out[bands.columns] = bands
        drawdown = self._drawdown(price)
        if include_drawdown:
            out["drawdown"] = drawdown
        out.attrs["max_drawdown"] = float(drawdown.max())
        out.attrs["feature_summary"] = {
            "price_col": price_col,
            "sma_short_window": sma_short_window,
            "sma_long_window": sma_long_window,
            "volatility_window": volatility_window,
            "bollinger_window": bollinger_window,
            "bollinger_num_std": bollinger_num_std,
            "annualize": annualize,
            "trading_days": trading_days,
            "include_drawdown": include_drawdown,
        }
        return out


__all__ = ["REQUIRED_OHLCV_COLUMNS", "BaseFeatureEngineer", "OHLCVFeatureEngineer"]
