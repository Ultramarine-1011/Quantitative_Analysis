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

    @staticmethod
    def _ema(price: pd.Series, span: int) -> pd.Series:
        return price.ewm(span=span, min_periods=span, adjust=False).mean().rename(f"ema_{span}")

    @staticmethod
    def _rsi(price: pd.Series, window: int) -> pd.Series:
        delta = price.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / float(window), min_periods=window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / float(window), min_periods=window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        # Flat prices (both avg_gain and avg_loss == 0) correspond to neutral RSI=50; zero loss with gain > 0 is 100.
        both_zero = (avg_gain == 0.0) & (avg_loss == 0.0)
        only_gain = (avg_gain > 0.0) & (avg_loss == 0.0)
        rsi = rsi.where(~only_gain, 100.0)
        rsi = rsi.where(~both_zero, 50.0)
        rsi.iloc[:window] = np.nan
        return rsi.rename(f"rsi_{window}").astype("float64")

    @staticmethod
    def _macd(price: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
        ema_fast = price.ewm(span=fast, min_periods=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, min_periods=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return pd.DataFrame(
            {
                "macd": macd_line.astype("float64"),
                "macd_signal": signal_line.astype("float64"),
                "macd_hist": hist.astype("float64"),
            },
            index=price.index,
        )

    def compute_ema(self, df: pd.DataFrame, price_col: str = "Close", span: int = 12) -> pd.Series:
        span = self._validate_positive_int(span, "span")
        price = self._get_price_series(self.validate_input(df), price_col)
        return self._ema(price, span)

    def compute_rsi(
        self, df: pd.DataFrame, price_col: str = "Close", window: int = 14
    ) -> pd.Series:
        window = self._validate_positive_int(window, "window")
        price = self._get_price_series(self.validate_input(df), price_col)
        return self._rsi(price, window)

    def compute_macd(
        self,
        df: pd.DataFrame,
        price_col: str = "Close",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        fast = self._validate_positive_int(fast, "fast")
        slow = self._validate_positive_int(slow, "slow")
        signal = self._validate_positive_int(signal, "signal")
        if fast >= slow:
            raise ValueError("MACD fast span must be smaller than slow span.")
        price = self._get_price_series(self.validate_input(df), price_col)
        return self._macd(price, fast, slow, signal)

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
        include_momentum_indicators: bool = False,
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ) -> pd.DataFrame:
        """Add SMA, log volatility, Bollinger bands, drawdown and optional RSI/MACD; preserve attrs.

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
        if include_momentum_indicators:
            rsi_window = self._validate_positive_int(rsi_window, "rsi_window")
            macd_fast = self._validate_positive_int(macd_fast, "macd_fast")
            macd_slow = self._validate_positive_int(macd_slow, "macd_slow")
            macd_signal = self._validate_positive_int(macd_signal, "macd_signal")
            if macd_fast >= macd_slow:
                raise ValueError("macd_fast must be smaller than macd_slow.")

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
        if include_momentum_indicators:
            out[f"rsi_{rsi_window}"] = self._rsi(price, rsi_window)
            macd_df = self._macd(price, macd_fast, macd_slow, macd_signal)
            out[macd_df.columns] = macd_df
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
            "include_momentum_indicators": include_momentum_indicators,
        }
        return out


__all__ = ["REQUIRED_OHLCV_COLUMNS", "BaseFeatureEngineer", "OHLCVFeatureEngineer"]
