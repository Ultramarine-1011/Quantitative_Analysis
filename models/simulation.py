"""Geometric Brownian motion simulation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from asset_catalog import get_asset_trading_days
from models.returns import _positive_integer, _resolve_price_column


@dataclass
class MonteCarloGBMResult:
    """GBM 蒙特卡洛模拟结果，便于绑图与展示元数据。"""

    historical_dates: pd.DatetimeIndex | pd.Index
    historical_close: np.ndarray
    future_dates: pd.DatetimeIndex
    quantiles_p05: np.ndarray
    quantiles_p50: np.ndarray
    quantiles_p95: np.ndarray
    sample_paths: np.ndarray
    mu_daily: float
    sigma_daily: float
    s0: float


def run_monte_carlo_gbm(
    df: pd.DataFrame,
    days: int = 252,
    num_simulations: int = 100,
    random_state: int | np.random.Generator | None = None,
    price_col: str | None = None,
    num_sample_paths: int = 5,
    trading_days: int | None = None,
    annual_drift: float | None = None,
) -> MonteCarloGBMResult:
    """基于历史对数收益估计参数，按 GBM 离散格式做未来路径蒙特卡洛模拟。

    默认由历史日对数收益估计样本标准差 :math:`\\hat\\sigma` 与漂移
    :math:`\\hat\\mu=\\bar R+\\hat\\sigma^2/2`；若显式传入 ``annual_drift``，
    则按 ``annual_drift / trading_days`` 设定日连续漂移率 ``mu_daily``。
    加密货币等 7×24 市场使用自然日（``freq="D"``）生成未来日期，其余使用工作日。
    """
    days = _positive_integer(days, "days")
    num_simulations = _positive_integer(num_simulations, "num_simulations")
    num_sample_paths = _positive_integer(num_sample_paths, "num_sample_paths")
    asset_type = str(getattr(df, "attrs", {}).get("asset_type", "") or "").strip().lower()
    eff_trading_days = (
        _positive_integer(trading_days, "trading_days")
        if trading_days is not None
        else get_asset_trading_days(asset_type or None)
    )
    if annual_drift is not None and (
        isinstance(annual_drift, bool)
        or not isinstance(annual_drift, Real)
        or not np.isfinite(float(annual_drift))
    ):
        raise ValueError("annual_drift must be finite when provided.")

    col = _resolve_price_column(df, price_col)
    prices = pd.to_numeric(df[col], errors="coerce").astype(float)
    if not isinstance(df.index, pd.DatetimeIndex):
        idx = pd.to_datetime(df.index, errors="coerce")
        if idx.isna().all():
            raise ValueError("DataFrame index must be convertible to datetimes.")
        prices.index = idx
    else:
        prices.index = df.index
    if prices.index.hasnans:
        raise ValueError("DataFrame index contains invalid datetimes.")
    prices = prices.sort_index(kind="stable")
    prices = prices.loc[~prices.index.duplicated(keep="last")].dropna()
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("GBM requires finite, strictly positive closing prices.")
    if prices.shape[0] < 3:
        raise ValueError("Need at least 3 valid closing prices to estimate GBM parameters.")
    # 用历史日对数收益估计 GBM 的漂移 mu 与波动 sigma。
    log_ret = np.log(prices / prices.shift(1)).dropna()
    if log_ret.shape[0] < 2:
        raise ValueError("Insufficient log returns after differencing.")
    sigma = float(log_ret.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Estimated daily volatility sigma must be finite and positive.")
    if annual_drift is None:
        mu = float(log_ret.mean() + 0.5 * sigma**2)
    else:
        mu = float(annual_drift) / float(eff_trading_days)
    s0 = float(prices.iloc[-1])
    if not np.isfinite(s0) or s0 <= 0:
        raise ValueError("Last closing price S0 must be finite and positive.")

    rng = np.random.default_rng(random_state)
    # 每条路径每天抽一个标准正态随机数，代表当天不可预测的市场冲击。
    z = rng.standard_normal(size=(num_simulations, days))
    increments = (mu - 0.5 * sigma**2) + sigma * z
    log_levels = np.log(s0) + np.cumsum(increments, axis=1)
    paths = np.exp(log_levels)

    # 对所有模拟路径按日期取分位数，得到未来价格的概率区间。
    p05, p50, p95 = np.percentile(paths, [5.0, 50.0, 95.0], axis=0)

    last_date = prices.index[-1]
    if not isinstance(last_date, pd.Timestamp):
        last_date = pd.Timestamp(last_date)
    if eff_trading_days >= 360:
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq="D")
    else:
        future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=days)

    k_show = min(num_sample_paths, num_simulations)
    sample_paths = paths[:k_show, :].copy()

    return MonteCarloGBMResult(
        historical_dates=prices.index,
        historical_close=prices.values.astype(float, copy=False),
        future_dates=future_dates,
        quantiles_p05=p05.astype(float, copy=False),
        quantiles_p50=p50.astype(float, copy=False),
        quantiles_p95=p95.astype(float, copy=False),
        sample_paths=sample_paths,
        mu_daily=mu,
        sigma_daily=sigma,
        s0=s0,
    )
