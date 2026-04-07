"""量化模型：价格归一化、GBM 蒙特卡洛路径与有效前沿随机权重模拟。

本模块与主链路 OHLCV 契约对齐：收盘价列优先使用 ``Close``，并兼容 ``close``。
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


def _resolve_price_column(df: pd.DataFrame, price_col: str | None) -> str:
    """解析价格列名：显式列名优先，否则 ``Close`` / ``close`` 二选一。"""
    if price_col is not None:
        if price_col not in df.columns:
            raise ValueError(f"price_col {price_col!r} not in DataFrame columns.")
        return price_col
    if "Close" in df.columns:
        return "Close"
    if "close" in df.columns:
        return "close"
    raise ValueError("DataFrame must contain a 'Close' or 'close' price column.")


def normalize_close_to_base_one(
    df: pd.DataFrame,
    price_col: str | None = None,
) -> pd.Series:
    """将收盘价序列归一化为以首期为 1 的净值曲线。

    Parameters
    ----------
    df
        含收盘价列的表；索引通常为日期。
    price_col
        价格列名；为 ``None`` 时自动选择 ``Close`` 或 ``close``。

    Returns
    -------
    pd.Series
        ``close / close.iloc[0]``，与 ``df`` 索引对齐。

    Raises
    ------
    ValueError
        若首价为零、缺失或非有限值。
    """
    col = _resolve_price_column(df, price_col)
    close = pd.to_numeric(df[col], errors="coerce")
    if close.empty:
        raise ValueError("Close series is empty.")
    first = close.iloc[0]
    if pd.isna(first) or not np.isfinite(first) or first == 0:
        raise ValueError(
            "First closing price must be finite, non-NaN, and non-zero for base-one normalization."
        )
    return close.astype(float) / float(first)


def _iqr_cross_check(sample_1d: np.ndarray) -> None:
    """用 ``numpy.percentile`` 与 ``scipy.stats.iqr`` 对四分位距做一致性校验。"""
    sample_1d = np.asarray(sample_1d, dtype=float)
    sample_1d = sample_1d[np.isfinite(sample_1d)]
    if sample_1d.size < 2:
        return
    q75, q25 = np.percentile(sample_1d, [75.0, 25.0])
    iqr_np = float(q75 - q25)
    iqr_sp = float(stats.iqr(sample_1d, rng=(25, 75)))
    if not np.isclose(iqr_np, iqr_sp, rtol=1e-12, atol=1e-10):
        raise RuntimeError(
            "NumPy percentile IQR and SciPy stats.iqr disagree; check inputs and library versions."
        )


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
) -> MonteCarloGBMResult:
    """基于历史对数收益估计参数，按 GBM 离散格式做未来路径蒙特卡洛模拟。

    日对数收益 :math:`R_t=\\ln(P_t/P_{t-1})`；样本标准差为 :math:`\\hat\\sigma`，
    漂移按 :math:`\\hat\\mu=\\bar R+\\hat\\sigma^2/2` 校准，使与
    :math:`dS=\\mu S\\,dt+\\sigma S\\,dW` 及
    :math:`\\ln(S_{t+1}/S_t)\\sim N((\\mu-\\sigma^2/2)\\Delta t,\\sigma^2\\Delta t)`
    （取 :math:`\\Delta t=1` 日）一致。

    模拟递推：:math:`S_{t+1}=S_t\\exp((\\mu-\\sigma^2/2)+\\sigma\\sqrt{\\Delta t}\\,Z)`，
    :math:`\\Delta t=1` 时即 :math:`S_{t+1}=S_t\\exp((\\mu-\\sigma^2/2)+\\sigma Z)`。

    沿时间轴的分位数由 ``numpy.percentile`` 计算；对首末步横截面价格与 SciPy
    ``stats.iqr`` 做四分位距交叉校验（生产仍以 NumPy 为准）。

    Parameters
    ----------
    df
        含收盘价与日期索引的表。
    days
        向前模拟的交易日个数（默认 252）。
    num_simulations
        路径条数。
    random_state
        可重复的随机种子或 ``numpy.random.Generator``。
    price_col
        价格列；``None`` 时自动 ``Close`` / ``close``。
    num_sample_paths
        返回用于展示的样本路径条数（不超过 ``num_simulations``）。

    Returns
    -------
    MonteCarloGBMResult
        历史序列、未来交易日索引、分位数带、若干示例路径及估计的日漂移、日波动与 S0。

    Raises
    ------
    ValueError
        数据不足以估计波动、波动非正、或 ``days`` / ``num_simulations`` 非法。
    """
    if days < 1 or num_simulations < 1:
        raise ValueError("days and num_simulations must be positive integers.")
    col = _resolve_price_column(df, price_col)
    prices = pd.to_numeric(df[col], errors="coerce").astype(float)
    if not isinstance(df.index, pd.DatetimeIndex):
        idx = pd.to_datetime(df.index, errors="coerce")
        if idx.isna().all():
            raise ValueError("DataFrame index must be convertible to datetimes.")
        prices.index = idx
    else:
        prices.index = df.index
    prices = prices.sort_index()
    prices = prices.dropna()
    if prices.shape[0] < 3:
        raise ValueError("Need at least 3 valid closing prices to estimate GBM parameters.")
    log_ret = np.log(prices / prices.shift(1)).dropna()
    if log_ret.shape[0] < 2:
        raise ValueError("Insufficient log returns after differencing.")
    sigma = float(log_ret.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Estimated daily volatility sigma must be finite and positive.")
    mu = float(log_ret.mean() + 0.5 * sigma**2)
    s0 = float(prices.iloc[-1])
    if not np.isfinite(s0) or s0 <= 0:
        raise ValueError("Last closing price S0 must be finite and positive.")

    rng = np.random.default_rng(random_state)
    z = rng.standard_normal(size=(num_simulations, days))
    increments = (mu - 0.5 * sigma**2) + sigma * z
    log_levels = np.log(s0) + np.cumsum(increments, axis=1)
    paths = np.exp(log_levels)

    p05, p50, p95 = (
        np.percentile(paths, q, axis=0) for q in (5.0, 50.0, 95.0)
    )
    _iqr_cross_check(paths[:, 0])
    _iqr_cross_check(paths[:, -1])

    last_date = prices.index[-1]
    if not isinstance(last_date, pd.Timestamp):
        last_date = pd.Timestamp(last_date)
    future_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=days)

    k_show = max(1, min(int(num_sample_paths), num_simulations))
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


@dataclass
class EfficientFrontierResult:
    """随机权重有效前沿模拟结果。"""

    volatility: np.ndarray
    returns: np.ndarray
    sharpe: np.ndarray
    weights: np.ndarray
    best_sharpe_idx: int


def generate_efficient_frontier(
    returns_df: pd.DataFrame,
    num_portfolios: int = 5000,
    risk_free_rate: float = 0.015,
    trading_days: int = 252,
    random_state: int | np.random.Generator | None = None,
    min_history: int = 30,
) -> EfficientFrontierResult:
    """在简单收益率宽表上随机生成权重，计算年化收益、波动与夏普比率。

    权重独立服从 Dirichlet(1,...,1)，保证非负且和为 1。组合日收益为
    ``returns_df @ w``；年化收益为日均值乘以 ``trading_days``，年化波动为
    日标准差乘以 ``sqrt(trading_days)``；夏普为
    ``(年化收益 - risk_free_rate) / 年化波动``（波动为 0 处为 NaN）。

    Parameters
    ----------
    returns_df
        列为资产、行为交易日的日**简单**收益率宽表（如 ``pct_change`` 后删首行）。
    num_portfolios
        随机组合个数。
    risk_free_rate
        年化无风险利率，用于夏普。
    trading_days
        每年的交易日数，用于年化。
    random_state
        随机种子或 ``Generator``，用于 Dirichlet 抽样。
    min_history
        有效样本最少行数，不足则报错。

    Returns
    -------
    EfficientFrontierResult
        各组合的波动、收益、夏普、权重矩阵及夏普最大（忽略 NaN）的索引。

    Raises
    ------
    ValueError
        资产数小于 2、历史不足、数据全为 NaN 或收益协方差矩阵病态/奇异。
    """
    if num_portfolios < 1:
        raise ValueError("num_portfolios must be at least 1.")
    if returns_df.shape[1] < 2:
        raise ValueError("Efficient frontier requires at least 2 assets (columns).")
    r = returns_df.copy()
    r = r.apply(pd.to_numeric, errors="coerce")
    r = r.dropna(how="all")
    if r.shape[0] < min_history:
        raise ValueError(
            f"Need at least {min_history} rows of joint return history after cleaning; got {r.shape[0]}."
        )
    if r.shape[1] < 2:
        raise ValueError("Fewer than 2 asset columns remain after dropping all-NaN rows.")
    r = r.dropna()
    if r.empty:
        raise ValueError("Returns are empty after dropping rows with any NaN.")
    if r.shape[0] < min_history:
        raise ValueError(
            f"Need at least {min_history} complete return rows; got {r.shape[0]} after dropna."
        )

    x = r.values.astype(float, copy=False)
    n_assets = x.shape[1]
    if not np.isfinite(x).any():
        raise ValueError("Return matrix contains no finite values.")

    cov = np.cov(x, rowvar=False)
    if cov.shape != (n_assets, n_assets):
        raise ValueError("Internal error: covariance shape mismatch.")
    eff_rank = np.linalg.matrix_rank(cov, tol=1e-10)
    if eff_rank < n_assets:
        raise ValueError(
            "Return covariance matrix is singular or rank-deficient; check for redundant or constant series."
        )
    cond = np.linalg.cond(cov)
    if not np.isfinite(cond) or cond > 1e12:
        raise ValueError(
            f"Return covariance matrix is ill-conditioned (cond={cond:.2e}); cannot stabilize frontier simulation."
        )

    rng = np.random.default_rng(random_state)
    w = rng.dirichlet(np.ones(n_assets), size=num_portfolios).astype(float, copy=False)

    port_daily = x @ w.T
    mean_d = np.mean(port_daily, axis=0)
    std_d = np.std(port_daily, axis=0, ddof=1)
    ann_ret = mean_d * trading_days
    ann_vol = std_d * np.sqrt(trading_days)
    sharpe = np.where(ann_vol > 0, (ann_ret - risk_free_rate) / ann_vol, np.nan)

    best_sharpe_idx = int(np.nanargmax(sharpe))

    return EfficientFrontierResult(
        volatility=ann_vol.astype(float, copy=False),
        returns=ann_ret.astype(float, copy=False),
        sharpe=sharpe.astype(float, copy=False),
        weights=w,
        best_sharpe_idx=best_sharpe_idx,
    )


__all__ = [
    "EfficientFrontierResult",
    "MonteCarloGBMResult",
    "generate_efficient_frontier",
    "normalize_close_to_base_one",
    "run_monte_carlo_gbm",
]
