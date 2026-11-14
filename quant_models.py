"""量化模型：价格归一化、GBM 蒙特卡洛路径与有效前沿随机权重模拟。

本模块与主链路 OHLCV 契约对齐：收盘价列优先使用 ``Close``，并兼容 ``close``。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd

from asset_catalog import get_asset_trading_days


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
    # 归一化后首日净值为 1，方便把不同价格量级的资产画在同一张图上比较。
    return close.astype(float) / float(first)


def _positive_integer(value: int, field_name: str, minimum: int = 1) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or int(value) != value
        or value < minimum
    ):
        raise ValueError(f"{field_name} must be an integer >= {minimum}.")
    return int(value)


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


@dataclass
class EfficientFrontierResult:
    """随机权重有效前沿模拟结果。"""

    volatility: np.ndarray
    returns: np.ndarray
    sharpe: np.ndarray
    weights: np.ndarray
    best_sharpe_idx: int


def clean_joint_returns(returns_df: pd.DataFrame, min_history: int = 30) -> pd.DataFrame:
    """对多资产日收益宽表做与有效前沿一致的联合清洗。

    ``to_numeric`` → 按行 ``dropna(how="all")`` → 再 ``dropna()``（任列为 NaN 的行删除），
    并在关键步骤校验 ``min_history`` 与至少 2 列资产。
    """
    # 多资产组合要求每个交易日都有所有资产收益，缺一个就无法正确相乘求组合收益。
    min_history = _positive_integer(min_history, "min_history", minimum=2)
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame.")
    if not returns_df.columns.is_unique or not returns_df.index.is_unique:
        raise ValueError("Return matrix must have unique columns and observations.")
    r = returns_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
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
    return r


def returns_correlation_matrix(returns_df: pd.DataFrame, min_history: int = 30) -> pd.DataFrame:
    """皮尔逊相关矩阵；样本与列顺序与 ``clean_joint_returns`` 一致。"""
    r = clean_joint_returns(returns_df, min_history=min_history)
    return r.corr(method="pearson")


RebalanceFreq = Literal["buy_and_hold", "daily", "monthly", "quarterly"]


def _is_rebalance_step(index: pd.Index, step: int, freq: RebalanceFreq) -> bool:
    if step == 0 or freq == "buy_and_hold":
        return False
    if freq == "daily":
        return True
    if not isinstance(index, pd.DatetimeIndex):
        return False
    prev_dt = index[step - 1]
    curr_dt = index[step]
    if freq == "monthly":
        return (curr_dt.year, curr_dt.month) != (prev_dt.year, prev_dt.month)
    if freq == "quarterly":
        return (curr_dt.year, (curr_dt.month - 1) // 3) != (
            prev_dt.year,
            (prev_dt.month - 1) // 3,
        )
    return False


def _simulate_portfolio_nav(
    returns_df: pd.DataFrame,
    target_weights: np.ndarray,
    *,
    rebalance_freq: RebalanceFreq,
    transaction_cost_bps: float,
    tradable_mask: pd.DataFrame | None = None,
) -> tuple[np.ndarray, float, float]:
    """Simulate portfolio NAV starting at pre-investment capital 1.0, retaining all return rows."""
    x = returns_df.to_numpy(dtype=float, copy=False)
    n_steps, n_assets = x.shape
    cost_rate = float(transaction_cost_bps) / 10000.0

    # Initial position establishment from cash (base NAV = 1.0).
    init_turnover = float(np.sum(np.abs(target_weights)))
    init_cost = cost_rate * init_turnover
    if init_cost >= 1.0:
        raise ValueError("Initial transaction cost exceeds 100% of starting capital.")
    total_cost = init_cost
    total_turnover = init_turnover
    sub_values = (1.0 - init_cost) * target_weights.copy()

    nav = np.empty(n_steps, dtype=float)
    mask_arr = (
        tradable_mask.reindex(returns_df.index, columns=returns_df.columns)
        .fillna(True)
        .to_numpy(dtype=bool)
        if tradable_mask is not None
        else None
    )

    pending_rebalance = False
    for step in range(n_steps):
        if _is_rebalance_step(returns_df.index, step, rebalance_freq):
            pending_rebalance = True
        if pending_rebalance:
            prev_equity = float(np.sum(sub_values))
            if prev_equity <= 0.0:
                raise ValueError("Portfolio equity fell to zero or below during backtest.")
            tradable_row = mask_arr[step] if mask_arr is not None else np.ones(n_assets, dtype=bool)
            if tradable_row.all():
                desired = prev_equity * target_weights
                turnover_amt = float(np.sum(np.abs(desired - sub_values)))
                step_turnover = turnover_amt / prev_equity
                step_cost = prev_equity * cost_rate * step_turnover
                after_fee = prev_equity - step_cost
                if after_fee <= 0.0:
                    raise ValueError(
                        "Portfolio equity fell to zero or below after transaction costs."
                    )
                sub_values = after_fee * target_weights
                total_turnover += step_turnover
                total_cost += step_cost / prev_equity
                pending_rebalance = False

        sub_values = sub_values * (1.0 + x[step])
        equity = float(np.sum(sub_values))
        if not np.isfinite(equity) or equity <= 0.0:
            raise ValueError("Portfolio equity fell to zero or below during backtest.")
        nav[step] = equity

    return nav, total_turnover, total_cost


def compute_backtest_metrics(
    nav_series: pd.Series,
    *,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.016,
) -> dict[str, float]:
    """Compute CAGR, total return, annualized volatility, max drawdown, Sharpe, Sortino, and Calmar."""
    s = pd.to_numeric(nav_series, errors="coerce").dropna().astype(float)
    if len(s) < 2 or s.iloc[0] <= 0.0 or s.iloc[-1] <= 0.0:
        return {
            "total_return": float("nan"),
            "cagr": float("nan"),
            "annualized_volatility": float("nan"),
            "max_drawdown": float("nan"),
            "sharpe_ratio": float("nan"),
            "sortino_ratio": float("nan"),
            "calmar_ratio": float("nan"),
        }
    total_return = float(s.iloc[-1] / s.iloc[0] - 1.0)
    if isinstance(s.index, pd.DatetimeIndex):
        days = float((s.index[-1] - s.index[0]).days)
        cagr = (
            float((s.iloc[-1] / s.iloc[0]) ** (365.25 / days) - 1.0) if days > 0.0 else float("nan")
        )
    else:
        years = float(len(s) - 1) / float(periods_per_year)
        cagr = (
            float((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0) if years > 0.0 else float("nan")
        )

    running_peak = s.cummax()
    mdd = float((1.0 - s / running_peak).max())
    simple_r = s.pct_change().dropna().to_numpy(dtype=float)
    ppy = float(periods_per_year)
    if simple_r.size >= 2:
        std_r = float(np.std(simple_r, ddof=1))
        ann_vol = std_r * float(np.sqrt(ppy))
        rf_per_period = float(risk_free_rate) / ppy
        excess = simple_r - rf_per_period
        sharpe = (float(np.mean(excess)) * ppy / ann_vol) if ann_vol > 0.0 else float("nan")
        down = np.minimum(excess, 0.0)
        down_dev = float(np.sqrt(np.mean(down * down))) * float(np.sqrt(ppy))
        sortino = (float(np.mean(excess)) * ppy / down_dev) if down_dev > 0.0 else float("nan")
    else:
        ann_vol = float("nan")
        sharpe = float("nan")
        sortino = float("nan")

    calmar = float(cagr / mdd) if (np.isfinite(cagr) and mdd > 0.0) else float("nan")
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": ann_vol,
        "max_drawdown": mdd,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
    }


def _infer_base_date(index: pd.Index, base_date: pd.Timestamp | None) -> object:
    if base_date is not None:
        ts = pd.Timestamp(base_date)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        if isinstance(index, pd.DatetimeIndex) and ts >= index[0]:
            raise ValueError("base_date must be strictly earlier than the first return date.")
        return ts
    if isinstance(index, pd.DatetimeIndex):
        return index[0] - pd.Timedelta(days=1)
    if isinstance(index, pd.RangeIndex) or pd.api.types.is_integer_dtype(index):
        return int(index[0]) - 1
    return "Base"


def backtest_portfolio(
    returns_df: pd.DataFrame,
    optimal_weights: np.ndarray | Sequence[float],
    *,
    min_history: int = 30,
    rebalance_freq: RebalanceFreq = "monthly",
    transaction_cost_bps: float = 0.0,
    tradable_mask: pd.DataFrame | None = None,
    base_date: pd.Timestamp | None = None,
    include_base_point: bool = True,
    periods_per_year: float = 252.0,
    risk_free_rate: float = 0.016,
) -> pd.DataFrame:
    """最优权重组合与等权组合的累计净值回测（基期 1.0，完整保留首期及全样本收益）。

    支持初始建仓成本、按周期再平衡（``buy_and_hold`` / ``daily`` / ``monthly`` / ``quarterly``）
    以及双边交易费用（bps）。若提供 ``tradable_mask``，非交易日绝不强行调仓。
    """
    if rebalance_freq not in ("buy_and_hold", "daily", "monthly", "quarterly"):
        raise ValueError(f"Unsupported rebalance_freq: {rebalance_freq!r}")
    if (
        isinstance(transaction_cost_bps, bool)
        or not np.isfinite(float(transaction_cost_bps))
        or float(transaction_cost_bps) < 0.0
    ):
        raise ValueError("transaction_cost_bps must be finite and non-negative.")

    r = clean_joint_returns(returns_df, min_history=min_history)
    w = np.asarray(optimal_weights, dtype=float).ravel()
    if w.size != r.shape[1]:
        raise ValueError(
            f"optimal_weights length ({w.size}) must match number of assets ({r.shape[1]})."
        )
    if not np.isfinite(w).all() or not np.isfinite(w.sum()) or abs(w.sum()) < 1e-12:
        raise ValueError("optimal_weights must be finite with a non-zero sum.")
    # 权重归一化保证所有资产权重加总为 1，即“满仓”组合。
    w = w / w.sum()
    w_eq = np.full(r.shape[1], 1.0 / float(r.shape[1]), dtype=float)

    nav_opt, opt_turnover, opt_cost = _simulate_portfolio_nav(
        r,
        w,
        rebalance_freq=rebalance_freq,
        transaction_cost_bps=float(transaction_cost_bps),
        tradable_mask=tradable_mask,
    )
    nav_eq, eq_turnover, eq_cost = _simulate_portfolio_nav(
        r,
        w_eq,
        rebalance_freq=rebalance_freq,
        transaction_cost_bps=float(transaction_cost_bps),
        tradable_mask=tradable_mask,
    )

    if include_base_point:
        anchor = _infer_base_date(r.index, base_date)
        full_index = pd.Index([anchor]).append(r.index)
        if isinstance(r.index, pd.DatetimeIndex):
            full_index = pd.DatetimeIndex(full_index, name=r.index.name or "Date")
        out = pd.DataFrame(
            {
                "Optimal_Portfolio": np.concatenate(([1.0], nav_opt)),
                "Equal_Weight_Portfolio": np.concatenate(([1.0], nav_eq)),
            },
            index=full_index,
        )
    else:
        out = pd.DataFrame(
            {
                "Optimal_Portfolio": nav_opt,
                "Equal_Weight_Portfolio": nav_eq,
            },
            index=r.index,
        )

    opt_perf = compute_backtest_metrics(
        out["Optimal_Portfolio"],
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    eq_perf = compute_backtest_metrics(
        out["Equal_Weight_Portfolio"],
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    opt_perf["total_turnover"] = opt_turnover
    opt_perf["total_cost_ratio"] = opt_cost
    eq_perf["total_turnover"] = eq_turnover
    eq_perf["total_cost_ratio"] = eq_cost
    out.attrs["performance"] = {
        "Optimal_Portfolio": opt_perf,
        "Equal_Weight_Portfolio": eq_perf,
        "rebalance_freq": rebalance_freq,
        "transaction_cost_bps": float(transaction_cost_bps),
        "periods_per_year": float(periods_per_year),
    }
    return out


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
    num_portfolios = _positive_integer(num_portfolios, "num_portfolios")
    trading_days = _positive_integer(trading_days, "trading_days")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    if returns_df.shape[1] < 2:
        raise ValueError("Efficient frontier requires at least 2 assets (columns).")
    r = clean_joint_returns(returns_df, min_history=min_history)

    x = r.values.astype(float, copy=False)
    n_assets = x.shape[1]
    # 协方差矩阵描述资产之间“同涨同跌”的程度，是组合波动率的核心输入。
    cov = np.cov(x, rowvar=False)
    if not np.isfinite(cov).all():
        raise ValueError("Return covariance matrix contains non-finite values.")
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
    # Dirichlet 抽样天然生成非负且总和为 1 的随机权重。
    w = rng.dirichlet(np.ones(n_assets), size=num_portfolios).astype(float, copy=False)

    # Portfolio moments avoid allocating a history × portfolios matrix.
    mean_d = np.einsum("ij,j->i", w, np.mean(x, axis=0))
    variance = np.einsum("ij,jk,ik->i", w, cov, w)
    std_d = np.sqrt(np.maximum(variance, 0.0))
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
    "RebalanceFreq",
    "backtest_portfolio",
    "clean_joint_returns",
    "compute_backtest_metrics",
    "generate_efficient_frontier",
    "normalize_close_to_base_one",
    "returns_correlation_matrix",
    "run_monte_carlo_gbm",
]
