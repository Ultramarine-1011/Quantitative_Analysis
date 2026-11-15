"""量化模型：价格归一化、GBM 蒙特卡洛路径与有效前沿随机权重模拟。

本模块与主链路 OHLCV 契约对齐：收盘价列优先使用 ``Close``，并兼容 ``close``。
"""

from __future__ import annotations

import importlib
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd

from analysis_metrics import RISK_FREE_RATE
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
    risk_free_rate: float = RISK_FREE_RATE,
    return_convention: Literal["simple", "log"] = "simple",
) -> dict[str, float]:
    """Compute CAGR, total return, annualized volatility, max drawdown, Sharpe, Sortino, and Calmar."""
    conv = str(return_convention or "simple").strip().lower()
    if conv not in ("simple", "log"):
        raise ValueError(
            f"Unsupported return_convention: {return_convention!r}; choose 'simple' or 'log'."
        )
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
    if conv == "log":
        ratio = (s / s.shift(1)).dropna().to_numpy(dtype=float)
        period_r = np.log(ratio[ratio > 0.0]) if ratio.size else ratio
        rf_per_period = float(np.log1p(risk_free_rate)) / ppy
    else:
        period_r = simple_r
        rf_per_period = float(risk_free_rate) / ppy

    if period_r.size >= 2:
        std_r = float(np.std(period_r, ddof=1))
        ann_vol = std_r * float(np.sqrt(ppy))
        excess = period_r - rf_per_period
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
    risk_free_rate: float = RISK_FREE_RATE,
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
        "return_convention": "simple",
    }
    return out


CovMethod = Literal["sample", "ledoit_wolf", "diagonal"]
WindowMode = Literal["rolling", "expanding"]
OptimizerObjective = Literal["max_sharpe", "min_variance"]


@dataclass
class CovarianceEstimate:
    """预期年化收益与协方差估计结果及条件数诊断。"""

    asset_names: list[str]
    mean_returns: np.ndarray
    cov_matrix: np.ndarray
    cov_method: CovMethod
    shrinkage: float | None
    condition_number: float
    min_eigenvalue: float
    ridge_applied: float


@dataclass
class PortfolioSolution:
    """单次约束组合优化求解结果。"""

    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe: float
    converged: bool
    solver_message: str


@dataclass
class ExactEfficientFrontierResult:
    """SLSQP 精确约束有效前沿、关键组合与可选随机散点。"""

    asset_names: list[str]
    frontier_volatility: np.ndarray
    frontier_returns: np.ndarray
    frontier_sharpe: np.ndarray
    frontier_weights: np.ndarray
    frontier_converged: np.ndarray
    min_variance: PortfolioSolution
    max_sharpe: PortfolioSolution
    cov_method: CovMethod
    shrinkage: float | None
    condition_number: float
    risk_free_rate: float
    min_weight: float
    max_weight: float
    random_volatility: np.ndarray | None = None
    random_returns: np.ndarray | None = None
    random_sharpe: np.ndarray | None = None
    random_weights: np.ndarray | None = None


@dataclass
class TrainTestSplitResult:
    """样本内优化、样本外静态验证与基准对照结果。"""

    train_returns: pd.DataFrame
    test_returns: pd.DataFrame
    split_date: object
    optimal_weights: np.ndarray
    in_sample_solution: PortfolioSolution
    in_sample_backtest: pd.DataFrame
    out_of_sample_backtest: pd.DataFrame
    in_sample_performance: dict[str, object]
    out_of_sample_performance: dict[str, object]


@dataclass
class WalkForwardResult:
    """无未来函数泄露的滚动/扩展窗口样本外组合回测结果。"""

    nav_df: pd.DataFrame
    weight_history: pd.DataFrame
    rebalance_dates: pd.Index
    performance: dict[str, object]
    window_mode: WindowMode
    train_window: int
    rebalance_freq: RebalanceFreq
    cov_method: CovMethod
    objective: OptimizerObjective


def _ledoit_wolf_cov_daily(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf 收缩协方差（优先调用 scikit-learn，缺省退回解析公式）。"""
    try:
        sklearn_cov = importlib.import_module("sklearn.covariance")
        lw = sklearn_cov.LedoitWolf(assume_centered=False).fit(x)
        return np.asarray(lw.covariance_, dtype=float), float(lw.shrinkage_)
    except ImportError:
        pass

    n_obs, n_assets = x.shape
    x_c = x - np.mean(x, axis=0)
    emp_cov = (x_c.T @ x_c) / float(n_obs)
    mu_iso = float(np.trace(emp_cov) / float(n_assets))
    delta_mat = emp_cov.copy()
    delta_mat.flat[:: n_assets + 1] -= mu_iso
    delta = float(np.sum(delta_mat * delta_mat) / float(n_assets))
    x2 = x_c * x_c
    # 用 ||x_k||^4 均值等价计算 (X^2)^T (X^2) 总和，时间复杂度 O(n*p)。
    beta_raw = (float(np.mean(np.sum(x2, axis=1) ** 2)) - float(np.sum(emp_cov * emp_cov))) / (
        float(n_assets) * float(n_obs)
    )
    beta = max(0.0, min(beta_raw, delta))
    shrinkage = 0.0 if delta <= 1e-30 else float(np.clip(beta / delta, 0.0, 1.0))
    shrunk = (1.0 - shrinkage) * emp_cov + shrinkage * mu_iso * np.eye(n_assets, dtype=float)
    return shrunk, shrinkage


def _estimate_covariance_from_array(
    x: np.ndarray,
    asset_names: list[str],
    *,
    cov_method: CovMethod = "sample",
    trading_days: int = 252,
) -> CovarianceEstimate:
    if cov_method not in ("sample", "ledoit_wolf", "diagonal"):
        raise ValueError(f"Unsupported cov_method: {cov_method!r}")
    trading_days = _positive_integer(trading_days, "trading_days")
    _, n_assets = x.shape
    if n_assets < 2:
        raise ValueError("Portfolio optimization requires at least 2 assets (columns).")

    mean_ann = np.mean(x, axis=0) * float(trading_days)
    shrinkage: float | None = None
    ridge_applied = 0.0

    if cov_method == "sample":
        cov_daily = np.cov(x, rowvar=False, ddof=1)
        if not np.isfinite(cov_daily).all():
            raise ValueError("Return covariance matrix contains non-finite values.")
        if cov_daily.shape != (n_assets, n_assets):
            raise ValueError("Internal error: covariance shape mismatch.")
        eff_rank = np.linalg.matrix_rank(cov_daily, tol=1e-10)
        if eff_rank < n_assets or bool(np.any(np.diag(cov_daily) <= 1e-15)):
            raise ValueError(
                "Return covariance matrix is singular or rank-deficient; check for redundant or constant series."
            )
        cond_daily = float(np.linalg.cond(cov_daily))
        if not np.isfinite(cond_daily) or cond_daily > 1e12:
            raise ValueError(
                f"Return covariance matrix is ill-conditioned (cond={cond_daily:.2e}); cannot stabilize frontier simulation."
            )
    elif cov_method == "ledoit_wolf":
        cov_daily, shrinkage = _ledoit_wolf_cov_daily(x)
        if not np.isfinite(cov_daily).all() or float(np.trace(cov_daily)) <= 1e-18:
            raise ValueError(
                "All assets have zero return variance; cannot estimate covariance matrix."
            )
        cov_daily = 0.5 * (cov_daily + cov_daily.T)
        eigs_d = np.linalg.eigvalsh(cov_daily)
        mean_var = float(np.mean(np.diag(cov_daily)))
        if float(eigs_d[0]) <= 1e-12 * max(mean_var, 1e-18):
            ridge_applied = max(1e-10 * mean_var, float(-eigs_d[0]) + 1e-12 * mean_var)
            cov_daily = cov_daily + ridge_applied * np.eye(n_assets, dtype=float)
    else:
        raw_cov = np.cov(x, rowvar=False, ddof=1)
        diag_var = np.diag(raw_cov).astype(float, copy=True)
        if not np.isfinite(diag_var).all() or float(np.max(diag_var)) <= 1e-18:
            raise ValueError(
                "All assets have zero return variance; cannot estimate covariance matrix."
            )
        pos = diag_var[diag_var > 1e-15]
        floor = 1e-8 * float(np.mean(pos))
        bad = diag_var <= 1e-15
        if bool(np.any(bad)):
            ridge_applied = floor
            diag_var = np.where(bad, floor, diag_var)
        cov_daily = np.diag(diag_var)

    cov_ann = 0.5 * (cov_daily + cov_daily.T) * float(trading_days)
    eigs_ann = np.linalg.eigvalsh(cov_ann)
    min_eig = float(eigs_ann[0])
    cond = float(eigs_ann[-1] / min_eig) if min_eig > 0.0 else float("inf")

    return CovarianceEstimate(
        asset_names=list(asset_names),
        mean_returns=mean_ann.astype(float, copy=False),
        cov_matrix=cov_ann.astype(float, copy=False),
        cov_method=cov_method,
        shrinkage=shrinkage,
        condition_number=cond,
        min_eigenvalue=min_eig,
        ridge_applied=ridge_applied * float(trading_days),
    )


def estimate_covariance_and_mean(
    returns_df: pd.DataFrame,
    *,
    cov_method: CovMethod = "sample",
    trading_days: int = 252,
    min_history: int = 30,
) -> CovarianceEstimate:
    """估计年化收益向量与协方差矩阵，支持 sample、Ledoit-Wolf 收缩与对角估计。"""
    r = clean_joint_returns(returns_df, min_history=min_history)
    x = r.to_numpy(dtype=float, copy=False)
    return _estimate_covariance_from_array(
        x,
        [str(c) for c in r.columns],
        cov_method=cov_method,
        trading_days=trading_days,
    )


def _validate_weight_bounds(
    n_assets: int, min_weight: float, max_weight: float
) -> tuple[float, float]:
    for val, name in ((min_weight, "min_weight"), (max_weight, "max_weight")):
        if isinstance(val, bool) or not isinstance(val, Real) or not np.isfinite(float(val)):
            raise ValueError(f"{name} must be a finite real number.")
    w_min = float(min_weight)
    w_max = float(max_weight)
    if w_min < 0.0 or w_max > 1.0 or w_min > w_max:
        raise ValueError("Long-only weight bounds require 0.0 <= min_weight <= max_weight <= 1.0.")
    if w_min * float(n_assets) > 1.0 + 1e-9:
        raise ValueError(
            f"Infeasible weight bounds: min_weight ({w_min:.4g}) * {n_assets} assets > 1.0."
        )
    if w_max * float(n_assets) < 1.0 - 1e-9:
        raise ValueError(
            f"Infeasible weight bounds: max_weight ({w_max:.4g}) * {n_assets} assets < 1.0."
        )
    return w_min, w_max


def _project_capped_simplex(
    vector: np.ndarray,
    lower: float,
    upper: float,
    total: float = 1.0,
) -> np.ndarray:
    """欧氏投影到带上下界单纯形 {w : sum(w) == total, lower <= w_i <= upper}。"""
    v = np.asarray(vector, dtype=float)
    n = v.size
    if abs(upper - lower) <= 1e-12:
        return np.full(n, total / float(n), dtype=float)
    lo = float(np.min(v) - upper - 1.0)
    hi = float(np.max(v) - lower + 1.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = float(np.sum(np.clip(v - mid, lower, upper)))
        if s > total:
            lo = mid
        else:
            hi = mid
    w = np.clip(v - 0.5 * (lo + hi), lower, upper)
    diff = total - float(np.sum(w))
    if abs(diff) > 0.0:
        room = (upper - w) if diff > 0.0 else (w - lower)
        room_sum = float(np.sum(room))
        if room_sum > 1e-15:
            w = np.clip(w + diff * (room / room_sum), lower, upper)
    return w.astype(float, copy=False)


def _greedy_extreme_return_weights(
    mean_ret: np.ndarray,
    lower: float,
    upper: float,
    *,
    maximize: bool,
) -> np.ndarray:
    n = mean_ret.size
    w = np.full(n, lower, dtype=float)
    rem = 1.0 - float(n) * lower
    order = np.argsort(-mean_ret if maximize else mean_ret, kind="mergesort")
    span = upper - lower
    for idx in order:
        if rem <= 1e-15:
            break
        add = min(span, rem)
        w[idx] += add
        rem -= add
    return _project_capped_simplex(w, lower, upper)


def _portfolio_stats(
    weights: np.ndarray,
    mean_ret: np.ndarray,
    cov_mat: np.ndarray,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    ret = float(mean_ret @ weights)
    var = max(float(weights @ cov_mat @ weights), 0.0)
    vol = math.sqrt(var)
    sharpe = (ret - risk_free_rate) / vol if vol > 0.0 else float("nan")
    return ret, vol, sharpe


def _is_feasible_solution(
    weights: np.ndarray,
    lower: float,
    upper: float,
    mean_ret: np.ndarray | None = None,
    target_return: float | None = None,
    tol: float = 1e-5,
) -> bool:
    if not np.isfinite(weights).all():
        return False
    if abs(float(np.sum(weights)) - 1.0) > tol:
        return False
    if float(np.min(weights)) < lower - tol or float(np.max(weights)) > upper + tol:
        return False
    if target_return is not None and mean_ret is not None:
        if float(mean_ret @ weights) < float(target_return) - tol:
            return False
    return True


def _deduplicate_starts(
    candidates: list[np.ndarray], lower: float, upper: float
) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for cand in candidates:
        proj = _project_capped_simplex(cand, lower, upper)
        if not any(np.allclose(proj, prev, atol=1e-6) for prev in unique):
            unique.append(proj)
    return unique


def _slsqp_minimize(scipy_optimize: object, *args: object, **kwargs: object) -> object:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Values in x were outside bounds during a minimize step.*",
            category=RuntimeWarning,
        )
        return scipy_optimize.minimize(*args, **kwargs)  # type: ignore[attr-defined]


def _solve_min_variance_core(
    mean_ret: np.ndarray,
    cov_mat: np.ndarray,
    risk_free_rate: float,
    lower: float,
    upper: float,
    *,
    target_return: float | None = None,
    x0: np.ndarray | None = None,
) -> PortfolioSolution:
    scipy_optimize = importlib.import_module("scipy.optimize")
    n = mean_ret.size
    if abs(upper - lower) <= 1e-12:
        w = np.full(n, 1.0 / float(n), dtype=float)
        if target_return is not None and float(mean_ret @ w) < float(target_return) - 1e-8:
            raise ValueError("Target return exceeds feasible portfolio bound under fixed weights.")
        ret, vol, sharpe = _portfolio_stats(w, mean_ret, cov_mat, risk_free_rate)
        return PortfolioSolution(w, ret, vol, sharpe, True, "Degenerate singleton box bounds.")

    scale = max(float(np.mean(np.diag(cov_mat))), 1e-8)
    cov_scaled = cov_mat / scale
    bounds = [(lower, upper)] * n
    constraints: list[dict[str, object]] = [
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
            "jac": lambda w: np.ones(n, dtype=float),
        }
    ]
    if target_return is not None:
        tgt = float(target_return)
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w, t=tgt: float(mean_ret @ w - t),
                "jac": lambda w: mean_ret,
            }
        )

    w_eq = np.full(n, 1.0 / float(n), dtype=float)
    inv_std = 1.0 / np.sqrt(np.maximum(np.diag(cov_mat), 1e-12))
    w_inv = inv_std / float(np.sum(inv_std))
    w_max_ret = _greedy_extreme_return_weights(mean_ret, lower, upper, maximize=True)
    raw_starts = [s for s in (x0, w_eq, w_inv, w_max_ret) if s is not None]
    starts = _deduplicate_starts(raw_starts, lower, upper)

    best_w: np.ndarray | None = None
    best_var = float("inf")
    last_msg = "No optimization run executed."

    for start in starts:
        res = _slsqp_minimize(
            scipy_optimize,
            lambda w: 0.5 * float(w @ cov_scaled @ w),
            start,
            jac=lambda w: cov_scaled @ w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 500, "disp": False},
        )
        last_msg = str(res.message)
        if res.success and _is_feasible_solution(
            res.x, lower, upper, mean_ret=mean_ret, target_return=target_return
        ):
            cand = _project_capped_simplex(res.x, lower, upper)
            if target_return is not None and float(mean_ret @ cand) < float(target_return) - 1e-5:
                cand = np.asarray(res.x, dtype=float)
            var = float(cand @ cov_mat @ cand)
            if var < best_var:
                best_var = var
                best_w = cand
            if x0 is not None and start is starts[0]:
                # 凸二次规划从上一前沿解热启动收敛后即为全局极小。
                break

    if best_w is None and target_return is not None:
        max_ret_val = float(mean_ret @ w_max_ret)
        if abs(float(target_return) - max_ret_val) <= 1e-6:
            ret, vol, sharpe = _portfolio_stats(w_max_ret, mean_ret, cov_mat, risk_free_rate)
            return PortfolioSolution(
                weights=w_max_ret,
                expected_return=ret,
                volatility=vol,
                sharpe=sharpe,
                converged=True,
                solver_message="Boundary vertex reached at maximum feasible return.",
            )

    if best_w is None:
        raise RuntimeError(f"Minimum variance optimization failed to converge: {last_msg}")

    ret, vol, sharpe = _portfolio_stats(best_w, mean_ret, cov_mat, risk_free_rate)
    return PortfolioSolution(
        weights=best_w,
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        converged=True,
        solver_message=last_msg,
    )


def _solve_tangency_core(
    mean_ret: np.ndarray,
    cov_mat: np.ndarray,
    risk_free_rate: float,
    lower: float,
    upper: float,
    *,
    gmv_weights: np.ndarray | None = None,
    extra_starts: Sequence[np.ndarray] = (),
) -> PortfolioSolution:
    scipy_optimize = importlib.import_module("scipy.optimize")
    n = mean_ret.size
    if abs(upper - lower) <= 1e-12:
        w = np.full(n, 1.0 / float(n), dtype=float)
        ret, vol, sharpe = _portfolio_stats(w, mean_ret, cov_mat, risk_free_rate)
        return PortfolioSolution(w, ret, vol, sharpe, True, "Degenerate singleton box bounds.")

    cov_scale = max(float(np.mean(np.diag(cov_mat))), 1e-12)
    cov_scaled = cov_mat / cov_scale
    bounds = [(lower, upper)] * n
    constraints = [
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
            "jac": lambda w: np.ones(n, dtype=float),
        }
    ]

    w_eq = np.full(n, 1.0 / float(n), dtype=float)
    w_max_ret = _greedy_extreme_return_weights(mean_ret, lower, upper, maximize=True)
    inv_std = 1.0 / np.sqrt(np.maximum(np.diag(cov_mat), 1e-12))
    w_inv = inv_std / float(np.sum(inv_std))
    single_sharpes = (mean_ret - risk_free_rate) / np.sqrt(np.maximum(np.diag(cov_mat), 1e-12))
    best_single = np.zeros(n, dtype=float)
    best_single[int(np.argmax(single_sharpes))] = 1.0
    best_ret_idx = np.zeros(n, dtype=float)
    best_ret_idx[int(np.argmax(mean_ret))] = 1.0

    raw_starts: list[np.ndarray] = []
    for cand in (gmv_weights, w_eq, w_max_ret, w_inv, best_single, best_ret_idx, *extra_starts):
        if cand is not None:
            raw_starts.append(np.asarray(cand, dtype=float))
    starts = _deduplicate_starts(raw_starts, lower, upper)

    def neg_scaled_sharpe_and_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        w_arr = np.asarray(w, dtype=float)
        excess = float(mean_ret @ w_arr) - risk_free_rate
        cov_w = cov_scaled @ w_arr
        var = max(float(w_arr @ cov_w), 1e-24)
        std = math.sqrt(var)
        obj = -excess / std
        grad = (-mean_ret * var + excess * cov_w) / (var * std)
        return obj, grad

    best_sol: PortfolioSolution | None = None
    last_msg = "No optimization run executed."

    for start in starts:
        res = _slsqp_minimize(
            scipy_optimize,
            lambda w: neg_scaled_sharpe_and_grad(w)[0],
            start,
            jac=lambda w: neg_scaled_sharpe_and_grad(w)[1],
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 600, "disp": False},
        )
        last_msg = str(res.message)
        if not res.success or not _is_feasible_solution(res.x, lower, upper):
            continue
        w_cand = _project_capped_simplex(res.x, lower, upper)
        ret, vol, sharpe = _portfolio_stats(w_cand, mean_ret, cov_mat, risk_free_rate)
        if not np.isfinite(sharpe):
            continue
        if (
            best_sol is None
            or sharpe > best_sol.sharpe + 1e-12
            or (abs(sharpe - best_sol.sharpe) <= 1e-12 and vol < best_sol.volatility)
        ):
            best_sol = PortfolioSolution(
                weights=w_cand,
                expected_return=ret,
                volatility=vol,
                sharpe=sharpe,
                converged=True,
                solver_message=last_msg,
            )

    if best_sol is None:
        raise RuntimeError(f"Tangency portfolio optimization failed to converge: {last_msg}")
    return best_sol


def solve_minimum_variance_portfolio(
    returns_df: pd.DataFrame,
    *,
    cov_method: CovMethod = "sample",
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = 252,
    min_history: int = 30,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    target_return: float | None = None,
) -> PortfolioSolution:
    """通过 SLSQP 求解约束最小方差组合（可选给定目标年化收益下限）。"""
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    if target_return is not None and not np.isfinite(float(target_return)):
        raise ValueError("target_return must be finite when provided.")
    est = estimate_covariance_and_mean(
        returns_df,
        cov_method=cov_method,
        trading_days=trading_days,
        min_history=min_history,
    )
    w_min, w_max = _validate_weight_bounds(len(est.asset_names), min_weight, max_weight)
    return _solve_min_variance_core(
        est.mean_returns,
        est.cov_matrix,
        float(risk_free_rate),
        w_min,
        w_max,
        target_return=target_return,
    )


def solve_tangency_portfolio(
    returns_df: pd.DataFrame,
    *,
    cov_method: CovMethod = "sample",
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = 252,
    min_history: int = 30,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> PortfolioSolution:
    """通过多起点 SLSQP 求解约束最大夏普（切线）组合。"""
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    est = estimate_covariance_and_mean(
        returns_df,
        cov_method=cov_method,
        trading_days=trading_days,
        min_history=min_history,
    )
    w_min, w_max = _validate_weight_bounds(len(est.asset_names), min_weight, max_weight)
    gmv = _solve_min_variance_core(
        est.mean_returns,
        est.cov_matrix,
        float(risk_free_rate),
        w_min,
        w_max,
    )
    return _solve_tangency_core(
        est.mean_returns,
        est.cov_matrix,
        float(risk_free_rate),
        w_min,
        w_max,
        gmv_weights=gmv.weights,
    )


def _sample_constrained_dirichlet(
    rng: np.random.Generator,
    num_portfolios: int,
    n_assets: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    if lower == 0.0 and upper == 1.0:
        return rng.dirichlet(np.ones(n_assets), size=num_portfolios).astype(float, copy=False)
    accepted: list[np.ndarray] = []
    batch = max(num_portfolios * 4, 256)
    for _ in range(8):
        draws = rng.dirichlet(np.ones(n_assets), size=batch)
        mask = np.all((draws >= lower - 1e-12) & (draws <= upper + 1e-12), axis=1)
        if np.any(mask):
            accepted.append(draws[mask])
            if sum(len(chunk) for chunk in accepted) >= num_portfolios:
                break
    if accepted:
        merged = np.vstack(accepted)
        if len(merged) >= num_portfolios:
            return merged[:num_portfolios].astype(float, copy=False)
    raw = rng.dirichlet(np.ones(n_assets), size=num_portfolios)
    projected = np.vstack([_project_capped_simplex(row, lower, upper) for row in raw])
    return projected.astype(float, copy=False)


def solve_exact_efficient_frontier(
    returns_df: pd.DataFrame,
    *,
    n_points: int = 30,
    cov_method: CovMethod = "ledoit_wolf",
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = 252,
    min_history: int = 30,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    num_random_portfolios: int = 0,
    random_state: int | np.random.Generator | None = None,
) -> ExactEfficientFrontierResult:
    """使用 SLSQP 求解真实均值-方差有效前沿包络线、GMV 与最大夏普组合。"""
    n_points = _positive_integer(n_points, "n_points", minimum=2)
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    est = estimate_covariance_and_mean(
        returns_df,
        cov_method=cov_method,
        trading_days=trading_days,
        min_history=min_history,
    )
    n_assets = len(est.asset_names)
    w_min, w_max = _validate_weight_bounds(n_assets, min_weight, max_weight)
    rf = float(risk_free_rate)

    gmv = _solve_min_variance_core(est.mean_returns, est.cov_matrix, rf, w_min, w_max)
    w_max_ret = _greedy_extreme_return_weights(est.mean_returns, w_min, w_max, maximize=True)
    max_ret = float(est.mean_returns @ w_max_ret)

    frontier_vols: list[float] = []
    frontier_rets: list[float] = []
    frontier_sharpes: list[float] = []
    frontier_weights: list[np.ndarray] = []
    frontier_conv: list[bool] = []

    if max_ret - gmv.expected_return <= 1e-10:
        frontier_vols.append(gmv.volatility)
        frontier_rets.append(gmv.expected_return)
        frontier_sharpes.append(gmv.sharpe)
        frontier_weights.append(gmv.weights)
        frontier_conv.append(gmv.converged)
    else:
        targets = np.linspace(gmv.expected_return, max_ret, n_points)
        prev_w = gmv.weights
        for idx, tgt in enumerate(targets):
            if idx == 0:
                sol = gmv
            else:
                try:
                    sol = _solve_min_variance_core(
                        est.mean_returns,
                        est.cov_matrix,
                        rf,
                        w_min,
                        w_max,
                        target_return=float(tgt),
                        x0=prev_w,
                    )
                except RuntimeError:
                    frontier_conv.append(False)
                    continue
            prev_w = sol.weights
            frontier_vols.append(sol.volatility)
            frontier_rets.append(sol.expected_return)
            frontier_sharpes.append(sol.sharpe)
            frontier_weights.append(sol.weights)
            frontier_conv.append(sol.converged)

    if not frontier_weights:
        raise RuntimeError(
            "Exact efficient frontier failed to converge at all target return points."
        )

    rand_vol: np.ndarray | None = None
    rand_ret: np.ndarray | None = None
    rand_sharpe: np.ndarray | None = None
    rand_weights: np.ndarray | None = None
    extra_seeds: list[np.ndarray] = list(frontier_weights)
    if num_random_portfolios > 0:
        num_rand = _positive_integer(num_random_portfolios, "num_random_portfolios")
        rng = np.random.default_rng(random_state)
        rw = _sample_constrained_dirichlet(rng, num_rand, n_assets, w_min, w_max)
        r_ret = np.einsum("ij,j->i", rw, est.mean_returns)
        r_var = np.maximum(np.einsum("ij,jk,ik->i", rw, est.cov_matrix, rw), 0.0)
        r_vol = np.sqrt(r_var)
        r_sharpe = np.where(r_vol > 0.0, (r_ret - rf) / r_vol, np.nan)
        rand_vol = r_vol.astype(float, copy=False)
        rand_ret = r_ret.astype(float, copy=False)
        rand_sharpe = r_sharpe.astype(float, copy=False)
        rand_weights = rw
        if np.any(np.isfinite(r_sharpe)):
            extra_seeds.append(rw[int(np.nanargmax(r_sharpe))])

    max_sharpe = _solve_tangency_core(
        est.mean_returns,
        est.cov_matrix,
        rf,
        w_min,
        w_max,
        gmv_weights=gmv.weights,
        extra_starts=extra_seeds,
    )

    return ExactEfficientFrontierResult(
        asset_names=est.asset_names,
        frontier_volatility=np.asarray(frontier_vols, dtype=float),
        frontier_returns=np.asarray(frontier_rets, dtype=float),
        frontier_sharpe=np.asarray(frontier_sharpes, dtype=float),
        frontier_weights=np.vstack(frontier_weights),
        frontier_converged=np.asarray(frontier_conv, dtype=bool),
        min_variance=gmv,
        max_sharpe=max_sharpe,
        cov_method=cov_method,
        shrinkage=est.shrinkage,
        condition_number=est.condition_number,
        risk_free_rate=rf,
        min_weight=w_min,
        max_weight=w_max,
        random_volatility=rand_vol,
        random_returns=rand_ret,
        random_sharpe=rand_sharpe,
        random_weights=rand_weights,
    )


def _optimize_weights_for_window(
    window_df: pd.DataFrame,
    *,
    objective: OptimizerObjective,
    cov_method: CovMethod,
    risk_free_rate: float,
    min_weight: float,
    max_weight: float,
    trading_days: int,
) -> PortfolioSolution:
    if objective not in ("max_sharpe", "min_variance"):
        raise ValueError(f"Unsupported objective: {objective!r}")
    est = _estimate_covariance_from_array(
        window_df.to_numpy(dtype=float, copy=False),
        [str(c) for c in window_df.columns],
        cov_method=cov_method,
        trading_days=trading_days,
    )
    w_min, w_max = _validate_weight_bounds(len(est.asset_names), min_weight, max_weight)
    gmv = _solve_min_variance_core(
        est.mean_returns,
        est.cov_matrix,
        risk_free_rate,
        w_min,
        w_max,
    )
    if objective == "min_variance":
        return gmv
    return _solve_tangency_core(
        est.mean_returns,
        est.cov_matrix,
        risk_free_rate,
        w_min,
        w_max,
        gmv_weights=gmv.weights,
    )


def split_train_test_returns(
    returns_df: pd.DataFrame,
    *,
    train_ratio: float = 0.7,
    split_date: pd.Timestamp | str | None = None,
    min_train_rows: int = 30,
    min_test_rows: int = 10,
    objective: OptimizerObjective = "max_sharpe",
    cov_method: CovMethod = "ledoit_wolf",
    risk_free_rate: float = RISK_FREE_RATE,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    rebalance_freq: RebalanceFreq = "monthly",
    transaction_cost_bps: float = 0.0,
    tradable_mask: pd.DataFrame | None = None,
    periods_per_year: float = 252.0,
) -> TrainTestSplitResult:
    """严格拆分训练集与测试集：仅用样本内收益求解最优权重，再独立做样本外静态回测。"""
    min_train_rows = _positive_integer(min_train_rows, "min_train_rows", minimum=5)
    min_test_rows = _positive_integer(min_test_rows, "min_test_rows", minimum=2)
    r = clean_joint_returns(returns_df, min_history=min_train_rows + min_test_rows)

    if split_date is not None:
        ts = pd.Timestamp(split_date)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        if not isinstance(r.index, pd.DatetimeIndex):
            raise ValueError("split_date requires returns_df to have a DatetimeIndex.")
        train_returns = r.loc[r.index < ts]
        test_returns = r.loc[r.index >= ts]
        resolved_split: object = ts
    else:
        if (
            isinstance(train_ratio, bool)
            or not isinstance(train_ratio, Real)
            or not np.isfinite(float(train_ratio))
            or float(train_ratio) <= 0.0
            or float(train_ratio) >= 1.0
        ):
            raise ValueError("train_ratio must be strictly between 0 and 1.")
        split_idx = int(math.floor(float(len(r)) * float(train_ratio)))
        train_returns = r.iloc[:split_idx]
        test_returns = r.iloc[split_idx:]
        resolved_split = r.index[split_idx] if split_idx < len(r) else r.index[-1]

    if len(train_returns) < min_train_rows:
        raise ValueError(
            f"Training split has {len(train_returns)} rows; requires at least {min_train_rows}."
        )
    if len(test_returns) < min_test_rows:
        raise ValueError(
            f"Test split has {len(test_returns)} rows; requires at least {min_test_rows}."
        )

    ann_days = max(int(round(float(periods_per_year))), 1)
    # 权重仅由 train_returns 拟合得出，测试集绝不参与协方差或均值估计。
    in_sample_sol = _optimize_weights_for_window(
        train_returns,
        objective=objective,
        cov_method=cov_method,
        risk_free_rate=float(risk_free_rate),
        min_weight=min_weight,
        max_weight=max_weight,
        trading_days=ann_days,
    )

    in_sample_bt = backtest_portfolio(
        train_returns,
        in_sample_sol.weights,
        min_history=min_train_rows,
        rebalance_freq=rebalance_freq,
        transaction_cost_bps=transaction_cost_bps,
        tradable_mask=tradable_mask,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    out_of_sample_bt = backtest_portfolio(
        test_returns,
        in_sample_sol.weights,
        min_history=min_test_rows,
        rebalance_freq=rebalance_freq,
        transaction_cost_bps=transaction_cost_bps,
        tradable_mask=tradable_mask,
        base_date=train_returns.index[-1] if isinstance(r.index, pd.DatetimeIndex) else None,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )

    return TrainTestSplitResult(
        train_returns=train_returns,
        test_returns=test_returns,
        split_date=resolved_split,
        optimal_weights=in_sample_sol.weights,
        in_sample_solution=in_sample_sol,
        in_sample_backtest=in_sample_bt,
        out_of_sample_backtest=out_of_sample_bt,
        in_sample_performance=dict(in_sample_bt.attrs.get("performance", {})),
        out_of_sample_performance=dict(out_of_sample_bt.attrs.get("performance", {})),
    )


def run_walk_forward_backtest(
    returns_df: pd.DataFrame,
    *,
    window_mode: WindowMode = "rolling",
    train_window: int = 60,
    rebalance_freq: RebalanceFreq = "monthly",
    cov_method: CovMethod = "ledoit_wolf",
    objective: OptimizerObjective = "max_sharpe",
    risk_free_rate: float = RISK_FREE_RATE,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    transaction_cost_bps: float = 10.0,
    periods_per_year: float = 252.0,
    tradable_mask: pd.DataFrame | None = None,
    min_test_rows: int = 5,
    include_base_point: bool = True,
) -> WalkForwardResult:
    """无未来信息泄露的滚动/扩展窗口样本外回测，并附等权与全样本后视参考线。"""
    if window_mode not in ("rolling", "expanding"):
        raise ValueError(f"Unsupported window_mode: {window_mode!r}")
    if rebalance_freq not in ("buy_and_hold", "daily", "monthly", "quarterly"):
        raise ValueError(f"Unsupported rebalance_freq: {rebalance_freq!r}")
    if (
        isinstance(transaction_cost_bps, bool)
        or not np.isfinite(float(transaction_cost_bps))
        or float(transaction_cost_bps) < 0.0
    ):
        raise ValueError("transaction_cost_bps must be finite and non-negative.")

    train_window = _positive_integer(train_window, "train_window", minimum=5)
    min_test_rows = _positive_integer(min_test_rows, "min_test_rows", minimum=2)
    r = clean_joint_returns(returns_df, min_history=train_window + min_test_rows)
    n_total, n_assets = r.shape
    ann_days = max(int(round(float(periods_per_year))), 1)
    cost_rate = float(transaction_cost_bps) / 10000.0

    t_start = train_window
    oos_len = n_total - t_start
    oos_returns = r.iloc[t_start:]
    x_oos = oos_returns.to_numpy(dtype=float, copy=False)

    mask_arr = (
        tradable_mask.reindex(r.index, columns=r.columns).fillna(True).to_numpy(dtype=bool)
        if tradable_mask is not None
        else None
    )

    # 首个 OOS 交易日 t_start 的建仓权重仅能看见 r.iloc[:t_start]，绝不包含 t_start 当日。
    init_start = 0 if window_mode == "expanding" else t_start - train_window
    init_sol = _optimize_weights_for_window(
        r.iloc[init_start:t_start],
        objective=objective,
        cov_method=cov_method,
        risk_free_rate=float(risk_free_rate),
        min_weight=min_weight,
        max_weight=max_weight,
        trading_days=ann_days,
    )
    current_target = init_sol.weights
    init_turnover = float(np.sum(np.abs(current_target)))
    init_cost = cost_rate * init_turnover
    if init_cost >= 1.0:
        raise ValueError("Initial transaction cost exceeds 100% of starting capital.")

    sub_values = (1.0 - init_cost) * current_target.copy()
    total_turnover = init_turnover
    total_cost_ratio = init_cost

    weight_rows: list[np.ndarray] = [current_target.copy()]
    rebalance_dates: list[object] = [r.index[t_start]]
    nav_oos = np.empty(oos_len, dtype=float)
    pending_rebalance = False

    for step_idx in range(oos_len):
        global_t = t_start + step_idx
        if step_idx > 0 and _is_rebalance_step(r.index, global_t, rebalance_freq):
            pending_rebalance = True
        if pending_rebalance:
            tradable_row = (
                mask_arr[global_t] if mask_arr is not None else np.ones(n_assets, dtype=bool)
            )
            if tradable_row.all():
                prev_equity = float(np.sum(sub_values))
                if prev_equity <= 0.0:
                    raise ValueError("Portfolio equity fell to zero or below during backtest.")
                win_start = 0 if window_mode == "expanding" else global_t - train_window
                sol = _optimize_weights_for_window(
                    r.iloc[win_start:global_t],
                    objective=objective,
                    cov_method=cov_method,
                    risk_free_rate=float(risk_free_rate),
                    min_weight=min_weight,
                    max_weight=max_weight,
                    trading_days=ann_days,
                )
                current_target = sol.weights
                desired = prev_equity * current_target
                step_turnover = float(np.sum(np.abs(desired - sub_values))) / prev_equity
                step_cost = prev_equity * cost_rate * step_turnover
                after_fee = prev_equity - step_cost
                if after_fee <= 0.0:
                    raise ValueError(
                        "Portfolio equity fell to zero or below after transaction costs."
                    )
                sub_values = after_fee * current_target
                total_turnover += step_turnover
                total_cost_ratio += step_cost / prev_equity
                weight_rows.append(current_target.copy())
                rebalance_dates.append(r.index[global_t])
                pending_rebalance = False

        sub_values = sub_values * (1.0 + x_oos[step_idx])
        equity = float(np.sum(sub_values))
        if not np.isfinite(equity) or equity <= 0.0:
            raise ValueError("Portfolio equity fell to zero or below during backtest.")
        nav_oos[step_idx] = equity

    w_eq = np.full(n_assets, 1.0 / float(n_assets), dtype=float)
    nav_eq, eq_turnover, eq_cost = _simulate_portfolio_nav(
        oos_returns,
        w_eq,
        rebalance_freq=rebalance_freq,
        transaction_cost_bps=float(transaction_cost_bps),
        tradable_mask=tradable_mask,
    )

    in_sample_sol = _optimize_weights_for_window(
        r,
        objective=objective,
        cov_method=cov_method,
        risk_free_rate=float(risk_free_rate),
        min_weight=min_weight,
        max_weight=max_weight,
        trading_days=ann_days,
    )
    nav_is, is_turnover, is_cost = _simulate_portfolio_nav(
        oos_returns,
        in_sample_sol.weights,
        rebalance_freq=rebalance_freq,
        transaction_cost_bps=float(transaction_cost_bps),
        tradable_mask=tradable_mask,
    )

    if include_base_point:
        anchor = r.index[t_start - 1]
        full_index = pd.Index([anchor]).append(oos_returns.index)
        if isinstance(oos_returns.index, pd.DatetimeIndex):
            full_index = pd.DatetimeIndex(full_index, name=oos_returns.index.name or "Date")
        nav_df = pd.DataFrame(
            {
                "OOS_Portfolio": np.concatenate(([1.0], nav_oos)),
                "Equal_Weight_Portfolio": np.concatenate(([1.0], nav_eq)),
                "In_Sample_Portfolio": np.concatenate(([1.0], nav_is)),
            },
            index=full_index,
        )
    else:
        nav_df = pd.DataFrame(
            {
                "OOS_Portfolio": nav_oos,
                "Equal_Weight_Portfolio": nav_eq,
                "In_Sample_Portfolio": nav_is,
            },
            index=oos_returns.index,
        )

    reb_idx = pd.Index(rebalance_dates, name=r.index.name or "Date")
    if isinstance(r.index, pd.DatetimeIndex):
        reb_idx = pd.DatetimeIndex(reb_idx, name=r.index.name or "Date")
    weight_history = pd.DataFrame(np.vstack(weight_rows), index=reb_idx, columns=r.columns)

    oos_perf = compute_backtest_metrics(
        nav_df["OOS_Portfolio"],
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    oos_perf["total_turnover"] = total_turnover
    oos_perf["total_cost_ratio"] = total_cost_ratio

    eq_perf = compute_backtest_metrics(
        nav_df["Equal_Weight_Portfolio"],
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    eq_perf["total_turnover"] = eq_turnover
    eq_perf["total_cost_ratio"] = eq_cost

    is_perf = compute_backtest_metrics(
        nav_df["In_Sample_Portfolio"],
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )
    is_perf["total_turnover"] = is_turnover
    is_perf["total_cost_ratio"] = is_cost
    is_perf["hindsight_only"] = 1.0

    perf: dict[str, object] = {
        "OOS_Portfolio": oos_perf,
        "Equal_Weight_Portfolio": eq_perf,
        "In_Sample_Portfolio": is_perf,
        "rebalance_count": len(rebalance_dates),
        "window_mode": window_mode,
        "train_window": train_window,
        "rebalance_freq": rebalance_freq,
        "cov_method": cov_method,
        "objective": objective,
        "transaction_cost_bps": float(transaction_cost_bps),
        "periods_per_year": float(periods_per_year),
        "return_convention": "simple",
        "in_sample_weights": in_sample_sol.weights.tolist(),
    }
    nav_df.attrs["performance"] = perf
    return WalkForwardResult(
        nav_df=nav_df,
        weight_history=weight_history,
        rebalance_dates=reb_idx,
        performance=perf,
        window_mode=window_mode,
        train_window=train_window,
        rebalance_freq=rebalance_freq,
        cov_method=cov_method,
        objective=objective,
    )


def generate_efficient_frontier(
    returns_df: pd.DataFrame,
    num_portfolios: int = 5000,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = 252,
    random_state: int | np.random.Generator | None = None,
    min_history: int = 30,
    cov_method: CovMethod = "sample",
) -> EfficientFrontierResult:
    """在简单收益率宽表上随机生成权重，计算年化收益、波动与夏普比率。

    权重独立服从 Dirichlet(1,...,1)，保证非负且和为 1；主要用于配合精确包络线做散点云可视化。
    年化约定：组合日均乘以 ``trading_days``、日标准差乘以 ``sqrt(trading_days)``；
    夏普为 ``(年化收益 - risk_free_rate) / 年化波动``（统一默认 ``RISK_FREE_RATE = 0.016``）。
    """
    num_portfolios = _positive_integer(num_portfolios, "num_portfolios")
    trading_days = _positive_integer(trading_days, "trading_days")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    if returns_df.shape[1] < 2:
        raise ValueError("Efficient frontier requires at least 2 assets (columns).")

    est = estimate_covariance_and_mean(
        returns_df,
        cov_method=cov_method,
        trading_days=trading_days,
        min_history=min_history,
    )
    n_assets = len(est.asset_names)

    rng = np.random.default_rng(random_state)
    # Dirichlet 抽样天然生成非负且总和为 1 的随机权重。
    w = rng.dirichlet(np.ones(n_assets), size=num_portfolios).astype(float, copy=False)

    ann_ret = np.einsum("ij,j->i", w, est.mean_returns)
    variance = np.einsum("ij,jk,ik->i", w, est.cov_matrix, w)
    ann_vol = np.sqrt(np.maximum(variance, 0.0))
    sharpe = np.where(ann_vol > 0, (ann_ret - float(risk_free_rate)) / ann_vol, np.nan)

    best_sharpe_idx = int(np.nanargmax(sharpe))

    return EfficientFrontierResult(
        volatility=ann_vol.astype(float, copy=False),
        returns=ann_ret.astype(float, copy=False),
        sharpe=sharpe.astype(float, copy=False),
        weights=w,
        best_sharpe_idx=best_sharpe_idx,
    )


__all__ = [
    "CovMethod",
    "CovarianceEstimate",
    "EfficientFrontierResult",
    "ExactEfficientFrontierResult",
    "MonteCarloGBMResult",
    "OptimizerObjective",
    "PortfolioSolution",
    "RebalanceFreq",
    "TrainTestSplitResult",
    "WalkForwardResult",
    "WindowMode",
    "backtest_portfolio",
    "clean_joint_returns",
    "compute_backtest_metrics",
    "estimate_covariance_and_mean",
    "generate_efficient_frontier",
    "normalize_close_to_base_one",
    "returns_correlation_matrix",
    "run_monte_carlo_gbm",
    "run_walk_forward_backtest",
    "solve_exact_efficient_frontier",
    "solve_minimum_variance_portfolio",
    "solve_tangency_portfolio",
    "split_train_test_returns",
]
