"""Portfolio NAV, train/test and walk-forward backtests."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd

from analysis_metrics import RISK_FREE_RATE
from models.covariance import CovMethod
from models.optimization import OptimizerObjective, PortfolioSolution, _optimize_weights_for_window
from models.returns import _positive_integer, clean_joint_returns

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
            "cumulative_return": float("nan"),
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
        "cumulative_return": total_return,
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


WindowMode = Literal["rolling", "expanding"]


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
