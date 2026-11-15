"""Shared Web/CLI diagnostics: calendar CAGR, log-return Sharpe/Sortino, Calmar, VaR/CVaR and drawdown."""

from __future__ import annotations

import importlib
import math
from typing import Any, Literal

import numpy as np
import pandas as pd

from asset_catalog import get_asset_trading_days
from feature_engineering import OHLCVFeatureEngineer

RISK_FREE_RATE = 0.016
TRADING_DAYS = 252
ReturnConvention = Literal["log", "simple"]


def format_percent_metric(value: float, signed: bool = False) -> str:
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return "N/A"
    return ("%+.2f%%" if signed else "%.2f%%") % (100.0 * float(value))


def format_ratio_metric(value: float) -> str:
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return "N/A"
    return "%.3f" % float(value)


def compute_cumulative_return(close: pd.Series) -> float:
    """计算区间累计收益率；非正或非有限起止价返回 NaN。"""
    if close.empty:
        return float("nan")
    p0 = float(close.iloc[0])
    pt = float(close.iloc[-1])
    if not (np.isfinite(p0) and np.isfinite(pt)) or p0 <= 0.0:
        return float("nan")
    return float(pt / p0 - 1.0)


def compute_cagr(close: pd.Series) -> float:
    """基于起止价格和自然日跨度计算 CAGR；非正价格或零日跨度安全返回 NaN。"""
    if close.empty:
        return float("nan")
    elapsed_days = float((close.index[-1] - close.index[0]).days)
    if elapsed_days <= 0.0:
        return float("nan")

    p0 = float(close.iloc[0])
    pt = float(close.iloc[-1])
    if not (np.isfinite(p0) and np.isfinite(pt)) or p0 <= 0.0 or pt <= 0.0:
        return float("nan")

    years = elapsed_days / 365.25
    # 与 CLI 一致：按自然日跨度折算年化复合增长率。
    return float((pt / p0) ** (1.0 / years) - 1.0)


def _normalize_return_convention(return_convention: str) -> ReturnConvention:
    conv = str(return_convention or "log").strip().lower()
    if conv not in ("log", "simple"):
        raise ValueError(
            f"Unsupported return_convention: {return_convention!r}; choose 'log' or 'simple'."
        )
    return conv  # type: ignore[return-value]


def _valid_period_returns(
    engineer: OHLCVFeatureEngineer,
    df: pd.DataFrame,
    *,
    return_convention: ReturnConvention = "log",
) -> pd.Series:
    conv = _normalize_return_convention(return_convention)
    try:
        return engineer.compute_returns(
            df, price_col="Close", periods=1, log=(conv == "log")
        ).dropna()
    except ValueError:
        return pd.Series(dtype="float64")


def _valid_log_returns(
    engineer: OHLCVFeatureEngineer,
    df: pd.DataFrame,
) -> pd.Series:
    return _valid_period_returns(engineer, df, return_convention="log")


def _daily_risk_free(
    risk_free_rate: float, trading_days: int, return_convention: ReturnConvention
) -> float:
    if return_convention == "simple":
        return float(risk_free_rate) / float(trading_days)
    return float(np.log1p(risk_free_rate)) / float(trading_days)


def compute_annualized_volatility(
    engineer: OHLCVFeatureEngineer,
    df: pd.DataFrame,
    trading_days: int = TRADING_DAYS,
    *,
    return_convention: ReturnConvention = "log",
) -> float:
    """基于日对数收益率（或简单收益率）计算样本年化波动率。"""
    trading_days = engineer._validate_positive_int(trading_days, "trading_days")
    returns = _valid_period_returns(engineer, df, return_convention=return_convention)
    if len(returns) < 2:
        return float("nan")
    vol = float(returns.std(ddof=1))
    if not np.isfinite(vol):
        return float("nan")
    return float(vol * math.sqrt(float(trading_days)))


def compute_sharpe_ratio(
    engineer: OHLCVFeatureEngineer,
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
    *,
    return_convention: ReturnConvention = "log",
) -> float:
    """使用日对数收益率（或简单收益率）计算年化 Sharpe。"""
    trading_days = engineer._validate_positive_int(trading_days, "trading_days")
    conv = _normalize_return_convention(return_convention)
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be finite and greater than -1.")
    returns = _valid_period_returns(engineer, df, return_convention=conv)
    if len(returns) < 2 or returns.nunique() == 1:
        return float("nan")

    daily_rf = _daily_risk_free(risk_free_rate, trading_days, conv)
    excess = returns - daily_rf
    volatility = float(returns.std(ddof=1))
    if not np.isfinite(volatility) or volatility == 0.0:
        return float("nan")

    sharpe = math.sqrt(float(trading_days)) * float(excess.mean()) / volatility
    return float(sharpe)


def compute_sortino_ratio(
    engineer: OHLCVFeatureEngineer,
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
    *,
    return_convention: ReturnConvention = "log",
) -> float:
    """使用日超额收益与下行半方差（MAR = 日无风险利率）计算年化 Sortino。"""
    trading_days = engineer._validate_positive_int(trading_days, "trading_days")
    conv = _normalize_return_convention(return_convention)
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be finite and greater than -1.")
    returns = _valid_period_returns(engineer, df, return_convention=conv)
    if len(returns) < 2 or returns.nunique() == 1:
        return float("nan")

    daily_rf = _daily_risk_free(risk_free_rate, trading_days, conv)
    excess = returns - daily_rf
    downside = np.minimum(excess.to_numpy(dtype=float), 0.0)
    downside_dev = float(np.sqrt(np.mean(downside * downside)))
    if not np.isfinite(downside_dev) or downside_dev <= 0.0:
        return float("nan")
    return float(math.sqrt(float(trading_days)) * float(excess.mean()) / downside_dev)


def compute_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """Calmar = CAGR / |Max Drawdown|，回撤为 0 或 CAGR 无效时返回 NaN。"""
    if not (np.isfinite(cagr) and np.isfinite(max_drawdown)) or abs(max_drawdown) <= 0.0:
        return float("nan")
    return float(cagr / abs(max_drawdown))


def compute_var_cvar(
    engineer: OHLCVFeatureEngineer,
    df: pd.DataFrame,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """基于历史日简单收益率分布计算样本期 VaR 与 CVaR（返回负值表示损失幅度）。"""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1.")
    simple_returns = engineer.compute_returns(df, price_col="Close", periods=1, log=False).dropna()
    if simple_returns.empty:
        return float("nan"), float("nan")
    arr = simple_returns.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    var = float(np.quantile(arr, 1.0 - float(confidence)))
    tail = arr[arr <= var]
    cvar = float(tail.mean()) if tail.size > 0 else var
    return var, cvar


def describe_bollinger_position(df: pd.DataFrame) -> str:
    """描述最新收盘价位于布林带的哪个区间。"""
    feature_summary = df.attrs.get("feature_summary", {})
    bollinger_window = int(feature_summary.get("bollinger_window", 20))
    upper_col = "bollinger_upper_%d" % bollinger_window
    lower_col = "bollinger_lower_%d" % bollinger_window

    if upper_col not in df.columns or lower_col not in df.columns:
        return "布林带数据缺失"

    latest = df.iloc[-1]
    latest_close = float(latest["Close"])
    upper = latest.get(upper_col)
    lower = latest.get(lower_col)
    if pd.isna(upper) or pd.isna(lower):
        return "布林带数据不足"
    if latest_close >= float(upper):
        return "突破上轨"
    if latest_close <= float(lower):
        return "触底下轨"
    return "均值回归中"


def get_quant_metrics(
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int | None = None,
    *,
    return_convention: ReturnConvention = "log",
    sma_short_window: int = 20,
    sma_long_window: int = 60,
    volatility_window: int = 20,
    bollinger_window: int = 20,
    bollinger_num_std: float = 2.0,
    include_momentum_indicators: bool = False,
) -> dict[str, Any]:
    """将 CLI/Web 共享数学诊断指标聚合输出，按资产类别自适应交易日年化因子与语义拦截。"""
    conv = _normalize_return_convention(return_convention)
    series_kind = str(df.attrs.get("series_kind", "price") or "price").strip().lower()
    asset_type = str(df.attrs.get("asset_type", "") or "").strip().lower()
    return_basis = str(df.attrs.get("return_basis", "") or "").strip().lower()
    is_yield_level = (
        series_kind == "yield_level"
        or return_basis == "yield_level"
        or asset_type == "bond_us_yield"
    )
    is_fx_rate = series_kind == "fx_rate" or return_basis == "fx_rate" or asset_type == "fx"
    effective_trading_days = (
        int(trading_days)
        if trading_days is not None
        else get_asset_trading_days(asset_type or None)
    )

    engineer = OHLCVFeatureEngineer()
    # 指标计算前先跑一遍特征工程，确保均线、布林带、回撤列都存在。
    featured_df = engineer.transform(
        df,
        sma_short_window=sma_short_window,
        sma_long_window=sma_long_window,
        volatility_window=volatility_window,
        bollinger_window=bollinger_window,
        bollinger_num_std=bollinger_num_std,
        trading_days=effective_trading_days,
        include_momentum_indicators=include_momentum_indicators,
    )
    if is_yield_level:
        series_kind = "yield_level"
    featured_df.attrs["series_kind"] = series_kind

    close = pd.to_numeric(featured_df["Close"], errors="coerce").astype("float64")
    if close.isna().any():
        raise ValueError("Close 列存在 NaN，无法计算指标。")

    metric_reasons: dict[str, str] = {}
    semantic_warnings: list[str] = []
    latest_close = float(close.iloc[-1])

    if is_yield_level:
        nan_val = float("nan")
        blocked_keys = (
            "cumulative_return",
            "cagr",
            "max_drawdown",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "var_95",
            "cvar_95",
        )
        for key in blocked_keys:
            metric_reasons[key] = "yield_level_not_return"
        cumulative_return = nan_val
        cagr = nan_val
        max_drawdown = nan_val
        annualized_volatility = nan_val
        sharpe_ratio = nan_val
        sortino_ratio = nan_val
        calmar_ratio = nan_val
        var_95 = nan_val
        cvar_95 = nan_val
        level_change = float(close.iloc[-1] - close.iloc[0]) if len(close) >= 1 else nan_val
        level_min = float(close.min())
        level_max = float(close.max())
        diffs = close.diff().dropna()
        level_diff_std = float(diffs.std(ddof=1)) if len(diffs) >= 2 else nan_val
        semantic_warnings.append(
            "收益率水平序列（yield_level）禁止计算资产持有期复利回报、CAGR、回撤与夏普比率；已改为输出水平区间统计。"
        )
    else:
        level_change = float("nan")
        level_min = float(close.min())
        level_max = float(close.max())
        level_diff_std = float("nan")

        cumulative_return = compute_cumulative_return(close)
        cagr = compute_cagr(close)
        max_drawdown = float(featured_df.attrs["max_drawdown"])
        annualized_volatility = compute_annualized_volatility(
            engineer, featured_df, trading_days=effective_trading_days, return_convention=conv
        )
        sharpe_ratio = compute_sharpe_ratio(
            engineer,
            featured_df,
            risk_free_rate=risk_free_rate,
            trading_days=effective_trading_days,
            return_convention=conv,
        )
        sortino_ratio = compute_sortino_ratio(
            engineer,
            featured_df,
            risk_free_rate=risk_free_rate,
            trading_days=effective_trading_days,
            return_convention=conv,
        )
        calmar_ratio = compute_calmar_ratio(cagr, max_drawdown)
        var_95, cvar_95 = compute_var_cvar(engineer, featured_df, confidence=0.95)

        period_returns = _valid_period_returns(engineer, featured_df, return_convention=conv)
        elapsed_days = float((close.index[-1] - close.index[0]).days)
        if not np.isfinite(cumulative_return):
            metric_reasons["cumulative_return"] = "non_positive_or_invalid_price"
        if not np.isfinite(cagr):
            metric_reasons["cagr"] = (
                "insufficient_observations"
                if len(close) < 2 or elapsed_days <= 0.0
                else "non_positive_price"
            )
        if not np.isfinite(annualized_volatility):
            metric_reasons["annualized_volatility"] = "insufficient_observations"
        if not np.isfinite(sharpe_ratio):
            metric_reasons["sharpe_ratio"] = (
                "insufficient_observations" if len(period_returns) < 2 else "zero_volatility"
            )
        if not np.isfinite(sortino_ratio):
            if len(period_returns) < 2:
                metric_reasons["sortino_ratio"] = "insufficient_observations"
            elif period_returns.nunique() == 1:
                metric_reasons["sortino_ratio"] = "zero_volatility"
            else:
                metric_reasons["sortino_ratio"] = "zero_downside_deviation"
        if not np.isfinite(calmar_ratio):
            metric_reasons["calmar_ratio"] = (
                "zero_max_drawdown"
                if np.isfinite(cagr) and abs(max_drawdown) <= 0.0
                else "cagr_unavailable"
            )
        if not np.isfinite(var_95):
            metric_reasons["var_95"] = "insufficient_observations"
        if not np.isfinite(cvar_95):
            metric_reasons["cvar_95"] = "insufficient_observations"

    metrics: dict[str, Any] = {
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "latest_close": latest_close,
        "latest_level": latest_close if is_yield_level else float("nan"),
        "level_change": level_change,
        "level_min": level_min,
        "level_max": level_max,
        "level_diff_std": level_diff_std,
        "trading_days": effective_trading_days,
        "periods_per_year": effective_trading_days,
        "return_convention": conv,
        "latest_signal": describe_bollinger_position(featured_df),
        "featured_df": featured_df,
        "series_kind": series_kind,
        "metric_reasons": metric_reasons,
        "semantic_warnings": semantic_warnings,
        "data_quality": dict(featured_df.attrs.get("data_quality", {})),
        "data_source": str(featured_df.attrs.get("data_source", "")),
        "currency": str(featured_df.attrs.get("currency", "")),
        "return_basis": str(featured_df.attrs.get("return_basis", "")),
    }
    if is_yield_level:
        metrics["metric_disclaimer"] = (
            "当前为收益率水平（yield_level）序列（%），并非债券持有期净值回报；"
            "已禁止计算累计收益率、CAGR、波动率、最大回撤、Sharpe、Sortino、Calmar 与 VaR/CVaR（显示 N/A），"
            "仅提供最新水平与区间点差变化供比较。"
        )
        metrics["cumulative_return_label"] = "水平变化（收益指标已禁用）"
        metrics["cagr_label"] = "水平年化（yield_level 禁算）"
    elif is_fx_rate:
        fx_caveat = (
            "即期汇率序列仅反映货币双边比价变动，不含货币利差与套息收益（Carry）；"
            "风险收益比率仅供参考，且不支持对汇率序列再作基准货币换算。"
        )
        semantic_warnings.append(fx_caveat)
        metrics["metric_disclaimer"] = fx_caveat
        metrics["cumulative_return_label"] = "累计汇率变动"
        metrics["cagr_label"] = "年化汇率变动"
    else:
        metrics["metric_disclaimer"] = ""
        metrics["cumulative_return_label"] = "累计收益率"
        metrics["cagr_label"] = "CAGR"
    return metrics


def _period_risk_free_rate(
    risk_free_rate: float,
    periods_per_year: float,
    return_convention: ReturnConvention,
) -> float:
    conv = _normalize_return_convention(return_convention)
    ann_periods = float(periods_per_year)
    if not np.isfinite(ann_periods) or ann_periods <= 0.0:
        raise ValueError("periods_per_year must be positive and finite.")
    if not np.isfinite(float(risk_free_rate)) or float(risk_free_rate) <= -1.0:
        raise ValueError("risk_free_rate must be finite and greater than -1.")
    if conv == "simple":
        return float(risk_free_rate) / ann_periods
    return float(np.log1p(float(risk_free_rate))) / ann_periods


def _to_finite_float_array(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    arr = numeric.to_numpy(dtype=float, copy=True)
    return arr[np.isfinite(arr)]


def _compound_cumulative_return(simple_returns: np.ndarray) -> float:
    if simple_returns.size == 0 or np.any(simple_returns <= -1.0):
        return float("nan")
    return float(np.expm1(float(np.sum(np.log1p(simple_returns)))))


def _scipy_stats() -> Any:
    return importlib.import_module("scipy.stats")


def _annualize_cumulative_return(
    cum_return: float, observations: int, periods_per_year: float
) -> float:
    if observations <= 0 or not np.isfinite(cum_return) or cum_return <= -1.0:
        return float("nan")
    return float((1.0 + float(cum_return)) ** (float(periods_per_year) / float(observations)) - 1.0)


def compute_return_distribution_stats(returns: pd.Series) -> dict[str, Any]:
    """计算收益率分布特征、正态性检验（Jarque-Bera）与均值 t 检验。"""
    arr = _to_finite_float_array(returns)
    n = int(arr.size)
    reasons: dict[str, str] = {}

    if n == 0:
        for key in (
            "mean",
            "median",
            "std",
            "skewness",
            "excess_kurtosis",
            "jarque_bera_stat",
            "jarque_bera_p_value",
            "win_rate",
            "best_period_return",
            "worst_period_return",
            "mean_t_stat",
            "mean_p_value",
        ):
            reasons[key] = "insufficient_observations"
        return {
            "observations": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "skewness": float("nan"),
            "excess_kurtosis": float("nan"),
            "jarque_bera_stat": float("nan"),
            "jarque_bera_p_value": float("nan"),
            "is_normal_at_5pct": None,
            "win_rate": float("nan"),
            "positive_periods": 0,
            "negative_periods": 0,
            "zero_periods": 0,
            "best_period_return": float("nan"),
            "worst_period_return": float("nan"),
            "mean_t_stat": float("nan"),
            "mean_p_value": float("nan"),
            "metric_reasons": reasons,
        }

    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    win_rate = float(np.mean(arr > 0.0))
    positive_periods = int(np.sum(arr > 0.0))
    negative_periods = int(np.sum(arr < 0.0))
    zero_periods = int(np.sum(arr == 0.0))
    best_ret = float(np.max(arr))
    worst_ret = float(np.min(arr))

    std_val = float(np.std(arr, ddof=1)) if n >= 2 else float("nan")
    if n < 2:
        reasons["std"] = "insufficient_observations"

    has_variation = n >= 2 and np.isfinite(std_val) and std_val > 1e-15

    skew_val = float("nan")
    kurt_val = float("nan")
    jb_stat = float("nan")
    jb_p = float("nan")
    is_normal: bool | None = None
    t_stat = float("nan")
    t_p = float("nan")

    if n < 2:
        reasons["mean_t_stat"] = "insufficient_observations"
        reasons["mean_p_value"] = "insufficient_observations"
    elif not has_variation:
        reasons["mean_t_stat"] = "zero_volatility"
        reasons["mean_p_value"] = "zero_volatility"
    else:
        scipy_stats = _scipy_stats()
        t_res = scipy_stats.ttest_1samp(arr, 0.0)
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)

    if n < 3:
        reasons["skewness"] = "insufficient_observations"
        reasons["jarque_bera_stat"] = "insufficient_observations"
        reasons["jarque_bera_p_value"] = "insufficient_observations"
    elif not has_variation:
        reasons["skewness"] = "zero_volatility"
        reasons["jarque_bera_stat"] = "zero_volatility"
        reasons["jarque_bera_p_value"] = "zero_volatility"
    else:
        scipy_stats = _scipy_stats()
        skew_val = float(scipy_stats.skew(arr, bias=False))
        jb_res = scipy_stats.jarque_bera(arr)
        jb_stat = float(jb_res.statistic)
        jb_p = float(jb_res.pvalue)
        if np.isfinite(jb_p):
            is_normal = bool(jb_p >= 0.05)

    if n < 4:
        reasons["excess_kurtosis"] = "insufficient_observations"
    elif not has_variation:
        reasons["excess_kurtosis"] = "zero_volatility"
    else:
        kurt_val = float(_scipy_stats().kurtosis(arr, fisher=True, bias=False))

    return {
        "observations": n,
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "skewness": skew_val,
        "excess_kurtosis": kurt_val,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_p_value": jb_p,
        "is_normal_at_5pct": is_normal,
        "win_rate": win_rate,
        "positive_periods": positive_periods,
        "negative_periods": negative_periods,
        "zero_periods": zero_periods,
        "best_period_return": best_ret,
        "worst_period_return": worst_ret,
        "mean_t_stat": t_stat,
        "mean_p_value": t_p,
        "metric_reasons": reasons,
    }


def compute_memmel_sharpe_test(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: float = TRADING_DAYS,
    risk_free_rate: float = RISK_FREE_RATE,
    return_convention: ReturnConvention = "simple",
) -> dict[str, Any]:
    """Memmel (2003) 渐近正态 Sharpe 比率差异检验（考虑收益相关性）。"""
    period_rf = _period_risk_free_rate(risk_free_rate, periods_per_year, return_convention)
    aligned = pd.concat(
        [
            pd.to_numeric(strategy_returns, errors="coerce"),
            pd.to_numeric(benchmark_returns, errors="coerce"),
        ],
        axis=1,
    ).dropna()
    if aligned.empty:
        return {
            "observations": 0,
            "z_stat": float("nan"),
            "p_value": float("nan"),
            "sharpe_strategy": float("nan"),
            "sharpe_benchmark": float("nan"),
            "sharpe_diff": float("nan"),
            "correlation": float("nan"),
            "method": "memmel_2003_asymptotic_normal",
            "reason": "insufficient_observations",
        }

    a = aligned.iloc[:, 0].to_numpy(dtype=float)
    b = aligned.iloc[:, 1].to_numpy(dtype=float)
    finite_mask = np.isfinite(a) & np.isfinite(b)
    a = a[finite_mask]
    b = b[finite_mask]
    n = int(a.size)
    if n < 5:
        return {
            "observations": n,
            "z_stat": float("nan"),
            "p_value": float("nan"),
            "sharpe_strategy": float("nan"),
            "sharpe_benchmark": float("nan"),
            "sharpe_diff": float("nan"),
            "correlation": float("nan"),
            "method": "memmel_2003_asymptotic_normal",
            "reason": "insufficient_observations",
        }

    ex_a = a - period_rf
    ex_b = b - period_rf
    std_a = float(np.std(ex_a, ddof=1))
    std_b = float(np.std(ex_b, ddof=1))
    if not (np.isfinite(std_a) and np.isfinite(std_b)) or std_a <= 1e-15 or std_b <= 1e-15:
        return {
            "observations": n,
            "z_stat": float("nan"),
            "p_value": float("nan"),
            "sharpe_strategy": float("nan"),
            "sharpe_benchmark": float("nan"),
            "sharpe_diff": float("nan"),
            "correlation": float("nan"),
            "method": "memmel_2003_asymptotic_normal",
            "reason": "zero_volatility",
        }

    sr_a = float(np.mean(ex_a)) / std_a
    sr_b = float(np.mean(ex_b)) / std_b
    ann_scale = math.sqrt(float(periods_per_year))
    sharpe_ann_a = sr_a * ann_scale
    sharpe_ann_b = sr_b * ann_scale
    sharpe_diff = sharpe_ann_a - sharpe_ann_b
    rho = float(np.clip(float(np.corrcoef(ex_a, ex_b)[0, 1]), -1.0, 1.0))

    theta = (
        2.0 * (1.0 - rho) + 0.5 * (sr_a * sr_a + sr_b * sr_b - 2.0 * sr_a * sr_b * rho * rho)
    ) / float(n)
    if np.allclose(a, b, atol=1e-14) or not np.isfinite(theta) or theta <= 1e-18:
        return {
            "observations": n,
            "z_stat": float("nan"),
            "p_value": float("nan"),
            "sharpe_strategy": sharpe_ann_a,
            "sharpe_benchmark": sharpe_ann_b,
            "sharpe_diff": sharpe_diff,
            "correlation": rho,
            "method": "memmel_2003_asymptotic_normal",
            "reason": "zero_difference_variance",
        }

    z_stat = float((sr_a - sr_b) / math.sqrt(theta))
    p_value = float(2.0 * _scipy_stats().norm.sf(abs(z_stat)))
    return {
        "observations": n,
        "z_stat": z_stat,
        "p_value": p_value,
        "sharpe_strategy": sharpe_ann_a,
        "sharpe_benchmark": sharpe_ann_b,
        "sharpe_diff": sharpe_diff,
        "correlation": rho,
        "method": "memmel_2003_asymptotic_normal",
        "reason": None,
    }


def compute_benchmark_relative_metrics(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    periods_per_year: float = TRADING_DAYS,
    risk_free_rate: float = RISK_FREE_RATE,
    return_convention: ReturnConvention = "simple",
) -> dict[str, Any]:
    """计算相对基准的超额收益、跟踪误差、信息比率、CAPM Alpha/Beta 与上下行捕获比率。"""
    conv = _normalize_return_convention(return_convention)
    period_rf = _period_risk_free_rate(risk_free_rate, periods_per_year, conv)
    joined = pd.concat(
        [
            pd.to_numeric(asset_returns, errors="coerce"),
            pd.to_numeric(benchmark_returns, errors="coerce"),
        ],
        axis=1,
    ).dropna()

    reasons: dict[str, str] = {}
    if joined.empty:
        return {
            "observations": 0,
            "cumulative_return_asset": float("nan"),
            "cumulative_return_benchmark": float("nan"),
            "excess_cumulative_return": float("nan"),
            "annualized_return_asset": float("nan"),
            "annualized_return_benchmark": float("nan"),
            "excess_annualized_return": float("nan"),
            "tracking_error": float("nan"),
            "information_ratio": float("nan"),
            "beta": float("nan"),
            "jensen_alpha": float("nan"),
            "correlation": float("nan"),
            "r_squared": float("nan"),
            "up_capture_ratio": float("nan"),
            "down_capture_ratio": float("nan"),
            "up_periods": 0,
            "down_periods": 0,
            "active_win_rate": float("nan"),
            "active_mean_t_stat": float("nan"),
            "active_mean_p_value": float("nan"),
            "memmel_sharpe_test": compute_memmel_sharpe_test(
                asset_returns,
                benchmark_returns,
                periods_per_year=periods_per_year,
                risk_free_rate=risk_free_rate,
                return_convention=conv,
            ),
            "metric_reasons": {"all": "no_overlapping_observations"},
        }

    r_a = joined.iloc[:, 0].to_numpy(dtype=float)
    r_b = joined.iloc[:, 1].to_numpy(dtype=float)
    finite_mask = np.isfinite(r_a) & np.isfinite(r_b)
    r_a = r_a[finite_mask]
    r_b = r_b[finite_mask]
    n = int(r_a.size)

    s_a = np.expm1(r_a) if conv == "log" else r_a
    s_b = np.expm1(r_b) if conv == "log" else r_b

    cum_a = _compound_cumulative_return(s_a)
    cum_b = _compound_cumulative_return(s_b)
    excess_cum = cum_a - cum_b if (np.isfinite(cum_a) and np.isfinite(cum_b)) else float("nan")
    ann_a = _annualize_cumulative_return(cum_a, n, periods_per_year)
    ann_b = _annualize_cumulative_return(cum_b, n, periods_per_year)
    excess_ann = ann_a - ann_b if (np.isfinite(ann_a) and np.isfinite(ann_b)) else float("nan")

    active = r_a - r_b
    active_win_rate = float(np.mean(active > 0.0)) if n >= 1 else float("nan")
    tracking_error = float("nan")
    information_ratio = float("nan")
    active_t_stat = float("nan")
    active_p_val = float("nan")

    if n < 2:
        reasons["tracking_error"] = "insufficient_observations"
        reasons["information_ratio"] = "insufficient_observations"
        reasons["active_mean_t_stat"] = "insufficient_observations"
        reasons["active_mean_p_value"] = "insufficient_observations"
    else:
        active_std = float(np.std(active, ddof=1))
        tracking_error = active_std * math.sqrt(float(periods_per_year))
        if active_std > 1e-15:
            information_ratio = float(
                math.sqrt(float(periods_per_year)) * float(np.mean(active)) / active_std
            )
            t_res = _scipy_stats().ttest_1samp(active, 0.0)
            active_t_stat = float(t_res.statistic)
            active_p_val = float(t_res.pvalue)
        else:
            reasons["information_ratio"] = "zero_tracking_error"
            reasons["active_mean_t_stat"] = "zero_tracking_error"
            reasons["active_mean_p_value"] = "zero_tracking_error"

    beta = float("nan")
    jensen_alpha = float("nan")
    corr = float("nan")
    r_squared = float("nan")
    if n < 3:
        for k in ("beta", "jensen_alpha", "correlation", "r_squared"):
            reasons[k] = "insufficient_observations"
    else:
        ex_a = r_a - period_rf
        ex_b = r_b - period_rf
        var_b = float(np.var(ex_b, ddof=1))
        var_a = float(np.var(ex_a, ddof=1))
        if var_b > 1e-18:
            cov_ab = float(np.cov(ex_a, ex_b, ddof=1)[0, 1])
            beta = cov_ab / var_b
            jensen_alpha = float(
                (float(np.mean(ex_a)) - beta * float(np.mean(ex_b))) * float(periods_per_year)
            )
            if var_a > 1e-18:
                corr = float(np.clip(cov_ab / math.sqrt(var_a * var_b), -1.0, 1.0))
                r_squared = float(corr * corr)
            else:
                reasons["correlation"] = "zero_asset_volatility"
                reasons["r_squared"] = "zero_asset_volatility"
        else:
            for k in ("beta", "jensen_alpha", "correlation", "r_squared"):
                reasons[k] = "zero_benchmark_volatility"

    def _capture_ratio(mask: np.ndarray, label: str) -> tuple[float, int]:
        count = int(np.sum(mask))
        if count == 0:
            reasons[label] = f"no_{label.replace('_ratio', '')}"
            return float("nan"), 0
        cum_sub_a = _compound_cumulative_return(s_a[mask])
        cum_sub_b = _compound_cumulative_return(s_b[mask])
        ann_sub_a = _annualize_cumulative_return(cum_sub_a, count, float(periods_per_year))
        ann_sub_b = _annualize_cumulative_return(cum_sub_b, count, float(periods_per_year))
        if not (np.isfinite(ann_sub_a) and np.isfinite(ann_sub_b)) or abs(ann_sub_b) <= 1e-15:
            reasons[label] = "degenerate_benchmark_capture_denominator"
            return float("nan"), count
        return float(ann_sub_a / ann_sub_b), count

    up_capture, up_periods = _capture_ratio(s_b > 0.0, "up_capture_ratio")
    down_capture, down_periods = _capture_ratio(s_b < 0.0, "down_capture_ratio")

    memmel = compute_memmel_sharpe_test(
        joined.iloc[:, 0],
        joined.iloc[:, 1],
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        return_convention=conv,
    )

    return {
        "observations": n,
        "cumulative_return_asset": cum_a,
        "cumulative_return_benchmark": cum_b,
        "excess_cumulative_return": excess_cum,
        "annualized_return_asset": ann_a,
        "annualized_return_benchmark": ann_b,
        "excess_annualized_return": excess_ann,
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "beta": beta,
        "jensen_alpha": jensen_alpha,
        "correlation": corr,
        "r_squared": r_squared,
        "up_capture_ratio": up_capture,
        "down_capture_ratio": down_capture,
        "up_periods": up_periods,
        "down_periods": down_periods,
        "active_win_rate": active_win_rate,
        "active_mean_t_stat": active_t_stat,
        "active_mean_p_value": active_p_val,
        "memmel_sharpe_test": memmel,
        "metric_reasons": reasons,
    }


def _summarize_rolling_series(series: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    clean = clean[np.isfinite(clean)]
    if clean.empty:
        return {
            "valid_points": 0,
            "latest": float("nan"),
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
        }
    return {
        "valid_points": int(len(clean)),
        "latest": float(clean.iloc[-1]),
        "min": float(clean.min()),
        "median": float(clean.median()),
        "max": float(clean.max()),
    }


def compute_rolling_comparison_stats(
    returns_df: pd.DataFrame,
    *,
    window: int = 20,
    periods_per_year: float = TRADING_DAYS,
    risk_free_rate: float = RISK_FREE_RATE,
    benchmark_col: str | None = None,
    return_convention: ReturnConvention = "simple",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """计算各资产滚动年化收益、滚动波动率、滚动 Sharpe 及相对滚动相关性/Beta。"""
    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise ValueError("rolling window must be an integer >= 2.")
    conv = _normalize_return_convention(return_convention)
    period_rf = _period_risk_free_rate(risk_free_rate, periods_per_year, conv)
    ann_periods = float(periods_per_year)
    sqrt_ann = math.sqrt(ann_periods)

    if returns_df.empty:
        return pd.DataFrame(index=returns_df.index), {
            "window": int(window),
            "assets": {},
            "pairs": {},
        }

    rolling_cols: dict[str, pd.Series] = {}
    asset_summaries: dict[str, Any] = {}

    for col in returns_df.columns:
        col_name = str(col)
        valid = pd.to_numeric(returns_df[col], errors="coerce").dropna()
        valid = valid[np.isfinite(valid)]
        if conv == "simple":
            log_equiv = np.log1p(valid.where(valid > -1.0))
        else:
            log_equiv = valid
        roll_ret = np.expm1(
            log_equiv.rolling(window, min_periods=window).sum() * (ann_periods / float(window))
        )
        roll_std = valid.rolling(window, min_periods=window).std(ddof=1)
        roll_vol = roll_std * sqrt_ann
        roll_mean_ex = (valid - period_rf).rolling(window, min_periods=window).mean()
        roll_sharpe = sqrt_ann * roll_mean_ex / roll_std.where(roll_std > 1e-15)

        ret_reindexed = roll_ret.reindex(returns_df.index)
        vol_reindexed = roll_vol.reindex(returns_df.index)
        sharpe_reindexed = roll_sharpe.reindex(returns_df.index)

        rolling_cols[f"{col_name}__rolling_return_ann"] = ret_reindexed
        rolling_cols[f"{col_name}__rolling_vol_ann"] = vol_reindexed
        rolling_cols[f"{col_name}__rolling_sharpe"] = sharpe_reindexed

        asset_summary: dict[str, Any] = {
            "window": int(window),
            "return_ann": _summarize_rolling_series(ret_reindexed),
            "volatility_ann": _summarize_rolling_series(vol_reindexed),
            "sharpe": _summarize_rolling_series(sharpe_reindexed),
        }

        if (
            benchmark_col is not None
            and benchmark_col in returns_df.columns
            and col_name != benchmark_col
        ):
            pair = pd.concat(
                [
                    pd.to_numeric(returns_df[col], errors="coerce"),
                    pd.to_numeric(returns_df[benchmark_col], errors="coerce"),
                ],
                axis=1,
            ).dropna()
            s_a = pair.iloc[:, 0]
            s_b = pair.iloc[:, 1]
            r_corr = s_a.rolling(window, min_periods=window).corr(s_b)
            var_b = s_b.rolling(window, min_periods=window).var(ddof=1)
            cov_ab = s_a.rolling(window, min_periods=window).cov(s_b)
            r_beta = cov_ab / var_b.where(var_b > 1e-18)

            corr_reindexed = r_corr.reindex(returns_df.index)
            beta_reindexed = r_beta.reindex(returns_df.index)
            rolling_cols[f"{col_name}__rolling_corr_vs_benchmark"] = corr_reindexed
            rolling_cols[f"{col_name}__rolling_beta_vs_benchmark"] = beta_reindexed
            asset_summary["corr_vs_benchmark"] = _summarize_rolling_series(corr_reindexed)
            asset_summary["beta_vs_benchmark"] = _summarize_rolling_series(beta_reindexed)

        asset_summaries[col_name] = asset_summary

    pair_summaries: dict[str, Any] = {}
    asset_cols = [str(c) for c in returns_df.columns if str(c) != benchmark_col]
    pair_idx = 0
    for i in range(len(asset_cols)):
        for j in range(i + 1, len(asset_cols)):
            c1, c2 = asset_cols[i], asset_cols[j]
            pair_key = f"{c1}__vs__{c2}"
            pair = pd.concat(
                [
                    pd.to_numeric(returns_df[c1], errors="coerce"),
                    pd.to_numeric(returns_df[c2], errors="coerce"),
                ],
                axis=1,
            ).dropna()
            r_corr = pair.iloc[:, 0].rolling(window, min_periods=window).corr(pair.iloc[:, 1])
            corr_reindexed = r_corr.reindex(returns_df.index)
            if len(asset_cols) <= 6 or pair_idx < 15:
                rolling_cols[f"{pair_key}__rolling_corr"] = corr_reindexed
            pair_summaries[pair_key] = _summarize_rolling_series(corr_reindexed)
            pair_idx += 1

    rolling_df = pd.DataFrame(rolling_cols, index=returns_df.index)
    if isinstance(returns_df.index, pd.DatetimeIndex) and not rolling_df.index.name:
        rolling_df.index.name = "Date"
    return rolling_df, {
        "window": int(window),
        "assets": asset_summaries,
        "pairs": pair_summaries,
    }
