"""Shared Web/CLI diagnostics: calendar CAGR, log-return Sharpe/Sortino, Calmar, VaR/CVaR and drawdown."""

from __future__ import annotations

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
