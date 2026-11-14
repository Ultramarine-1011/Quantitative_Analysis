"""Shared Web/CLI diagnostics: calendar CAGR, log-return Sharpe/Sortino, Calmar, VaR/CVaR and drawdown."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from asset_catalog import get_asset_trading_days
from feature_engineering import OHLCVFeatureEngineer

RISK_FREE_RATE = 0.016
TRADING_DAYS = 252


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


def _valid_log_returns(
    engineer: OHLCVFeatureEngineer,
    df: pd.DataFrame,
) -> pd.Series:
    try:
        return engineer.compute_returns(df, price_col="Close", periods=1, log=True).dropna()
    except ValueError:
        return pd.Series(dtype="float64")


def compute_annualized_volatility(
    engineer: OHLCVFeatureEngineer,
    df: pd.DataFrame,
    trading_days: int = TRADING_DAYS,
) -> float:
    """基于日对数收益率计算样本年化波动率。"""
    trading_days = engineer._validate_positive_int(trading_days, "trading_days")
    log_returns = _valid_log_returns(engineer, df)
    if len(log_returns) < 2:
        return float("nan")
    vol = float(log_returns.std(ddof=1))
    if not np.isfinite(vol):
        return float("nan")
    return float(vol * math.sqrt(float(trading_days)))


def compute_sharpe_ratio(
    engineer: OHLCVFeatureEngineer,
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> float:
    """使用日对数收益率计算年化 Sharpe。"""
    trading_days = engineer._validate_positive_int(trading_days, "trading_days")
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be finite and greater than -1.")
    log_returns = _valid_log_returns(engineer, df)
    if len(log_returns) < 2 or log_returns.nunique() == 1:
        return float("nan")

    # Sharpe = 年化超额收益均值 / 超额收益波动率；这里用日对数收益口径。
    daily_rf_log = np.log1p(risk_free_rate) / float(trading_days)
    excess_log_returns = log_returns - daily_rf_log
    volatility = float(log_returns.std(ddof=1))
    if not np.isfinite(volatility) or volatility == 0.0:
        return float("nan")

    sharpe = math.sqrt(float(trading_days)) * float(excess_log_returns.mean()) / volatility
    return float(sharpe)


def compute_sortino_ratio(
    engineer: OHLCVFeatureEngineer,
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> float:
    """使用日对数超额收益与下行半方差（MAR = 日无风险利率）计算年化 Sortino。"""
    trading_days = engineer._validate_positive_int(trading_days, "trading_days")
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise ValueError("risk_free_rate must be finite and greater than -1.")
    log_returns = _valid_log_returns(engineer, df)
    if len(log_returns) < 2 or log_returns.nunique() == 1:
        return float("nan")

    daily_rf_log = np.log1p(risk_free_rate) / float(trading_days)
    excess = log_returns - daily_rf_log
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
    sma_short_window: int = 20,
    sma_long_window: int = 60,
    volatility_window: int = 20,
    bollinger_window: int = 20,
    bollinger_num_std: float = 2.0,
    include_momentum_indicators: bool = False,
) -> dict[str, Any]:
    """将 CLI/Web 共享数学诊断指标聚合输出，按资产类别自适应交易日年化因子。"""
    series_kind = str(df.attrs.get("series_kind", "price") or "price").strip().lower()
    asset_type = str(df.attrs.get("asset_type", "") or "").strip().lower()
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
    featured_df.attrs["series_kind"] = series_kind

    close = pd.to_numeric(featured_df["Close"], errors="coerce").astype("float64")
    if close.isna().any():
        raise ValueError("Close 列存在 NaN，无法计算指标。")

    cagr = compute_cagr(close)
    max_drawdown = float(featured_df.attrs["max_drawdown"])
    var_95, cvar_95 = compute_var_cvar(engineer, featured_df, confidence=0.95)

    metrics: dict[str, Any] = {
        "cumulative_return": compute_cumulative_return(close),
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "annualized_volatility": compute_annualized_volatility(
            engineer, featured_df, trading_days=effective_trading_days
        ),
        "sharpe_ratio": compute_sharpe_ratio(
            engineer,
            featured_df,
            risk_free_rate=risk_free_rate,
            trading_days=effective_trading_days,
        ),
        "sortino_ratio": compute_sortino_ratio(
            engineer,
            featured_df,
            risk_free_rate=risk_free_rate,
            trading_days=effective_trading_days,
        ),
        "calmar_ratio": compute_calmar_ratio(cagr, max_drawdown),
        "var_95": var_95,
        "cvar_95": cvar_95,
        "trading_days": effective_trading_days,
        "latest_signal": describe_bollinger_position(featured_df),
        "featured_df": featured_df,
        "series_kind": series_kind,
        "data_quality": dict(featured_df.attrs.get("data_quality", {})),
        "data_source": str(featured_df.attrs.get("data_source", "")),
        "currency": str(featured_df.attrs.get("currency", "")),
        "return_basis": str(featured_df.attrs.get("return_basis", "")),
    }
    if series_kind == "yield_level":
        # 收益率水平不是资产净值，显示时需要换一套更准确的指标文案。
        metrics["metric_disclaimer"] = (
            "以下为收益率**水平**序列的统计（末/初比与波动），不代表债券持有期回报；"
            "夏普等指标基于水平序列的对数变化，仅作相对比较参考。"
        )
        metrics["cumulative_return_label"] = "水平相对变化（Close末/初-1）"
        metrics["cagr_label"] = "水平年化缩放（非票息回报）"
    else:
        metrics["metric_disclaimer"] = ""
        metrics["cumulative_return_label"] = "累计收益率"
        metrics["cagr_label"] = "CAGR"
    return metrics
