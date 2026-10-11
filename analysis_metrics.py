"""Shared Web/CLI diagnostics: calendar CAGR, log-return Sharpe and drawdown."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from feature_engineering import OHLCVFeatureEngineer

RISK_FREE_RATE = 0.016
TRADING_DAYS = 252


def format_percent_metric(value: float, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return ("%+.2f%%" if signed else "%.2f%%") % (100.0 * float(value))


def format_ratio_metric(value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return "%.3f" % float(value)


def compute_cumulative_return(close: pd.Series) -> float:
    """计算区间累计收益率。"""
    return float(close.iloc[-1] / close.iloc[0] - 1.0)


def compute_cagr(close: pd.Series) -> float:
    """基于起止价格和自然日跨度计算 CAGR。"""
    elapsed_days = float((close.index[-1] - close.index[0]).days)
    if elapsed_days <= 0:
        return float("nan")

    years = elapsed_days / 365.25
    # 与 CLI 一致：按自然日跨度折算年化复合增长率。
    return float((close.iloc[-1] / close.iloc[0]) ** (1.0 / years) - 1.0)


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
    log_returns = engineer.compute_returns(df, price_col="Close", periods=1, log=True).dropna()
    if log_returns.empty or log_returns.nunique() == 1:
        return float("nan")

    # Sharpe = 年化超额收益均值 / 超额收益波动率；这里用日对数收益口径。
    daily_rf_log = np.log1p(risk_free_rate) / float(trading_days)
    excess_log_returns = log_returns - daily_rf_log
    volatility = float(log_returns.std())
    if not np.isfinite(volatility) or volatility == 0.0:
        return float("nan")

    sharpe = math.sqrt(float(trading_days)) * float(excess_log_returns.mean()) / volatility
    return float(sharpe)


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
    trading_days: int = TRADING_DAYS,
) -> dict[str, Any]:
    """将 CLI 数学诊断逻辑迁移为 Web 指标字典输出。"""
    series_kind = str(df.attrs.get("series_kind", "price") or "price").strip().lower()
    engineer = OHLCVFeatureEngineer()
    # 指标计算前先跑一遍特征工程，确保均线、布林带、回撤列都存在。
    featured_df = engineer.transform(df, trading_days=trading_days)
    featured_df.attrs["series_kind"] = series_kind

    close = pd.to_numeric(featured_df["Close"], errors="coerce").astype("float64")
    if close.isna().any():
        raise ValueError("Close 列存在 NaN，无法计算指标。")

    metrics: dict[str, Any] = {
        "cumulative_return": compute_cumulative_return(close),
        "cagr": compute_cagr(close),
        "max_drawdown": float(featured_df.attrs["max_drawdown"]),
        "sharpe_ratio": compute_sharpe_ratio(
            engineer,
            featured_df,
            risk_free_rate=risk_free_rate,
            trading_days=trading_days,
        ),
        "latest_signal": describe_bollinger_position(featured_df),
        "featured_df": featured_df,
        "series_kind": series_kind,
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
