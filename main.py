"""Quantitative analysis system entrypoint."""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any

import numpy as np
import pandas as pd

from china_equity_entry import (
    CHINA_EQUITY_ASSET_TYPES,
    DEFAULT_CHINA_EQUITY_SYMBOLS,
    format_china_equity_user_message,
    load_china_equity_ohlcv,
    normalize_china_equity_asset_type,
    resolve_analysis_date_window,
)
from data_fetcher import DataFetchError, apply_default_network_proxy_policy
from feature_engineering import OHLCVFeatureEngineer
from visualizer import MplfinanceVisualizer

LOOKBACK_YEARS = 3
RISK_FREE_RATE = 0.016
TRADING_DAYS = 252


def _compute_cumulative_return(close: pd.Series) -> float:
    """Compute cumulative simple return from the first to the last observation."""
    return float(close.iloc[-1] / close.iloc[0] - 1.0)


def _compute_cagr(close: pd.Series) -> float:
    """Compute CAGR based on start/end prices and calendar elapsed time."""
    elapsed_days = float((close.index[-1] - close.index[0]).days)
    if elapsed_days <= 0:
        return float("nan")
    years = elapsed_days / 365.25
    if years <= 0:
        return float("nan")
    return float((close.iloc[-1] / close.iloc[0]) ** (1.0 / years) - 1.0)


def _compute_sharpe_ratio(
    engineer: OHLCVFeatureEngineer,
    df: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    trading_days: int = TRADING_DAYS,
) -> float:
    """Compute annualized Sharpe ratio using daily log returns."""
    log_returns = engineer.compute_returns(df, price_col="Close", periods=1, log=True).dropna()
    if log_returns.empty:
        return float("nan")

    daily_rf_log = np.log1p(risk_free_rate) / float(trading_days)
    excess_log_returns = log_returns - daily_rf_log
    volatility = float(excess_log_returns.std())
    if not np.isfinite(volatility) or volatility == 0.0:
        return float("nan")

    sharpe = math.sqrt(float(trading_days)) * float(excess_log_returns.mean()) / volatility
    return float(sharpe)


def _describe_bollinger_position(df: pd.DataFrame) -> str:
    """Describe where the latest close is relative to the Bollinger Bands."""
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


def build_diagnostic_report(
    symbol: str,
    df: pd.DataFrame,
    engineer: OHLCVFeatureEngineer,
    risk_free_rate: float = RISK_FREE_RATE,
) -> str:
    """Build the asset math diagnostic report for console output."""
    close = pd.to_numeric(df["Close"], errors="coerce").astype("float64")
    if close.isna().any():
        raise ValueError("Close column contains NaN values; cannot build diagnostic report.")

    cumulative_return = _compute_cumulative_return(close)
    cagr = _compute_cagr(close)
    max_drawdown = df.attrs.get("max_drawdown")
    if max_drawdown is None:
        max_drawdown = engineer.compute_max_drawdown(df, price_col="Close")
    sharpe_ratio = _compute_sharpe_ratio(engineer, df, risk_free_rate=risk_free_rate)
    bollinger_position = _describe_bollinger_position(df)

    return "\n".join(
        [
            "【资产数学诊断报告】",
            "标的代码: %s" % symbol,
            "样本区间: %s 至 %s"
            % (
                df.index[0].strftime("%Y-%m-%d"),
                df.index[-1].strftime("%Y-%m-%d"),
            ),
            "期间累计收益率: %s" % _format_percent(cumulative_return),
            "年化收益率 (CAGR): %s" % _format_percent(cagr),
            "最大回撤: %s" % _format_percent(-abs(float(max_drawdown))),
            "夏普比率 (Rf=1.6%%): %s" % _format_ratio(sharpe_ratio),
            "当前布林带位置: %s" % bollinger_position,
        ]
    )


def _format_percent(value: float) -> str:
    """Format a decimal value as percentage text."""
    if value is None or pd.isna(value):
        return "N/A"
    return "%.2f%%" % (100.0 * float(value))


def _format_ratio(value: float) -> str:
    """Format a ratio metric for console reporting."""
    if value is None or pd.isna(value):
        return "N/A"
    return "%.3f" % float(value)


def run_analysis(
    asset_type: str = "etf",
    symbol: str | None = None,
    start_date: object | None = None,
    end_date: object | None = None,
) -> dict[str, Any]:
    """Run the end-to-end quantitative analysis workflow for 境内 ETF / A 股."""
    apply_default_network_proxy_policy()
    asset_key = normalize_china_equity_asset_type(asset_type)
    sym = symbol if symbol is not None else DEFAULT_CHINA_EQUITY_SYMBOLS[asset_key]
    start_ts, end_ts = resolve_analysis_date_window(
        start_date,
        end_date,
        lookback_years=LOOKBACK_YEARS,
    )

    engineer = OHLCVFeatureEngineer()
    visualizer = MplfinanceVisualizer()

    raw_df = load_china_equity_ohlcv(
        asset_key,
        sym,
        start_ts,
        end_ts,
        adjust="hfq",
    )
    featured_df = engineer.transform(raw_df)
    featured_df.attrs["symbol"] = sym
    featured_df.attrs["asset_type"] = asset_key
    featured_df.attrs["analysis_start"] = start_ts.strftime("%Y-%m-%d")
    featured_df.attrs["analysis_end"] = end_ts.strftime("%Y-%m-%d")

    figure = visualizer.plot(featured_df, show=True)
    report = build_diagnostic_report(symbol=sym, df=featured_df, engineer=engineer)
    print(report)

    return {
        "raw_df": raw_df,
        "featured_df": featured_df,
        "figure": figure,
        "report": report,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="量化诊断 CLI：境内 A 股 ETF 与 A 股个股（与 Web 共用校验与数据路由）。",
    )
    parser.add_argument(
        "--asset-type",
        choices=CHINA_EQUITY_ASSET_TYPES,
        default="etf",
        metavar="TYPE",
        help="资产类型: etf 或 ashare（默认 etf）",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="标的代码；默认 ETF 为 510300，A 股为 600519",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="开始日期 YYYY-MM-DD；默认约为结束日期往前 3 年",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="结束日期 YYYY-MM-DD；默认今天",
    )
    return parser.parse_args(argv)


def main() -> int:
    """CLI entrypoint."""
    args = _parse_args()
    try:
        run_analysis(
            asset_type=args.asset_type,
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
        )
        return 0
    except (ImportError, DataFetchError, ValueError, RuntimeError, TypeError) as exc:
        print(format_china_equity_user_message(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
