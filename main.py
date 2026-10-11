"""Quantitative analysis system entrypoint."""

from __future__ import annotations

import argparse
import sys
from typing import Any

import pandas as pd

from asset_resolver import (
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

from analysis_metrics import (
    TRADING_DAYS as TRADING_DAYS,
    RISK_FREE_RATE,
    compute_cumulative_return as _compute_cumulative_return,
    compute_cagr as _compute_cagr,
    compute_sharpe_ratio as _compute_sharpe_ratio,
    describe_bollinger_position as _describe_bollinger_position,
    format_percent_metric as _format_percent,
    format_ratio_metric as _format_ratio,
)

LOOKBACK_YEARS = 3


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

    # 报告指标和 Web 看板保持同一口径：收益、回撤、Sharpe、布林带位置。
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
            "夏普比率 (Rf=%.1f%%): %s" % (risk_free_rate * 100.0, _format_ratio(sharpe_ratio)),
            "当前布林带位置: %s" % bollinger_position,
        ]
    )


def run_analysis(
    asset_type: str = "etf",
    symbol: str | None = None,
    start_date: object | None = None,
    end_date: object | None = None,
) -> dict[str, Any]:
    """Run the end-to-end quantitative analysis workflow for 境内 ETF / A 股."""
    apply_default_network_proxy_policy()
    # CLI 只开放境内 ETF/A 股；资产类型先规范化，避免大小写或空格造成分支错误。
    asset_key = normalize_china_equity_asset_type(asset_type)
    sym = symbol if symbol is not None else DEFAULT_CHINA_EQUITY_SYMBOLS[asset_key]
    start_ts, end_ts = resolve_analysis_date_window(
        start_date,
        end_date,
        lookback_years=LOOKBACK_YEARS,
    )

    engineer = OHLCVFeatureEngineer()
    visualizer = MplfinanceVisualizer()

    # 典型流水线：拉取标准 OHLCV -> 生成技术指标 -> 绘图 -> 输出文本报告。
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
