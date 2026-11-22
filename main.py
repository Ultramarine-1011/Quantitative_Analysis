"""Quantitative analysis system CLI entrypoint covering all 13 asset classes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from analysis_metrics import (
    RISK_FREE_RATE,
    TRADING_DAYS as TRADING_DAYS,
    compute_cagr as _compute_cagr,  # noqa: F401 - re-exported for parity tests
)
from artifacts.transaction import FileTransaction
from asset_catalog import (
    ASSET_RETURN_BASIS_NOTES,
    DEFAULT_SYMBOLS,
    SUPPORTED_ASSET_TYPES,
    get_asset_trading_days,
    resolve_adjustment,
)
from asset_loader import load_asset_data
from asset_resolver import (
    CHINA_EQUITY_ASSET_TYPES,
    format_diagnosis_user_message,
    load_china_equity_ohlcv,
    resolve_analysis_date_window,
)
from contracts import AnalysisRequest
from currency import convert_ohlcv_to_base_currency
from data_fetcher import DataFetchError, apply_default_network_proxy_policy
from feature_engineering import OHLCVFeatureEngineer
from ohlcv import REQUIRED_OHLCV_COLUMNS
from reporting.console import build_diagnostic_report as build_diagnostic_report
from services.analysis_service import analyze_single_asset
from visualizer import MplfinanceVisualizer

LOOKBACK_YEARS = 3
_DEFAULT_CHINA_EQUITY_LOADER = load_china_equity_ohlcv
_convert_ohlcv_to_base_currency = convert_ohlcv_to_base_currency


def _export_ohlcv_csv(df: pd.DataFrame, export_path: str) -> Path:
    """导出标准化 UTF-8-SIG OHLCV CSV 文件。"""
    target = Path(export_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    subset = df.loc[:, list(REQUIRED_OHLCV_COLUMNS)].sort_index().copy()
    if not isinstance(subset.index, pd.DatetimeIndex):
        subset.index = pd.to_datetime(subset.index, errors="coerce")
    subset.index.name = "Date"
    with FileTransaction() as txn:
        txn.stage(
            target,
            lambda tmp: subset.to_csv(
                tmp, index=True, date_format="%Y-%m-%d", encoding="utf-8-sig"
            ),
        )
        txn.commit()
    return target


def run_analysis(
    asset_type: str = "etf",
    symbol: str | None = None,
    start_date: object | None = None,
    end_date: object | None = None,
    *,
    adjust: str | None = None,
    proxy_url: str | None = None,
    base_currency: str | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
    save_chart: str | None = None,
    export_csv: str | None = None,
    show: bool = True,
    return_figure: bool = True,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Run the end-to-end quantitative analysis workflow across all supported asset types."""
    apply_default_network_proxy_policy()
    asset_key = str(asset_type).strip().lower()
    if asset_key not in SUPPORTED_ASSET_TYPES:
        raise ValueError(
            "Unsupported asset_type: %r. Choose from: %s"
            % (asset_type, ", ".join(SUPPORTED_ASSET_TYPES))
        )

    sym = (
        str(symbol).strip()
        if symbol is not None and str(symbol).strip()
        else DEFAULT_SYMBOLS[asset_key]
    )
    resolved_adjust = resolve_adjustment(asset_key, adjust)
    start_ts, end_ts = resolve_analysis_date_window(
        start_date,
        end_date,
        lookback_years=LOOKBACK_YEARS,
    )

    if asset_key in CHINA_EQUITY_ASSET_TYPES and (
        load_china_equity_ohlcv is not _DEFAULT_CHINA_EQUITY_LOADER
    ):

        def _active_loader(
            req_asset: str,
            req_sym: str,
            req_start: object,
            req_end: object,
            **kwargs: Any,
        ) -> pd.DataFrame:
            return load_china_equity_ohlcv(
                req_asset,
                req_sym,
                req_start,
                req_end,
                adjust=resolved_adjust,
            )

    else:

        def _active_loader(
            req_asset: str,
            req_sym: str,
            req_start: object,
            req_end: object,
            **kwargs: Any,
        ) -> pd.DataFrame:
            call_kwargs: dict[str, Any] = {
                "proxy_url": kwargs.get("proxy_url"),
                "adjust": (
                    resolved_adjust
                    if (adjust is not None or asset_key in CHINA_EQUITY_ASSET_TYPES)
                    else None
                ),
            }
            if kwargs.get("force_refresh"):
                call_kwargs["force_refresh"] = True
            return load_asset_data(
                req_asset,
                req_sym,
                req_start,
                req_end,
                **call_kwargs,
            )

    request = AnalysisRequest(
        asset_type=asset_key,
        symbol=sym,
        start_date=start_ts.strftime("%Y-%m-%d"),
        end_date=end_ts.strftime("%Y-%m-%d"),
        adjust=resolved_adjust,
        base_currency=(base_currency or "LOCAL"),
        risk_free_rate=risk_free_rate,
        trading_days=get_asset_trading_days(asset_key),
        force_refresh=force_refresh,
        proxy_url=proxy_url,
    )
    analysis_result = analyze_single_asset(request, loader=_active_loader)
    featured_df = analysis_result.data_frame
    assert featured_df is not None
    raw_df = featured_df.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()
    raw_df.attrs = dict(featured_df.attrs)

    service_metrics: dict[str, Any] = {
        **analysis_result.metrics,
        "featured_df": featured_df,
        "series_kind": featured_df.attrs.get("series_kind", analysis_result.asset.series_kind),
        "metric_reasons": dict(analysis_result.metric_reasons),
        "data_quality": analysis_result.quality.to_dict(),
        "data_source": analysis_result.asset.data_source or "",
        "currency": analysis_result.asset.output_currency or "",
        "return_basis": analysis_result.asset.return_basis,
        "metric_disclaimer": analysis_result.disclaimers[-1]
        if (
            analysis_result.disclaimers
            and analysis_result.disclaimers[-1]
            != ASSET_RETURN_BASIS_NOTES.get(analysis_result.asset.return_basis)
        )
        else "",
    }

    engineer = OHLCVFeatureEngineer()
    figure = None
    if return_figure or show or save_chart:
        figure = MplfinanceVisualizer().plot(featured_df, show=show, savepath=save_chart)
    csv_path = _export_ohlcv_csv(raw_df, export_csv) if export_csv is not None else None
    report = build_diagnostic_report(
        symbol=sym,
        df=featured_df,
        engineer=engineer,
        risk_free_rate=risk_free_rate,
        metrics=service_metrics,
    )
    print(report)

    return {
        "raw_df": raw_df,
        "featured_df": featured_df,
        "figure": figure,
        "report": report,
        "saved_csv": str(csv_path) if csv_path is not None else None,
        "analysis_result": analysis_result,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="全资产定量分析 CLI：支持 13 类资产的数据抓取、口径切换、指标诊断与图表导出。",
    )
    parser.add_argument(
        "--asset-type",
        choices=SUPPORTED_ASSET_TYPES,
        default="etf",
        metavar="TYPE",
        help=f"资产类型（默认 etf）；可选: {', '.join(SUPPORTED_ASSET_TYPES)}",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="标的代码（缺省使用对应资产类别默认示例，如 510300 / AAPL / GC / BTC/USDT）",
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
    parser.add_argument(
        "--adjust",
        default=None,
        help="历史行情复权口径（如 hfq / qfq / 空字符串 '' 表示不复权；按资产接口实际支持校验）",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="网络代理地址（如 http://127.0.0.1:7890），用于加密货币或跨市场数据源",
    )
    parser.add_argument(
        "--base-currency",
        choices=("CNY", "USD"),
        default=None,
        help="可选单标的计价折算目标货币（CNY 或 USD）",
    )
    parser.add_argument(
        "--save-chart",
        default=None,
        help="可选图表输出路径（如 outputs/chart.png）",
    )
    parser.add_argument(
        "--export-csv",
        default=None,
        help="可选标准化 OHLCV CSV 导出路径（如 outputs/ohlcv.csv）",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="仅生成/保存图表而不阻塞弹窗显示（适合无头脚本与批处理）",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="忽略本地持久缓存，强制重新拉取最新行情并刷新缓存",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _parse_args(argv)
    try:
        run_analysis(
            asset_type=args.asset_type,
            symbol=args.symbol,
            start_date=args.start,
            end_date=args.end,
            adjust=args.adjust,
            proxy_url=args.proxy,
            base_currency=args.base_currency,
            save_chart=args.save_chart,
            export_csv=args.export_csv,
            show=not args.no_show,
            return_figure=False,
            force_refresh=args.refresh,
        )
        return 0
    except (ImportError, DataFetchError, ValueError, RuntimeError, TypeError) as exc:
        print(format_diagnosis_user_message(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
