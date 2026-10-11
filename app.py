"""Streamlit entry point: widgets, caching and display; calculations live in library modules."""

from __future__ import annotations

import importlib
import io
import re
from typing import Any, Callable

import pandas as pd

from asset_catalog import (
    SUPPORTED_ASSET_TYPES,
    ASSET_LABELS,
    DEFAULT_SYMBOLS,
    ASSET_TYPE_GUIDES,
    format_asset_guide_markdown,
    build_ef_multi_asset_codes_reference_df,
    _TAB2_SYSTEM_MATH_MARKDOWN,
)
from asset_loader import (
    GOLD_SYMBOL_ALIASES,
    load_asset_data as _load_asset_data,
    load_multiple_etfs_close as _load_multiple_etfs_close,
    load_multi_leg_close_wide as _load_multi_leg_close_wide,
    parse_multi_asset_legs,
    multi_asset_leg_column_name,
)
from asset_resolver import EMPTY_SYMBOL_FETCH_HINT, format_diagnosis_user_message
from data_fetcher import DataFetchError, EmptySymbolDataError, apply_default_network_proxy_policy
from ohlcv import REQUIRED_OHLCV_COLUMNS
from analysis_metrics import (
    RISK_FREE_RATE,
    TRADING_DAYS,
    get_quant_metrics,
    compute_cumulative_return as _compute_cumulative_return,
    compute_cagr as _compute_cagr,
    compute_sharpe_ratio as _compute_sharpe_ratio,
    describe_bollinger_position as _describe_bollinger_position,
    format_percent_metric as _format_percent_metric,
    format_ratio_metric as _format_ratio_metric,
)
from charts import (
    build_price_volume_figure,
    build_tab3_normalize_figure,
    build_tab3_gbm_figure,
    build_tab3_efficient_frontier_figure,
    build_tab3_correlation_heatmap,
    build_tab3_portfolio_backtest_figure,
)
from quant_models import (
    EfficientFrontierResult,
    backtest_portfolio,
    clean_joint_returns,
    generate_efficient_frontier,
    normalize_close_to_base_one,
    run_monte_carlo_gbm,
)

LOOKBACK_YEARS = 3


def _identity_cache_data(
    *args: Any, **kwargs: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """在未安装 Streamlit 时提供无副作用的缓存装饰器占位。"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator


try:
    st = importlib.import_module("streamlit")
except ImportError:
    st = None


cache_data = st.cache_data if st is not None else _identity_cache_data


def _require_streamlit() -> Any:
    """在真正渲染页面前检查 Streamlit 依赖。"""
    if st is None:
        raise ImportError(
            "运行 `app.py` 需要安装 streamlit。请先执行 `pip install streamlit plotly akshare ccxt`。"
        )
    return st


# Cache at the UI boundary, leaving all library modules usable offline.
load_asset_data = cache_data(ttl=3600, show_spinner=False)(_load_asset_data)
load_multiple_etfs_close = cache_data(ttl=3600, show_spinner=False)(_load_multiple_etfs_close)


@cache_data(ttl=3600, show_spinner=False)
def load_multi_leg_close_wide(
    legs_tuple: tuple[tuple[str, str], ...],
    start_date: object,
    end_date: object,
    proxy_url: str | None = None,
) -> pd.DataFrame:
    """Keep each leg's single-asset cache shared with the diagnostic dashboard."""
    return _load_multi_leg_close_wide(
        legs_tuple,
        start_date,
        end_date,
        proxy_url,
        loader=load_asset_data,
    )


def _sync_sidebar_state(streamlit: Any, asset_type: str) -> None:
    """让侧边栏默认值在资产类型切换时保持联动。"""
    session_state = streamlit.session_state
    last_asset_type = session_state.get("_last_asset_type")
    if "symbol_input" not in session_state:
        session_state["symbol_input"] = DEFAULT_SYMBOLS[asset_type]
    if "proxy_url_input" not in session_state:
        session_state["proxy_url_input"] = ""

    if last_asset_type != asset_type:
        session_state["symbol_input"] = DEFAULT_SYMBOLS[asset_type]
        if asset_type != "crypto":
            session_state["proxy_url_input"] = ""

    session_state["_last_asset_type"] = asset_type


_WIN_FILENAME_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _sanitize_symbol_for_filename(display_symbol: str) -> str:
    """净化代码片段，避免 Windows 非法字符并将路径分隔符替换为下划线。"""
    text = str(display_symbol).strip() or "symbol"
    text = text.replace("/", "_")
    text = _WIN_FILENAME_FORBIDDEN.sub("_", text)
    text = text.strip(" .")
    return text or "symbol"


def _cleaned_ohlcv_to_csv_bytes_and_filename(
    df: pd.DataFrame,
    asset_type: str,
    display_symbol: str,
    start_date: object,
    end_date: object,
) -> tuple[bytes, str]:
    """将 load_asset_data 返回的清洗后 OHLCV 导出为 UTF-8-SIG CSV 字节与安全文件名。"""
    subset = df.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()
    subset = subset.sort_index()
    if not isinstance(subset.index, pd.DatetimeIndex):
        subset.index = pd.to_datetime(subset.index, errors="coerce")
    subset.index.name = "Date"
    text_buffer = io.StringIO()
    subset.to_csv(text_buffer, index=True, date_format="%Y-%m-%d")
    csv_bytes = text_buffer.getvalue().encode("utf-8-sig")
    at_key = str(asset_type).strip().lower()
    sym_part = _sanitize_symbol_for_filename(display_symbol)
    at_part = _sanitize_symbol_for_filename(at_key)
    start_s = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    filename = "%s_%s_%s_%s_ohlcv.csv" % (at_part, sym_part, start_s, end_s)
    return csv_bytes, filename


def _analysis_result(
    streamlit: Any,
    name: str,
    parameters: tuple,
    requested: bool,
    compute: Callable[[], Any],
    spinner: str,
) -> Any:
    """Retain one result per action and discard it when its inputs change."""
    results = streamlit.session_state.setdefault("_analysis_results", {})
    saved = results.get(name, {})
    if saved.get("parameters") != parameters:
        saved = {}
        results.pop(name, None)
    if requested:
        saved = {"parameters": parameters}
        try:
            with streamlit.spinner(spinner):
                saved["value"] = compute()
        except (ImportError, DataFetchError, RuntimeError, TypeError, ValueError) as exc:
            saved["error"] = format_diagnosis_user_message(exc)
        results[name] = saved
    if "error" in saved:
        streamlit.error(saved["error"])
    return saved.get("value")


def _plot_chart(streamlit: Any, builder: Callable[..., Any], *args: Any) -> None:
    """Keep chart dependency/validation errors in the same user-facing error format."""
    try:
        streamlit.plotly_chart(builder(*args), use_container_width=True)
    except (ImportError, TypeError, ValueError) as exc:
        streamlit.error(format_diagnosis_user_message(exc))


def _render_diagnosis(streamlit: Any, df: pd.DataFrame, metrics: dict, query: tuple) -> None:
    asset_type, symbol, start_date, end_date, _ = query
    display_symbol = str(df.attrs.get("symbol", symbol)).strip() or symbol
    streamlit.subheader("%s · %s" % (ASSET_LABELS[asset_type], display_symbol))
    streamlit.caption(
        "样本区间：%s 至 %s · %d 条记录"
        % (df.index[0].strftime("%Y-%m-%d"), df.index[-1].strftime("%Y-%m-%d"), len(df))
    )
    unit_cap = str(df.attrs.get("unit_caption", "") or "").strip()
    if unit_cap:
        streamlit.caption("单位与口径：%s" % unit_cap)
    if metrics.get("metric_disclaimer"):
        streamlit.warning(metrics["metric_disclaimer"])

    columns = streamlit.columns(3)
    columns[0].metric(metrics["cagr_label"], _format_percent_metric(metrics["cagr"]))
    columns[1].metric(
        "最大回撤", _format_percent_metric(-abs(metrics["max_drawdown"]), signed=True)
    )
    columns[2].metric("Sharpe", _format_ratio_metric(metrics["sharpe_ratio"]))
    streamlit.info(
        "最新信号：%s | %s：%s"
        % (
            metrics["latest_signal"],
            metrics["cumulative_return_label"],
            _format_percent_metric(metrics["cumulative_return"], signed=True),
        )
    )
    _plot_chart(streamlit, build_price_volume_figure, metrics["featured_df"])
    csv_bytes, csv_filename = _cleaned_ohlcv_to_csv_bytes_and_filename(
        df, asset_type, display_symbol, start_date, end_date
    )
    streamlit.download_button(
        "下载清洗后 OHLCV（CSV）",
        data=csv_bytes,
        file_name=csv_filename,
        mime="text/csv",
        key="download_cleaned_ohlcv_csv",
        help="包含 Date、Open、High、Low、Close、Volume；可用于复现看板指标。",
    )
    with streamlit.expander("查看清洗后与特征后数据", expanded=False):
        streamlit.markdown("**清洗后 OHLCV 数据尾部**")
        streamlit.dataframe(df.tail(20), width="stretch")
        streamlit.markdown("**特征工程结果尾部**")
        streamlit.dataframe(metrics["featured_df"].tail(20), width="stretch")
        streamlit.caption("标准列: %s" % ", ".join(list(REQUIRED_OHLCV_COLUMNS)))


def _render_advanced(
    streamlit: Any,
    query: tuple,
    load_single: Callable[[], pd.DataFrame],
    single_valid: bool,
    dates_valid: bool,
) -> None:
    _, _, start_date, end_date, proxy_url = query
    streamlit.caption("单标的分析使用参数栏；组合分析可独立配置标的。计算结果保留至对应参数变更。")
    with streamlit.expander("基准归一化", expanded=True):
        streamlit.markdown(r"收盘价除以样本首日价格，得到期初为 1 的净值：$N_t = P_t / P_0$。")
        requested = streamlit.button(
            "显示归一化净值曲线", key="tab3_btn_normalize", disabled=not single_valid
        )

        def normalize():
            df = load_single()
            return normalize_close_to_base_one(df)

        normalized = _analysis_result(
            streamlit, "normalize", query, requested, normalize, "正在加载行情…"
        )
        if normalized is not None:
            _plot_chart(streamlit, build_tab3_normalize_figure, normalized.index, normalized)

    with streamlit.expander("GBM 蒙特卡洛", expanded=True):
        streamlit.markdown(
            "根据历史日对数收益估计漂移与波动，模拟未来 **252** 个交易日。"
            "分位数带与示例路径展示模拟分布。"
        )
        requested = streamlit.button("运行 GBM 模拟", key="tab3_btn_gbm", disabled=not single_valid)
        mc = _analysis_result(
            streamlit,
            "gbm",
            query,
            requested,
            lambda: run_monte_carlo_gbm(
                load_single(),
                days=TRADING_DAYS,
                num_simulations=500,
                random_state=42,
                num_sample_paths=5,
            ),
            "正在加载数据并模拟路径…",
        )
        if mc is not None:
            streamlit.caption(
                "日漂移 μ≈%.6f · 日波动 σ≈%.6f · 起点 S₀=%.4f"
                % (mc.mu_daily, mc.sigma_daily, mc.s0)
            )
            _plot_chart(streamlit, build_tab3_gbm_figure, mc)

    streamlit.subheader("组合有效前沿")
    streamlit.markdown(
        "每行输入 `资产类型,代码`，至少两个不同标的。"
        "也支持 `类型|代码` 或用英文分号分隔；示例：`etf,510300`。"
    )
    ef_default = "\n".join(("etf,510300", "etf,510500", "us_stock,AAPL"))
    ef_legs_text = streamlit.text_area(
        "多标的组合（至少 2 条，不重复）",
        value=ef_default,
        height=120,
        key="tab3_ef_legs_input",
    )
    requested = streamlit.button("运行协方差矩阵优化", key="tab3_btn_ef", disabled=not dates_valid)
    with streamlit.expander("资产类型与代码对照", expanded=False):
        streamlit.dataframe(
            build_ef_multi_asset_codes_reference_df(),
            width="stretch",
            hide_index=True,
            height=360,
        )
    streamlit.caption(
        "组合使用参数栏的起止日期。混合市场按共同有效收益样本计算；"
        "如含加密货币，可在参数栏切换到加密货币后填写代理。"
    )

    def optimize():
        legs = parse_multi_asset_legs(ef_legs_text)
        wide_close = load_multi_leg_close_wide(legs, start_date, end_date, proxy_url=proxy_url)
        returns = clean_joint_returns(wide_close.pct_change(fill_method=None).dropna())
        if returns.shape[1] < 2:
            raise ValueError("有效资产列不足，请检查标的与日期区间。")
        ef = generate_efficient_frontier(
            returns,
            num_portfolios=5000,
            risk_free_rate=RISK_FREE_RATE,
            trading_days=TRADING_DAYS,
            random_state=42,
        )
        weights = ef.weights[ef.best_sharpe_idx]
        return (
            returns.corr(),
            ef,
            pd.DataFrame([weights], columns=returns.columns),
            backtest_portfolio(returns, weights),
        )

    result = _analysis_result(
        streamlit,
        "frontier",
        (start_date, end_date, proxy_url, ef_legs_text.strip()),
        requested,
        optimize,
        "正在拉取多标的收盘价并模拟组合…",
    )
    if result is not None:
        corr, ef, weights, backtest = result
        _plot_chart(streamlit, build_tab3_correlation_heatmap, corr)
        with streamlit.expander("查看相关系数数据"):
            streamlit.dataframe(corr.style.format("{:.3f}"), width="stretch")
        _plot_chart(streamlit, build_tab3_efficient_frontier_figure, ef)
        streamlit.markdown("**最大夏普比率组合的权重**")
        streamlit.dataframe(weights.style.format("{:.2%}"), width="stretch", hide_index=True)
        streamlit.markdown("**净值回测曲线**")
        streamlit.caption("最优与等权组合共用同一联合收益样本。")
        _plot_chart(streamlit, build_tab3_portfolio_backtest_figure, backtest)


def render_app() -> None:
    """渲染 Streamlit 量化诊断应用。"""
    streamlit = _require_streamlit()
    apply_default_network_proxy_policy()
    streamlit.set_page_config(page_title="量化诊断应用", layout="wide")
    streamlit.markdown(
        """<style>
        .block-container {padding-top: 3.5rem; padding-bottom: 2rem;}
        [data-testid="stAppViewContainer"] h1 {font-size: clamp(1.65rem, 3vw, 2.25rem);}
        </style>""",
        unsafe_allow_html=True,
    )
    streamlit.title("量化诊断应用")
    streamlit.caption("覆盖 13 类资产的历史行情、风险收益诊断与组合分析。")

    today = pd.Timestamp.today().normalize().date()
    default_start = (pd.Timestamp(today) - pd.DateOffset(years=LOOKBACK_YEARS)).date()
    streamlit.sidebar.header("参数设置")
    asset_type = streamlit.sidebar.selectbox(
        "资产类型",
        options=list(SUPPORTED_ASSET_TYPES),
        format_func=lambda key: ASSET_LABELS[key],
        key="asset_type",
    )
    _sync_sidebar_state(streamlit, asset_type)
    code_label = "收益率列（留空默认 10Y）" if asset_type == "bond_us_yield" else "代码"
    symbol = streamlit.sidebar.text_input(code_label, key="symbol_input").strip()
    start_date = streamlit.sidebar.date_input(
        "开始日期", value=default_start, key="start_date_input"
    )
    end_date = streamlit.sidebar.date_input("结束日期", value=today, key="end_date_input")
    proxy_url = None
    if asset_type == "crypto":
        proxy_url = streamlit.sidebar.text_input(
            "代理地址", key="proxy_url_input", placeholder="http://127.0.0.1:33210"
        )
        streamlit.sidebar.caption("代理仅用于加密货币数据。")

    dates_present = start_date is not None and end_date is not None
    dates_valid = dates_present and start_date <= end_date
    single_valid = dates_valid and (bool(symbol) or asset_type == "bond_us_yield")
    if not dates_present:
        streamlit.sidebar.error("请选择开始和结束日期。")
    elif not dates_valid:
        streamlit.sidebar.error("开始日期不能晚于结束日期。")
    if not symbol and asset_type != "bond_us_yield":
        streamlit.sidebar.error("请输入资产代码。")
    with streamlit.sidebar.expander("资产类型说明", expanded=False):
        streamlit.markdown(format_asset_guide_markdown(asset_type))
    clicked = streamlit.sidebar.button(
        "开始诊断", width="stretch", type="primary", disabled=not single_valid
    )
    query = (asset_type, symbol, start_date, end_date, proxy_url)

    def load_single():
        df = load_asset_data(
            asset_type=asset_type,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            proxy_url=proxy_url,
        )
        if df.empty:
            raise EmptySymbolDataError(EMPTY_SYMBOL_FETCH_HINT)
        return df

    tab1, tab2, tab3 = streamlit.tabs(["量化诊断看板", "数学原理", "进阶金融算法"])
    with tab1:

        def diagnose():
            df = load_single()
            return df, get_quant_metrics(df)

        result = _analysis_result(
            streamlit, "diagnosis", query, clicked, diagnose, "正在加载数据并计算量化指标…"
        )
        if result is None:
            streamlit.info("在参数栏选择资产和日期，点击「开始诊断」查看指标与行情图。")
            streamlit.caption(
                "当前选择：%s · %s · %s 至 %s"
                % (ASSET_LABELS[asset_type], symbol or "10Y", start_date, end_date)
            )
        else:
            _render_diagnosis(streamlit, *result, query)
    with tab2:
        streamlit.markdown(_TAB2_SYSTEM_MATH_MARKDOWN)
    with tab3:
        _render_advanced(streamlit, query, load_single, single_valid, dates_valid)


if __name__ == "__main__":
    render_app()
