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
from data_fetcher import DataFetchError, apply_default_network_proxy_policy
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


def render_app() -> None:
    """渲染 Streamlit 量化诊断应用。"""
    streamlit = _require_streamlit()
    apply_default_network_proxy_policy()
    streamlit.set_page_config(page_title="量化诊断应用", layout="wide")
    streamlit.title("量化诊断应用")
    streamlit.caption(
        "支持 A 股、公募、贵金属、加密、美股/港股/全球指数、国债与美债收益率、外盘商品、REITs、外汇等；"
        "统一 OHLCV 契约下的诊断与图表（收益率水平类资产见看板说明）。"
    )

    today = pd.Timestamp.today().normalize().date()
    default_start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=LOOKBACK_YEARS)).date()
    streamlit.sidebar.header("参数设置")
    # 侧边栏只收集“资产类型、代码、日期、代理”这些最小必要输入。
    asset_type = streamlit.sidebar.selectbox(
        "资产类型",
        options=list(SUPPORTED_ASSET_TYPES),
        format_func=lambda key: ASSET_LABELS[key],
        key="asset_type",
    )
    _sync_sidebar_state(streamlit, asset_type)
    code_label = "代码"
    if asset_type == "bond_us_yield":
        code_label = "收益率列（默认 10Y=美债10年，可填表中完整列名）"
    symbol = streamlit.sidebar.text_input(code_label, key="symbol_input").strip()
    start_date = streamlit.sidebar.date_input(
        "开始日期",
        value=default_start,
    )
    end_date = streamlit.sidebar.date_input(
        "结束日期",
        value=today,
    )

    proxy_url: str | None = None
    if asset_type == "crypto":
        proxy_url = streamlit.sidebar.text_input(
            "代理地址",
            key="proxy_url_input",
            placeholder="http://127.0.0.1:33210",
        )
        streamlit.sidebar.caption("代理仅在加密货币模式下生效。")

    with streamlit.sidebar.expander("资产类型说明", expanded=False):
        streamlit.markdown(format_asset_guide_markdown(asset_type))

    clicked = streamlit.sidebar.button("开始诊断", use_container_width=True)

    tab1, tab2, tab3 = streamlit.tabs(["📊 量化诊断看板", "📚 系统数学原理解析", "🧮 进阶金融算法"])

    with tab2:
        streamlit.markdown(_TAB2_SYSTEM_MATH_MARKDOWN)

    with tab3:
        streamlit.markdown(
            "本页使用**左侧**资产类型、代码与起止日期；各区块**独立按钮**拉数与计算，无需先点击「开始诊断」。"
        )
        streamlit.caption(
            "有效前沿（模块 B）支持在下方按行配置多标的（类型与侧栏「资产类型」一致）；"
            "不同市场日历 **outer 对齐 + 前向填充** 后与 ETF 批量路径语义相同。"
            "混合资产时重叠交易日可能变短；夏普等横比需谨慎（参见侧栏各类资产说明）。"
            "GBM 与归一化仍用侧栏单一标的。"
        )

        streamlit.subheader("0 · 基准归一化")
        streamlit.markdown(
            r"将收盘价序列归一化为以样本首日价格为 1 的净值：$N_t = P_t / P_{t_0}$（$P$ 为 `Close`）。"
        )
        if streamlit.button(
            "显示归一化净值曲线", key="tab3_btn_normalize", use_container_width=True
        ):
            try:
                with streamlit.spinner("正在加载行情…"):
                    # 进阶算法页也复用同一条 load_asset_data 数据入口，避免口径不一致。
                    df_norm = load_asset_data(
                        asset_type=asset_type,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        proxy_url=proxy_url,
                    )
                if df_norm.empty:
                    streamlit.error(EMPTY_SYMBOL_FETCH_HINT)
                else:
                    series_n = normalize_close_to_base_one(df_norm)
                    fig_n = build_tab3_normalize_figure(df_norm.index, series_n)
                    streamlit.plotly_chart(fig_n, use_container_width=True)
            except (ImportError, DataFetchError, RuntimeError, TypeError, ValueError) as exc:
                streamlit.error(format_diagnosis_user_message(exc))

        streamlit.subheader("A · GBM 蒙特卡洛")
        streamlit.markdown(
            "基于历史日对数收益估计波动与漂移，按几何布朗运动离散递推未来 **252** 个交易日路径；"
            "图中为分位数带与若干示例路径。"
        )
        if streamlit.button("运行 GBM 模拟", key="tab3_btn_gbm", use_container_width=True):
            try:
                with streamlit.spinner("正在加载数据并模拟路径…"):
                    # GBM 使用历史 Close 估计参数，输出的是模拟分布而不是确定预测。
                    df_gbm = load_asset_data(
                        asset_type=asset_type,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        proxy_url=proxy_url,
                    )
                if df_gbm.empty:
                    streamlit.error(EMPTY_SYMBOL_FETCH_HINT)
                else:
                    mc = run_monte_carlo_gbm(
                        df_gbm,
                        days=TRADING_DAYS,
                        num_simulations=500,
                        random_state=42,
                        num_sample_paths=5,
                    )
                    streamlit.caption(
                        "估计参数：日漂移 μ≈%.6f，日波动 σ≈%.6f，起点 S₀=%.4f。"
                        % (mc.mu_daily, mc.sigma_daily, mc.s0)
                    )
                    fig_g = build_tab3_gbm_figure(mc)
                    streamlit.plotly_chart(fig_g, use_container_width=True)
            except (ImportError, DataFetchError, RuntimeError, TypeError, ValueError) as exc:
                streamlit.error(format_diagnosis_user_message(exc))

        streamlit.subheader("B · 有效前沿（随机权重）")
        streamlit.markdown(
            "每行一条：`资产类型,代码` 或 `资产类型|代码`（与侧栏类型键一致，如 `etf`、`us_stock`、`crypto`）；"
            "同一行可用英文分号 `;` 写多条。列名为 `类型:代码`（如 `crypto:BTC/USDT`）。"
            "若含 **crypto**，侧栏代理与「加密货币」模式相同（请将侧栏资产类型选为加密货币以填写代理）。"
        )
        streamlit.markdown(
            "**资产类型与代码对照**（输入格式：`类型键,代码`；下列默认代码与侧栏「代码」默认值一致）"
        )
        streamlit.dataframe(
            build_ef_multi_asset_codes_reference_df(),
            use_container_width=True,
            hide_index=True,
            height=min(520, 40 + len(SUPPORTED_ASSET_TYPES) * 36),
        )
        ef_default = "\n".join(
            (
                "etf,%s" % DEFAULT_SYMBOLS.get("etf", "510300"),
                "etf,510500",
                "us_stock,%s" % DEFAULT_SYMBOLS.get("us_stock", "AAPL"),
            )
        )
        ef_legs_text = streamlit.text_area(
            "多标的组合（至少 2 条，不重复）",
            value=ef_default,
            height=120,
            key="tab3_ef_legs_input",
        )
        if streamlit.button("运行协方差矩阵优化", key="tab3_btn_ef", use_container_width=True):
            try:
                legs = parse_multi_asset_legs(ef_legs_text)
                with streamlit.spinner("正在拉取多标的收盘价并模拟组合…"):
                    # 有效前沿需要多资产 Close 宽表，每列是一只资产的价格序列。
                    wide_close = load_multi_leg_close_wide(
                        legs,
                        start_date,
                        end_date,
                        proxy_url=proxy_url,
                    )
                returns_wide = wide_close.pct_change(fill_method=None).dropna()
                if returns_wide.shape[1] < 2:
                    streamlit.error("有效资产列不足，请检查标的与日期区间。")
                else:
                    streamlit.caption(
                        "联合样本中日收益需各列同时非缺失；重叠过少或序列近似线性相关时优化可能失败。"
                    )
                    r_joint = clean_joint_returns(returns_wide)
                    corr_df = r_joint.corr(method="pearson")
                    fig_corr = build_tab3_correlation_heatmap(corr_df)
                    streamlit.plotly_chart(fig_corr, use_container_width=True)
                    ef_res: EfficientFrontierResult = generate_efficient_frontier(
                        r_joint,
                        num_portfolios=5000,
                        risk_free_rate=RISK_FREE_RATE,
                        trading_days=TRADING_DAYS,
                        random_state=42,
                    )
                    fig_ef = build_tab3_efficient_frontier_figure(ef_res)
                    streamlit.plotly_chart(fig_ef, use_container_width=True)
                    w_best = ef_res.weights[ef_res.best_sharpe_idx]
                    weights_df = pd.DataFrame(
                        [w_best],
                        columns=list(r_joint.columns),
                    )
                    streamlit.markdown("**最大夏普比率组合的权重**")
                    streamlit.dataframe(weights_df, use_container_width=True)
                    streamlit.markdown("")
                    streamlit.markdown("**净值回测曲线**")
                    streamlit.caption(
                        "上图权重对应红色曲线；灰色为等权基准；与热力图、前沿共用同一联合收益样本。"
                    )
                    bt_df = backtest_portfolio(r_joint, w_best)
                    fig_bt = build_tab3_portfolio_backtest_figure(bt_df)
                    streamlit.plotly_chart(fig_bt, use_container_width=True)
            except (ImportError, DataFetchError, RuntimeError, TypeError, ValueError) as exc:
                streamlit.error(format_diagnosis_user_message(exc))

    with tab1:
        if not clicked:
            streamlit.info(
                "在左侧选择资产和时间区间后点击“开始诊断”，即可查看 Web 版量化分析结果。"
            )
            with streamlit.expander("资产类型说明（与左侧「资产类型」联动）", expanded=False):
                streamlit.markdown(format_asset_guide_markdown(asset_type))
        else:
            try:
                with streamlit.spinner("正在加载数据并计算量化指标..."):
                    # 主看板流程：拉数 -> 特征工程/指标 -> 图表；异常统一转成用户文案。
                    df = load_asset_data(
                        asset_type=asset_type,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        proxy_url=proxy_url,
                    )
                    if df.empty:
                        streamlit.error(EMPTY_SYMBOL_FETCH_HINT)
                        return
                    metrics = get_quant_metrics(df)
                    figure = build_price_volume_figure(metrics["featured_df"])
            except (ImportError, DataFetchError, RuntimeError, TypeError, ValueError) as exc:
                streamlit.error(format_diagnosis_user_message(exc))
                return

            display_symbol = str(df.attrs.get("symbol", symbol)).strip() or symbol
            streamlit.subheader("%s · %s" % (ASSET_LABELS[asset_type], display_symbol))
            unit_cap = str(df.attrs.get("unit_caption", "") or "").strip()
            if unit_cap:
                streamlit.caption("单位与口径：%s" % unit_cap)
            streamlit.caption(
                "样本区间：%s 至 %s，共 %d 条记录。"
                % (
                    df.index[0].strftime("%Y-%m-%d"),
                    df.index[-1].strftime("%Y-%m-%d"),
                    len(df),
                )
            )
            if metrics.get("metric_disclaimer"):
                streamlit.warning(metrics["metric_disclaimer"])
            csv_bytes, csv_filename = _cleaned_ohlcv_to_csv_bytes_and_filename(
                df,
                asset_type=asset_type,
                display_symbol=display_symbol,
                start_date=start_date,
                end_date=end_date,
            )
            # 下载的是清洗后原始 OHLCV，便于用户自己复现看板指标。
            streamlit.download_button(
                label="下载清洗后 OHLCV（CSV）",
                data=csv_bytes,
                file_name=csv_filename,
                mime="text/csv",
                use_container_width=False,
                key="download_cleaned_ohlcv_csv",
            )
            streamlit.caption(
                "CSV 含 Date 与 Open、High、Low、Close、Volume 五列，与看板清洗后数据一致（不含均线、布林带等特征列）。"
            )

            metric_col_1, metric_col_2, metric_col_3 = streamlit.columns(3)
            metric_col_1.metric(metrics["cagr_label"], _format_percent_metric(metrics["cagr"]))
            metric_col_2.metric(
                "最大回撤", _format_percent_metric(-abs(metrics["max_drawdown"]), signed=True)
            )
            metric_col_3.metric("Sharpe", _format_ratio_metric(metrics["sharpe_ratio"]))

            streamlit.info(
                "最新信号：%s | %s：%s"
                % (
                    metrics["latest_signal"],
                    metrics["cumulative_return_label"],
                    _format_percent_metric(metrics["cumulative_return"], signed=True),
                )
            )
            streamlit.plotly_chart(figure, use_container_width=True)

            with streamlit.expander("查看清洗后与特征后数据", expanded=False):
                streamlit.markdown("**清洗后 OHLCV 数据尾部**")
                streamlit.dataframe(df.tail(20), use_container_width=True)
                streamlit.markdown("**特征工程结果尾部**")
                streamlit.dataframe(metrics["featured_df"].tail(20), use_container_width=True)
                streamlit.caption("标准列: %s" % ", ".join(list(REQUIRED_OHLCV_COLUMNS)))


if __name__ == "__main__":
    render_app()
