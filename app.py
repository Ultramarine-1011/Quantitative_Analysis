"""Streamlit entry point: widgets, caching and display; calculations live in library modules."""

from __future__ import annotations

import importlib
import io
import re
from collections.abc import Callable
from typing import Any

import pandas as pd

from analysis_metrics import (
    RISK_FREE_RATE,
    TRADING_DAYS,
    compute_cagr as _compute_cagr,
    compute_cumulative_return as _compute_cumulative_return,
    compute_sharpe_ratio as _compute_sharpe_ratio,
    describe_bollinger_position as _describe_bollinger_position,
    format_percent_metric as _format_percent_metric,
    format_ratio_metric as _format_ratio_metric,
    get_quant_metrics,
)
from asset_catalog import (
    _TAB2_SYSTEM_MATH_MARKDOWN,
    ADJUSTMENT_LABELS,
    ASSET_LABELS,
    ASSET_RETURN_BASIS_NOTES,
    DEFAULT_ADJUSTMENT,
    DEFAULT_SYMBOLS,
    PRESET_PORTFOLIOS,
    SUPPORTED_ASSET_TYPES,
    build_ef_multi_asset_codes_reference_df,
    format_asset_guide_markdown,
    get_supported_adjustments,
)
from asset_loader import (
    GOLD_SYMBOL_ALIASES,
    load_and_align_multi_asset_portfolio as _load_and_align_multi_asset_portfolio,
    load_asset_data as _load_asset_data,
    load_multi_leg_close_wide as _load_multi_leg_close_wide,
    load_multiple_etfs_close as _load_multiple_etfs_close,
    multi_asset_leg_column_name,
    parse_multi_asset_legs,
)
from asset_resolver import EMPTY_SYMBOL_FETCH_HINT, format_diagnosis_user_message
from charts import (
    build_price_volume_figure,
    build_tab3_correlation_heatmap,
    build_tab3_efficient_frontier_figure,
    build_tab3_gbm_figure,
    build_tab3_multi_normalize_figure,
    build_tab3_normalize_figure,
    build_tab3_portfolio_backtest_figure,
    build_tab3_walk_forward_figure,
    build_tab3_weight_history_figure,
)
from currency import SUPPORTED_BASE_CURRENCIES
from data_fetcher import DataFetchError, EmptySymbolDataError, apply_default_network_proxy_policy
from ohlcv import REQUIRED_OHLCV_COLUMNS
from quant_models import (
    CovMethod,
    OptimizerObjective,
    RebalanceFreq,
    clean_joint_returns,
    normalize_close_to_base_one,
    returns_correlation_matrix,
    run_monte_carlo_gbm,
    run_walk_forward_backtest,
    solve_exact_efficient_frontier,
    split_train_test_returns,
)

LOOKBACK_YEARS = 3
_VAL_MODE_SPLIT = "样本内+样本外拆分 (Train/Test)"
_VAL_MODE_ROLLING = "滚动 Walk-Forward (Rolling OOS)"
_VAL_MODE_EXPANDING = "扩展 Walk-Forward (Expanding OOS)"
_CALENDAR_LABELS = {
    "common_trading_dates": "共同交易日交集（无零填充偏差）",
    "weekly_friday": "周五周频采样交集",
    "mtm_outer": "逐日盯市（非交易日不调仓）",
}
_COV_LABELS = {
    "ledoit_wolf": "Ledoit-Wolf 收缩估计（推荐）",
    "sample": "经典样本协方差",
    "diagonal": "对角方差矩阵（低估相关性噪音）",
}
_REBAL_LABELS = {
    "monthly": "按月再平衡",
    "quarterly": "按季再平衡",
    "daily": "每日再平衡",
    "buy_and_hold": "初始建仓后买入持有",
}
_OBJECTIVE_LABELS = {
    "max_sharpe": "最大夏普比率 (Tangency)",
    "min_variance": "全局最小方差 (GMV)",
}


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
    """保持代码与复权选项随资产类别联动，并持久化代理地址防止切换丢失。"""
    session_state = streamlit.session_state
    last_asset_type = session_state.get("_last_asset_type")
    if "symbol_input" not in session_state:
        session_state["symbol_input"] = DEFAULT_SYMBOLS[asset_type]
    if "proxy_url_input" not in session_state:
        session_state["proxy_url_input"] = ""

    supported_adj = get_supported_adjustments(asset_type)
    if last_asset_type != asset_type:
        session_state["symbol_input"] = DEFAULT_SYMBOLS[asset_type]
        if supported_adj:
            session_state["adjust_input"] = DEFAULT_ADJUSTMENT[asset_type]
    elif supported_adj and session_state.get("adjust_input") not in supported_adj:
        session_state["adjust_input"] = DEFAULT_ADJUSTMENT[asset_type]

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


def _plot_chart(streamlit: Any, builder: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """Keep chart dependency/validation errors in the same user-facing error format."""
    try:
        streamlit.plotly_chart(builder(*args, **kwargs), use_container_width=True)
    except (ImportError, TypeError, ValueError) as exc:
        streamlit.error(format_diagnosis_user_message(exc))


def _format_backtest_comparison_table(
    rows: list[tuple[str, dict[str, Any]]],
) -> pd.DataFrame:
    """整理多组策略回测绩效对比表，便于 UI 直观对比样本内与样本外。"""
    records = []
    for label, stats in rows:
        records.append(
            {
                "策略/样本阶段": label,
                "累计收益": _format_percent_metric(
                    stats.get("total_return", float("nan")), signed=True
                ),
                "年化收益 (CAGR)": _format_percent_metric(
                    stats.get("cagr", float("nan")), signed=True
                ),
                "年化波动": _format_percent_metric(
                    stats.get("annualized_volatility", float("nan"))
                ),
                "最大回撤": _format_percent_metric(
                    -abs(float(stats.get("max_drawdown", float("nan"))))
                    if pd.notna(stats.get("max_drawdown"))
                    else float("nan"),
                    signed=True,
                ),
                "Sharpe": _format_ratio_metric(stats.get("sharpe_ratio", float("nan"))),
                "Sortino": _format_ratio_metric(stats.get("sortino_ratio", float("nan"))),
                "Calmar": _format_ratio_metric(stats.get("calmar_ratio", float("nan"))),
                "总换手": _format_ratio_metric(stats.get("total_turnover", float("nan"))),
                "累计费率损耗": _format_percent_metric(stats.get("total_cost_ratio", float("nan"))),
            }
        )
    return pd.DataFrame(records)


def _render_diagnosis(streamlit: Any, df: pd.DataFrame, metrics: dict, query: tuple) -> None:
    asset_type, symbol, start_date, end_date, *_ = query
    display_symbol = str(df.attrs.get("symbol", symbol)).strip() or symbol
    streamlit.subheader("%s · %s" % (ASSET_LABELS[asset_type], display_symbol))

    source_text = metrics.get("data_source") or "N/A"
    ccy_text = metrics.get("currency") or "N/A"
    adj_text = str(df.attrs.get("adjust", "") or "") or "默认/不适用"
    cache_text = " · 本地缓存命中" if df.attrs.get("cache_hit") else ""
    streamlit.caption(
        "样本区间：%s 至 %s · %d 条记录 · 数据源：%s · 计价：%s · 复权：%s%s"
        % (
            df.index[0].strftime("%Y-%m-%d"),
            df.index[-1].strftime("%Y-%m-%d"),
            len(df),
            source_text,
            ccy_text,
            adj_text,
            cache_text,
        )
    )
    unit_cap = str(df.attrs.get("unit_caption", "") or "").strip()
    if unit_cap:
        streamlit.caption("单位与口径：%s" % unit_cap)
    return_basis = str(metrics.get("return_basis") or "")
    if return_basis in ("futures_continuous", "price_index", "fx_rate", "adjusted_or_price"):
        basis_note = ASSET_RETURN_BASIS_NOTES.get(return_basis)
        if basis_note:
            streamlit.caption("口径说明：%s" % basis_note)
    if metrics.get("metric_disclaimer"):
        streamlit.warning(metrics["metric_disclaimer"])

    quality = metrics.get("data_quality") or {}
    dropped = int(quality.get("dropped_rows", 0) or 0)
    ffilled = int(quality.get("ffilled_cells", 0) or 0)
    if dropped > 0 or ffilled > 0:
        streamlit.warning(
            "数据质量提示：已清理异常/缺失记录 %d 行，前向填补 %d 单元格（策略：%s）。"
            % (dropped, ffilled, quality.get("missing_policy", "drop"))
        )

    columns = streamlit.columns(3)
    columns[0].metric(metrics["cagr_label"], _format_percent_metric(metrics["cagr"]))
    columns[1].metric(
        "最大回撤", _format_percent_metric(-abs(metrics["max_drawdown"]), signed=True)
    )
    columns[2].metric("Sharpe", _format_ratio_metric(metrics["sharpe_ratio"]))

    featured_df = metrics["featured_df"]
    momentum_note = ""
    if "rsi_14" in featured_df.columns and "macd_hist" in featured_df.columns:
        rsi_val = featured_df["rsi_14"].iloc[-1]
        macd_hist = featured_df["macd_hist"].iloc[-1]
        if pd.notna(rsi_val) and pd.notna(macd_hist):
            momentum_note = " | RSI(14)：%.1f | MACD柱：%+.4f" % (float(rsi_val), float(macd_hist))

    streamlit.info(
        "最新信号：%s | %s：%s | 年化波动：%s | Sortino：%s | Calmar：%s | 95%% 日 VaR / CVaR：%s / %s%s"
        % (
            metrics["latest_signal"],
            metrics["cumulative_return_label"],
            _format_percent_metric(metrics["cumulative_return"], signed=True),
            _format_percent_metric(metrics["annualized_volatility"]),
            _format_ratio_metric(metrics["sortino_ratio"]),
            _format_ratio_metric(metrics["calmar_ratio"]),
            _format_percent_metric(metrics["var_95"], signed=True),
            _format_percent_metric(metrics["cvar_95"], signed=True),
            momentum_note,
        )
    )
    _plot_chart(streamlit, build_price_volume_figure, featured_df)
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
    with streamlit.expander("查看清洗后数据、特征工程与质量诊断", expanded=False):
        if quality:
            streamlit.json(quality)
        streamlit.markdown("**清洗后 OHLCV 数据尾部**")
        streamlit.dataframe(df.tail(20), width="stretch")
        streamlit.markdown("**特征工程结果尾部**")
        streamlit.dataframe(featured_df.tail(20), width="stretch")
        streamlit.caption("标准列: %s" % ", ".join(list(REQUIRED_OHLCV_COLUMNS)))


def _apply_preset_template() -> None:
    if st is None:
        return
    chosen = st.session_state.get("tab3_preset_template")
    if chosen in PRESET_PORTFOLIOS:
        st.session_state["tab3_ef_legs_input"] = "\n".join(PRESET_PORTFOLIOS[chosen])


def _render_advanced(
    streamlit: Any,
    query: tuple,
    load_single: Callable[[], pd.DataFrame],
    single_valid: bool,
    dates_valid: bool,
) -> None:
    _, _, start_date, end_date, proxy_url, _, _, _, _, rf_pct, _ = query
    rf_rate = float(rf_pct) / 100.0
    streamlit.caption(
        "单标的分析复用参数栏设置；多标的组合支持跨市场币种换算、精准前沿与无泄露样本外回测。"
    )

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
        streamlit.markdown("根据历史日对数收益估计漂移与波动，模拟未来价格分位带与路径分布。")
        gbm_cols = streamlit.columns(3)
        gbm_days = int(
            gbm_cols[0].number_input(
                "模拟交易日数",
                min_value=20,
                max_value=1000,
                value=TRADING_DAYS,
                step=21,
                key="tab3_gbm_days",
            )
        )
        drift_mode = gbm_cols[1].selectbox(
            "漂移率估计模式",
            options=["按历史样本估算", "自定义年化漂移 (%)"],
            key="tab3_gbm_drift_mode",
        )
        custom_drift_pct = float(
            gbm_cols[2].number_input(
                "自定义年化漂移 (%)",
                min_value=-50.0,
                max_value=100.0,
                value=8.0,
                step=1.0,
                disabled=(drift_mode == "按历史样本估算"),
                key="tab3_gbm_drift_pct",
            )
        )
        annual_drift = (custom_drift_pct / 100.0) if drift_mode == "自定义年化漂移 (%)" else None

        requested = streamlit.button("运行 GBM 模拟", key="tab3_btn_gbm", disabled=not single_valid)
        gbm_params = query + (gbm_days, drift_mode, annual_drift)
        mc = _analysis_result(
            streamlit,
            "gbm",
            gbm_params,
            requested,
            lambda: run_monte_carlo_gbm(
                load_single(),
                days=gbm_days,
                num_simulations=500,
                random_state=42,
                num_sample_paths=5,
                annual_drift=annual_drift,
            ),
            "正在加载数据并模拟路径…",
        )
        if mc is not None:
            streamlit.caption(
                "模拟 %d 步 · 日漂移 μ≈%.6f · 日波动 σ≈%.6f · 起点 S₀=%.4f"
                % (gbm_days, mc.mu_daily, mc.sigma_daily, mc.s0)
            )
            _plot_chart(streamlit, build_tab3_gbm_figure, mc)

    streamlit.subheader("跨市场组合有效前沿与样本外回测")
    streamlit.selectbox(
        "组合配置预设模板",
        options=list(PRESET_PORTFOLIOS.keys()),
        key="tab3_preset_template",
        on_change=_apply_preset_template,
        help="选择预设组合将自动更新下方标的列表。",
    )
    streamlit.session_state.setdefault(
        "tab3_ef_legs_input",
        "\n".join(PRESET_PORTFOLIOS["自定义（默认跨市示例）"]),
    )
    streamlit.markdown(
        "每行输入 `资产类型,代码`，至少两个不同标的（支持 `类型|代码` 或 `;` 分隔）。"
    )
    ef_legs_text = streamlit.text_area(
        "多标的组合（至少 2 条，不重复）",
        height=110,
        key="tab3_ef_legs_input",
    )

    c_top1, c_top2, c_top3 = streamlit.columns(3)
    base_currency = c_top1.selectbox(
        "统一基准货币",
        options=list(SUPPORTED_BASE_CURRENCIES),
        index=0,
        key="tab3_base_currency",
    )
    calendar_mode = c_top2.selectbox(
        "估值日历对齐口径",
        options=["common_trading_dates", "weekly_friday", "mtm_outer"],
        format_func=lambda k: _CALENDAR_LABELS[k],
        key="tab3_calendar_mode",
    )
    val_mode = c_top3.selectbox(
        "样本外验证模式",
        options=[_VAL_MODE_SPLIT, _VAL_MODE_ROLLING, _VAL_MODE_EXPANDING],
        key="tab3_val_mode",
    )

    with streamlit.expander("协方差收缩、权重约束与再平衡交易成本设置", expanded=False):
        o1, o2, o3 = streamlit.columns(3)
        cov_method: CovMethod = o1.selectbox(
            "协方差估计方法",
            options=["ledoit_wolf", "sample", "diagonal"],
            format_func=lambda k: _COV_LABELS[k],
            key="tab3_cov_method",
        )
        objective: OptimizerObjective = o2.selectbox(
            "优化目标",
            options=["max_sharpe", "min_variance"],
            format_func=lambda k: _OBJECTIVE_LABELS[k],
            key="tab3_objective",
        )
        rebalance_freq: RebalanceFreq = o3.selectbox(
            "再平衡频率",
            options=["monthly", "quarterly", "daily", "buy_and_hold"],
            format_func=lambda k: _REBAL_LABELS[k],
            key="tab3_rebalance_freq",
        )

        b1, b2, b3, b4 = streamlit.columns(4)
        min_weight = float(
            b1.number_input(
                "单资产下限权重",
                min_value=0.0,
                max_value=0.5,
                value=0.0,
                step=0.05,
                format="%.2f",
                key="tab3_min_weight",
            )
        )
        max_weight = float(
            b2.number_input(
                "单资产上限权重",
                min_value=0.05,
                max_value=1.0,
                value=1.0,
                step=0.05,
                format="%.2f",
                key="tab3_max_weight",
            )
        )
        tx_bps = float(
            b3.number_input(
                "双边交易成本 (bps)",
                min_value=0.0,
                max_value=200.0,
                value=10.0,
                step=2.5,
                format="%.1f",
                key="tab3_tx_bps",
            )
        )
        train_ratio = float(
            b4.slider(
                "Train 占比（拆分模式）",
                min_value=0.40,
                max_value=0.85,
                value=0.70,
                step=0.05,
                key="tab3_train_ratio",
            )
        )
        w1, w2, w3 = streamlit.columns(3)
        train_window = int(
            w1.number_input(
                "滚动训练窗口期数（Walk-Forward）",
                min_value=20,
                max_value=500,
                value=60,
                step=10,
                key="tab3_train_window",
            )
        )
        allow_usdt_peg = bool(
            w2.checkbox(
                "允许 USDT≈USD 1:1 锚定",
                value=False,
                key="tab3_allow_usdt_peg",
                help="开启前默认尝试获取市场 USDT/USD 汇价；若数据源受限可显式启用锚定假定。",
            )
        )
        show_extra_charts = bool(
            w3.checkbox(
                "显示多资产走势对比与权重演化图",
                value=False,
                key="tab3_show_extra_charts",
            )
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
        "组合共享左侧起止日期与代理设置；若包含期货连续价格、纯价格指数或美债收益率，系统将按口径规范处理并给出风险提示。"
    )

    def _cached_fx_loader(pair_symbol: str, s_dt: object, e_dt: object) -> pd.DataFrame:
        force_refresh = bool(streamlit.session_state.get("force_refresh_input", False))
        if force_refresh:
            return _load_asset_data(
                asset_type="fx",
                symbol=pair_symbol,
                start_date=s_dt,
                end_date=e_dt,
                proxy_url=proxy_url,
                force_refresh=True,
            )
        return load_asset_data(
            asset_type="fx",
            symbol=pair_symbol,
            start_date=s_dt,
            end_date=e_dt,
            proxy_url=proxy_url,
        )

    def optimize():
        force_refresh = bool(streamlit.session_state.get("force_refresh_input", False))
        if force_refresh:
            load_asset_data.clear()
            load_multi_leg_close_wide.clear()
        legs = parse_multi_asset_legs(ef_legs_text)
        aligned = _load_and_align_multi_asset_portfolio(
            legs,
            start_date,
            end_date,
            proxy_url=proxy_url,
            calendar_mode=calendar_mode,
            base_currency=base_currency,
            allow_usdt_usd_peg=allow_usdt_peg,
            loader=None if force_refresh else load_asset_data,
            fx_loader=_cached_fx_loader,
            force_refresh=force_refresh,
        )
        min_hist = 10 if calendar_mode == "weekly_friday" else 30
        ppy = float(aligned.backtest_periods_per_year)
        ann_days = max(int(round(ppy)), 1)
        corr = returns_correlation_matrix(
            aligned.correlation_returns_base,
            min_history=min(min_hist, len(aligned.correlation_returns_base)),
        )
        returns_all = clean_joint_returns(
            aligned.returns_base,
            min_history=min(min_hist, len(aligned.returns_base)),
        )

        if val_mode == _VAL_MODE_SPLIT:
            split_res = split_train_test_returns(
                returns_all,
                train_ratio=train_ratio,
                min_train_rows=10 if calendar_mode == "weekly_friday" else 30,
                min_test_rows=5 if calendar_mode == "weekly_friday" else 10,
                objective=objective,
                cov_method=cov_method,
                risk_free_rate=rf_rate,
                min_weight=min_weight,
                max_weight=max_weight,
                rebalance_freq=rebalance_freq,
                transaction_cost_bps=tx_bps,
                tradable_mask=aligned.tradable_mask,
                periods_per_year=ppy,
            )
            exact_ef = solve_exact_efficient_frontier(
                split_res.train_returns,
                n_points=25,
                cov_method=cov_method,
                risk_free_rate=rf_rate,
                trading_days=ann_days,
                min_history=len(split_res.train_returns),
                min_weight=min_weight,
                max_weight=max_weight,
                num_random_portfolios=1000,
                random_state=42,
            )
            return {
                "mode": val_mode,
                "aligned": aligned,
                "corr": corr,
                "exact_ef": exact_ef,
                "split_res": split_res,
            }

        win_mode = "rolling" if val_mode == _VAL_MODE_ROLLING else "expanding"
        wf_res = run_walk_forward_backtest(
            returns_all,
            window_mode=win_mode,
            train_window=train_window,
            rebalance_freq=rebalance_freq,
            cov_method=cov_method,
            objective=objective,
            risk_free_rate=rf_rate,
            min_weight=min_weight,
            max_weight=max_weight,
            transaction_cost_bps=tx_bps,
            periods_per_year=ppy,
            tradable_mask=aligned.tradable_mask,
        )
        latest_slice = returns_all.iloc[-train_window:] if win_mode == "rolling" else returns_all
        exact_ef = solve_exact_efficient_frontier(
            latest_slice,
            n_points=25,
            cov_method=cov_method,
            risk_free_rate=rf_rate,
            trading_days=ann_days,
            min_history=len(latest_slice),
            min_weight=min_weight,
            max_weight=max_weight,
            num_random_portfolios=1000,
            random_state=42,
        )
        return {
            "mode": val_mode,
            "aligned": aligned,
            "corr": corr,
            "exact_ef": exact_ef,
            "wf_res": wf_res,
        }

    ef_params = (
        start_date,
        end_date,
        proxy_url,
        ef_legs_text.strip(),
        base_currency,
        calendar_mode,
        allow_usdt_peg,
        cov_method,
        objective,
        rebalance_freq,
        min_weight,
        max_weight,
        tx_bps,
        train_ratio,
        train_window,
        val_mode,
        rf_rate,
    )
    result = _analysis_result(
        streamlit,
        "frontier",
        ef_params,
        requested,
        optimize,
        "正在统一基准货币、求解精确前沿并运行样本外回测…",
    )
    if result is not None:
        aligned = result["aligned"]
        corr = result["corr"]
        exact_ef = result["exact_ef"]

        summary = aligned.overlap_summary
        streamlit.caption(
            "统一估值区间：%s 至 %s（%d 个估值点 · 基准货币：%s · 日历口径：%s）"
            % (
                summary.get("effective_start", "—"),
                summary.get("effective_end", "—"),
                int(summary.get("return_periods", 0)),
                aligned.base_currency,
                aligned.calendar_mode,
            )
        )
        for warn_msg in aligned.warnings:
            streamlit.warning(warn_msg)

        _plot_chart(streamlit, build_tab3_correlation_heatmap, corr)
        with streamlit.expander("查看相关系数与对齐明细", expanded=False):
            streamlit.dataframe(corr.style.format("{:.3f}"), width="stretch")
            streamlit.json(summary)

        _plot_chart(streamlit, build_tab3_efficient_frontier_figure, exact_ef)
        shrink_text = (
            f" · Ledoit-Wolf 收缩强度 δ={exact_ef.shrinkage:.3f}"
            if exact_ef.shrinkage is not None
            else ""
        )
        streamlit.caption(
            "协方差矩阵条件数 cond=%.2f%s · 权重约束 [%.0f%%, %.0f%%]"
            % (
                exact_ef.condition_number,
                shrink_text,
                exact_ef.min_weight * 100.0,
                exact_ef.max_weight * 100.0,
            )
        )

        w_table = pd.DataFrame(
            [exact_ef.max_sharpe.weights, exact_ef.min_variance.weights],
            index=["最大夏普组合 (Tangency)", "全局最小方差 (GMV)"],
            columns=exact_ef.asset_names,
        )
        streamlit.markdown("**精确优化目标权重**")
        streamlit.dataframe(w_table.style.format("{:.2%}"), width="stretch")

        if result["mode"] == _VAL_MODE_SPLIT:
            split_res = result["split_res"]
            oos_bt = split_res.out_of_sample_backtest
            _plot_chart(
                streamlit,
                build_tab3_portfolio_backtest_figure,
                oos_bt,
                title=f"样本外独立净值回测（自 {split_res.split_date:%Y-%m-%d} 起 · {aligned.base_currency} · 期初=1.0）",
                optimal_label="样本外策略（训练期拟合权重）",
            )
            perf_df = _format_backtest_comparison_table(
                [
                    (
                        "样本内拟合 (In-Sample)",
                        split_res.in_sample_performance.get("Optimal_Portfolio", {}),
                    ),
                    (
                        "样本外投资 (Out-of-Sample)",
                        split_res.out_of_sample_performance.get("Optimal_Portfolio", {}),
                    ),
                    (
                        "样本外等权基准 (Equal-Weight)",
                        split_res.out_of_sample_performance.get("Equal_Weight_Portfolio", {}),
                    ),
                ]
            )
            streamlit.markdown("**样本内拟合 vs 样本外真实绩效对比**")
            streamlit.dataframe(perf_df, width="stretch", hide_index=True)
        else:
            wf_res = result["wf_res"]
            _plot_chart(
                streamlit,
                build_tab3_walk_forward_figure,
                wf_res,
                base_currency=aligned.base_currency,
            )
            perf = wf_res.performance
            perf_df = _format_backtest_comparison_table(
                [
                    ("样本外 Walk-Forward 策略", perf.get("OOS_Portfolio", {})),
                    ("样本外等权基准", perf.get("Equal_Weight_Portfolio", {})),
                    ("全样本后视参考 (含前视偏差)", perf.get("In_Sample_Portfolio", {})),
                ]
            )
            streamlit.markdown("**滚动/扩展 Walk-Forward 绩效对比**")
            streamlit.dataframe(perf_df, width="stretch", hide_index=True)
            if show_extra_charts:
                _plot_chart(streamlit, build_tab3_weight_history_figure, wf_res.weight_history)

        if show_extra_charts:
            _plot_chart(
                streamlit,
                build_tab3_multi_normalize_figure,
                aligned.base_prices,
                base_currency=aligned.base_currency,
            )


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
    streamlit.caption("覆盖 13 类资产的历史行情、风险收益诊断、精确有效前沿与无泄露样本外回测。")

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

    supported_adj = get_supported_adjustments(asset_type)
    selected_adjust = ""
    if supported_adj:
        selected_adjust = str(
            streamlit.sidebar.selectbox(
                "历史复权口径",
                options=list(supported_adj),
                format_func=lambda v: ADJUSTMENT_LABELS.get(v, v),
                key="adjust_input",
            )
        )
    else:
        streamlit.sidebar.caption("当前资产类别无复权切换选项（按接口原始口径加载）。")

    start_date = streamlit.sidebar.date_input(
        "开始日期", value=default_start, key="start_date_input"
    )
    end_date = streamlit.sidebar.date_input("结束日期", value=today, key="end_date_input")
    proxy_raw = streamlit.sidebar.text_input(
        "代理地址（可选）",
        key="proxy_url_input",
        placeholder="http://127.0.0.1:7890",
        help="切换资产类别不会清空代理地址；用于加密货币及跨境市场。",
    )
    proxy_url = proxy_raw.strip() or None

    with streamlit.sidebar.expander("高级指标与利率参数", expanded=False):
        sma_short = int(
            streamlit.number_input(
                "短期均线窗口",
                min_value=2,
                max_value=120,
                value=20,
                step=1,
                key="sma_short_input",
            )
        )
        sma_long = int(
            streamlit.number_input(
                "长期均线窗口",
                min_value=5,
                max_value=365,
                value=60,
                step=1,
                key="sma_long_input",
            )
        )
        bb_window = int(
            streamlit.number_input(
                "布林带窗口",
                min_value=5,
                max_value=120,
                value=20,
                step=1,
                key="bb_window_input",
            )
        )
        rf_pct = float(
            streamlit.number_input(
                "无风险利率 Rf (%)",
                min_value=0.0,
                max_value=20.0,
                value=float(RISK_FREE_RATE * 100.0),
                step=0.1,
                format="%.2f",
                key="rf_pct_input",
            )
        )
        include_momentum = bool(
            streamlit.checkbox(
                "计算 RSI(14) 与 MACD",
                value=False,
                key="include_momentum_input",
            )
        )
        force_refresh = bool(
            streamlit.checkbox(
                "强制刷新行情（跳过本地持久缓存）",
                value=False,
                key="force_refresh_input",
                help="勾选后在本次诊断或组合优化时绕过本地 Parquet 磁盘与内存缓存并重新抓取。",
            )
        )

    dates_present = start_date is not None and end_date is not None
    dates_valid = dates_present and start_date <= end_date
    sma_valid = sma_short < sma_long
    single_valid = dates_valid and sma_valid and (bool(symbol) or asset_type == "bond_us_yield")
    if not dates_present:
        streamlit.sidebar.error("请选择开始和结束日期。")
    elif not dates_valid:
        streamlit.sidebar.error("开始日期不能晚于结束日期。")
    if not sma_valid:
        streamlit.sidebar.error("短期均线窗口必须严格小于长期均线窗口。")
    if not symbol and asset_type != "bond_us_yield":
        streamlit.sidebar.error("请输入资产代码。")
    with streamlit.sidebar.expander("资产类型说明", expanded=False):
        streamlit.markdown(format_asset_guide_markdown(asset_type))
    clicked = streamlit.sidebar.button(
        "开始诊断", width="stretch", type="primary", disabled=not single_valid
    )

    adj_param = (
        None
        if (not supported_adj or selected_adjust == DEFAULT_ADJUSTMENT.get(asset_type))
        else selected_adjust
    )
    query = (
        asset_type,
        symbol,
        start_date,
        end_date,
        proxy_url,
        adj_param,
        sma_short,
        sma_long,
        bb_window,
        rf_pct,
        include_momentum,
    )

    def load_single():
        if force_refresh:
            load_asset_data.clear()
            df = _load_asset_data(
                asset_type=asset_type,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                proxy_url=proxy_url,
                adjust=adj_param,
                force_refresh=True,
            )
        else:
            df = load_asset_data(
                asset_type=asset_type,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                proxy_url=proxy_url,
                adjust=adj_param,
            )
        if df.empty:
            raise EmptySymbolDataError(EMPTY_SYMBOL_FETCH_HINT)
        return df

    tab1, tab2, tab3 = streamlit.tabs(["量化诊断看板", "数学原理", "进阶金融算法"])
    with tab1:

        def diagnose():
            df = load_single()
            return df, get_quant_metrics(
                df,
                risk_free_rate=rf_pct / 100.0,
                sma_short_window=sma_short,
                sma_long_window=sma_long,
                bollinger_window=bb_window,
                include_momentum_indicators=include_momentum,
            )

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
