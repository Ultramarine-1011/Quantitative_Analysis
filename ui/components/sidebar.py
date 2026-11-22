"""Sidebar configuration panel for theme, global analysis window, and single-asset controls."""

from __future__ import annotations

import datetime as dt
from typing import Any

from asset_catalog import (
    ASSET_LABELS,
    DEFAULT_SYMBOLS,
    SUPPORTED_ADJUSTMENTS,
    SUPPORTED_ASSET_TYPES,
    format_asset_guide_markdown,
    get_asset_trading_days,
)
from asset_resolver import resolve_analysis_date_window
from contracts import SUPPORTED_RETURN_CONVENTIONS
from ui.labels import ADJUST_LABELS, BASE_CURRENCY_LABELS, RETURN_CONVENTION_LABELS
from ui.theme import THEME_MODES, THEME_SIDEBAR_LABELS, normalize_theme_mode


def sync_symbol_with_asset_type(st_module: Any) -> None:
    """Update default symbol & adaptive trading_days smoothly when ``asset_type`` changes."""
    if st_module is None:
        return
    new_type = str(st_module.session_state.get("asset_type", "etf"))
    prev_type = str(st_module.session_state.get("_prev_asset_type", "etf"))
    current_sym = str(st_module.session_state.get("symbol_input", "")).strip()
    prev_default = DEFAULT_SYMBOLS.get(prev_type, "")
    if not current_sym or current_sym == prev_default:
        st_module.session_state["symbol_input"] = DEFAULT_SYMBOLS.get(new_type, "")

    prev_td = get_asset_trading_days(prev_type)
    cur_td = st_module.session_state.get("single_trading_days_input")
    if cur_td is None or int(cur_td) == prev_td:
        st_module.session_state["single_trading_days_input"] = get_asset_trading_days(new_type)

    st_module.session_state["_prev_asset_type"] = new_type


def render_sidebar(st_module: Any) -> dict[str, Any]:
    """Render theme switcher, global window/runtime controls, and single-asset configuration."""
    default_start, default_end = resolve_analysis_date_window(None, None)

    st_module.sidebar.markdown(
        """
        <div class="qa-sidebar-brand">
          <div class="qa-sidebar-brand-title">Quant Insight 控制台</div>
          <div class="qa-sidebar-brand-sub">全局分析窗口、外观与单资产探索配置</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if hasattr(st_module.sidebar, "segmented_control"):
        raw_theme = st_module.sidebar.segmented_control(
            "界面主题",
            options=THEME_MODES,
            default="auto",
            format_func=lambda m: THEME_SIDEBAR_LABELS.get(m, m),
            key="ui_theme_mode",
            width="stretch",
        )
    else:
        raw_theme = st_module.sidebar.radio(
            "界面主题",
            options=THEME_MODES,
            index=0,
            horizontal=True,
            format_func=lambda m: THEME_SIDEBAR_LABELS.get(m, m),
            key="ui_theme_mode",
        )
    theme_mode = normalize_theme_mode(raw_theme)
    resolved_theme = theme_mode

    st_module.sidebar.markdown(
        '<div class="qa-sidebar-section">全局分析区间与基准利率</div>', unsafe_allow_html=True
    )

    start_val = st_module.sidebar.date_input(
        "开始日期",
        value=default_start.date(),
        key="start_date_input",
    )
    end_val = st_module.sidebar.date_input(
        "结束日期",
        value=default_end.date(),
        key="end_date_input",
    )

    date_error: str | None = None
    start_str: str | None = None
    end_str: str | None = None
    if (
        start_val is None
        or end_val is None
        or isinstance(start_val, tuple)
        or isinstance(end_val, tuple)
    ):
        date_error = "请选择开始和结束日期。"
    elif isinstance(start_val, dt.date) and isinstance(end_val, dt.date):
        if start_val > end_val:
            date_error = "开始日期不能晚于结束日期，请调整后再运行。"
        else:
            start_str = start_val.strftime("%Y-%m-%d")
            end_str = end_val.strftime("%Y-%m-%d")
    else:
        date_error = "请选择开始和结束日期。"

    if date_error:
        st_module.sidebar.error(date_error)

    rf_pct = float(
        st_module.sidebar.number_input(
            "无风险年化利率 Rf (%)",
            min_value=0.0,
            max_value=20.0,
            value=1.6,
            step=0.1,
            key="rf_pct_input",
            help="用于夏普、Sortino、特雷诺及切线组合计算；默认 1.6% 与 CLI 保持一致。",
        )
    )

    st_module.sidebar.markdown(
        '<div class="qa-sidebar-section">📈 单资产探索参数</div>', unsafe_allow_html=True
    )

    if "asset_type" not in st_module.session_state:
        st_module.session_state["asset_type"] = "etf"
    if "_prev_asset_type" not in st_module.session_state:
        st_module.session_state["_prev_asset_type"] = st_module.session_state["asset_type"]
    if "symbol_input" not in st_module.session_state:
        st_module.session_state["symbol_input"] = DEFAULT_SYMBOLS["etf"]
    if "single_trading_days_input" not in st_module.session_state:
        st_module.session_state["single_trading_days_input"] = get_asset_trading_days("etf")

    asset_type = str(
        st_module.sidebar.selectbox(
            "资产类别",
            options=SUPPORTED_ASSET_TYPES,
            format_func=lambda key: ASSET_LABELS.get(key, key),
            key="asset_type",
            on_change=lambda: sync_symbol_with_asset_type(st_module),
        )
    )
    symbol = str(
        st_module.sidebar.text_input(
            "资产代码",
            key="symbol_input",
            help="支持基金/ETF/A股/美股/港股/全球指数/加密货币/汇率/债券/REITs。",
        )
    ).strip()

    with st_module.sidebar.expander("💡 常见代码格式与品种提示", expanded=False):
        st_module.markdown(format_asset_guide_markdown(asset_type))

    supported_adjusts = SUPPORTED_ADJUSTMENTS.get(asset_type)
    if supported_adjusts:
        cur_adj = st_module.session_state.get("adjust_input")
        if cur_adj not in supported_adjusts:
            st_module.session_state["adjust_input"] = supported_adjusts[0]
        adjust: str | None = str(
            st_module.sidebar.selectbox(
                "复权口径",
                options=supported_adjusts,
                format_func=lambda code: ADJUST_LABELS.get(code, code),
                key="adjust_input",
            )
        )
    else:
        st_module.sidebar.selectbox(
            "复权口径",
            options=("该资产不适用复权",),
            index=0,
            disabled=True,
            key="adjust_input_disabled",
        )
        adjust = None

    single_base_currency = str(
        st_module.sidebar.selectbox(
            "估值基准货币",
            options=("LOCAL", "CNY", "USD"),
            index=0,
            format_func=lambda c: BASE_CURRENCY_LABELS.get(c, c),
            key="single_base_currency_input",
            help="默认保持资产原生货币 LOCAL（与 `quant analyze` 默认口径一致）。",
        )
    )

    with st_module.sidebar.expander("🔧 单资产高级技术与统计参数", expanded=False):
        single_return_convention = str(
            st_module.selectbox(
                "收益率计算口径",
                options=SUPPORTED_RETURN_CONVENTIONS,
                index=0,
                format_func=lambda m: RETURN_CONVENTION_LABELS.get(m, m),
                key="single_return_convention_input",
                help="单资产默认使用时间可加的对数收益率 (log)，可切换为简单收益率。",
            )
        )
        sma_short = int(
            st_module.number_input(
                "短期均线窗口 (SMA Short)", 2, 120, 20, step=1, key="sma_short_input"
            )
        )
        sma_long = int(
            st_module.number_input(
                "长期均线窗口 (SMA Long)", 5, 250, 60, step=5, key="sma_long_input"
            )
        )
        if sma_short >= sma_long:
            st_module.caption("⚠️ 短期均线窗口应严格小于长期均线窗口。")
        bb_window = int(
            st_module.number_input(
                "布林带滚动窗口 (BB Window)", 2, 120, 20, step=1, key="bb_window_input"
            )
        )
        single_trading_days = int(
            st_module.number_input(
                "年化基准天数 (Trading Days)",
                min_value=1,
                max_value=366,
                step=1,
                key="single_trading_days_input",
                help="股票/ETF 默认 252，外汇默认 260，加密货币默认 365。",
            )
        )
        include_momentum = bool(
            st_module.checkbox(
                "计算并绘制动量副图 (RSI-14 / MACD)", value=False, key="include_momentum_input"
            )
        )

    symbol_missing = (not symbol) and (asset_type != "bond_us_yield")
    if symbol_missing:
        st_module.sidebar.warning("请输入有效的资产代码（美债利率支持留空默认为 10Y）。")

    single_clicked = st_module.sidebar.button(
        "🔍 运行单资产分析",
        key="single_run_btn",
        type="primary",
        width="stretch",
        disabled=bool(date_error or symbol_missing),
    )

    st_module.sidebar.markdown(
        '<div class="qa-sidebar-section">数据源与网络环境</div>', unsafe_allow_html=True
    )
    with st_module.sidebar.expander("🌐 网络代理、缓存与稳定币口径", expanded=False):
        proxy_raw = st_module.text_input(
            "HTTP/SOCKS 代理地址 (可选)",
            value="",
            placeholder="http://127.0.0.1:7890",
            key="proxy_url_input",
        )
        force_refresh = st_module.checkbox(
            "跳过本地缓存（强制在线刷新）",
            value=False,
            key="force_refresh_input",
        )
        allow_usdt_peg = st_module.checkbox(
            "允许加密货币按 USDT=USD (1:1) 近似折算",
            value=False,
            key="allow_usdt_peg_input",
            help="默认严格抓取实际 USDT/USD 汇率；若不可得时可显式勾选允许 1:1 近似。",
        )

    return {
        "theme_mode": theme_mode,
        "resolved_theme_mode": resolved_theme,
        "dates_valid": date_error is None,
        "date_error": date_error,
        "start_str": start_str,
        "end_str": end_str,
        "rf_rate": rf_pct / 100.0,
        "proxy_url": proxy_raw.strip() or None,
        "force_refresh": force_refresh,
        "allow_usdt_peg": allow_usdt_peg,
        "asset_type": asset_type,
        "symbol": symbol,
        "adjust": adjust,
        "single_base_currency": single_base_currency,
        "single_return_convention": single_return_convention,
        "sma_short": sma_short,
        "sma_long": sma_long,
        "bb_window": bb_window,
        "single_trading_days": single_trading_days,
        "include_momentum": include_momentum,
        "single_clicked": single_clicked,
    }
