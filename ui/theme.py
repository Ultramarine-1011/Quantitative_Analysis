"""Design system, responsive Light/Dark/Auto CSS, and Plotly theme harmonization."""

from __future__ import annotations

from typing import Any, Literal

ThemeMode = Literal["auto", "dark", "light"]

THEME_MODES: tuple[ThemeMode, ...] = ("auto", "dark", "light")
THEME_MODE_LABELS: dict[str, str] = {
    "auto": "跟随系统",
    "dark": "深色终端",
    "light": "浅色研报",
}
THEME_SIDEBAR_LABELS: dict[str, str] = {
    "auto": "自动",
    "dark": "深色",
    "light": "浅色",
}

_FONT_STACK = (
    "'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text', "
    "'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif"
)
_MONO_STACK = (
    "'JetBrains Mono', 'SF Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, monospace"
)


def normalize_theme_mode(value: object) -> ThemeMode:
    """Normalize user theme selection into ``auto``, ``dark``, or ``light``."""
    raw = str(value or "auto").strip().lower()
    if raw == "dark":
        return "dark"
    if raw == "light":
        return "light"
    return "auto"


def _auto_adaptive_tokens() -> str:
    """Theme-neutral translucent tokens that inherit Streamlit's active Light/Dark canvas."""
    return """
        --qa-bg: transparent;
        --qa-bg-elevated: rgba(148, 163, 184, 0.06);
        --qa-surface: rgba(148, 163, 184, 0.07);
        --qa-surface-solid: var(--background-color, rgba(148, 163, 184, 0.08));
        --qa-surface-alt: rgba(148, 163, 184, 0.11);
        --qa-sidebar-bg: transparent;
        --qa-header-bg: transparent;
        --qa-border: rgba(128, 138, 152, 0.24);
        --qa-border-strong: rgba(128, 138, 152, 0.38);
        --qa-text: inherit;
        --qa-text-secondary: inherit;
        --qa-text-muted: rgba(110, 122, 138, 0.95);
        --qa-accent: #2563eb;
        --qa-accent-hover: #3b82f6;
        --qa-accent-soft: rgba(37, 99, 235, 0.12);
        --qa-positive: #059669;
        --qa-positive-soft: rgba(16, 185, 129, 0.14);
        --qa-negative: #e11d48;
        --qa-negative-soft: rgba(225, 29, 72, 0.14);
        --qa-warning: #d97706;
        --qa-warning-soft: rgba(245, 158, 11, 0.14);
        --qa-shadow: 0 4px 16px -8px rgba(15, 23, 42, 0.12);
        --qa-hero-grad: linear-gradient(135deg, rgba(37, 99, 235, 0.10) 0%, rgba(148, 163, 184, 0.05) 55%, rgba(8, 145, 178, 0.08) 100%);
        --qa-input-bg: transparent;
    """


def _token_vars(*, dark: bool) -> str:
    if dark:
        return """
        --qa-bg: #090e1a;
        --qa-bg-elevated: #0f172a;
        --qa-surface: rgba(15, 23, 42, 0.88);
        --qa-surface-solid: #111b2e;
        --qa-surface-alt: #17233a;
        --qa-sidebar-bg: #0c1322;
        --qa-header-bg: rgba(9, 14, 26, 0.82);
        --qa-border: rgba(148, 163, 184, 0.18);
        --qa-border-strong: rgba(148, 163, 184, 0.34);
        --qa-text: #f1f5f9;
        --qa-text-secondary: #cbd5e1;
        --qa-text-muted: #94a3b8;
        --qa-accent: #3b82f6;
        --qa-accent-hover: #60a5fa;
        --qa-accent-soft: rgba(59, 130, 246, 0.16);
        --qa-positive: #34d399;
        --qa-positive-soft: rgba(16, 185, 129, 0.16);
        --qa-negative: #fb7185;
        --qa-negative-soft: rgba(244, 63, 94, 0.16);
        --qa-warning: #fbbf24;
        --qa-warning-soft: rgba(245, 158, 11, 0.15);
        --qa-shadow: 0 10px 28px -12px rgba(2, 6, 23, 0.65);
        --qa-hero-grad: linear-gradient(135deg, rgba(30, 58, 138, 0.32) 0%, rgba(15, 23, 42, 0.94) 55%, rgba(14, 116, 144, 0.18) 100%);
        --qa-input-bg: #131f35;
        """
    return """
        --qa-bg: #f4f6fa;
        --qa-bg-elevated: #ffffff;
        --qa-surface: #ffffff;
        --qa-surface-solid: #ffffff;
        --qa-surface-alt: #f8fafc;
        --qa-sidebar-bg: #eef2f8;
        --qa-header-bg: rgba(244, 246, 250, 0.86);
        --qa-border: rgba(15, 23, 42, 0.10);
        --qa-border-strong: rgba(15, 23, 42, 0.22);
        --qa-text: #0f172a;
        --qa-text-secondary: #334155;
        --qa-text-muted: #64748b;
        --qa-accent: #2563eb;
        --qa-accent-hover: #1d4ed8;
        --qa-accent-soft: rgba(37, 99, 235, 0.10);
        --qa-positive: #059669;
        --qa-positive-soft: rgba(5, 150, 105, 0.10);
        --qa-negative: #dc2626;
        --qa-negative-soft: rgba(220, 38, 38, 0.10);
        --qa-warning: #b45309;
        --qa-warning-soft: rgba(217, 119, 6, 0.12);
        --qa-shadow: 0 8px 24px -12px rgba(15, 23, 42, 0.12);
        --qa-hero-grad: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(255, 255, 255, 0.96) 55%, rgba(8, 145, 178, 0.06) 100%);
        --qa-input-bg: #ffffff;
    """


def resolve_streamlit_auto_theme(st_module: Any) -> ThemeMode | None:
    """Detect active Streamlit frontend theme (``'light'`` / ``'dark'``) if available."""
    if st_module is None:
        return None
    try:
        theme_ctx = getattr(getattr(st_module, "context", None), "theme", None)
        t_type = getattr(theme_ctx, "type", None)
        if t_type in {"dark", "light"}:
            return t_type
    except Exception:
        return None
    return None


def _explicit_surface_overrides(
    theme_mode: ThemeMode,
    *,
    resolved_auto: ThemeMode | None = None,
) -> str:
    """Force consistent Streamlit container, sidebar, and input surfaces in explicit dark/light modes."""
    if theme_mode == "auto":
        if resolved_auto == "dark":
            return """
            :root {
                __DARK_TOKENS__
            }
            """
        if resolved_auto == "light":
            return """
            :root {
                __LIGHT_TOKENS__
            }
            """
        return """
        @media (prefers-color-scheme: dark) {
            :root {
                --qa-shadow: 0 8px 24px -10px rgba(2, 6, 23, 0.45);
            }
        }
        """
    scheme = "dark" if theme_mode == "dark" else "light"
    st_bg = "#090e1a" if theme_mode == "dark" else "#f4f6fa"
    st_sbg = "#0c1322" if theme_mode == "dark" else "#eef2f8"
    st_txt = "#f1f5f9" if theme_mode == "dark" else "#0f172a"
    df_dark_rule = (
        """
        [data-testid="stDataFrame"] > div {
            filter: invert(0.91) hue-rotate(180deg) brightness(1.06) contrast(0.95);
        }
        """
        if theme_mode == "dark"
        else ""
    )
    return f"""
        {df_dark_rule}
        :root, .stApp {{
            color-scheme: {scheme};
            --background-color: {st_bg};
            --secondary-background-color: {st_sbg};
            --text-color: {st_txt};
        }}
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            background-color: var(--qa-bg) !important;
            color: var(--qa-text) !important;
        }}
        [data-testid="stHeader"] {{
            background-color: var(--qa-header-bg) !important;
            backdrop-filter: blur(8px);
        }}
        section[data-testid="stSidebar"] {{
            background-color: var(--qa-sidebar-bg) !important;
            border-right: 1px solid var(--qa-border) !important;
        }}
        section[data-testid="stSidebar"] *, [data-testid="stMain"] label,
        [data-testid="stMain"] .stMarkdown, [data-testid="stMain"] p,
        [data-testid="stMain"] h1, [data-testid="stMain"] h2,
        [data-testid="stMain"] h3, [data-testid="stMain"] h4 {{
            color: var(--qa-text);
        }}
        [data-testid="stMain"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
            color: var(--qa-text-muted) !important;
        }}
        [data-baseweb="input"],
        [data-baseweb="base-input"],
        [data-baseweb="textarea"],
        [data-baseweb="input"] > div,
        [data-baseweb="base-input"] > div,
        [data-baseweb="textarea"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="select"] > div > div,
        [data-testid="stSelectbox"] > div,
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stSelectbox"] [data-baseweb="select"],
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stNumberInputContainer"],
        [data-testid="stDateInputField"],
        [data-testid="stDateInputField"] > div,
        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stNumberInput"] button,
        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-baseweb="popover"],
        [data-baseweb="popover"] ul,
        [data-baseweb="popover"] li {{
            background-color: var(--qa-input-bg) !important;
            color: var(--qa-text) !important;
            border-color: var(--qa-border-strong) !important;
            -webkit-text-fill-color: var(--qa-text) !important;
        }}
        [data-testid="stMain"] code,
        section[data-testid="stSidebar"] code {{
            background-color: var(--qa-accent-soft) !important;
            color: var(--qa-accent) !important;
            padding: 0.08rem 0.35rem;
            border-radius: 5px;
        }}
        [data-testid="stDateInputField"] *,
        [data-testid="stSelectbox"] span,
        [data-testid="stSelectbox"] div,
        [data-testid="stSelectbox"] svg,
        [data-testid="stNumberInputStepDown"] svg,
        [data-testid="stNumberInputStepUp"] svg {{
            color: var(--qa-text) !important;
            fill: var(--qa-text-muted) !important;
        }}
        [data-testid="stDateInputCalendarButton"],
        [data-testid="stDateInputCalendarButton"] svg {{
            color: var(--qa-text-muted) !important;
            background-color: transparent !important;
        }}
        [data-testid="stButtonGroup"] {{
            background-color: transparent !important;
        }}
        [data-testid="stButtonGroup"] button {{
            background-color: var(--qa-input-bg) !important;
            color: var(--qa-text-secondary) !important;
            border: 1px solid var(--qa-border-strong) !important;
        }}
        [data-testid="stButtonGroup"] button[aria-checked="true"],
        [data-testid="stButtonGroup"] button[data-testid="stBaseButton-segmented_controlActive"] {{
            background-color: var(--qa-accent-soft) !important;
            color: var(--qa-accent) !important;
            border-color: var(--qa-accent) !important;
        }}
        [data-testid="stButtonGroup"] button * {{
            color: inherit !important;
        }}
        [data-testid="stExpander"] {{
            background-color: var(--qa-surface) !important;
            border: 1px solid var(--qa-border) !important;
            border-radius: 12px !important;
        }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary p {{
            color: var(--qa-text) !important;
        }}
        [data-testid="stButton"] > button:not([kind="primary"]),
        [data-testid="stDownloadButton"] > button {{
            background-color: var(--qa-surface-solid) !important;
            color: var(--qa-text) !important;
            border: 1px solid var(--qa-border-strong) !important;
        }}
        [data-testid="stButton"] > button:not([kind="primary"]):hover,
        [data-testid="stDownloadButton"] > button:hover {{
            border-color: var(--qa-accent) !important;
            color: var(--qa-accent) !important;
        }}
        [data-baseweb="tab-list"] [role="tab"] {{
            color: var(--qa-text-muted) !important;
        }}
        [data-baseweb="tab-list"] [role="tab"][aria-selected="true"] {{
            color: var(--qa-text) !important;
        }}
        [data-testid="stAlert"],
        [data-testid="stAlert"] > div,
        [data-testid="stAlertContainer"] {{
            background-color: var(--qa-surface-alt) !important;
            color: var(--qa-text) !important;
            border-radius: 10px !important;
        }}
        [data-testid="stAlertContainer"] {{
            border: 1px solid var(--qa-border-strong) !important;
        }}
        [data-testid="stAlert"] *,
        [data-testid="stAlertContainer"] * {{
            color: var(--qa-text) !important;
        }}
    """


def build_theme_css(
    theme_mode: ThemeMode = "auto",
    *,
    resolved_auto: ThemeMode | None = None,
) -> str:
    """Compile complete CSS stylesheet for the chosen theme mode."""
    mode = normalize_theme_mode(theme_mode)
    if mode == "auto" and resolved_auto not in {"dark", "light"}:
        base_tokens = _auto_adaptive_tokens()
    else:
        base_is_dark = mode == "dark" or (mode == "auto" and resolved_auto == "dark")
        base_tokens = _token_vars(dark=base_is_dark)
    overrides = (
        _explicit_surface_overrides(mode, resolved_auto=resolved_auto)
        .replace("__DARK_TOKENS__", _token_vars(dark=True))
        .replace("__LIGHT_TOKENS__", _token_vars(dark=False))
    )
    return f"""
    <style>
    :root {{
        --qa-font-sans: {_FONT_STACK};
        --qa-font-mono: {_MONO_STACK};
        --qa-radius-lg: 14px;
        --qa-radius-md: 10px;
        --qa-radius-sm: 8px;
        {base_tokens}
    }}
    {overrides}

    html, body, .stApp {{
        font-family: var(--qa-font-sans);
    }}

    /* Scroll stability & desktop canvas spacing */
    [data-testid="stMain"] {{
        overflow-y: auto;
        overflow-anchor: none;
        overscroll-behavior-y: contain;
    }}
    [data-testid="stMainBlockContainer"] {{
        flex-shrink: 0;
        max-width: 1520px;
        padding-top: 1.45rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }}

    /* Preserve clickable Streamlit header toolbar controls */
    header[data-testid="stHeader"],
    header[data-testid="stHeader"] [data-testid="stToolbar"] {{
        pointer-events: none;
    }}
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] a,
    header[data-testid="stHeader"] [data-testid="stToolbarActions"],
    header[data-testid="stHeader"] [data-testid="stStatusWidget"],
    header[data-testid="stHeader"] [data-testid="stMainMenu"],
    header[data-testid="stHeader"] [data-testid="stAppDeployButton"],
    header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"] {{
        pointer-events: auto;
    }}

    /* Sidebar visual refinement */
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        padding-top: 0.65rem;
    }}
    .qa-sidebar-brand {{
        padding: 0.62rem 0.8rem;
        border-radius: var(--qa-radius-md);
        border: 1px solid var(--qa-border);
        background: var(--qa-surface);
        margin-bottom: 0.55rem;
    }}
    .qa-sidebar-brand-title {{
        font-size: 0.96rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin: 0;
        color: var(--qa-text);
    }}
    .qa-sidebar-brand-sub {{
        font-size: 0.74rem;
        color: var(--qa-text-muted);
        margin: 0.14rem 0 0 0;
    }}
    .qa-sidebar-section {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--qa-text-muted);
        margin: 0.55rem 0 0.28rem 0;
        padding-bottom: 0.2rem;
        border-bottom: 1px solid var(--qa-border);
    }}

    /* Masthead / Hero Bar */
    .qa-masthead {{
        padding: 0.95rem 1.25rem;
        border-radius: var(--qa-radius-lg);
        border: 1px solid var(--qa-border);
        background: var(--qa-hero-grad);
        box-shadow: var(--qa-shadow);
        margin-bottom: 0.75rem;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 0.85rem;
    }}
    .qa-masthead-left {{
        min-width: 260px;
        flex: 1 1 520px;
    }}
    .qa-masthead-kicker {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--qa-accent);
        margin-bottom: 0.2rem;
    }}
    .qa-masthead-title {{
        font-size: 1.42rem;
        font-weight: 750;
        letter-spacing: -0.02em;
        line-height: 1.22;
        margin: 0 0 0.28rem 0;
        color: var(--qa-text);
    }}
    .qa-masthead-sub {{
        font-size: 0.86rem;
        color: var(--qa-text-muted);
        margin: 0;
        line-height: 1.45;
    }}
    .qa-masthead-badges {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.42rem;
        align-items: center;
        justify-content: flex-end;
    }}
    .qa-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.26rem 0.62rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        border: 1px solid var(--qa-border);
        background: var(--qa-surface);
        color: var(--qa-text-secondary);
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }}
    .qa-pill--accent {{
        border-color: rgba(59, 130, 246, 0.35);
        background: var(--qa-accent-soft);
        color: var(--qa-accent);
    }}
    .qa-pill--ok {{
        border-color: rgba(16, 185, 129, 0.35);
        background: var(--qa-positive-soft);
        color: var(--qa-positive);
    }}
    .qa-pill--warn {{
        border-color: rgba(245, 158, 11, 0.35);
        background: var(--qa-warning-soft);
        color: var(--qa-warning);
    }}

    /* Workspace Navigation Tabs & Nested Chart Tabs */
    [data-testid="stMain"] [data-baseweb="tab-list"] {{
        gap: 0.4rem;
        border-bottom: 1px solid var(--qa-border);
        padding-bottom: 0.15rem;
        margin-bottom: 0.65rem;
    }}
    [data-testid="stMain"] [data-baseweb="tab-list"] [role="tab"] {{
        border-radius: var(--qa-radius-sm) var(--qa-radius-sm) 0 0;
        padding: 0.5rem 0.95rem;
        font-weight: 600;
        font-size: 0.9rem;
        transition: background-color 0.15s ease, color 0.15s ease;
    }}
    [data-testid="stMain"] [data-baseweb="tab-list"] [role="tab"]:hover {{
        background: var(--qa-accent-soft);
    }}
    [data-testid="stMain"] [data-baseweb="tab-list"] [role="tab"][aria-selected="true"] {{
        background: var(--qa-accent-soft);
        color: var(--qa-accent);
    }}
    [data-testid="stTabs"] [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        border-bottom: none;
        background: var(--qa-surface-alt);
        border: 1px solid var(--qa-border);
        border-radius: var(--qa-radius-md);
        padding: 0.25rem;
        gap: 0.25rem;
        flex-wrap: wrap;
    }}
    [data-testid="stTabs"] [data-testid="stTabs"] [data-baseweb="tab-list"] [role="tab"] {{
        border-radius: var(--qa-radius-sm);
        padding: 0.36rem 0.78rem;
        font-size: 0.83rem;
        border: none !important;
    }}
    [data-testid="stTabs"] [data-testid="stTabs"] [data-baseweb="tab-list"] [role="tab"][aria-selected="true"] {{
        background: var(--qa-accent);
        color: #ffffff !important;
    }}
    [data-testid="stTabs"] [data-testid="stTabs"] [data-baseweb="tab-list"] [role="tab"][aria-selected="true"] * {{
        color: #ffffff !important;
    }}

    /* Primary Metrics (st.metric) */
    div[data-testid="stMetric"] {{
        border: 1px solid var(--qa-border);
        border-radius: var(--qa-radius-md);
        padding: 0.8rem 1rem;
        background: var(--qa-surface);
        box-shadow: var(--qa-shadow);
        transition: border-color 0.15s ease, transform 0.15s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        border-color: var(--qa-border-strong);
    }}
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--qa-text-muted);
        letter-spacing: 0.01em;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-family: var(--qa-font-mono);
        font-variant-numeric: tabular-nums;
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--qa-text);
    }}

    /* Secondary KPI Responsive Grid */
    .qa-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(156px, 1fr));
        gap: 0.65rem;
        margin: 0.65rem 0 0.9rem 0;
    }}
    .qa-kpi-card {{
        border: 1px solid var(--qa-border);
        background: var(--qa-surface);
        border-radius: var(--qa-radius-md);
        padding: 0.68rem 0.82rem;
        min-width: 0;
    }}
    .qa-kpi-label {{
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--qa-text-muted);
        margin-bottom: 0.22rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .qa-kpi-value {{
        font-family: var(--qa-font-mono);
        font-variant-numeric: tabular-nums;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--qa-text);
        line-height: 1.25;
        word-break: break-word;
    }}
    .qa-kpi-value--pos {{ color: var(--qa-positive); }}
    .qa-kpi-value--neg {{ color: var(--qa-negative); }}
    .qa-kpi-value--warn {{ color: var(--qa-warning); }}
    .qa-kpi-value--muted {{ color: var(--qa-text-muted); }}
    .qa-kpi-sub {{
        font-size: 0.72rem;
        color: var(--qa-text-muted);
        margin-top: 0.16rem;
    }}

    /* Context Strip & Section Headings */
    .qa-context-strip {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem 0.9rem;
        padding: 0.65rem 0.9rem;
        border-radius: var(--qa-radius-md);
        border: 1px solid var(--qa-border);
        background: var(--qa-surface);
        margin-bottom: 0.65rem;
    }}
    .qa-context-title {{
        font-weight: 700;
        font-size: 0.98rem;
        color: var(--qa-text);
        display: flex;
        align-items: center;
        gap: 0.45rem;
        flex-wrap: wrap;
    }}
    .qa-code-tag {{
        font-family: var(--qa-font-mono);
        font-size: 0.82rem;
        padding: 0.12rem 0.45rem;
        border-radius: 6px;
        background: var(--qa-accent-soft);
        color: var(--qa-accent);
        font-weight: 600;
    }}
    .qa-section-head {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0.9rem 0 0.45rem 0;
        padding-bottom: 0.28rem;
        border-bottom: 1px solid var(--qa-border);
    }}
    .qa-section-title {{
        font-size: 1.02rem;
        font-weight: 700;
        color: var(--qa-text);
        margin: 0;
    }}
    .qa-section-sub {{
        font-size: 0.78rem;
        color: var(--qa-text-muted);
        margin: 0;
    }}

    /* Empty State & Preview Badges */
    .qa-empty-card {{
        border: 1px dashed var(--qa-border-strong);
        border-radius: var(--qa-radius-lg);
        background: var(--qa-surface);
        padding: 1.25rem 1.35rem;
        margin: 0.5rem 0 0.85rem 0;
    }}
    .qa-empty-title {{
        font-size: 1.04rem;
        font-weight: 700;
        color: var(--qa-text);
        margin: 0 0 0.35rem 0;
    }}
    .qa-empty-sub {{
        font-size: 0.86rem;
        color: var(--qa-text-muted);
        margin: 0 0 0.65rem 0;
        line-height: 1.5;
    }}
    .qa-chip-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin: 0.25rem 0 0.5rem 0;
    }}
    .qa-chip {{
        font-family: var(--qa-font-mono);
        font-size: 0.74rem;
        padding: 0.16rem 0.5rem;
        border-radius: 6px;
        border: 1px solid var(--qa-border);
        background: var(--qa-surface-alt);
        color: var(--qa-text-secondary);
    }}

    /* Chart & Export Panels */
    .qa-chart-note {{
        font-size: 0.8rem;
        color: var(--qa-text-muted);
        margin: 0.15rem 0 0.35rem 0;
    }}
    .qa-export-head {{
        padding: 0.65rem 0.9rem;
        border-radius: var(--qa-radius-md);
        border: 1px solid var(--qa-border);
        background: var(--qa-surface);
        margin: 0.8rem 0 0.55rem 0;
    }}

    /* Buttons & Dataframes */
    [data-testid="stButton"] > button[kind="primary"] {{
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        border: 1px solid rgba(37, 99, 235, 0.8) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: var(--qa-radius-sm) !important;
        box-shadow: 0 4px 12px -4px rgba(37, 99, 235, 0.45);
    }}
    [data-testid="stButton"] > button[kind="primary"]:hover:not(:disabled) {{
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        border-color: #60a5fa !important;
    }}
    [data-testid="stButton"] > button[kind="primary"]:disabled {{
        opacity: 0.48 !important;
        box-shadow: none !important;
    }}
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--qa-border);
        border-radius: var(--qa-radius-md);
        overflow: hidden;
    }}

    @media (max-width: 900px) {{
        [data-testid="stMainBlockContainer"] {{
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1.1rem !important;
        }}
        .qa-masthead {{
            padding: 0.8rem 0.95rem;
        }}
        .qa-masthead-title {{
            font-size: 1.18rem;
        }}
        .qa-masthead-badges {{
            justify-content: flex-start;
        }}
        .qa-kpi-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
    }}
    @media (max-width: 520px) {{
        .qa-kpi-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    </style>
    """


def apply_ui_plotly_theme(figure: Any, theme_mode: ThemeMode = "auto") -> Any:
    """Adapt a Plotly Figure's typography, grids, and hoverlabel to the active UI theme.

    Preserves transparent ``paper_bgcolor`` and ``plot_bgcolor`` so charts seamlessly
    match cards and existing figure invariants.
    """
    mode = normalize_theme_mode(theme_mode)
    fig = figure.__class__(figure) if hasattr(figure, "update_layout") else figure

    if mode == "dark":
        font_color = "#e2e8f0"
        muted_color = "#94a3b8"
        grid_color = "rgba(148, 163, 184, 0.16)"
        zero_color = "rgba(148, 163, 184, 0.30)"
        hover_bg = "rgba(15, 23, 42, 0.94)"
        hover_border = "rgba(96, 165, 250, 0.55)"
        hover_text = "#f8fafc"
        rs_bg = "rgba(30, 41, 59, 0.85)"
        rs_active = "rgba(37, 99, 235, 0.85)"
    elif mode == "light":
        font_color = "#0f172a"
        muted_color = "#475569"
        grid_color = "rgba(100, 116, 139, 0.18)"
        zero_color = "rgba(100, 116, 139, 0.32)"
        hover_bg = "rgba(255, 255, 255, 0.96)"
        hover_border = "rgba(37, 99, 235, 0.45)"
        hover_text = "#0f172a"
        rs_bg = "rgba(241, 245, 249, 0.95)"
        rs_active = "rgba(37, 99, 235, 0.16)"
    else:
        # Auto keeps font color inherited from Streamlit theme while polishing hover & controls.
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": _FONT_STACK},
            hoverlabel={
                "font": {"family": _FONT_STACK, "size": 12},
                "namelength": -1,
            },
        )
        return fig

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": _FONT_STACK, "color": font_color},
        title={"font": {"color": font_color}},
        legend={"font": {"color": font_color}},
        hoverlabel={
            "bgcolor": hover_bg,
            "bordercolor": hover_border,
            "font": {"family": _FONT_STACK, "size": 12, "color": hover_text},
            "namelength": -1,
        },
    )
    fig.update_xaxes(
        gridcolor=grid_color,
        zerolinecolor=zero_color,
        tickfont={"color": muted_color},
        title_font={"color": font_color},
    )
    fig.update_yaxes(
        gridcolor=grid_color,
        zerolinecolor=zero_color,
        tickfont={"color": muted_color},
        title_font={"color": font_color},
    )
    # Apply rangeselector theme only when an xaxis already defines rangeselector buttons.
    for axis_name in ("xaxis", "xaxis2", "xaxis3", "xaxis4"):
        axis_obj = getattr(fig.layout, axis_name, None)
        if axis_obj is not None and getattr(axis_obj, "rangeselector", None) is not None:
            rs = axis_obj.rangeselector
            if getattr(rs, "buttons", None):
                axis_obj.rangeselector.update(
                    bgcolor=rs_bg,
                    activecolor=rs_active,
                    font={"color": font_color, "size": 11},
                )

    # Update subplot titles without overwriting explicit cell annotation colors (e.g. heatmap).
    if fig.layout.annotations:
        for ann in fig.layout.annotations:
            ann_font = getattr(ann, "font", None)
            existing_color = getattr(ann_font, "color", None) if ann_font is not None else None
            if not existing_color:
                ann.update(font={"color": font_color})
    return fig


def inject_ui_theme_css(st_module: Any, theme_mode: ThemeMode = "auto") -> None:
    """Inject compiled theme CSS into the active Streamlit session."""
    if st_module is None:
        return
    st_module.markdown(
        build_theme_css(theme_mode, resolved_auto=None),
        unsafe_allow_html=True,
    )
