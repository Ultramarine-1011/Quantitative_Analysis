"""Neutral desktop-inspired Light/Dark/Auto CSS design system and Plotly theme harmonization."""

from __future__ import annotations

from typing import Any, Literal

ThemeMode = Literal["auto", "dark", "light"]

THEME_MODES: tuple[ThemeMode, ...] = ("auto", "dark", "light")
THEME_MODE_LABELS: dict[str, str] = {
    "auto": "跟随系统",
    "dark": "深色模式",
    "light": "浅色模式",
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
    """Theme-neutral tokens inheriting Streamlit's active Light/Dark canvas."""
    return """
        --qa-bg: transparent;
        --qa-bg-elevated: rgba(128, 134, 145, 0.05);
        --qa-surface: rgba(128, 134, 145, 0.055);
        --qa-surface-solid: var(--background-color, rgba(128, 134, 145, 0.06));
        --qa-surface-alt: rgba(128, 134, 145, 0.09);
        --qa-sidebar-bg: transparent;
        --qa-header-bg: transparent;
        --qa-border: rgba(128, 134, 145, 0.20);
        --qa-border-strong: rgba(128, 134, 145, 0.34);
        --qa-text: inherit;
        --qa-text-secondary: inherit;
        --qa-text-muted: rgba(113, 120, 132, 0.96);
        --qa-accent: #2563eb;
        --qa-accent-hover: #3b82f6;
        --qa-accent-soft: rgba(37, 99, 235, 0.11);
        --qa-positive: #059669;
        --qa-positive-soft: rgba(16, 185, 129, 0.13);
        --qa-negative: #dc2626;
        --qa-negative-soft: rgba(220, 38, 38, 0.13);
        --qa-warning: #d97706;
        --qa-warning-soft: rgba(245, 158, 11, 0.13);
        --qa-shadow: none;
        --qa-input-bg: transparent;
    """


def _token_vars(*, dark: bool) -> str:
    if dark:
        return """
        --qa-bg: #111214;
        --qa-bg-elevated: #16171a;
        --qa-surface: #181a1e;
        --qa-surface-solid: #1a1c20;
        --qa-surface-alt: #202227;
        --qa-sidebar-bg: #141518;
        --qa-header-bg: rgba(17, 18, 20, 0.88);
        --qa-border: rgba(255, 255, 255, 0.09);
        --qa-border-strong: rgba(255, 255, 255, 0.18);
        --qa-text: #eceef2;
        --qa-text-secondary: #c8ccd4;
        --qa-text-muted: #8b919d;
        --qa-accent: #60a5fa;
        --qa-accent-hover: #93c5fd;
        --qa-accent-soft: rgba(96, 165, 250, 0.14);
        --qa-positive: #34d399;
        --qa-positive-soft: rgba(16, 185, 129, 0.15);
        --qa-negative: #fb7185;
        --qa-negative-soft: rgba(244, 63, 94, 0.15);
        --qa-warning: #fbbf24;
        --qa-warning-soft: rgba(245, 158, 11, 0.14);
        --qa-shadow: none;
        --qa-input-bg: #16181c;
        """
    return """
        --qa-bg: #f7f7f8;
        --qa-bg-elevated: #ffffff;
        --qa-surface: #ffffff;
        --qa-surface-solid: #ffffff;
        --qa-surface-alt: #f1f2f4;
        --qa-sidebar-bg: #f2f3f5;
        --qa-header-bg: rgba(247, 247, 248, 0.90);
        --qa-border: rgba(17, 20, 28, 0.09);
        --qa-border-strong: rgba(17, 20, 28, 0.18);
        --qa-text: #181a1f;
        --qa-text-secondary: #374151;
        --qa-text-muted: #6b7280;
        --qa-accent: #2563eb;
        --qa-accent-hover: #1d4ed8;
        --qa-accent-soft: rgba(37, 99, 235, 0.09);
        --qa-positive: #059669;
        --qa-positive-soft: rgba(5, 150, 105, 0.10);
        --qa-negative: #dc2626;
        --qa-negative-soft: rgba(220, 38, 38, 0.10);
        --qa-warning: #b45309;
        --qa-warning-soft: rgba(217, 119, 6, 0.12);
        --qa-shadow: none;
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
    """Force consistent Streamlit container, sidebar, popover, and input surfaces in explicit modes."""
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
                --qa-border: rgba(255, 255, 255, 0.11);
                --qa-border-strong: rgba(255, 255, 255, 0.22);
            }
        }
        """
    scheme = "dark" if theme_mode == "dark" else "light"
    st_bg = "#111214" if theme_mode == "dark" else "#f7f7f8"
    st_sbg = "#141518" if theme_mode == "dark" else "#f2f3f5"
    st_txt = "#eceef2" if theme_mode == "dark" else "#181a1f"
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
        [data-baseweb="popover"] > div,
        [data-baseweb="popover"] [role="dialog"],
        [data-baseweb="popover"] ul,
        [data-baseweb="popover"] li,
        [data-baseweb="calendar"],
        [data-baseweb="calendar"] div,
        [data-baseweb="calendar"] button,
        [data-baseweb="calendar"] select,
        [data-baseweb="calendar"] [role="gridcell"],
        [data-testid="stPopoverBody"] {{
            background-color: var(--qa-input-bg) !important;
            color: var(--qa-text) !important;
            border-color: var(--qa-border-strong) !important;
            -webkit-text-fill-color: var(--qa-text) !important;
        }}
        [data-baseweb="calendar"] [role="gridcell"]::before,
        [data-baseweb="calendar"] [role="gridcell"]::after {{
            background-color: transparent !important;
        }}
        [data-baseweb="calendar"] [aria-selected="true"],
        [data-baseweb="calendar"] [aria-selected="true"]::after {{
            background-color: #2563eb !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border-radius: 8px !important;
        }}
        [data-baseweb="calendar"] svg {{
            color: var(--qa-text-muted) !important;
            fill: var(--qa-text-muted) !important;
        }}
        [data-testid="stDateInputCalendar"],
        [data-testid="stDateInputCalendar"] [role="application"],
        [data-testid="stDateInputCalendar"] div,
        [data-testid="stDateInputCalendar"] span,
        [data-testid="stDateInputCalendar"] table,
        [data-testid="stDateInputCalendar"] th,
        [data-testid="stDateInputCalendar"] td,
        [data-testid="stDateInputCalendar"] button,
        [data-testid="stDateInputCalendar"] select,
        [data-testid="stDateInputCalendar"] [role="gridcell"],
        [data-testid="stDateInputCalendar"] [role="button"] {{
            background-color: var(--qa-input-bg) !important;
            color: var(--qa-text) !important;
            border-color: var(--qa-border-strong) !important;
            -webkit-text-fill-color: var(--qa-text) !important;
        }}
        [data-testid="stDateInputCalendar"] {{
            border: 1px solid var(--qa-border-strong) !important;
        }}
        [data-testid="stDateInputCalendar"] [role="button"]:hover {{
            background-color: var(--qa-surface-alt) !important;
        }}
        [data-testid="stDateInputCalendar"] [aria-selected="true"],
        [data-testid="stDateInputCalendar"] [aria-selected="true"] [role="button"],
        [data-testid="stDateInputCalendar"] [data-selected="true"] {{
            background-color: #2563eb !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border-radius: 6px !important;
        }}
        [data-testid="stDateInputCalendar"] [aria-disabled="true"],
        [data-testid="stDateInputCalendar"] [data-outside-month="true"] {{
            color: var(--qa-text-muted) !important;
            -webkit-text-fill-color: var(--qa-text-muted) !important;
            opacity: 0.55;
        }}
        [data-testid="stDateInputCalendar"] svg {{
            color: var(--qa-text-muted) !important;
            fill: var(--qa-text-muted) !important;
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
        [data-testid="stExpander"],
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] details > summary {{
            background-color: var(--qa-surface) !important;
            color: var(--qa-text) !important;
            border-color: var(--qa-border) !important;
            border-radius: 10px !important;
        }}
        [data-testid="stExpander"] summary:hover {{
            background-color: var(--qa-surface-alt) !important;
        }}
        [data-testid="stExpander"] summary *,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary p {{
            color: var(--qa-text) !important;
        }}
        [data-testid="stExpander"] summary svg {{
            color: var(--qa-text-muted) !important;
            fill: var(--qa-text-muted) !important;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: var(--qa-surface) !important;
            border-color: var(--qa-border) !important;
        }}
        [data-testid="stButton"] button:not([kind="primary"]),
        [data-testid="stDownloadButton"] button,
        [data-testid="stPopover"] button {{
            background-color: var(--qa-surface-solid) !important;
            color: var(--qa-text) !important;
            border: 1px solid var(--qa-border-strong) !important;
        }}
        [data-testid="stPopover"] button *,
        [data-testid="stPopover"] button svg {{
            color: var(--qa-text) !important;
            fill: var(--qa-text-muted) !important;
        }}
        [data-testid="stButton"] button:not([kind="primary"]):hover,
        [data-testid="stDownloadButton"] button:hover,
        [data-testid="stPopover"] button:hover {{
            border-color: var(--qa-accent) !important;
            color: var(--qa-accent) !important;
            background-color: var(--qa-surface-alt) !important;
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
            border-radius: 8px !important;
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
    """Compile complete neutral desktop CSS stylesheet for the chosen theme mode."""
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
        --qa-radius-lg: 12px;
        --qa-radius-md: 8px;
        --qa-radius-sm: 6px;
        {base_tokens}
    }}
    {overrides}

    html, body, .stApp {{
        font-family: var(--qa-font-sans);
    }}

    /* Scroll stability & clean desktop canvas spacing */
    [data-testid="stMain"] {{
        overflow-y: auto;
        overflow-anchor: none;
        overscroll-behavior-y: contain;
    }}
    [data-testid="stMainBlockContainer"] {{
        flex-shrink: 0;
        max-width: 1440px;
        padding-top: 3.45rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 1.75rem !important;
        padding-right: 1.75rem !important;
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

    /* Lightweight Desktop Rail Sidebar */
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        padding-top: 0.75rem;
    }}
    .qa-sidebar-brand {{
        padding: 0.45rem 0.65rem;
        border-radius: var(--qa-radius-md);
        border: 1px solid var(--qa-border);
        background: var(--qa-surface);
        margin-bottom: 0.6rem;
    }}
    .qa-sidebar-brand-title {{
        font-size: 0.94rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        margin: 0;
        color: var(--qa-text);
    }}
    .qa-sidebar-brand-sub {{
        font-size: 0.72rem;
        color: var(--qa-text-muted);
        margin: 0.1rem 0 0 0;
    }}
    .qa-sidebar-section {{
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--qa-text-muted);
        margin: 0.65rem 0 0.3rem 0;
        padding-bottom: 0.18rem;
        border-bottom: 1px solid var(--qa-border);
    }}
    section[data-testid="stSidebar"] [data-testid="stButton"] > button {{
        justify-content: flex-start !important;
        text-align: left !important;
        font-size: 0.85rem !important;
        padding: 0.38rem 0.68rem !important;
        border-radius: var(--qa-radius-sm) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {{
        background: var(--qa-accent-soft) !important;
        color: var(--qa-accent) !important;
        border: 1px solid rgba(37, 99, 235, 0.38) !important;
        font-weight: 650 !important;
        box-shadow: none !important;
    }}

    /* Slim Workspace Topbar */
    .qa-topbar {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem 0.9rem;
        padding: 0.48rem 0.85rem;
        border-radius: var(--qa-radius-md);
        border: 1px solid var(--qa-border);
        background: var(--qa-surface);
        margin-bottom: 0.75rem;
    }}
    .qa-topbar-left {{
        display: flex;
        align-items: center;
        gap: 0.45rem;
        flex-wrap: wrap;
        min-width: 0;
    }}
    .qa-topbar-title {{
        font-size: 0.98rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--qa-text);
    }}
    .qa-topbar-sep {{
        color: var(--qa-text-muted);
        font-size: 0.86rem;
    }}
    .qa-topbar-context {{
        font-size: 0.86rem;
        font-weight: 500;
        color: var(--qa-text-muted);
    }}
    .qa-topbar-badges {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.38rem;
        align-items: center;
    }}
    .qa-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.18rem 0.52rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        border: 1px solid var(--qa-border);
        background: var(--qa-surface-alt);
        color: var(--qa-text-secondary);
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }}
    .qa-pill--accent {{
        border-color: rgba(37, 99, 235, 0.32);
        background: var(--qa-accent-soft);
        color: var(--qa-accent);
    }}
    .qa-pill--ok {{
        border-color: rgba(16, 185, 129, 0.32);
        background: var(--qa-positive-soft);
        color: var(--qa-positive);
    }}
    .qa-pill--warn {{
        border-color: rgba(245, 158, 11, 0.35);
        background: var(--qa-warning-soft);
        color: var(--qa-warning);
    }}
    .qa-pill--danger {{
        border-color: rgba(220, 38, 38, 0.35);
        background: var(--qa-negative-soft);
        color: var(--qa-negative);
    }}

    /* Home Screen */
    .qa-home-hero {{
        padding: 0.9rem 0 0.8rem 0;
    }}
    .qa-home-title {{
        font-size: 1.72rem;
        font-weight: 720;
        letter-spacing: -0.025em;
        color: var(--qa-text);
        margin: 0 0 0.25rem 0;
    }}
    .qa-home-sub {{
        font-size: 0.92rem;
        color: var(--qa-text-muted);
        margin: 0;
        line-height: 1.5;
    }}
    .qa-btn-spacer {{
        height: 1.68rem;
    }}
    .qa-resume-title {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--qa-text-muted);
        margin-bottom: 0.15rem;
    }}
    .qa-resume-main {{
        font-size: 0.95rem;
        font-weight: 650;
        color: var(--qa-text);
    }}
    .qa-resume-meta {{
        font-size: 0.8rem;
        color: var(--qa-text-muted);
        margin: 0.12rem 0 0.5rem 0;
        font-variant-numeric: tabular-nums;
    }}

    /* Results-First Summary Bar & Takeaway & Delta Strip */
    .qa-summary-inline {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.45rem;
        padding: 0.42rem 0.65rem;
        border-radius: var(--qa-radius-sm);
        border: 1px solid var(--qa-border);
        background: var(--qa-surface);
        min-height: 2.38rem;
    }}
    .qa-summary-strong {{
        font-size: 0.86rem;
        font-weight: 650;
        color: var(--qa-text);
    }}
    .qa-summary-meta {{
        font-size: 0.78rem;
        color: var(--qa-text-muted);
        font-variant-numeric: tabular-nums;
    }}
    .qa-takeaway {{
        border: 1px solid var(--qa-border);
        border-left: 3px solid var(--qa-accent);
        background: var(--qa-surface);
        border-radius: var(--qa-radius-md);
        padding: 0.68rem 0.95rem;
        margin: 0.45rem 0 0.75rem 0;
    }}
    .qa-takeaway-head {{
        font-size: 0.94rem;
        font-weight: 650;
        color: var(--qa-text);
        line-height: 1.45;
    }}
    .qa-takeaway-detail {{
        font-size: 0.78rem;
        color: var(--qa-text-muted);
        margin-top: 0.18rem;
        font-variant-numeric: tabular-nums;
    }}
    .qa-delta-strip {{
        border: 1px solid var(--qa-border);
        background: var(--qa-surface);
        border-radius: var(--qa-radius-md);
        padding: 0.62rem 0.85rem;
        margin: 0.35rem 0 0.75rem 0;
    }}
    .qa-delta-header {{
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.35rem;
        margin-bottom: 0.45rem;
    }}
    .qa-delta-title {{
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--qa-text);
    }}
    .qa-delta-meta {{
        font-size: 0.75rem;
        color: var(--qa-text-muted);
        font-variant-numeric: tabular-nums;
    }}
    .qa-delta-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
        gap: 0.5rem;
    }}
    .qa-delta-item {{
        border: 1px solid var(--qa-border);
        border-radius: var(--qa-radius-sm);
        background: var(--qa-surface-alt);
        padding: 0.45rem 0.6rem;
    }}
    .qa-delta-label {{
        font-size: 0.72rem;
        color: var(--qa-text-muted);
        font-weight: 600;
    }}
    .qa-delta-values {{
        font-family: var(--qa-font-mono);
        font-variant-numeric: tabular-nums;
        font-size: 0.84rem;
        color: var(--qa-text);
        margin-top: 0.12rem;
    }}
    .qa-delta-diff {{
        font-family: var(--qa-font-mono);
        font-variant-numeric: tabular-nums;
        font-size: 0.75rem;
        font-weight: 650;
        margin-top: 0.1rem;
    }}

    /* Visual Asset Picker */
    .qa-picker-label {{
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--qa-text-muted);
        margin-bottom: 0.35rem;
    }}
    .qa-picker-empty {{
        border: 1px dashed var(--qa-border-strong);
        border-radius: var(--qa-radius-sm);
        padding: 0.55rem 0.75rem;
        font-size: 0.82rem;
        color: var(--qa-text-muted);
        margin-bottom: 0.45rem;
    }}
    .qa-asset-pill {{
        border: 1px solid var(--qa-border);
        border-radius: var(--qa-radius-sm);
        background: var(--qa-surface-alt);
        padding: 0.35rem 0.55rem;
        display: flex;
        flex-direction: column;
        min-height: 2.38rem;
        justify-content: center;
    }}
    .qa-asset-pill-name {{
        font-size: 0.81rem;
        font-weight: 600;
        color: var(--qa-text);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .qa-asset-pill-code {{
        font-family: var(--qa-font-mono);
        font-size: 0.69rem;
        color: var(--qa-text-muted);
    }}

    /* Chart Sub-Tabs */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {{
        border-bottom: none;
        background: var(--qa-surface-alt);
        border: 1px solid var(--qa-border);
        border-radius: var(--qa-radius-md);
        padding: 0.22rem;
        gap: 0.22rem;
        flex-wrap: wrap;
        margin-bottom: 0.55rem;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-list"] [role="tab"] {{
        border-radius: var(--qa-radius-sm);
        padding: 0.32rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 600;
        border: none !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-list"] [role="tab"][aria-selected="true"] {{
        background: var(--qa-surface);
        color: var(--qa-text) !important;
        border: 1px solid var(--qa-border) !important;
    }}

    /* Primary Metrics (st.metric) */
    div[data-testid="stMetric"] {{
        border: 1px solid var(--qa-border);
        border-radius: var(--qa-radius-md);
        padding: 0.72rem 0.92rem;
        background: var(--qa-surface);
    }}
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--qa-text-muted);
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-family: var(--qa-font-mono);
        font-variant-numeric: tabular-nums;
        font-size: 1.48rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--qa-text);
    }}

    /* Secondary KPI Responsive Grid */
    .qa-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(156px, 1fr));
        gap: 0.55rem;
        margin: 0.55rem 0 0.8rem 0;
    }}
    .qa-kpi-card {{
        border: 1px solid var(--qa-border);
        background: var(--qa-surface);
        border-radius: var(--qa-radius-md);
        padding: 0.6rem 0.78rem;
        min-width: 0;
    }}
    .qa-kpi-label {{
        font-size: 0.74rem;
        font-weight: 600;
        color: var(--qa-text-muted);
        margin-bottom: 0.18rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .qa-kpi-value {{
        font-family: var(--qa-font-mono);
        font-variant-numeric: tabular-nums;
        font-size: 1.02rem;
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
        font-size: 0.71rem;
        color: var(--qa-text-muted);
        margin-top: 0.14rem;
    }}

    /* Section Headings, Empty State & Chips */
    .qa-section-head {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 0.95rem 0 0.45rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid var(--qa-border);
    }}
    .qa-section-title {{
        font-size: 0.98rem;
        font-weight: 700;
        color: var(--qa-text);
        margin: 0;
    }}
    .qa-section-sub {{
        font-size: 0.76rem;
        color: var(--qa-text-muted);
        margin: 0;
    }}
    .qa-empty-card {{
        border: 1px dashed var(--qa-border-strong);
        border-radius: var(--qa-radius-md);
        background: var(--qa-surface);
        padding: 1rem 1.15rem;
        margin: 0.45rem 0 0.75rem 0;
    }}
    .qa-empty-title {{
        font-size: 0.98rem;
        font-weight: 680;
        color: var(--qa-text);
        margin: 0 0 0.28rem 0;
    }}
    .qa-empty-sub {{
        font-size: 0.84rem;
        color: var(--qa-text-muted);
        margin: 0 0 0.55rem 0;
        line-height: 1.5;
    }}
    .qa-chip-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.32rem;
        margin: 0.25rem 0 0.4rem 0;
    }}
    .qa-chip {{
        font-family: var(--qa-font-mono);
        font-size: 0.72rem;
        padding: 0.14rem 0.46rem;
        border-radius: var(--qa-radius-sm);
        border: 1px solid var(--qa-border);
        background: var(--qa-surface-alt);
        color: var(--qa-text-secondary);
    }}
    .qa-chart-note {{
        font-size: 0.78rem;
        color: var(--qa-text-muted);
        margin: 0.1rem 0 0.3rem 0;
    }}

    /* Flat Primary Buttons & Clean Dataframes */
    [data-testid="stMain"] [data-testid="stButton"] > button[kind="primary"] {{
        background: #2563eb !important;
        border: 1px solid #1d4ed8 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: var(--qa-radius-sm) !important;
        box-shadow: none !important;
    }}
    [data-testid="stMain"] [data-testid="stButton"] > button[kind="primary"]:hover:not(:disabled) {{
        background: #1d4ed8 !important;
        border-color: #1e40af !important;
    }}
    [data-testid="stButton"] > button[kind="primary"]:disabled {{
        opacity: 0.45 !important;
    }}
    [data-testid="stMain"] [data-testid="stHorizontalBlock"] [data-testid="stButton"] button {{
        padding-inline: 0.45rem !important;
        font-size: 0.84rem !important;
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
            padding-top: 3.1rem !important;
        }}
        .qa-btn-spacer {{
            display: none;
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


def _separate_legend_and_rangeselector(fig: Any) -> None:
    """Prevent horizontal legend from colliding with rangeselector buttons on narrower panes."""
    if not hasattr(fig, "layout") or not hasattr(fig, "update_layout"):
        return
    xaxis = getattr(fig.layout, "xaxis", None)
    rs = getattr(xaxis, "rangeselector", None) if xaxis is not None else None
    rs_buttons = getattr(rs, "buttons", None) if rs is not None else None
    if not rs_buttons or len(getattr(fig, "data", ()) or ()) < 4:
        return
    cur_margin = fig.layout.margin.to_plotly_json() if getattr(fig.layout, "margin", None) else {}
    has_xtitle = False
    for ax_name in ("xaxis", "xaxis2", "xaxis3"):
        ax = getattr(fig.layout, ax_name, None)
        t = getattr(getattr(ax, "title", None), "text", None) if ax is not None else None
        if t:
            has_xtitle = True
    cur_b = int(cur_margin.get("b") or 38)
    cur_t = int(cur_margin.get("t") or 62)
    cur_margin["b"] = max(cur_b, 92 if has_xtitle else 76)
    cur_margin["t"] = max(cur_t, 56)
    fig.update_layout(
        margin=cur_margin,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18 if has_xtitle else -0.12,
            "x": 0.0,
            "xanchor": "left",
        },
    )


def apply_ui_plotly_theme(figure: Any, theme_mode: ThemeMode = "auto") -> Any:
    """Adapt a Plotly Figure's typography, grids, and hoverlabel to the active UI theme."""
    mode = normalize_theme_mode(theme_mode)
    fig = figure.__class__(figure) if hasattr(figure, "update_layout") else figure
    _separate_legend_and_rangeselector(fig)

    if mode == "dark":
        font_color = "#eceef2"
        muted_color = "#8b919d"
        grid_color = "rgba(255, 255, 255, 0.08)"
        zero_color = "rgba(255, 255, 255, 0.18)"
        hover_bg = "rgba(22, 24, 28, 0.96)"
        hover_border = "rgba(96, 165, 250, 0.50)"
        hover_text = "#f8fafc"
        rs_bg = "rgba(32, 34, 39, 0.92)"
        rs_active = "rgba(37, 99, 235, 0.85)"
    elif mode == "light":
        font_color = "#181a1f"
        muted_color = "#6b7280"
        grid_color = "rgba(17, 20, 28, 0.08)"
        zero_color = "rgba(17, 20, 28, 0.18)"
        hover_bg = "rgba(255, 255, 255, 0.97)"
        hover_border = "rgba(37, 99, 235, 0.40)"
        hover_text = "#181a1f"
        rs_bg = "rgba(241, 242, 244, 0.95)"
        rs_active = "rgba(37, 99, 235, 0.14)"
    else:
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
    if (
        getattr(fig.layout, "xaxis", None) is not None
        and getattr(fig.layout.xaxis, "rangeselector", None) is not None
    ):
        fig.update_xaxes(
            rangeselector={
                "bgcolor": rs_bg,
                "activecolor": rs_active,
                "font": {"color": font_color, "size": 11},
            }
        )
    if getattr(fig.layout, "annotations", None):
        updated_annotations = []
        for ann in fig.layout.annotations:
            ann_font = dict(ann.font.to_plotly_json()) if ann.font else {}
            if not ann_font.get("color"):
                ann_font["color"] = muted_color
            ann_dict = ann.to_plotly_json()
            ann_dict["font"] = ann_font
            updated_annotations.append(ann_dict)
        fig.update_layout(annotations=updated_annotations)
    return fig


def inject_ui_theme_css(st_module: Any, theme_mode: ThemeMode = "auto") -> None:
    """Inject responsive UI theme CSS into the active Streamlit document."""
    if st_module is None:
        return
    mode = normalize_theme_mode(theme_mode)
    # In 'auto' mode, keep translucent theme-neutral tokens so text always inherits
    # Streamlit's actual active canvas color without dark-text/light-canvas mismatches.
    st_module.markdown(
        build_theme_css(mode, resolved_auto=None),
        unsafe_allow_html=True,
    )
