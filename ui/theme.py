"""Claude Dark CSS design system and Plotly chart harmonization."""

from __future__ import annotations

from typing import Any

CLAUDE_DARK_THEME = "claude-dark"

CLAUDE_DARK_TOKENS: dict[str, str] = {
    "bg_deep": "#1f1e1d",
    "bg_primary": "#262624",
    "bg_secondary": "#30302e",
    "bg_panel": "#30302e",
    "table_bg_alt": "#2b2b29",
    "table_hover_bg": "#353430",
    "text_heading": "#faf9f5",
    "text_normal": "#e8e6dc",
    "text_muted": "#c2c0b6",
    "text_faint": "#9c9a92",
    "border_faint": "#3d3d3a",
    "border_muted": "#3d3d3a",
    "border_divider": "#3d3d3a",
    "border_normal": "#4d4c48",
    "border_bold": "#4d4c48",
    "border_button": "#4d4c48",
    "border_quote": "#d97757",
    "accent": "#d97757",
    "accent_hover": "#c56547",
    "accent_blue": "#6a9bcc",
    "accent_green": "#788c5d",
    "color_red": "#c46a5a",
    "color_yellow": "#d4a574",
    "color_green": "#8fa67a",
    "color_blue": "#6a9bcc",
    "color_purple": "#c4a484",
    "color_magenta": "#c4897a",
    "link_color": "#8fb4d6",
    "link_hover_color": "#d97757",
    "code_inline_bg": "#1f1e1d",
    "code_inline_fg": "#f0d7cc",
    "radius_small": "2px",
    "radius_medium": "4px",
    "radius_large": "6px",
}

_SERIF_FONT_STACK = (
    'Lora, Georgia, "Noto Serif SC", "Source Han Serif SC", "Source Han Serif CN", '
    '"思源宋体", "Songti SC", "STSong", SimSun, PMingLiU, serif'
)
_HEADING_FONT_STACK = (
    '"Source Sans 3", -apple-system, "Segoe UI", "PingFang SC", '
    '"Noto Sans SC", "Source Han Sans SC", "Microsoft YaHei", sans-serif'
)
_MONO_FONT_STACK = '"JetBrains Mono", Consolas, Menlo, monospace'

_CLAUDE_SERIF_URL = (
    "https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700"
    "&family=Noto+Serif+SC:wght@400;600;700"
    "&family=Source+Sans+3:ital,wght@0,400..700;1,400..700&display=swap"
)
_JETBRAINS_MONO_URL = "https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,100..800;1,100..800&display=swap"


def build_theme_css() -> str:
    """Compile the complete Claude Dark CSS stylesheet matching ``claude-dark-mpe``."""
    t = CLAUDE_DARK_TOKENS
    return f"""
    <style>
    @import url("{_CLAUDE_SERIF_URL}");
    @import url("{_JETBRAINS_MONO_URL}");

    :root, .stApp {{
        color-scheme: dark;

        --bg-deep: {t["bg_deep"]};
        --bg-primary: {t["bg_primary"]};
        --bg-secondary: {t["bg_secondary"]};
        --bg-panel: {t["bg_panel"]};

        --text-heading: {t["text_heading"]};
        --text-normal: {t["text_normal"]};
        --text-muted: {t["text_muted"]};
        --text-faint: {t["text_faint"]};

        --border-faint: {t["border_faint"]};
        --border-muted: {t["border_muted"]};
        --border-normal: {t["border_normal"]};
        --border-bold: {t["border_bold"]};
        --border-divider: {t["border_divider"]};
        --border-button: {t["border_button"]};
        --border-block: {t["border_faint"]};
        --border-quote: {t["border_quote"]};

        --accent: {t["accent"]};
        --accent-hover: {t["accent_hover"]};
        --accent-blue: {t["accent_blue"]};
        --accent-green: {t["accent_green"]};

        --color-red: {t["color_red"]};
        --color-yellow: {t["color_yellow"]};
        --color-green: {t["color_green"]};
        --color-blue: {t["color_blue"]};
        --color-purple: {t["color_purple"]};
        --color-magenta: {t["color_magenta"]};

        --selection: rgb(217 119 87 / 0.32);
        --text-mark-bg: rgb(217 119 87 / 0.22);

        --link-color: {t["link_color"]};
        --link-hover-color: {t["link_hover_color"]};

        --table-border: {t["border_faint"]};
        --table-border-strong: {t["border_bold"]};
        --table-header-bg: {t["bg_secondary"]};
        --table-bg: transparent;
        --table-bg-alt: {t["table_bg_alt"]};
        --table-hover-bg: {t["table_hover_bg"]};

        --code-inline-bg: {t["code_inline_bg"]};
        --code-inline-fg: {t["code_inline_fg"]};
        --code-bg: {t["bg_deep"]};
        --code-fg: {t["text_normal"]};

        --font-family: {_SERIF_FONT_STACK};
        --heading-font-family: {_HEADING_FONT_STACK};
        --code-font-family: {_MONO_FONT_STACK};

        --font-size: 17.5px;
        --ui-font-size: 13.5px;
        --line-height: 1.78;
        --letter-spacing: 0.004em;
        --font-weight: 400;
        --font-weight-bold: 600;
        --font-weight-heading: 560;

        --code-font-size: 13px;
        --code-line-height: 1.65;
        --code-inline-font-size: 0.86em;

        --radius-small: {t["radius_small"]};
        --radius-medium: {t["radius_medium"]};
        --radius-large: {t["radius_large"]};

        --background-color: var(--bg-primary);
        --secondary-background-color: var(--bg-secondary);
        --text-color: var(--text-normal);
    }}

    /* === Canvas & Base Typography === */
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {{
        background-color: var(--bg-primary) !important;
        color: var(--text-normal) !important;
        font-family: var(--font-family);
        font-weight: var(--font-weight);
        text-rendering: optimizeLegibility;
    }}
    ::selection {{
        background-color: var(--selection);
        color: var(--text-heading);
    }}

    /* Scrollbars */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: var(--bg-primary);
    }}
    ::-webkit-scrollbar-thumb {{
        background: var(--border-bold);
        border-radius: var(--radius-medium);
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--text-faint);
    }}

    /* Scroll stability & desktop canvas spacing */
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

    /* Header Bar */
    [data-testid="stHeader"] {{
        background-color: rgb(38 38 36 / 0.90) !important;
        backdrop-filter: blur(8px);
        border-bottom: 1px solid rgb(61 61 58 / 0.65);
    }}
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
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] svg {{
        color: var(--text-muted) !important;
    }}
    header[data-testid="stHeader"] button:hover {{
        color: var(--text-heading) !important;
        background-color: var(--table-hover-bg) !important;
    }}

    /* Headings & UI Controls Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-family: var(--heading-font-family) !important;
        color: var(--text-heading) !important;
        letter-spacing: -0.018em;
        line-height: 1.28;
        font-weight: var(--font-weight-heading);
    }}
    strong, b {{
        font-weight: var(--font-weight-bold);
        color: var(--text-heading);
    }}
    del, s {{
        color: var(--text-faint);
    }}
    mark {{
        background-color: var(--text-mark-bg);
        color: var(--text-heading);
        border-radius: var(--radius-small);
        padding: 1px 2px;
    }}

    /* Markdown Prose */
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] {{
        font-family: var(--font-family);
        color: var(--text-normal);
        line-height: var(--line-height);
        letter-spacing: var(--letter-spacing);
    }}
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] li {{
        font-family: var(--font-family);
        font-size: 15.5px;
        line-height: var(--line-height);
        letter-spacing: var(--letter-spacing);
        color: var(--text-normal);
    }}
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] h1 {{
        font-size: 1.65rem;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid rgb(217 119 87 / 0.55);
        margin-top: 1.25rem;
        margin-bottom: 0.75rem;
    }}
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] h2 {{
        font-size: 1.32rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid var(--border-bold);
        margin-top: 1.2rem;
        margin-bottom: 0.65rem;
    }}
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] li {{
        font-size: var(--font-size);
        line-height: var(--line-height);
    }}
    [data-testid="stMarkdownContainer"] a {{
        color: var(--link-color);
        text-decoration: none;
    }}
    [data-testid="stMarkdownContainer"] a:hover {{
        color: var(--link-hover-color);
        text-decoration: underline;
    }}
    [data-testid="stMarkdownContainer"] hr {{
        background-color: transparent;
        height: 0;
        border: none;
        border-bottom: 1px solid var(--border-muted);
        margin: 1.25rem 0;
    }}
    [data-testid="stMarkdownContainer"] blockquote {{
        background-color: var(--bg-secondary);
        color: var(--text-muted);
        font-style: normal;
        border: 1px solid var(--border-muted);
        border-left: 2px solid var(--border-quote);
        border-radius: var(--radius-medium);
        padding: 0.5rem 1.25rem;
        margin: 1rem 0 1.25rem;
    }}
    [data-testid="stMarkdownContainer"] blockquote p {{
        color: var(--text-muted);
    }}

    /* Inline Code & Code Blocks / JSON */
    [data-testid="stMain"] code,
    section[data-testid="stSidebar"] code,
    [data-testid="stPopoverBody"] code,
    [role="dialog"] code {{
        font-family: var(--code-font-family) !important;
        font-size: var(--code-inline-font-size) !important;
        line-height: var(--code-line-height);
        background-color: var(--code-inline-bg) !important;
        color: var(--code-inline-fg) !important;
        border: 1px solid var(--border-faint) !important;
        border-radius: var(--radius-small) !important;
        padding: 2px 5px 1px !important;
        box-shadow: none !important;
    }}
    [data-testid="stMain"] pre,
    [data-testid="stCode"],
    [data-testid="stJson"] {{
        font-family: var(--code-font-family) !important;
        font-size: var(--code-font-size) !important;
        line-height: var(--code-line-height) !important;
        background-color: var(--code-bg) !important;
        color: var(--code-fg) !important;
        border: 1px solid var(--border-muted) !important;
        border-radius: var(--radius-large) !important;
    }}
    [data-testid="stMain"] pre code {{
        border: none !important;
        padding: 0 !important;
        background-color: transparent !important;
        color: inherit !important;
        font-size: var(--code-font-size) !important;
    }}

    /* Captions & Labels */
    [data-testid="stMain"] label,
    section[data-testid="stSidebar"] label,
    [data-testid="stWidgetLabel"] *,
    [data-testid="stButton"] button,
    [data-testid="stDownloadButton"] button,
    [data-testid="stPopover"] button,
    [data-testid="stTabs"] [role="tab"],
    [data-testid="stExpander"] summary,
    [data-baseweb="input"],
    [data-baseweb="select"],
    [data-baseweb="popover"],
    [role="dialog"] {{
        font-family: var(--heading-font-family) !important;
    }}
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p {{
        font-size: var(--ui-font-size) !important;
        font-weight: 500;
        color: var(--text-normal) !important;
    }}
    [data-testid="stMain"] [data-testid="stCaptionContainer"],
    [data-testid="stMain"] [data-testid="stCaptionContainer"] *,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {{
        font-family: var(--heading-font-family) !important;
        font-size: 12.5px !important;
        line-height: 1.5 !important;
        color: var(--text-muted) !important;
    }}

    /* === Sidebar Navigation Rail === */
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-panel) !important;
        border-right: 1px solid var(--border-muted) !important;
        color: var(--text-normal) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        padding-top: 0.75rem;
    }}
    section[data-testid="stSidebar"] * {{
        color: var(--text-normal);
    }}
    .qa-sidebar-brand {{
        padding: 0.52rem 0.72rem;
        border-radius: var(--radius-medium);
        border: 1px solid var(--border-muted);
        background: var(--bg-primary);
        margin-bottom: 0.65rem;
    }}
    .qa-sidebar-brand-title {{
        font-family: var(--heading-font-family);
        font-size: 0.98rem;
        font-weight: 600;
        letter-spacing: -0.015em;
        margin: 0;
        color: var(--text-heading) !important;
    }}
    .qa-sidebar-brand-sub {{
        font-family: var(--heading-font-family);
        font-size: 0.74rem;
        color: var(--text-muted) !important;
        margin: 0.1rem 0 0 0;
    }}
    .qa-sidebar-section {{
        font-family: var(--heading-font-family);
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-faint) !important;
        margin: 0.7rem 0 0.32rem 0;
        padding-bottom: 0.2rem;
        border-bottom: 1px solid var(--border-muted);
    }}
    section[data-testid="stSidebar"] [data-testid="stButton"] > button {{
        justify-content: flex-start !important;
        text-align: left !important;
        font-size: var(--ui-font-size) !important;
        padding: 0.4rem 0.68rem !important;
        border-radius: var(--radius-medium) !important;
        background-color: transparent !important;
        border: 1px solid transparent !important;
        color: var(--text-muted) !important;
        box-shadow: none !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stButton"] > button:hover {{
        background-color: var(--table-hover-bg) !important;
        border-color: var(--border-muted) !important;
        color: var(--text-heading) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {{
        background: rgb(217 119 87 / 0.16) !important;
        color: var(--accent) !important;
        border: 1px solid rgb(217 119 87 / 0.48) !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stButton"] > button {{
        justify-content: center !important;
        text-align: center !important;
        padding: 0.28rem 0.35rem !important;
        font-size: 0.78rem !important;
        background-color: var(--bg-primary) !important;
        border: 1px solid var(--border-button) !important;
        color: var(--text-muted) !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stButton"] > button:hover {{
        background-color: var(--table-hover-bg) !important;
        border-color: var(--accent) !important;
        color: var(--text-heading) !important;
    }}

    /* === Forms, Inputs, Selectboxes & Checkboxes === */
    [data-baseweb="input"],
    [data-baseweb="base-input"],
    [data-baseweb="textarea"],
    [data-baseweb="input"] > div,
    [data-baseweb="base-input"] > div,
    [data-baseweb="textarea"] > div,
    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div > div,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSelectbox"] .react-aria-ComboBox [role="group"],
    [data-testid="stSelectbox"] input,
    [data-testid="stNumberInputContainer"],
    [data-testid="stDateInputField"],
    [data-testid="stDateInputField"] > div,
    [data-testid="stNumberInput"] [data-baseweb="input"],
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stTextArea"] textarea {{
        background-color: var(--bg-deep) !important;
        color: var(--text-normal) !important;
        -webkit-text-fill-color: var(--text-normal) !important;
        border-color: var(--border-normal) !important;
        border-radius: var(--radius-medium) !important;
        font-size: var(--ui-font-size) !important;
    }}
    [data-baseweb="input"]:focus-within,
    [data-baseweb="textarea"]:focus-within,
    [data-baseweb="select"] > div:focus-within,
    [data-testid="stSelectbox"] .react-aria-ComboBox [role="group"]:focus-within,
    [data-testid="stNumberInputContainer"]:focus-within,
    [data-testid="stDateInputField"]:focus-within {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px rgb(217 119 87 / 0.35) !important;
    }}
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {{
        color: var(--text-faint) !important;
        -webkit-text-fill-color: var(--text-faint) !important;
    }}
    [data-testid="stNumberInput"] button {{
        background-color: var(--bg-deep) !important;
        color: var(--text-muted) !important;
        border-color: var(--border-normal) !important;
    }}
    [data-testid="stNumberInput"] button:hover {{
        background-color: var(--table-hover-bg) !important;
        color: var(--text-heading) !important;
    }}
    [data-testid="stDateInputField"] *,
    [data-testid="stSelectbox"] span,
    [data-testid="stSelectbox"] div {{
        color: var(--text-normal) !important;
    }}
    [data-testid="stSelectbox"] svg,
    [data-testid="stNumberInputStepDown"] svg,
    [data-testid="stNumberInputStepUp"] svg,
    [data-testid="stDateInputCalendarButton"] svg {{
        color: var(--text-muted) !important;
        fill: var(--text-muted) !important;
    }}
    [data-testid="stDateInputCalendarButton"] {{
        background-color: transparent !important;
    }}
    [data-testid="stCheckbox"] input[type="checkbox"],
    input[type="checkbox"] {{
        accent-color: var(--accent) !important;
    }}
    [data-testid="stCheckbox"] label,
    [data-testid="stCheckbox"] p {{
        color: var(--text-normal) !important;
        font-size: var(--ui-font-size) !important;
    }}

    /* === Popovers, Dropdown Menus, Calendar & Dialogs === */
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"],
    [role="listbox"],
    .react-aria-Popover,
    .react-aria-ListBox,
    [data-testid="stSelectboxVirtualDropdown"],
    [data-testid="stPopoverBody"],
    [role="dialog"],
    [data-testid="stDialog"],
    [data-testid="stDialog"] > div {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-normal) !important;
        border-color: var(--border-bold) !important;
        border-radius: var(--radius-large) !important;
        box-shadow: none !important;
    }}
    [data-baseweb="popover"] ul,
    [data-baseweb="menu"] {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-bold) !important;
    }}
    [data-baseweb="popover"] li,
    [role="option"] {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-normal) !important;
        font-size: var(--ui-font-size) !important;
    }}
    [data-baseweb="popover"] li:hover,
    [role="option"]:hover,
    [role="option"][aria-selected="true"] {{
        background-color: var(--table-hover-bg) !important;
        color: var(--text-heading) !important;
    }}
    [data-baseweb="calendar"],
    [data-baseweb="calendar"] div,
    [data-baseweb="calendar"] button,
    [data-baseweb="calendar"] select,
    [data-baseweb="calendar"] [role="gridcell"],
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
        background-color: var(--bg-secondary) !important;
        color: var(--text-normal) !important;
        border-color: var(--border-bold) !important;
        -webkit-text-fill-color: var(--text-normal) !important;
    }}
    [data-testid="stDateInputCalendar"] {{
        border: 1px solid var(--border-bold) !important;
        border-radius: var(--radius-large) !important;
    }}
    [data-baseweb="calendar"] [role="gridcell"]::before,
    [data-baseweb="calendar"] [role="gridcell"]::after {{
        background-color: transparent !important;
    }}
    [data-testid="stDateInputCalendar"] [role="button"]:hover {{
        background-color: var(--table-hover-bg) !important;
    }}
    [data-baseweb="calendar"] [aria-selected="true"],
    [data-baseweb="calendar"] [aria-selected="true"]::after,
    [data-testid="stDateInputCalendar"] [aria-selected="true"],
    [data-testid="stDateInputCalendar"] [aria-selected="true"] [role="button"],
    [data-testid="stDateInputCalendar"] [data-selected="true"] {{
        background-color: var(--accent) !important;
        color: var(--text-heading) !important;
        -webkit-text-fill-color: var(--text-heading) !important;
        border-radius: var(--radius-medium) !important;
    }}
    [data-testid="stDateInputCalendar"] [aria-disabled="true"],
    [data-testid="stDateInputCalendar"] [data-outside-month="true"] {{
        color: var(--text-faint) !important;
        -webkit-text-fill-color: var(--text-faint) !important;
        opacity: 0.6;
    }}
    [data-baseweb="calendar"] svg,
    [data-testid="stDateInputCalendar"] svg {{
        color: var(--text-muted) !important;
        fill: var(--text-muted) !important;
    }}

    /* === Buttons === */
    [data-testid="stPopoverBody"] [data-testid="stDownloadButton"] button,
    [data-testid="stPopoverBody"] [data-testid="stButton"] button:not([kind="primary"]) {{
        background-color: var(--bg-primary) !important;
        color: var(--text-normal) !important;
        border: 1px solid var(--border-button) !important;
        border-radius: var(--radius-medium) !important;
        font-size: var(--ui-font-size) !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }}
    [data-testid="stMain"] [data-testid="stButton"] button:not([kind="primary"]),
    [data-testid="stDownloadButton"] button,
    [data-testid="stPopover"] button {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-normal) !important;
        border: 1px solid var(--border-button) !important;
        border-radius: var(--radius-medium) !important;
        font-size: var(--ui-font-size) !important;
        font-weight: 500 !important;
        box-shadow: none !important;
    }}
    [data-testid="stPopover"] button *,
    [data-testid="stPopover"] button svg {{
        color: var(--text-normal) !important;
        fill: var(--text-muted) !important;
    }}
    [data-testid="stMain"] [data-testid="stButton"] button:not([kind="primary"]):hover:not(:disabled),
    [data-testid="stDownloadButton"] button:hover:not(:disabled),
    [data-testid="stPopover"] button:hover:not(:disabled) {{
        border-color: var(--accent) !important;
        color: var(--text-heading) !important;
        background-color: var(--table-hover-bg) !important;
    }}
    [data-testid="stMain"] [data-testid="stButton"] > button[kind="primary"] {{
        background: var(--accent) !important;
        border: 1px solid var(--accent-hover) !important;
        color: var(--text-heading) !important;
        font-size: var(--ui-font-size) !important;
        font-weight: 600 !important;
        border-radius: var(--radius-medium) !important;
        box-shadow: none !important;
    }}
    [data-testid="stMain"] [data-testid="stButton"] > button[kind="primary"]:hover:not(:disabled) {{
        background: var(--accent-hover) !important;
        border-color: #b5583c !important;
        color: var(--text-heading) !important;
    }}
    [data-testid="stButton"] > button:disabled,
    [data-testid="stDownloadButton"] > button:disabled {{
        opacity: 0.45 !important;
    }}
    [data-testid="stMain"] [data-testid="stHorizontalBlock"] [data-testid="stButton"] button {{
        padding-inline: 0.5rem !important;
    }}

    /* === Containers & Expanders === */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-muted) !important;
        border-radius: var(--radius-large) !important;
        box-shadow: none !important;
    }}
    [data-testid="stExpander"],
    [data-testid="stExpander"] details {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-normal) !important;
        border: 1px solid var(--border-muted) !important;
        border-radius: var(--radius-large) !important;
        box-shadow: none !important;
    }}
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details > summary {{
        background-color: var(--bg-secondary) !important;
        color: var(--text-heading) !important;
        border-color: var(--border-muted) !important;
        border-radius: var(--radius-large) !important;
        font-size: var(--ui-font-size) !important;
        font-weight: 560 !important;
    }}
    [data-testid="stExpander"] summary:hover {{
        background-color: var(--table-hover-bg) !important;
    }}
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {{
        color: var(--text-heading) !important;
    }}
    [data-testid="stExpander"] summary svg {{
        color: var(--text-muted) !important;
        fill: var(--text-muted) !important;
    }}

    /* === Callouts / Alerts (Claude Dark Callout Tokens) === */
    [data-testid="stAlert"] {{
        box-shadow: none !important;
        border-radius: var(--radius-medium) !important;
    }}
    [data-testid="stAlertContainer"] {{
        background-color: rgb(106 155 204 / 0.10) !important;
        border: 1px solid rgb(106 155 204 / 0.38) !important;
        border-left: 2px solid rgb(106 155 204 / 0.75) !important;
        border-radius: var(--radius-medium) !important;
        color: var(--text-normal) !important;
    }}
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
        background-color: rgb(212 165 116 / 0.10) !important;
        border-color: rgb(212 165 116 / 0.40) !important;
        border-left-color: rgb(212 165 116 / 0.75) !important;
    }}
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{
        background-color: rgb(196 106 90 / 0.11) !important;
        border-color: rgb(196 106 90 / 0.42) !important;
        border-left-color: rgb(196 106 90 / 0.80) !important;
    }}
    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
        background-color: rgb(143 166 122 / 0.10) !important;
        border-color: rgb(143 166 122 / 0.38) !important;
        border-left-color: rgb(143 166 122 / 0.75) !important;
    }}
    [data-testid="stAlert"] *,
    [data-testid="stAlertContainer"] * {{
        color: var(--text-normal) !important;
        font-family: var(--heading-font-family) !important;
        font-size: var(--ui-font-size) !important;
        line-height: 1.55 !important;
    }}
    [data-testid="stAlertContainer"] [data-testid="stAlertContentInfo"] svg {{
        color: var(--color-blue) !important;
        fill: var(--color-blue) !important;
    }}
    [data-testid="stAlertContainer"] [data-testid="stAlertContentWarning"] svg {{
        color: var(--color-yellow) !important;
        fill: var(--color-yellow) !important;
    }}
    [data-testid="stAlertContainer"] [data-testid="stAlertContentError"] svg {{
        color: var(--color-red) !important;
        fill: var(--color-red) !important;
    }}
    [data-testid="stAlertContainer"] [data-testid="stAlertContentSuccess"] svg {{
        color: var(--color-green) !important;
        fill: var(--color-green) !important;
    }}

    /* === Tables & DataFrames === */
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--table-border) !important;
        border-radius: var(--radius-medium) !important;
        background-color: var(--bg-primary) !important;
        overflow: hidden;
    }}
    [data-testid="stMarkdownContainer"] table {{
        margin-block: 1rem 1.25rem;
        width: 100%;
        max-width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        overflow: hidden;
        border-radius: var(--radius-medium);
        font-family: var(--heading-font-family);
        font-size: var(--ui-font-size);
    }}
    [data-testid="stMarkdownContainer"] table thead {{
        background-color: var(--table-header-bg);
    }}
    [data-testid="stMarkdownContainer"] table th {{
        color: var(--text-heading);
        font-weight: var(--font-weight-bold);
        text-align: left;
        line-height: 1.25;
        padding: 7px 12px;
        border-bottom: 1px solid var(--table-border);
        border-right: 1px solid var(--table-border);
    }}
    [data-testid="stMarkdownContainer"] table tr:first-of-type > th {{
        border-top: 1px solid var(--table-border);
    }}
    [data-testid="stMarkdownContainer"] table tr:last-of-type > th {{
        border-bottom: 2px solid var(--table-border-strong);
    }}
    [data-testid="stMarkdownContainer"] table td {{
        background-color: var(--table-bg);
        color: var(--text-normal);
        padding: 4px 12px;
        line-height: 1.5;
        border-bottom: 1px solid var(--table-border);
        border-right: 1px solid var(--table-border);
    }}
    [data-testid="stMarkdownContainer"] table th:first-of-type,
    [data-testid="stMarkdownContainer"] table td:first-of-type {{
        border-left: 1px solid var(--table-border);
    }}
    [data-testid="stMarkdownContainer"] table tr:nth-child(even) > td {{
        background-color: var(--table-bg-alt);
    }}
    [data-testid="stMarkdownContainer"] table tr:hover > td {{
        background-color: var(--table-hover-bg);
    }}

    /* === Workspace Topbar & Status Pills === */
    .qa-topbar {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem 0.9rem;
        padding: 0.52rem 0.88rem;
        border-radius: var(--radius-large);
        border: 1px solid var(--border-muted);
        background: var(--bg-secondary);
        margin-bottom: 0.78rem;
        font-family: var(--heading-font-family);
    }}
    .qa-topbar-left {{
        display: flex;
        align-items: center;
        gap: 0.48rem;
        flex-wrap: wrap;
        min-width: 0;
    }}
    .qa-topbar-title {{
        font-family: var(--heading-font-family);
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: -0.015em;
        color: var(--text-heading);
    }}
    .qa-topbar-sep {{
        color: var(--text-faint);
        font-size: 0.86rem;
    }}
    .qa-topbar-context {{
        font-family: var(--heading-font-family);
        font-size: 0.86rem;
        font-weight: 500;
        color: var(--text-muted);
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
        padding: 0.18rem 0.54rem;
        border-radius: var(--radius-medium);
        font-family: var(--heading-font-family);
        font-size: 0.74rem;
        font-weight: 600;
        border: 1px solid var(--border-normal);
        background: var(--bg-primary);
        color: var(--text-muted);
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }}
    .qa-pill--accent {{
        border-color: rgb(217 119 87 / 0.45);
        background: rgb(217 119 87 / 0.14);
        color: var(--accent);
    }}
    .qa-pill--ok {{
        border-color: rgb(143 166 122 / 0.45);
        background: rgb(143 166 122 / 0.14);
        color: var(--color-green);
    }}
    .qa-pill--warn {{
        border-color: rgb(212 165 116 / 0.45);
        background: rgb(212 165 116 / 0.14);
        color: var(--color-yellow);
    }}
    .qa-pill--danger {{
        border-color: rgb(196 106 90 / 0.45);
        background: rgb(196 106 90 / 0.14);
        color: var(--color-red);
    }}

    /* === Home Screen === */
    .qa-home-hero {{
        padding: 0.85rem 0 0.85rem 0;
    }}
    .qa-home-title {{
        font-family: var(--heading-font-family);
        font-size: 1.85rem;
        font-weight: var(--font-weight-heading);
        letter-spacing: -0.018em;
        line-height: 1.28;
        color: var(--text-heading);
        margin: 0 0 0.5rem 0;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid rgb(217 119 87 / 0.55);
    }}
    .qa-home-sub {{
        font-family: var(--font-family);
        font-size: 0.98rem;
        color: var(--text-muted);
        margin: 0;
        line-height: var(--line-height);
        letter-spacing: var(--letter-spacing);
    }}
    .qa-btn-spacer {{
        height: 1.72rem;
    }}
    .qa-resume-title {{
        font-family: var(--heading-font-family);
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-faint);
        margin-bottom: 0.16rem;
    }}
    .qa-resume-main {{
        font-family: var(--heading-font-family);
        font-size: 0.96rem;
        font-weight: 600;
        color: var(--text-heading);
    }}
    .qa-resume-meta {{
        font-family: var(--code-font-family);
        font-size: 0.78rem;
        color: var(--text-muted);
        margin: 0.14rem 0 0.52rem 0;
        font-variant-numeric: tabular-nums;
    }}

    /* === Results-First Summary Bar, Takeaway & Delta Strip === */
    .qa-summary-inline {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.48rem;
        padding: 0.44rem 0.7rem;
        border-radius: var(--radius-medium);
        border: 1px solid var(--border-muted);
        background: var(--bg-secondary);
        min-height: 2.38rem;
        font-family: var(--heading-font-family);
    }}
    .qa-summary-strong {{
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-heading);
    }}
    .qa-summary-meta {{
        font-size: 0.78rem;
        color: var(--text-muted);
        font-variant-numeric: tabular-nums;
    }}
    .qa-takeaway {{
        border: 1px solid var(--border-muted);
        border-left: 2px solid var(--border-quote);
        background: var(--bg-secondary);
        border-radius: var(--radius-medium);
        padding: 0.72rem 1.05rem;
        margin: 0.48rem 0 0.78rem 0;
    }}
    .qa-takeaway-head {{
        font-family: var(--font-family);
        font-size: 0.98rem;
        font-weight: 600;
        color: var(--text-heading);
        line-height: 1.55;
    }}
    .qa-takeaway-detail {{
        font-family: var(--heading-font-family);
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 0.22rem;
        font-variant-numeric: tabular-nums;
    }}
    .qa-delta-strip {{
        border: 1px solid var(--border-muted);
        background: var(--bg-secondary);
        border-radius: var(--radius-large);
        padding: 0.65rem 0.88rem;
        margin: 0.35rem 0 0.78rem 0;
    }}
    .qa-delta-header {{
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.35rem;
        margin-bottom: 0.48rem;
        font-family: var(--heading-font-family);
    }}
    .qa-delta-title {{
        font-size: 0.84rem;
        font-weight: 600;
        color: var(--text-heading);
    }}
    .qa-delta-meta {{
        font-size: 0.75rem;
        color: var(--text-muted);
        font-variant-numeric: tabular-nums;
    }}
    .qa-delta-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
        gap: 0.5rem;
    }}
    .qa-delta-item {{
        border: 1px solid var(--border-muted);
        border-radius: var(--radius-medium);
        background: var(--table-bg-alt);
        padding: 0.45rem 0.62rem;
    }}
    .qa-delta-label {{
        font-family: var(--heading-font-family);
        font-size: 0.72rem;
        color: var(--text-muted);
        font-weight: 600;
    }}
    .qa-delta-values {{
        font-family: var(--code-font-family);
        font-variant-numeric: tabular-nums;
        font-size: 0.84rem;
        color: var(--text-normal);
        margin-top: 0.14rem;
    }}
    .qa-delta-diff {{
        font-family: var(--code-font-family);
        font-variant-numeric: tabular-nums;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.12rem;
    }}

    /* === Visual Asset Picker === */
    .qa-picker-label {{
        font-family: var(--heading-font-family);
        font-size: var(--ui-font-size);
        font-weight: 600;
        color: var(--text-muted);
        margin-bottom: 0.35rem;
    }}
    .qa-picker-empty {{
        border: 1px dashed var(--border-bold);
        border-radius: var(--radius-medium);
        padding: 0.58rem 0.78rem;
        font-family: var(--heading-font-family);
        font-size: var(--ui-font-size);
        color: var(--text-muted);
        background: var(--bg-primary);
        margin-bottom: 0.45rem;
    }}
    .qa-asset-pill {{
        border: 1px solid var(--border-muted);
        border-radius: var(--radius-medium);
        background: var(--bg-primary);
        padding: 0.36rem 0.58rem;
        display: flex;
        flex-direction: column;
        min-height: 2.38rem;
        justify-content: center;
    }}
    .qa-asset-pill-name {{
        font-family: var(--heading-font-family);
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-heading);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .qa-asset-pill-code {{
        font-family: var(--code-font-family);
        font-size: 0.69rem;
        color: var(--text-faint);
    }}

    /* === Chart Sub-Tabs === */
    [data-testid="stTabs"] [data-baseweb="tab-list"],
    [data-testid="stTabs"] [role="tablist"] {{
        border-bottom: none !important;
        background: var(--bg-deep) !important;
        border: 1px solid var(--border-muted) !important;
        border-radius: var(--radius-large) !important;
        padding: 0.22rem !important;
        gap: 0.22rem !important;
        flex-wrap: wrap;
        margin-bottom: 0.55rem;
        box-shadow: none !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-list"] [role="tab"],
    [data-testid="stTabs"] [role="tablist"] [role="tab"],
    [data-testid="stTabs"] [data-testid="stTab"] {{
        border-radius: var(--radius-medium) !important;
        padding: 0.34rem 0.78rem !important;
        height: auto !important;
        font-family: var(--heading-font-family) !important;
        font-size: var(--ui-font-size) !important;
        font-weight: 600 !important;
        color: var(--text-muted) !important;
        border: 1px solid transparent !important;
        background: transparent !important;
    }}
    [data-testid="stTabs"] [data-testid="stTab"] *,
    [data-testid="stTabs"] [role="tab"] * {{
        color: inherit !important;
        font-family: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-list"] [role="tab"]:hover,
    [data-testid="stTabs"] [role="tablist"] [role="tab"]:hover,
    [data-testid="stTabs"] [data-testid="stTab"]:hover {{
        color: var(--text-heading) !important;
        background: var(--table-hover-bg) !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-list"] [role="tab"][aria-selected="true"],
    [data-testid="stTabs"] [role="tablist"] [role="tab"][aria-selected="true"],
    [data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {{
        background: var(--bg-secondary) !important;
        color: var(--text-heading) !important;
        border: 1px solid var(--border-bold) !important;
    }}
    [data-testid="stTabs"] [data-testid="stTab"]::after,
    [data-testid="stTabs"] [data-testid="stTab"]::before,
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-baseweb="tab-border"],
    [data-testid="stTabs"] [role="tablist"] > div:not([data-testid="stTab"]):not([role="tab"]) {{
        display: none !important;
        background-color: transparent !important;
    }}

    /* === Primary Metrics (st.metric) & Secondary KPI Strip === */
    div[data-testid="stMetric"] {{
        border: 1px solid var(--border-muted);
        border-radius: var(--radius-large);
        padding: 0.72rem 0.92rem;
        background: var(--bg-secondary);
    }}
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
        font-family: var(--heading-font-family);
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--text-muted);
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-family: var(--code-font-family);
        font-variant-numeric: tabular-nums;
        font-size: 1.46rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: var(--text-heading);
    }}
    .qa-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(156px, 1fr));
        gap: 0.55rem;
        margin: 0.55rem 0 0.8rem 0;
    }}
    .qa-kpi-card {{
        border: 1px solid var(--border-muted);
        background: var(--bg-secondary);
        border-radius: var(--radius-medium);
        padding: 0.62rem 0.8rem;
        min-width: 0;
    }}
    .qa-kpi-label {{
        font-family: var(--heading-font-family);
        font-size: 0.74rem;
        font-weight: 600;
        color: var(--text-muted);
        margin-bottom: 0.18rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .qa-kpi-value {{
        font-family: var(--code-font-family);
        font-variant-numeric: tabular-nums;
        font-size: 1.02rem;
        font-weight: 600;
        color: var(--text-heading);
        line-height: 1.25;
        word-break: break-word;
    }}
    .qa-kpi-value--pos {{ color: var(--color-green); }}
    .qa-kpi-value--neg {{ color: var(--color-red); }}
    .qa-kpi-value--warn {{ color: var(--color-yellow); }}
    .qa-kpi-value--muted {{ color: var(--text-faint); }}
    .qa-kpi-sub {{
        font-family: var(--heading-font-family);
        font-size: 0.71rem;
        color: var(--text-faint);
        margin-top: 0.16rem;
    }}

    /* === Section Headings, Empty State & Chips === */
    .qa-section-head {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--border-bold);
    }}
    .qa-section-title {{
        font-family: var(--heading-font-family);
        font-size: 1.02rem;
        font-weight: var(--font-weight-heading);
        letter-spacing: -0.015em;
        color: var(--text-heading);
        margin: 0;
    }}
    .qa-section-sub {{
        font-family: var(--heading-font-family);
        font-size: 0.76rem;
        color: var(--text-faint);
        margin: 0;
    }}
    .qa-empty-card {{
        border: 1px dashed var(--border-bold);
        border-radius: var(--radius-large);
        background: var(--bg-secondary);
        padding: 1rem 1.15rem;
        margin: 0.45rem 0 0.75rem 0;
    }}
    .qa-empty-title {{
        font-family: var(--heading-font-family);
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-heading);
        margin: 0 0 0.3rem 0;
    }}
    .qa-empty-sub {{
        font-family: var(--font-family);
        font-size: 0.88rem;
        color: var(--text-muted);
        margin: 0 0 0.58rem 0;
        line-height: 1.6;
    }}
    .qa-chip-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.32rem;
        margin: 0.25rem 0 0.4rem 0;
    }}
    .qa-chip {{
        font-family: var(--code-font-family);
        font-size: 0.72rem;
        padding: 0.16rem 0.48rem;
        border-radius: var(--radius-small);
        border: 1px solid var(--border-muted);
        background: var(--bg-deep);
        color: var(--code-inline-fg);
    }}
    .qa-chart-note {{
        font-family: var(--heading-font-family);
        font-size: 0.78rem;
        color: var(--text-muted);
        margin: 0.1rem 0 0.3rem 0;
    }}

    /* === Plotly Modebar === */
    .js-plotly-plot .plotly .modebar {{
        background: transparent !important;
    }}
    .js-plotly-plot .plotly .modebar-btn path {{
        fill: var(--text-faint) !important;
    }}
    .js-plotly-plot .plotly .modebar-btn:hover path {{
        fill: var(--text-heading) !important;
    }}
    .js-plotly-plot .plotly .modebar-btn.active path {{
        fill: var(--accent) !important;
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


def apply_ui_plotly_theme(figure: Any) -> Any:
    """Harmonize a Plotly Figure's chrome, typography, grids, and hoverlabel to Claude Dark."""
    fig = figure.__class__(figure) if hasattr(figure, "update_layout") else figure
    _separate_legend_and_rangeselector(fig)

    t = CLAUDE_DARK_TOKENS
    font_color = t["text_normal"]
    heading_color = t["text_heading"]
    muted_color = t["text_muted"]
    faint_color = t["text_faint"]
    grid_color = t["border_muted"]
    zero_color = t["border_bold"]

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": _HEADING_FONT_STACK, "color": font_color},
        title={"font": {"family": _HEADING_FONT_STACK, "color": heading_color}},
        legend={
            "font": {"family": _HEADING_FONT_STACK, "color": muted_color},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={
            "bgcolor": t["bg_secondary"],
            "bordercolor": t["border_bold"],
            "font": {"family": _HEADING_FONT_STACK, "size": 12, "color": heading_color},
            "namelength": -1,
        },
    )
    fig.update_xaxes(
        gridcolor=grid_color,
        zerolinecolor=zero_color,
        linecolor=grid_color,
        tickfont={"color": faint_color, "family": _HEADING_FONT_STACK},
        title_font={"color": muted_color, "family": _HEADING_FONT_STACK},
    )
    fig.update_yaxes(
        gridcolor=grid_color,
        zerolinecolor=zero_color,
        linecolor=grid_color,
        tickfont={"color": faint_color, "family": _HEADING_FONT_STACK},
        title_font={"color": muted_color, "family": _HEADING_FONT_STACK},
    )

    xaxis = getattr(fig.layout, "xaxis", None)
    if xaxis is not None and getattr(xaxis, "showspikes", False):
        fig.update_xaxes(spikecolor="rgba(156, 154, 146, 0.65)")

    if xaxis is not None and getattr(xaxis, "rangeselector", None) is not None:
        fig.update_xaxes(
            rangeselector={
                "bgcolor": t["bg_secondary"],
                "activecolor": "rgba(217, 119, 87, 0.32)",
                "bordercolor": t["border_bold"],
                "borderwidth": 1,
                "font": {"color": font_color, "size": 11, "family": _HEADING_FONT_STACK},
            }
        )

    for trace in getattr(fig, "data", ()) or ():
        if hasattr(trace, "colorbar") and getattr(trace.colorbar, "len", None) is not None:
            trace.update(
                colorbar={
                    "tickfont": {"color": faint_color, "family": _HEADING_FONT_STACK},
                    "title": {"font": {"color": muted_color, "family": _HEADING_FONT_STACK}},
                }
            )
        marker = getattr(trace, "marker", None)
        cb = getattr(marker, "colorbar", None) if marker is not None else None
        if cb is not None and getattr(cb, "len", None) is not None:
            trace.update(
                marker_colorbar={
                    "tickfont": {"color": faint_color, "family": _HEADING_FONT_STACK},
                    "title": {"font": {"color": muted_color, "family": _HEADING_FONT_STACK}},
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


def inject_ui_theme_css(st_module: Any) -> None:
    """Inject the Claude Dark CSS stylesheet into the active Streamlit document."""
    if st_module is None:
        return
    st_module.markdown(build_theme_css(), unsafe_allow_html=True)
