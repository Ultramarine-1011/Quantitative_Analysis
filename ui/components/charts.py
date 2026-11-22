"""Tabbed Plotly chart explorer component with skipped-chart semantic disclosures."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from typing import Any

from reporting import format_reason_label

_PLOTLY_CONFIG: dict[str, Any] = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def render_chart_explorer(
    st_module: Any,
    figs: Mapping[str, tuple[str, str, Any]],
    skipped: Mapping[str, str],
    order: Sequence[str],
    labels: Mapping[str, str],
    *,
    key_prefix: str,
) -> None:
    """Render ordered Plotly figures inside compact tabs and disclose any skipped charts."""
    if st_module is None:
        return
    available = [ct for ct in order if ct in figs]
    if len(available) == 1:
        ct = available[0]
        title, subtitle, fig = figs[ct]
        heading = labels.get(ct, title)
        st_module.markdown(f"**{heading}**")
        if subtitle:
            st_module.markdown(
                f'<div class="qa-chart-note">{html.escape(subtitle)}</div>',
                unsafe_allow_html=True,
            )
        st_module.plotly_chart(
            fig,
            width="stretch",
            key=f"{key_prefix}_plotly_{ct}_0",
            config=_PLOTLY_CONFIG,
        )
    elif len(available) > 1:
        tab_titles = [labels.get(ct, figs[ct][0]) for ct in available]
        tabs = st_module.tabs(tab_titles)
        for idx, (tab, ct) in enumerate(zip(tabs, available, strict=False)):
            _, subtitle, fig = figs[ct]
            with tab:
                if subtitle:
                    st_module.markdown(
                        f'<div class="qa-chart-note">{html.escape(subtitle)}</div>',
                        unsafe_allow_html=True,
                    )
                st_module.plotly_chart(
                    fig,
                    width="stretch",
                    key=f"{key_prefix}_plotly_{ct}_{idx}",
                    config=_PLOTLY_CONFIG,
                )

    if skipped:
        reasons = [
            f"`{labels.get(ct, ct)}` 已跳过（{format_reason_label(reason)}）"
            for ct, reason in skipped.items()
        ]
        st_module.info("图表口径限制提示：" + "；".join(reasons))
