"""Unified Markdown/HTML/JSON/CSV report popover, downloads, and inline preview."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from contracts import AnalysisResult, MultiAssetResult
from ui.components.tables import render_section_heading
from ui.services_bridge import get_or_build_report_exports

_EXTRA_CSV_LABELS: dict[str, str] = {
    "weights": "权重与风险贡献 (.csv)",
    "backtest": "回测净值序列 (.csv)",
    "returns": "对齐收益率 (.csv)",
    "correlation": "相关系数矩阵 (.csv)",
    "covariance": "年化协方差矩阵 (.csv)",
}


def _render_download_buttons(
    container: Any,
    exports: Mapping[str, Any],
    result: AnalysisResult | MultiAssetResult,
    *,
    file_prefix: str,
    key_prefix: str,
) -> None:
    c1, c2 = container.columns(2)
    c1.download_button(
        label="Markdown 报告 (.md)",
        data=exports["md"].encode("utf-8"),
        file_name=f"{file_prefix}_report.md",
        mime="text/markdown",
        key=f"{key_prefix}_dl_md",
        width="stretch",
    )
    c2.download_button(
        label="HTML 网页报告 (.html)",
        data=exports["html"].encode("utf-8"),
        file_name=f"{file_prefix}_report.html",
        mime="text/html",
        key=f"{key_prefix}_dl_html",
        width="stretch",
    )
    c3, c4 = container.columns(2)
    c3.download_button(
        label="结构化结果 (.json)",
        data=exports["json"].encode("utf-8"),
        file_name=f"{file_prefix}_result.json",
        mime="application/json",
        key=f"{key_prefix}_dl_json",
        width="stretch",
    )
    csv_label = "OHLCV 明细 (.csv)" if isinstance(result, AnalysisResult) else "对齐价格 (.csv)"
    c4.download_button(
        label=csv_label,
        data=exports["csv"],
        file_name=f"{file_prefix}_data.csv",
        mime="text/csv",
        key=f"{key_prefix}_dl_csv",
        width="stretch",
    )
    if isinstance(result, MultiAssetResult):
        extra_csvs = exports.get("extra_csvs") or {}
        if extra_csvs:
            container.caption("附加分析明细表")
            for frame_key, payload in extra_csvs.items():
                container.download_button(
                    label=_EXTRA_CSV_LABELS.get(frame_key, f"{frame_key}.csv"),
                    data=payload,
                    file_name=f"{file_prefix}_{frame_key}.csv",
                    mime="text/csv",
                    key=f"{key_prefix}_dl_extra_{frame_key}",
                    width="stretch",
                )


def render_export_popover(
    st_module: Any,
    state: dict[str, Any],
    result: AnalysisResult | MultiAssetResult,
    *,
    file_prefix: str,
    skipped_charts: Mapping[str, str] | None,
    key_prefix: str,
) -> dict[str, Any]:
    """Render a compact popover button in the action bar with all report & CSV exports."""
    if st_module is None:
        return {}
    exports = get_or_build_report_exports(state, result, skipped_charts=skipped_charts)
    if hasattr(st_module, "popover"):
        with st_module.popover("导出报告与数据", width="stretch"):
            st_module.caption("报告与导出口径严格对齐，参数或区间变化时自动重新生成。")
            _render_download_buttons(
                st_module, exports, result, file_prefix=file_prefix, key_prefix=key_prefix
            )
    else:
        with st_module.expander("导出报告与数据", expanded=False):
            _render_download_buttons(
                st_module, exports, result, file_prefix=file_prefix, key_prefix=key_prefix
            )
    return exports


def render_report_preview_drawer(
    st_module: Any,
    state: dict[str, Any],
    result: AnalysisResult | MultiAssetResult,
    *,
    skipped_charts: Mapping[str, str] | None,
) -> None:
    """Render collapsible inline Markdown report preview without duplicating download keys."""
    if st_module is None:
        return
    exports = get_or_build_report_exports(state, result, skipped_charts=skipped_charts)
    with st_module.expander("在线预览完整研究报告 (Markdown)", expanded=False):
        st_module.markdown(exports["md"])


def render_report_and_download_section(
    st_module: Any,
    state: dict[str, Any],
    result: AnalysisResult | MultiAssetResult,
    *,
    file_prefix: str,
    skipped_charts: Mapping[str, str] | None,
    key_prefix: str,
) -> None:
    """Render unified Markdown/HTML/JSON/CSV download buttons and inline report preview."""
    if st_module is None:
        return
    exports = get_or_build_report_exports(state, result, skipped_charts=skipped_charts)

    render_section_heading(
        st_module,
        "研究报告与数据导出",
        "严格对齐 result.json 与报告渲染口径 · 参数变更自动更新缓存",
    )
    _render_download_buttons(
        st_module, exports, result, file_prefix=file_prefix, key_prefix=key_prefix
    )
    with st_module.expander("在线预览研究报告 (Markdown)", expanded=False):
        st_module.markdown(exports["md"])
