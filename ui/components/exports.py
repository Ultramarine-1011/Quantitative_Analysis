"""Unified Markdown/HTML/JSON/CSV report preview and download panel."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from contracts import AnalysisResult, MultiAssetResult
from ui.components.tables import render_section_heading
from ui.services_bridge import get_or_build_report_exports


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
        "📑 统一研究报告与数据导出",
        "严格对齐 result.json 与报告渲染口径 · 参数或图表变更自动更新缓存",
    )

    cols = st_module.columns(4)
    cols[0].download_button(
        label="📄 Markdown 报告 (.md)",
        data=exports["md"].encode("utf-8"),
        file_name=f"{file_prefix}_report.md",
        mime="text/markdown",
        key=f"{key_prefix}_dl_md",
        width="stretch",
    )
    cols[1].download_button(
        label="🌐 HTML 报告 (.html)",
        data=exports["html"].encode("utf-8"),
        file_name=f"{file_prefix}_report.html",
        mime="text/html",
        key=f"{key_prefix}_dl_html",
        width="stretch",
    )
    cols[2].download_button(
        label="🧾 结构化 JSON (.json)",
        data=exports["json"].encode("utf-8"),
        file_name=f"{file_prefix}_result.json",
        mime="application/json",
        key=f"{key_prefix}_dl_json",
        width="stretch",
    )
    csv_label = (
        "📊 OHLCV 指标明细 (.csv)" if isinstance(result, AnalysisResult) else "📊 对齐价格 (.csv)"
    )
    cols[3].download_button(
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
            dl_cols = st_module.columns(len(extra_csvs))
            label_map = {
                "weights": "⚖️ 权重与风险贡献 (.csv)",
                "backtest": "📈 回测净值序列 (.csv)",
                "returns": "📉 对齐收益率 (.csv)",
                "correlation": "🔗 相关系数矩阵 (.csv)",
                "covariance": "📐 年化协方差矩阵 (.csv)",
            }
            for col, (frame_key, payload) in zip(dl_cols, extra_csvs.items(), strict=False):
                col.download_button(
                    label=label_map.get(frame_key, f"{frame_key}.csv"),
                    data=payload,
                    file_name=f"{file_prefix}_{frame_key}.csv",
                    mime="text/csv",
                    key=f"{key_prefix}_dl_extra_{frame_key}",
                    width="stretch",
                )

    with st_module.expander("👁️ 在线预览研究报告 (Markdown)", expanded=False):
        st_module.markdown(exports["md"])
