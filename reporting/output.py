"""Dispatch requested renderers and publish their files with the result bundle."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from artifacts.store import write_result_outputs
from contracts import AnalysisResult, MultiAssetResult


def write_outputs(
    result: AnalysisResult | MultiAssetResult,
    *,
    out_dir: Path | None,
    write_csv: bool,
    export_csv: Path | None,
    include_data: bool,
    compact: bool,
    report_formats: Sequence[str] = (),
    chart_formats: Sequence[str] = (),
    chart_types: Sequence[str] | None = None,
) -> AnalysisResult | MultiAssetResult:
    files: dict[Path, bytes] = {}
    attachments = {}
    charts = ()
    skipped = {}

    def collect(target: Path, payload: bytes) -> Path:
        files[target] = payload
        return target

    if out_dir is not None:
        root = out_dir.resolve()
        if chart_formats:
            from reporting.chart_renderer import render_charts

            rendered = render_charts(
                result, root, formats=chart_formats, chart_types=chart_types, file_writer=collect
            )
            charts, skipped = rendered.charts, rendered.skipped
            if rendered.warnings:
                result = replace(
                    result, warnings=tuple(dict.fromkeys((*result.warnings, *rendered.warnings)))
                )
        reports = ()
        if report_formats:
            from reporting.report_renderer import render_report

            reports = render_report(
                result,
                root,
                formats=report_formats,
                charts=charts,
                skipped_charts=skipped,
                file_writer=collect,
            ).reports
        attachments = {
            item.artifact_key: (item.path.relative_to(root).as_posix(), files[item.path])
            for item in (*charts, *reports)
        }

    return write_result_outputs(
        result,
        out_dir=out_dir,
        write_csv=write_csv,
        export_csv=export_csv,
        include_data=include_data,
        compact=compact,
        attachments=attachments,
    )
