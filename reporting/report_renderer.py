"""Independent Markdown and HTML report renderer for single-asset, compare, and portfolio results.

Consumes AnalysisResult / MultiAssetResult directly (or via load_result_from_artifacts) with zero
duplicate data fetching or quantitative recomputation and zero heavyweight plotting dependencies.
"""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contracts import AnalysisResult, MultiAssetResult
from reporting.chart_renderer import ChartArtifact
from reporting.formatting import (
    describe_unit,
    format_asset_type_label,
    format_drawdown_percent,
    format_integer,
    format_metric_value,
    format_p_value,
    format_percent,
    format_price_with_unit,
    format_ratio,
    format_reason_label,
    is_valid_number,
)

SUPPORTED_REPORT_FORMATS: tuple[str, ...] = ("md", "html")

_FORMAT_ALIASES: dict[str, str] = {
    "md": "md",
    "markdown": "md",
    "html": "html",
    "htm": "html",
}

_SINGLE_METRIC_LABELS: dict[str, str] = {
    "latest_close": "最新收盘价",
    "latest_level": "最新利率水平",
    "level_change": "区间绝对变动",
    "level_min": "区间最低水平",
    "level_max": "区间最高水平",
    "level_diff_std": "日变动标准差",
    "cumulative_return": "区间累计收益率",
    "cagr": "年化复合收益率 (CAGR)",
    "annualized_volatility": "年化波动率",
    "max_drawdown": "历史最大回撤 (水下)",
    "sharpe_ratio": "Sharpe 比率",
    "sortino_ratio": "Sortino 比率",
    "calmar_ratio": "Calmar 比率",
    "var_95": "历史 VaR (95%, 单期)",
    "cvar_95": "条件 CVaR (95%, 单期)",
    "trading_days": "年化交易周期",
    "return_convention": "收益率口径",
    "latest_signal": "最新技术信号",
}


@dataclass(frozen=True)
class ReportArtifact:
    """Rendered report output metadata and text payload."""

    format: str
    path: Path
    artifact_key: str
    content: str


@dataclass(frozen=True)
class ReportRenderOutput:
    """Collection of generated reports and non-fatal rendering notes."""

    reports: tuple[ReportArtifact, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Badge:
    label: str
    tone: str  # "neutral" | "ok" | "warn" | "danger" | "info"


@dataclass(frozen=True)
class _Callout:
    tone: str  # "info" | "warn" | "danger"
    title: str
    body: str


@dataclass(frozen=True)
class _SectionBlock:
    kind: str  # "table" | "kv" | "bullets" | "paragraph"
    title: str | None = None
    headers: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    items: tuple[tuple[str, str], ...] = ()
    bullets: tuple[str, ...] = ()
    text: str = ""
    numeric_cols: tuple[int, ...] = ()


@dataclass(frozen=True)
class _Section:
    heading: str
    blocks: tuple[_SectionBlock, ...]


@dataclass(frozen=True)
class _ChartGroup:
    chart_type: str
    title: str
    description: str
    files_by_format: dict[str, Path]


@dataclass(frozen=True)
class _ReportDocument:
    title: str
    subtitle: str
    badges: tuple[_Badge, ...]
    callouts: tuple[_Callout, ...]
    sections: tuple[_Section, ...]
    chart_groups: tuple[_ChartGroup, ...]
    skipped_charts: dict[str, str]
    warnings: tuple[str, ...]
    disclaimers: tuple[str, ...]
    metadata_footer: str = ""


def parse_report_formats(raw: str | None) -> tuple[str, ...]:
    """Parse ``--report`` CLI/API value into ordered unique report formats."""
    if raw is None:
        return ()
    cleaned = str(raw).strip().lower()
    if not cleaned or cleaned == "none":
        return ()
    if cleaned == "all":
        return SUPPORTED_REPORT_FORMATS

    tokens = [part.strip() for part in cleaned.split(",") if part.strip()]
    if not tokens:
        return ()
    if "none" in tokens and len(tokens) > 1:
        raise ValueError("Invalid --report value: 'none' cannot be combined with other formats.")

    resolved: list[str] = []
    for token in tokens:
        if token == "all":
            for fmt in SUPPORTED_REPORT_FORMATS:
                if fmt not in resolved:
                    resolved.append(fmt)
            continue
        mapped = _FORMAT_ALIASES.get(token)
        if mapped is None:
            raise ValueError(
                f"Unsupported report format {token!r}. Allowed values: md, html, all, none."
            )
        if mapped not in resolved:
            resolved.append(mapped)
    return tuple(resolved)


def _group_charts(charts: Sequence[ChartArtifact]) -> tuple[_ChartGroup, ...]:
    order: list[str] = []
    grouped: dict[str, list[ChartArtifact]] = {}
    for ch in charts:
        if ch.chart_type not in grouped:
            order.append(ch.chart_type)
            grouped[ch.chart_type] = []
        grouped[ch.chart_type].append(ch)

    out: list[_ChartGroup] = []
    for ct in order:
        items = grouped[ct]
        by_fmt = {item.format: item.path for item in items}
        out.append(
            _ChartGroup(
                chart_type=ct,
                title=items[0].title,
                description=items[0].description,
                files_by_format=by_fmt,
            )
        )
    return tuple(out)


def _chart_href(path: Path, base_dir: Path | None) -> str:
    """Compute a portable relative URL from *base_dir* (e.g. ``charts/price.png``)."""
    if base_dir is not None:
        try:
            rel = path.resolve().relative_to(base_dir.resolve())
            return rel.as_posix()
        except ValueError:
            pass
    if path.parent.name == "charts":
        return f"charts/{path.name}"
    return path.name


def _yes_no(value: Any) -> str:
    return "是" if bool(value) else "否"


def _quality_table_rows(q: Any) -> tuple[tuple[str, ...], ...]:
    """Build structured rows for a single QualityReport (accepts dataclass or mapping)."""
    get = q.get if isinstance(q, Mapping) else lambda k, d=None: getattr(q, k, d)
    anomalies = (
        bool(q.has_anomalies)
        if not isinstance(q, Mapping)
        else any(
            int(get(k, 0) or 0) > 0
            for k in (
                "dropped_invalid_rows",
                "filled_price_rows",
                "missing_volume_rows",
                "non_positive_price_rows",
                "negative_volume_rows",
                "ohlc_anomaly_rows",
            )
        )
    )
    return (
        ("审计口径 (quality_scope)", str(get("quality_scope", "slice"))),
        ("缺失值策略 (missing_policy)", str(get("missing_policy", "ffill_drop"))),
        ("有效区间", f"{get('start_date', '')} ~ {get('end_date', '')}"),
        ("有效观测行数 (total_rows)", format_integer(get("total_rows", 0))),
        ("剔除无效行 (dropped_invalid_rows)", format_integer(get("dropped_invalid_rows", 0))),
        ("价格前向填充行 (filled_price_rows)", format_integer(get("filled_price_rows", 0))),
        ("缺失成交量行 (missing_volume_rows)", format_integer(get("missing_volume_rows", 0))),
        ("零成交量行 (zero_volume_rows)", format_integer(get("zero_volume_rows", 0))),
        ("非正价格行 (non_positive_price_rows)", format_integer(get("non_positive_price_rows", 0))),
        ("负成交量行 (negative_volume_rows)", format_integer(get("negative_volume_rows", 0))),
        ("OHLC 结构异常行 (ohlc_anomaly_rows)", format_integer(get("ohlc_anomaly_rows", 0))),
        ("真零收益行 (true_zero_return_rows)", format_integer(get("true_zero_return_rows", 0))),
        (
            "最大日历间隔 (max_calendar_gap_days)",
            f"{format_integer(get('max_calendar_gap_days', 0))} 天",
        ),
        ("是否检出质量异常", "是（已降级提示）" if anomalies else "否（通过）"),
    )


def _format_memmel_test_fields(
    mem: Mapping[str, Any] | None,
) -> tuple[str, str, str, str, str, str]:
    """Format Memmel (2003) Sharpe test fields with significance derived from valid p-values."""
    if not isinstance(mem, Mapping):
        return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"
    sharpe_strategy = mem["sharpe_strategy"] if "sharpe_strategy" in mem else mem.get("sharpe_a")
    sharpe_benchmark = mem["sharpe_benchmark"] if "sharpe_benchmark" in mem else mem.get("sharpe_b")
    p_val = mem.get("p_value")
    if is_valid_number(p_val):
        sig_str = "显著差异 (p < 0.05)" if float(p_val) < 0.05 else "无显著差异"
    else:
        sig_str = "N/A"
    return (
        format_ratio(sharpe_strategy),
        format_ratio(sharpe_benchmark),
        format_ratio(mem.get("sharpe_diff"), signed=True),
        format_ratio(mem.get("z_stat"), decimals=3),
        format_p_value(p_val),
        sig_str,
    )


def _describe_portfolio_validation(
    *,
    val_mode: str,
    strategy: str,
    is_oos: bool,
    has_bias: bool,
) -> tuple[str, str, _Callout, str, str]:
    """Return consistent badge text/tone, callout, bias label, and row suffix by validation mode."""
    mode_label = val_mode or ("in_sample" if not is_oos else "out_of_sample")
    if val_mode == "in_sample" or not is_oos:
        if has_bias:
            return (
                f"验证: {mode_label} (存在前视偏差)",
                "danger",
                _Callout(
                    tone="danger",
                    title="前视偏差警告 (Lookahead Bias Warning)",
                    body="当前运行处于 `in_sample` 模式，最优权重与回测收益共享同一历史全样本区间。所展示绩效仅代表事后样本内拟合参考，不可视为真实可投预期收益。",
                ),
                "是（请勿实盘外推）",
                "In-Sample 样本内 [前视偏差]",
            )
        bias_desc = (
            "否（外生固定权重全样本回测，非样本外评估）"
            if strategy == "custom"
            else "否（样本内全样本回测，非样本外评估）"
        )
        callout_body = (
            "当前组合使用 `custom` 外生固定权重在完整历史区间直接回测（`in_sample`），未经过时间切分或滚动样本外隔离，属于全样本评估而非样本外回测；若权重源于事后观察，仍可能存在选择偏差。"
            if strategy == "custom"
            else "当前运行处于样本内全样本回测模式（`in_sample`），未经过时间切分或滚动样本外隔离，所展示结果仅作历史全样本对照参考，非样本外回测。"
        )
        return (
            f"验证: {mode_label} (样本内全样本 | 非 OOS)",
            "warn",
            _Callout(
                tone="warn",
                title="样本内全样本回测提示 (In-Sample Non-OOS Backtest)",
                body=callout_body,
            ),
            bias_desc,
            "In-Sample 样本内 [非 OOS]",
        )

    if has_bias:
        return (
            f"验证: {mode_label} (含前视偏差风险)",
            "danger",
            _Callout(
                tone="danger",
                title="前视偏差警告 (Lookahead Bias Warning)",
                body="当前回测结果标记存在前视偏差，所展示绩效不可视为严格无偏样本外预测。",
            ),
            "是（请勿实盘外推）",
            f"{mode_label} [前视偏差风险]",
        )

    if val_mode.startswith("walk_forward"):
        return (
            f"验证: {mode_label} (严格因果 OOS)",
            "ok",
            _Callout(
                tone="info",
                title="严格因果 Walk-Forward 验证与后视参考区分",
                body="`OOS_Portfolio` 每次调仓严格只使用历史因果训练窗口数据；`In_Sample_Portfolio`（若列示）仅作为全样本事后上限对比参照。",
            ),
            "否（因果滚动/扩展估计）",
            "OOS 样本外",
        )

    if val_mode == "train_test":
        if strategy == "custom":
            return (
                f"验证: {mode_label} (外生权重测试集 OOS)",
                "ok",
                _Callout(
                    tone="info",
                    title="时间切分测试集验证 (Train-Test OOS)",
                    body="外生自定义权重在切分日之后的未见测试集样本外开展净值评估，与切分日前训练集统计隔离；若权重源于全样本事后选定，仍需结合训练集对照表审视潜在选择偏差。",
                ),
                "否（外生权重测试集评估）",
                "OOS 样本外",
            )
        return (
            f"验证: {mode_label} (严格因果 OOS)",
            "ok",
            _Callout(
                tone="info",
                title="严格时间切分样本外验证 (Train-Test OOS)",
                body="策略权重与协方差矩阵仅在切分日之前的训练样本中估计，并在之后完全未见的测试集样本外开展净值回测。",
            ),
            "否（因果估计）",
            "OOS 样本外",
        )

    return (
        f"验证: {mode_label} (样本外评估)",
        "ok",
        _Callout(
            tone="info",
            title=f"样本外验证 ({mode_label})",
            body="组合绩效在非样本内区间上评估。",
        ),
        "否（样本外区间评估）",
        "OOS 样本外",
    )


def _build_single_asset_document(
    result: AnalysisResult,
    *,
    charts: Sequence[ChartArtifact],
    skipped_charts: Mapping[str, str],
) -> _ReportDocument:
    meta = result.asset
    q = result.quality
    params = result.parameters
    metrics = result.metrics
    reasons = result.metric_reasons
    units = result.metric_units
    snapshot = result.latest_snapshot
    is_yield = meta.series_kind == "yield_level"

    type_label = format_asset_type_label(meta.asset_type)
    badges: list[_Badge] = [
        _Badge(f"状态: {result.status.upper()}", "warn" if result.status == "degraded" else "ok"),
        _Badge(f"口径: {meta.series_kind}", "info"),
        _Badge(
            f"计价: {meta.output_currency or meta.native_currency or 'NATIVE'}",
            "neutral",
        ),
        _Badge(f"收益定义: {params.get('return_convention', 'simple')}", "neutral"),
    ]

    callouts: list[_Callout] = []
    if result.status == "degraded":
        callouts.append(
            _Callout(
                tone="warn",
                title="降级提示 (Degraded Result)",
                body="当前分析存在数据质量异常、指标阻断原因或跨币种/后备数据源告警，请在引用指标前核对下方风险提示与质量审计表。",
            )
        )
    if is_yield:
        callouts.append(
            _Callout(
                tone="info",
                title="收益率水平序列语义说明 (yield_level_not_return)",
                body="当前标的为国债/利率收益率水平（以百分点 % 表示），不代表可复利资产价格，因此所有总收益、CAGR、回撤与风险比率指标已按语义阻断为 N/A，图表仅展示水平走势。",
            )
        )
    elif meta.series_kind == "fx_rate":
        callouts.append(
            _Callout(
                tone="info",
                title="外汇标价语义说明 (fx_rate)",
                body="外汇序列为汇率标价变动（非带息Carry总回报），收益率与风险指标按汇率点位变化计算，不支持再按其他计价货币二次折算。",
            )
        )

    # Executive summary bullets
    exec_bullets: list[str] = [
        f"**分析标的**：`{meta.canonical_symbol}`（{type_label}，复权=`{meta.adjust}`，收益基础=`{meta.return_basis}`）。",
        f"**有效区间与样本**：`{q.start_date}` 至 `{q.end_date}`，共 `{q.total_rows:,}` 条有效观测；数据源 `{meta.data_source or 'N/A'}`。",
    ]
    if is_yield:
        exec_bullets.append(
            f"**最新水平与区间极值**：最新利率水平 {format_price_with_unit(metrics.get('latest_level'), series_kind='yield_level')}，"
            f"区间变动 {format_price_with_unit(metrics.get('level_change'), series_kind='yield_level', signed=True, is_change=True)}，"
            f"区间范围 [{format_price_with_unit(metrics.get('level_min'), series_kind='yield_level')} ~ "
            f"{format_price_with_unit(metrics.get('level_max'), series_kind='yield_level')}]。"
        )
    else:
        exec_bullets.append(
            f"**历史表现摘要（非未来预测）**：最新收盘 {format_price_with_unit(metrics.get('latest_close'), series_kind=meta.series_kind, output_currency=meta.output_currency)}，"
            f"累计收益 {format_percent(metrics.get('cumulative_return'), signed=True)}，"
            f"CAGR {format_percent(metrics.get('cagr'), signed=True)}，"
            f"年化波动 {format_percent(metrics.get('annualized_volatility'))}，"
            f"最大水下回撤 {format_drawdown_percent(metrics.get('max_drawdown'))}，"
            f"Sharpe {format_ratio(metrics.get('sharpe_ratio'))}。"
        )
    if meta.fx_converted:
        exec_bullets.append(
            f"**汇率换算**：原生计价 `{meta.native_currency}` 已通过 `{meta.fx_pair}` 折算为 `{meta.output_currency}`。"
        )

    summary_section = _Section(
        heading="1. 执行摘要 (Executive Summary)",
        blocks=(_SectionBlock(kind="bullets", title="核心结论", bullets=tuple(exec_bullets)),),
    )

    provenance_items: tuple[tuple[str, str], ...] = (
        ("资产类别 (asset_type)", type_label),
        ("规范代码 (canonical_symbol)", f"`{meta.canonical_symbol}`"),
        ("原始代码 (symbol)", f"`{meta.symbol}`"),
        ("复权口径 (adjust)", f"`{meta.adjust}`"),
        ("收益基础 (return_basis)", f"`{meta.return_basis}`"),
        ("序列类型 (series_kind)", f"`{meta.series_kind}`"),
        ("原生币种 (native_currency)", str(meta.native_currency or "N/A")),
        ("输出口径 (output_currency)", str(meta.output_currency or "NATIVE")),
        ("是否汇率折算 (fx_converted)", _yes_no(meta.fx_converted)),
        ("汇率对 (fx_pair)", str(meta.fx_pair or "-")),
        ("年化周期 (trading_days)", f"{meta.trading_days} 天/年"),
        ("数据源 (data_source)", str(meta.data_source or "N/A")),
        ("拉取时间 (fetched_at)", str(meta.fetched_at or "N/A")),
        ("命中本地缓存 (cache_hit)", _yes_no(meta.cache_hit)),
    )
    provenance_section = _Section(
        heading="2. 资产元数据与数据来源 (Asset & Data Provenance)",
        blocks=(_SectionBlock(kind="kv", items=provenance_items),),
    )

    # Metrics table
    preferred_order = (
        (
            "latest_level",
            "level_change",
            "level_min",
            "level_max",
            "level_diff_std",
            "cumulative_return",
            "cagr",
            "annualized_volatility",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "var_95",
            "cvar_95",
        )
        if is_yield
        else (
            "latest_close",
            "cumulative_return",
            "cagr",
            "annualized_volatility",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "var_95",
            "cvar_95",
        )
    )
    seen_keys: set[str] = set()
    ordered_keys = [k for k in preferred_order if k in metrics or k in reasons]
    for k in metrics:
        if k not in ordered_keys and k not in {
            "trading_days",
            "return_convention",
            "latest_signal",
        }:
            ordered_keys.append(k)

    metric_rows: list[tuple[str, ...]] = []
    for key in ordered_keys:
        if key in seen_keys:
            continue
        seen_keys.add(key)
        raw_val = metrics.get(key)
        unit = units.get(key)
        reason = reasons.get(key)
        formatted = format_metric_value(
            key,
            raw_val,
            unit=unit,
            reason=reason,
            series_kind=meta.series_kind,
            output_currency=meta.output_currency,
            include_reason=False,
        )
        note = (
            format_reason_label(reason)
            if reason and formatted == "N/A"
            else (
                "水下负数展示 (JSON 内部为 >=0 深度)"
                if key == "max_drawdown" and formatted != "N/A"
                else "历史统计量"
            )
        )
        metric_rows.append(
            (
                _SINGLE_METRIC_LABELS.get(key, key),
                formatted,
                describe_unit(unit),
                note,
            )
        )

    metrics_blocks: list[_SectionBlock] = [
        _SectionBlock(
            kind="table",
            title="核心定量指标（基于历史样本，不构成预测）",
            headers=("指标", "数值", "单位", "说明 / 阻断原因"),
            rows=tuple(metric_rows),
            numeric_cols=(1,),
        )
    ]

    if snapshot:
        snap_rows: list[tuple[str, ...]] = []
        for sk, sv in snapshot.items():
            if sk == "date":
                snap_rows.append(("最新估值日 (date)", str(sv)))
            elif sk in {"close", "Close"}:
                snap_rows.append(
                    (
                        "最新收盘/水平",
                        format_price_with_unit(
                            sv, series_kind=meta.series_kind, output_currency=meta.output_currency
                        ),
                    )
                )
            elif sk.startswith("sma_") or sk.startswith("bollinger_"):
                snap_rows.append(
                    (
                        sk,
                        format_price_with_unit(
                            sv, series_kind=meta.series_kind, output_currency=meta.output_currency
                        ),
                    )
                )
            elif sk.startswith("volatility_"):
                snap_rows.append((f"{sk} (年化)", format_percent(sv)))
            elif sk == "drawdown":
                snap_rows.append(("当前水下回撤", format_drawdown_percent(sv)))
            elif sk == "latest_signal":
                snap_rows.append(("技术信号状态", str(sv)))
            elif is_valid_number(sv):
                snap_rows.append((sk, format_ratio(sv)))
            elif sv is not None:
                snap_rows.append((sk, str(sv)))
        if snap_rows:
            metrics_blocks.append(
                _SectionBlock(
                    kind="kv",
                    title="最新截面快照与技术特征 (Latest Snapshot)",
                    items=tuple(snap_rows),
                )
            )

    metrics_section = _Section(
        heading="3. 定量指标与最新快照 (Core Metrics & Snapshot)",
        blocks=tuple(metrics_blocks),
    )

    quality_section = _Section(
        heading="4. 数据质量审计 (Data Quality Audit)",
        blocks=(
            _SectionBlock(
                kind="table",
                headers=("检查项", "结果"),
                rows=_quality_table_rows(q),
            ),
        ),
    )

    method_items: tuple[tuple[str, str], ...] = (
        ("分析命令", f"`{result.command}`"),
        ("收益率计算口径", f"`{params.get('return_convention', 'simple')}`"),
        ("无风险利率假设 (年化)", format_percent(params.get("risk_free_rate", 0.02))),
        ("年化周期数", f"{meta.trading_days} 天/年"),
        ("均线窗口 (SMA)", f"{params.get('sma_short', 20)}D / {params.get('sma_long', 60)}D"),
        ("波动率滚动窗口", f"{params.get('volatility_window', 20)}D"),
        (
            "回撤展示约定",
            "报告及图表统一展示为非正水下回撤 (<= 0%)；result.json 底层字段 `max_drawdown` 存储为非负深度 (>= 0)。",
        ),
        (
            "历史统计限定声明",
            "所有指标均属于样本区间历史统计（Historical Realized），非统计学前瞻估计或实盘收益承诺。",
        ),
    )
    method_section = _Section(
        heading="5. 模型方法与关键假设 (Methodology & Assumptions)",
        blocks=(_SectionBlock(kind="kv", items=method_items),),
    )

    return _ReportDocument(
        title=f"单资产定量研究报告：{meta.canonical_symbol}",
        subtitle=f"{type_label} | 区间 {q.start_date} ~ {q.end_date} | 口径 {meta.output_currency or meta.native_currency or 'NATIVE'}",
        badges=tuple(badges),
        callouts=tuple(callouts),
        sections=(
            summary_section,
            provenance_section,
            metrics_section,
            quality_section,
            method_section,
        ),
        chart_groups=_group_charts(charts),
        skipped_charts=dict(skipped_charts),
        warnings=result.warnings,
        disclaimers=result.disclaimers,
        metadata_footer=f"Schema v{result.schema_version} | Software v{result.software_version} | Command: {result.command}",
    )


def _build_compare_document(
    result: MultiAssetResult,
    *,
    charts: Sequence[ChartArtifact],
    skipped_charts: Mapping[str, str],
) -> _ReportDocument:
    meth = result.methodology
    ds = result.data_summary
    metrics = result.metrics
    base_ccy = str(meth.get("base_currency") or "LOCAL")
    mixed_local = bool(meth.get("mixed_local_currencies", False))
    bench_leg = (
        str(result.benchmark.get("leg_name"))
        if isinstance(result.benchmark, Mapping) and result.benchmark.get("leg_name")
        else None
    )

    badges: list[_Badge] = [
        _Badge(f"状态: {result.status.upper()}", "warn" if result.status == "degraded" else "ok"),
        _Badge(f"日历对齐: {meth.get('calendar_mode', 'N/A')}", "info"),
        _Badge(f"计价货币: {base_ccy}", "danger" if mixed_local else "neutral"),
        _Badge(f"收益定义: {meth.get('return_convention', 'simple')}", "neutral"),
    ]
    if bench_leg:
        badges.append(_Badge(f"基准: {bench_leg}", "info"))

    callouts: list[_Callout] = []
    if mixed_local:
        callouts.append(
            _Callout(
                tone="danger",
                title="混合原生货币警报 (Mixed Local Currencies)",
                body="当前处于 base_currency='LOCAL' 且入参标的跨越多个不同原生币种；跨币种价格未经过汇率折算，收益率比较不包含汇率损益，建议改用 `--base-currency CNY` 或 `--base-currency USD`。",
            )
        )
    if result.status == "degraded" and not mixed_local:
        callouts.append(
            _Callout(
                tone="warn",
                title="降级提示 (Degraded Result)",
                body="部分标的存在缺失对齐损失、样本量偏小或质量异常，请核对下方数据质量表与告警信息。",
            )
        )

    rankings = metrics.get("rankings", {})
    top_sharpe = rankings.get("by_sharpe") or []
    top_cagr = rankings.get("by_cagr") or []
    pairs = metrics.get("correlation", {}).get("pairs") or []

    exec_bullets = [
        f"**对比标的清单**：`{', '.join(result.asset_order)}`（基准：`{bench_leg or '无'}`）。",
        f"**对齐样本窗口**：`{ds.get('effective_start', '')}` ~ `{ds.get('effective_end', '')}`，估值点 `{ds.get('valuation_points', 0)}` 期，同时有效收益率样本 `{ds.get('concurrent_return_periods', 0)}` 期。",
    ]
    if top_sharpe:
        exec_bullets.append(
            f"**历史风险调整收益排序 (Sharpe, 历史实现)**：`{' > '.join(str(x) for x in top_sharpe)}`。"
        )
    if top_cagr:
        exec_bullets.append(
            f"**历史年化复合收益排序 (CAGR, 历史实现)**：`{' > '.join(str(x) for x in top_cagr)}`。"
        )
    if pairs:
        sorted_pairs = sorted(
            [p for p in pairs if is_valid_number(p.get("pearson"))],
            key=lambda p: float(p["pearson"]),
        )
        if sorted_pairs:
            low_p = sorted_pairs[0]
            high_p = sorted_pairs[-1]
            exec_bullets.append(
                f"**线性相关性极值 (Pearson)**：最低配对 `{low_p['asset_a']} / {low_p['asset_b']}` ({format_ratio(low_p['pearson'])})，"
                f"最高配对 `{high_p['asset_a']} / {high_p['asset_b']}` ({format_ratio(high_p['pearson'])})。"
            )

    summary_section = _Section(
        heading="1. 执行摘要 (Executive Summary)",
        blocks=(_SectionBlock(kind="bullets", bullets=tuple(exec_bullets)),),
    )

    # Universe & Quality table
    asset_rows: list[tuple[str, ...]] = []
    raw_counts = ds.get("raw_counts", {})
    dropped_days = ds.get("dropped_non_overlap_days_by_asset", {})
    for leg, meta in result.assets.items():
        qual = result.quality.get(leg)
        q_flag = "异常" if (qual is not None and qual.has_anomalies) else "正常"
        asset_rows.append(
            (
                leg,
                format_asset_type_label(meta.asset_type),
                meta.canonical_symbol,
                f"{meta.native_currency or '?'} -> {meta.output_currency or meta.native_currency or '?'}",
                meta.fx_pair or "-",
                meta.return_basis,
                meta.data_source or "N/A",
                format_integer(raw_counts.get(leg, qual.total_rows if qual else 0)),
                format_integer(dropped_days.get(leg, 0)),
                q_flag,
            )
        )

    universe_section = _Section(
        heading="2. 标的清单、货币口径与样本对齐 (Universe & Alignment)",
        blocks=(
            _SectionBlock(
                kind="table",
                headers=(
                    "Leg",
                    "类别",
                    "代码",
                    "币种转换",
                    "FX 对",
                    "收益基础",
                    "数据源",
                    "原始行数",
                    "非重叠丢弃",
                    "质量状态",
                ),
                rows=tuple(asset_rows),
                numeric_cols=(7, 8),
            ),
        ),
    )

    # Per-asset performance table
    per_asset = metrics.get("per_asset", {})
    reasons_map = result.metric_reasons if isinstance(result.metric_reasons, Mapping) else {}
    perf_rows: list[tuple[str, ...]] = []
    for leg in per_asset:
        m = per_asset[leg]
        leg_tag = f"{leg} (基准)" if bench_leg and leg == bench_leg else leg
        perf_rows.append(
            (
                leg_tag,
                format_percent(m.get("cumulative_return"), signed=True),
                format_percent(m.get("cagr"), signed=True),
                format_percent(m.get("annualized_volatility")),
                format_drawdown_percent(m.get("max_drawdown")),
                format_ratio(m.get("sharpe_ratio")),
                format_ratio(m.get("sortino_ratio")),
                format_ratio(m.get("calmar_ratio")),
                format_percent(m.get("var_95"), signed=True),
                format_percent(m.get("cvar_95"), signed=True),
            )
        )

    reason_notes: list[str] = []
    for leg, leg_reasons in reasons_map.items():
        if isinstance(leg_reasons, Mapping):
            for mk, mr in leg_reasons.items():
                reason_notes.append(f"`{leg}.{mk}`: {format_reason_label(str(mr))}")

    perf_blocks: list[_SectionBlock] = [
        _SectionBlock(
            kind="table",
            title="历史区间收益与风险指标（基于共同对齐口径，不代表未来收益）",
            headers=(
                "标的",
                "累计收益",
                "CAGR",
                "年化波动",
                "最大回撤",
                "Sharpe",
                "Sortino",
                "Calmar",
                "VaR(95%)",
                "CVaR(95%)",
            ),
            rows=tuple(perf_rows),
            numeric_cols=(1, 2, 3, 4, 5, 6, 7, 8, 9),
        )
    ]
    if reason_notes:
        perf_blocks.append(
            _SectionBlock(kind="bullets", title="缺失/阻断指标说明", bullets=tuple(reason_notes))
        )

    perf_section = _Section(
        heading="3. 横向业绩与风险比较 (Historical Cross-Asset Metrics)",
        blocks=tuple(perf_blocks),
    )

    # Distribution & statistical inference
    dist_map = metrics.get("distribution", {})
    dist_rows: list[tuple[str, ...]] = []
    for leg, d in dist_map.items():
        is_norm = d.get("is_normal_at_5pct")
        norm_label = (
            "未拒绝正态"
            if is_norm is True
            else ("拒绝正态 (非正态)" if is_norm is False else "N/A")
        )
        dist_rows.append(
            (
                leg,
                format_integer(d.get("observations")),
                format_percent(d.get("mean"), decimals=3, signed=True),
                format_percent(d.get("std"), decimals=3),
                format_ratio(d.get("skewness")),
                format_ratio(d.get("excess_kurtosis")),
                format_ratio(d.get("jarque_bera_stat")),
                format_p_value(d.get("jarque_bera_p_value")),
                norm_label,
                format_ratio(d.get("mean_t_stat")),
                format_p_value(d.get("mean_p_value")),
                format_percent(d.get("win_rate")),
            )
        )

    dist_section = _Section(
        heading="4. 收益率统计分布与正态性检验 (Statistical Inference)",
        blocks=(
            _SectionBlock(
                kind="table",
                title="高阶矩、Jarque-Bera 正态性检验与均值 t 检验 (5% 显著性水平)",
                headers=(
                    "标的",
                    "样本数",
                    "单期均值",
                    "单期标准差",
                    "偏度",
                    "超额峰度",
                    "JB 统计量",
                    "JB p 值",
                    "正态结论",
                    "均值 t 值",
                    "均值 p 值",
                    "胜率",
                ),
                rows=tuple(dist_rows),
                numeric_cols=(1, 2, 3, 4, 5, 6, 7, 9, 10, 11),
            ),
        ),
    )

    # Correlation, covariance & rolling
    corr_blocks: list[_SectionBlock] = []
    if pairs:
        pair_rows = tuple(
            (
                f"{p['asset_a']} vs {p['asset_b']}",
                format_ratio(p.get("pearson"), decimals=4),
                format_ratio(p.get("spearman"), decimals=4),
            )
            for p in pairs
        )
        corr_blocks.append(
            _SectionBlock(
                kind="table",
                title="两两相关系数对比 (Pearson 线性 vs Spearman 秩相关)",
                headers=("资产配对", "Pearson ρ", "Spearman ρs"),
                rows=pair_rows,
                numeric_cols=(1, 2),
            )
        )

    corr_df = result.frames.get("correlation")
    if corr_df is not None and not corr_df.empty:
        c_headers = ("资产", *(str(c) for c in corr_df.columns))
        c_rows = tuple(
            (
                str(idx),
                *(format_ratio(corr_df.loc[idx, col], decimals=3) for col in corr_df.columns),
            )
            for idx in corr_df.index
        )
        corr_blocks.append(
            _SectionBlock(
                kind="table",
                title="Pearson 相关系数矩阵",
                headers=c_headers,
                rows=c_rows,
                numeric_cols=tuple(range(1, len(c_headers))),
            )
        )

    cov_info = metrics.get("covariance", {})
    if cov_info:
        cov_items = (
            ("协方差估计方法", f"`{cov_info.get('method', 'N/A')}`"),
            ("Ledoit-Wolf 收缩强度", format_ratio(cov_info.get("shrinkage"), decimals=4)),
            (
                "矩阵条件数 (Condition Number)",
                format_ratio(cov_info.get("condition_number"), decimals=2),
            ),
            (
                "最小特征值 (Min Eigenvalue)",
                format_ratio(cov_info.get("min_eigenvalue"), decimals=6),
            ),
            ("是否施加 Ridge 正则修复", _yes_no(cov_info.get("ridge_applied"))),
        )
        corr_blocks.append(_SectionBlock(kind="kv", title="协方差矩阵病态性诊断", items=cov_items))

    rolling_sum = metrics.get("rolling_summary", {})
    r_assets = rolling_sum.get("assets", {}) if isinstance(rolling_sum, Mapping) else {}
    if r_assets:
        r_rows: list[tuple[str, ...]] = []
        for leg, rdata in r_assets.items():
            ret_s = rdata.get("return_ann", {})
            vol_s = rdata.get("volatility_ann", {})
            sr_s = rdata.get("sharpe", {})
            r_rows.append(
                (
                    str(leg),
                    format_integer(ret_s.get("valid_points", 0)),
                    f"{format_percent(ret_s.get('latest'), signed=True)} (中位 {format_percent(ret_s.get('median'), signed=True)})",
                    f"{format_percent(vol_s.get('latest'))} (中位 {format_percent(vol_s.get('median'))})",
                    f"{format_ratio(sr_s.get('latest'))} (区间 [{format_ratio(sr_s.get('min'))}, {format_ratio(sr_s.get('max'))}])",
                )
            )
        corr_blocks.append(
            _SectionBlock(
                kind="table",
                title=f"滚动统计摘要 (窗口 = {rolling_sum.get('window', meth.get('rolling_window', 'N/A'))} 期)",
                headers=(
                    "标的",
                    "有效滚动点数",
                    "最新/中位年化收益",
                    "最新/中位年化波动",
                    "最新/区间 Sharpe",
                ),
                rows=tuple(r_rows),
                numeric_cols=(1,),
            )
        )

    corr_section = _Section(
        heading="5. 相关性、协方差估计与滚动统计 (Correlation & Rolling)",
        blocks=tuple(corr_blocks),
    )

    sections: list[_Section] = [
        summary_section,
        universe_section,
        perf_section,
        dist_section,
        corr_section,
    ]

    # Benchmark relative section (optional)
    rel_map = metrics.get("benchmark_comparison")
    if isinstance(rel_map, Mapping) and rel_map:
        rel_rows: list[tuple[str, ...]] = []
        memmel_rows: list[tuple[str, ...]] = []
        for leg, rel in rel_map.items():
            if not isinstance(rel, Mapping):
                continue
            rel_rows.append(
                (
                    str(leg),
                    format_percent(rel.get("excess_cumulative_return"), signed=True),
                    format_percent(rel.get("excess_annualized_return"), signed=True),
                    format_percent(rel.get("tracking_error")),
                    format_ratio(rel.get("information_ratio")),
                    format_ratio(rel.get("beta")),
                    format_percent(rel.get("jensen_alpha"), signed=True),
                    format_ratio(rel.get("r_squared")),
                    format_percent(rel.get("up_capture_ratio")),
                    format_percent(rel.get("down_capture_ratio")),
                )
            )
            mem = rel.get("memmel_sharpe_test")
            if isinstance(mem, Mapping):
                s_strat, s_bench, s_diff, z_str, p_str, sig_str = _format_memmel_test_fields(mem)
                memmel_rows.append(
                    (
                        str(leg),
                        s_strat,
                        s_bench,
                        s_diff,
                        z_str,
                        p_str,
                        sig_str,
                    )
                )
        bench_blocks: list[_SectionBlock] = [
            _SectionBlock(
                kind="table",
                title=f"相对于基准 `{bench_leg}` 的超额绩效与捕获率",
                headers=(
                    "标的",
                    "超额累计",
                    "超额年化",
                    "跟踪误差",
                    "信息比率",
                    "Beta",
                    "Jensen α",
                    "R²",
                    "上行捕获",
                    "下行捕获",
                ),
                rows=tuple(rel_rows),
                numeric_cols=(1, 2, 3, 4, 5, 6, 7, 8, 9),
            )
        ]
        if memmel_rows:
            bench_blocks.append(
                _SectionBlock(
                    kind="table",
                    title="Memmel (2003) 渐近正态夏普比率相等性检验",
                    headers=(
                        "标的 vs 基准",
                        "标的 Sharpe",
                        "基准 Sharpe",
                        "Sharpe 差",
                        "Z 统计量",
                        "双侧 p 值",
                        "5% 显著性",
                    ),
                    rows=tuple(memmel_rows),
                    numeric_cols=(1, 2, 3, 4, 5),
                )
            )
        sections.append(
            _Section(
                heading="6. 相对基准归因与 Memmel 夏普假设检验 (Benchmark Attribution & Inference)",
                blocks=tuple(bench_blocks),
            )
        )

    sec_num = len(sections) + 1
    method_items = (
        ("日历对齐模式 (calendar_mode)", f"`{meth.get('calendar_mode', '')}`"),
        ("统一计价货币 (base_currency)", f"`{base_ccy}` (混合本地货币={_yes_no(mixed_local)})"),
        ("收益率口径 (return_convention)", f"`{meth.get('return_convention', 'simple')}`"),
        (
            "有效年化周期 (effective_periods_per_year)",
            format_ratio(meth.get("effective_periods_per_year"), decimals=1),
        ),
        ("协方差收缩估计 (cov_method)", f"`{meth.get('cov_method', '')}`"),
        ("滚动统计窗口", f"{meth.get('rolling_window', '')} 期"),
        (
            "MTM 相关性采样说明",
            "mtm_outer 模式下相关性与协方差基于各腿同时有有效报价的交集收益率估计。",
        ),
        ("回撤展示约定", "所有表格及水下回撤图均以 <= 0% 展示。"),
    )
    sections.append(
        _Section(
            heading=f"{sec_num}. 模型方法与口径假设 (Methodology & Conventions)",
            blocks=(_SectionBlock(kind="kv", items=method_items),),
        )
    )

    return _ReportDocument(
        title=f"多资产定量比较研究报告 ({len(result.asset_order)} 标的)",
        subtitle=f"区间 {ds.get('effective_start', '')} ~ {ds.get('effective_end', '')} | 对齐 {meth.get('calendar_mode', '')} | 计价 {base_ccy}",
        badges=tuple(badges),
        callouts=tuple(callouts),
        sections=tuple(sections),
        chart_groups=_group_charts(charts),
        skipped_charts=dict(skipped_charts),
        warnings=result.warnings,
        disclaimers=result.disclaimers,
        metadata_footer=f"Schema v{result.schema_version} | Software v{result.software_version} | Command: {result.command}",
    )


def _build_portfolio_document(
    result: MultiAssetResult,
    *,
    charts: Sequence[ChartArtifact],
    skipped_charts: Mapping[str, str],
) -> _ReportDocument:
    meth = result.methodology
    ds = result.data_summary
    metrics = result.metrics
    strategy = str(meth.get("strategy") or "")
    val_mode = str(meth.get("validation_mode") or "")
    is_oos = val_mode != "in_sample" and bool(meth.get("is_out_of_sample", True))
    has_bias = bool(meth.get("has_lookahead_bias", False))
    base_ccy = str(meth.get("base_currency") or "LOCAL")
    mixed_local = bool(meth.get("mixed_local_currencies", False))

    val_badge_label, val_badge_tone, val_callout, bias_row_label, perf_row_suffix = (
        _describe_portfolio_validation(
            val_mode=val_mode,
            strategy=strategy,
            is_oos=is_oos,
            has_bias=has_bias,
        )
    )

    badges: list[_Badge] = [
        _Badge(f"状态: {result.status.upper()}", "warn" if result.status == "degraded" else "ok"),
        _Badge(f"策略: {strategy}", "info"),
        _Badge(val_badge_label, val_badge_tone),
        _Badge(f"计价: {base_ccy}", "danger" if mixed_local else "neutral"),
        _Badge(
            f"调仓: {meth.get('rebalance_freq', '')} / {meth.get('transaction_cost_bps', 0)} bps",
            "neutral",
        ),
    ]

    callouts: list[_Callout] = [val_callout]
    if mixed_local:
        callouts.append(
            _Callout(
                tone="danger",
                title="混合原生货币警报 (Mixed Local Currencies)",
                body="当前组合存在多种原生货币且未指定统一计价货币，收益加权与再平衡未包含跨币种汇率换算，请改用 `--base-currency CNY` 或 `USD`。",
            )
        )

    bt_metrics = metrics.get("backtest", {})
    val_summary = metrics.get("validation", {})
    primary_bt_key = next(
        (key for key in ("OOS_Portfolio", "Portfolio", "Optimal_Portfolio") if key in bt_metrics),
        "Portfolio" if val_mode == "in_sample" else "OOS_Portfolio",
    )
    primary_perf = bt_metrics.get(primary_bt_key, {})
    if (
        (not isinstance(primary_perf, Mapping) or not primary_perf)
        and (val_mode == "in_sample" or not is_oos)
        and isinstance(val_summary, Mapping)
        and isinstance(val_summary.get("performance"), Mapping)
    ):
        primary_perf = val_summary["performance"]
    eq_perf = bt_metrics.get("Equal_Weight_Portfolio", {})

    exec_bullets = [
        f"**组合资产池**：`{', '.join(result.asset_order)}`（策略 `{strategy}`，协方差 `{meth.get('cov_method', '')}`，调仓 `{meth.get('rebalance_freq', '')}`，双边成本 `{meth.get('transaction_cost_bps', 0)} bps`）。",
        f"**数据与回测窗口**：全对齐样本 `{ds.get('effective_start', '')} ~ {ds.get('effective_end', '')}`（`{ds.get('return_periods', 0)}` 期）；回测评估窗口 `{ds.get('backtest_start', '')} ~ {ds.get('backtest_end', '')}`（`{ds.get('backtest_points', 0)}` 点）。",
        f"**{'样本外 OOS 回测' if is_oos else '样本内全样本回测参考'}表现 vs 等权基准 (1/N)**："
        f"CAGR {format_percent(primary_perf.get('cagr'), signed=True)} vs {format_percent(eq_perf.get('cagr'), signed=True)}，"
        f"年化波动 {format_percent(primary_perf.get('annualized_volatility'))} vs {format_percent(eq_perf.get('annualized_volatility'))}，"
        f"最大回撤 {format_drawdown_percent(primary_perf.get('max_drawdown'))} vs {format_drawdown_percent(eq_perf.get('max_drawdown'))}，"
        f"Sharpe {format_ratio(primary_perf.get('sharpe_ratio'))} vs {format_ratio(eq_perf.get('sharpe_ratio'))}。",
    ]
    rc = metrics.get("risk_contributions", {})
    if isinstance(rc, Mapping) and is_valid_number(rc.get("diversification_ratio")):
        exec_bullets.append(
            f"**最新权重估计快照风控特征**（窗口 `{meth.get('estimation_window', '')}`：`{meth.get('estimation_start', '')} ~ {meth.get('estimation_end', '')}`）："
            f"组合波动率 {format_percent(rc.get('portfolio_volatility'))}，分散化比率 DR = {format_ratio(rc.get('diversification_ratio'))}，"
            f"有效持仓数（权重/风险）= {format_ratio(rc.get('effective_n_weights'))} / {format_ratio(rc.get('effective_n_risk'))}。"
        )

    summary_section = _Section(
        heading="1. 执行摘要 (Executive Summary)",
        blocks=(_SectionBlock(kind="bullets", bullets=tuple(exec_bullets)),),
    )

    # Validation separation section
    val_items: list[tuple[str, str]] = [
        ("验证模式 (validation_mode)", f"`{val_mode}`"),
        ("是否严格样本外 (is_out_of_sample)", _yes_no(is_oos)),
        ("是否含前视偏差 (has_lookahead_bias)", bias_row_label),
        (
            "权重快照估计窗口",
            f"`{meth.get('estimation_window', '')}` ({meth.get('estimation_start', '')} ~ {meth.get('estimation_end', '')})",
        ),
        (
            "回测评估区间",
            f"{ds.get('backtest_start', '')} ~ {ds.get('backtest_end', '')} ({ds.get('backtest_points', 0)} 个估值点)",
        ),
    ]
    if val_mode == "train_test" and isinstance(val_summary, Mapping):
        val_items.extend(
            [
                ("切分日期 (split_date)", str(val_summary.get("split_date", ""))),
                (
                    "训练集 (In-Sample)",
                    f"{val_summary.get('train_start', '')} ~ {val_summary.get('train_end', '')} ({val_summary.get('train_rows', 0)} 期)",
                ),
                (
                    "测试集 (Out-of-Sample)",
                    f"{val_summary.get('test_start', '')} ~ {val_summary.get('test_end', '')} ({val_summary.get('test_rows', 0)} 期)",
                ),
            ]
        )
    elif val_mode.startswith("walk_forward") and isinstance(val_summary, Mapping):
        val_items.extend(
            [
                (
                    "滚动窗口模式",
                    f"`{val_summary.get('window_mode', '')}` (train_window = {val_summary.get('train_window', '')} 期)",
                ),
                ("样本外再平衡次数", format_integer(val_summary.get("rebalance_count", 0))),
                (
                    "OOS 起止区间",
                    f"{val_summary.get('oos_start', '')} ~ {val_summary.get('oos_end', '')} ({val_summary.get('oos_periods', 0)} 期)",
                ),
            ]
        )

    val_blocks: list[_SectionBlock] = [_SectionBlock(kind="kv", items=tuple(val_items))]
    if val_mode == "train_test" and isinstance(val_summary.get("in_sample_performance"), Mapping):
        is_p = val_summary["in_sample_performance"]
        oos_p = val_summary.get("out_of_sample_performance", primary_perf)
        val_blocks.append(
            _SectionBlock(
                kind="table",
                title="样本内估计期 (Train) vs 样本外回测期 (Test) 衰减对照",
                headers=(
                    "阶段口径",
                    "累计收益",
                    "CAGR",
                    "年化波动",
                    "最大回撤",
                    "Sharpe",
                    "Calmar",
                ),
                rows=(
                    (
                        "训练集样本内 (In-Sample 权重拟合期)",
                        format_percent(is_p.get("cumulative_return"), signed=True),
                        format_percent(is_p.get("cagr"), signed=True),
                        format_percent(is_p.get("annualized_volatility")),
                        format_drawdown_percent(is_p.get("max_drawdown")),
                        format_ratio(is_p.get("sharpe_ratio")),
                        format_ratio(is_p.get("calmar_ratio")),
                    ),
                    (
                        "测试集样本外 (Out-of-Sample 因果回测)",
                        format_percent(oos_p.get("cumulative_return"), signed=True),
                        format_percent(oos_p.get("cagr"), signed=True),
                        format_percent(oos_p.get("annualized_volatility")),
                        format_drawdown_percent(oos_p.get("max_drawdown")),
                        format_ratio(oos_p.get("sharpe_ratio")),
                        format_ratio(oos_p.get("calmar_ratio")),
                    ),
                ),
                numeric_cols=(1, 2, 3, 4, 5, 6),
            )
        )

    validation_section = _Section(
        heading="2. 验证架构与因果隔离边界 (Validation & Epistemic Separation)",
        blocks=tuple(val_blocks),
    )

    # Portfolio weights & risk decomposition
    w_df = result.frames.get("weights")
    w_rows: list[tuple[str, ...]] = []
    if w_df is not None and not w_df.empty:
        for asset in w_df.index:
            row = w_df.loc[asset]
            w_rows.append(
                (
                    str(asset),
                    format_percent(row.get("weight")),
                    format_percent(row.get("equal_weight")),
                    format_percent(row.get("min_variance_weight")),
                    format_percent(row.get("max_sharpe_weight")),
                    format_percent(row.get("expected_return_ann"), signed=True),
                    format_percent(row.get("volatility_ann")),
                    format_ratio(row.get("marginal_risk_contribution"), decimals=4),
                    format_percent(row.get("risk_contribution")),
                    format_percent(row.get("pct_risk_contribution")),
                )
            )
    else:
        p_weights = metrics.get("portfolio", {}).get("weights", {})
        prc_map = rc.get("pct_risk_contribution", {}) if isinstance(rc, Mapping) else {}
        mrc_map = rc.get("marginal_risk_contribution", {}) if isinstance(rc, Mapping) else {}
        rc_map = rc.get("risk_contribution", {}) if isinstance(rc, Mapping) else {}
        for asset in result.asset_order:
            w_rows.append(
                (
                    asset,
                    format_percent(p_weights.get(asset)),
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    format_ratio(mrc_map.get(asset), decimals=4),
                    format_percent(rc_map.get(asset)),
                    format_percent(prc_map.get(asset)),
                )
            )

    p_meta = metrics.get("portfolio", {})
    w_blocks: list[_SectionBlock] = [
        _SectionBlock(
            kind="table",
            title=f"权重快照与 Euler 风险贡献拆解（基于估计窗口 `{meth.get('estimation_window', '')}`）",
            headers=(
                "资产",
                "目标权重",
                "等权重",
                "GMV 权重",
                "MaxSharpe",
                "估计年化收益",
                "估计年化波动",
                "边际风险贡献 MRC",
                "风险贡献 RC",
                "风险贡献占比%",
            ),
            rows=tuple(w_rows),
            numeric_cols=(1, 2, 3, 4, 5, 6, 7, 8, 9),
        ),
        _SectionBlock(
            kind="kv",
            title="优化求解与分散度汇总",
            items=(
                (
                    "优化器收敛状态",
                    f"{_yes_no(p_meta.get('converged'))} ({p_meta.get('solver_message', '')})",
                ),
                (
                    "估计期预期年化收益",
                    format_percent(p_meta.get("estimation_expected_return"), signed=True),
                ),
                ("估计期年化波动率", format_percent(p_meta.get("estimation_volatility"))),
                ("估计期 Sharpe 比率", format_ratio(p_meta.get("estimation_sharpe"))),
                ("加权资产平均波动率", format_percent(rc.get("weighted_asset_volatility"))),
                (
                    "分散化比率 (Diversification Ratio)",
                    format_ratio(rc.get("diversification_ratio")),
                ),
                (
                    "有效权重资产数 (Effective N Weights)",
                    format_ratio(rc.get("effective_n_weights")),
                ),
                ("有效风险资产数 (Effective N Risk)", format_ratio(rc.get("effective_n_risk"))),
                ("相对等风险预算最大偏离", format_percent(rc.get("max_equal_risk_deviation"))),
            ),
        ),
    ]
    weights_section = _Section(
        heading="3. 组合权重配置与 Euler 风险贡献 (Weights & Risk Decomposition)",
        blocks=tuple(w_blocks),
    )

    # Backtest performance comparison
    perf_compare_rows: list[tuple[str, ...]] = []
    series_entries: list[tuple[str, Mapping[str, Any]]] = [
        (
            f"{strategy} ({perf_row_suffix})",
            primary_perf,
        ),
        ("Equal Weight (1/N 等权基准)", eq_perf),
    ]
    if "In_Sample_Portfolio" in bt_metrics and isinstance(
        bt_metrics["In_Sample_Portfolio"], Mapping
    ):
        series_entries.append(
            ("In-Sample Reference [后视全样本参考-不可实盘]", bt_metrics["In_Sample_Portfolio"])
        )
    if isinstance(result.benchmark, Mapping):
        bench_perf = result.benchmark.get("performance")
        if not isinstance(bench_perf, Mapping):
            bench_perf = result.benchmark.get("metrics")
        if isinstance(bench_perf, Mapping) and bench_perf:
            series_entries.append(
                (
                    f"Benchmark ({result.benchmark.get('leg_name', 'Bench')})",
                    bench_perf,
                )
            )

    for s_label, s_perf in series_entries:
        perf_compare_rows.append(
            (
                s_label,
                format_percent(s_perf.get("cumulative_return"), signed=True),
                format_percent(s_perf.get("cagr"), signed=True),
                format_percent(s_perf.get("annualized_volatility")),
                format_drawdown_percent(s_perf.get("max_drawdown")),
                format_ratio(s_perf.get("sharpe_ratio")),
                format_ratio(s_perf.get("sortino_ratio")),
                format_ratio(s_perf.get("calmar_ratio")),
                format_percent(s_perf.get("total_turnover")),
                format_percent(s_perf.get("total_cost_ratio")),
            )
        )

    bt_blocks: list[_SectionBlock] = [
        _SectionBlock(
            kind="table",
            title="净值回测绩效对照（均含设定的单期交易成本与非可交易跳过约束）",
            headers=(
                "组合 / 基准",
                "累计收益",
                "CAGR",
                "年化波动",
                "最大回撤",
                "Sharpe",
                "Sortino",
                "Calmar",
                "累计换手",
                "累计成本比",
            ),
            rows=tuple(perf_compare_rows),
            numeric_cols=(1, 2, 3, 4, 5, 6, 7, 8, 9),
        )
    ]

    rel_eq = bt_metrics.get("relative_to_equal_weight")
    if isinstance(rel_eq, Mapping):
        eq_mem = rel_eq.get("memmel_sharpe_test")
        es_strat, es_bench, es_diff, ez_str, ep_str, esig_str = _format_memmel_test_fields(
            eq_mem if isinstance(eq_mem, Mapping) else None
        )
        bt_blocks.append(
            _SectionBlock(
                kind="kv",
                title="目标策略相对等权基准 (1/N) 统计归因与 Memmel 检验",
                items=(
                    (
                        "超额累计收益",
                        format_percent(rel_eq.get("excess_cumulative_return"), signed=True),
                    ),
                    (
                        "超额年化收益",
                        format_percent(rel_eq.get("excess_annualized_return"), signed=True),
                    ),
                    ("跟踪误差 (Tracking Error)", format_percent(rel_eq.get("tracking_error"))),
                    ("信息比率 (Information Ratio)", format_ratio(rel_eq.get("information_ratio"))),
                    (
                        "Memmel Sharpe (策略 vs 等权)",
                        f"{es_strat} vs {es_bench} (差 {es_diff})",
                    ),
                    (
                        "Memmel 检验 (Z / p / 5%显著性)",
                        f"z = {ez_str}, p = {ep_str} ({esig_str})",
                    ),
                ),
            )
        )

    if isinstance(result.benchmark, Mapping):
        bench_rel = result.benchmark.get("relative_to_strategy")
        if not isinstance(bench_rel, Mapping) and isinstance(
            result.benchmark.get("relative_comparison"), Mapping
        ):
            bench_rel = result.benchmark["relative_comparison"].get("portfolio")
        if isinstance(bench_rel, Mapping) and bench_rel:
            b_mem = bench_rel.get("memmel_sharpe_test")
            bs_strat, bs_bench, bs_diff, bz_str, bp_str, bsig_str = _format_memmel_test_fields(
                b_mem if isinstance(b_mem, Mapping) else None
            )
            eff_start = result.benchmark.get("effective_start")
            eff_end = result.benchmark.get("effective_end")
            eff_pts = result.benchmark.get("valuation_points")
            span_note = (
                f"（重叠基准区间 {eff_start} ~ {eff_end}，共 {format_integer(eff_pts)} 点）"
                if eff_start and eff_end and is_valid_number(eff_pts)
                else ""
            )
            bt_blocks.append(
                _SectionBlock(
                    kind="kv",
                    title=f"目标策略相对外部基准 `{result.benchmark.get('leg_name', '')}` 归因{span_note}",
                    items=(
                        (
                            "超额累计收益",
                            format_percent(bench_rel.get("excess_cumulative_return"), signed=True),
                        ),
                        (
                            "超额年化收益",
                            format_percent(bench_rel.get("excess_annualized_return"), signed=True),
                        ),
                        (
                            "信息比率 / 跟踪误差",
                            f"IR = {format_ratio(bench_rel.get('information_ratio'))}, TE = {format_percent(bench_rel.get('tracking_error'))}",
                        ),
                        (
                            "Beta / Jensen α / R²",
                            f"β = {format_ratio(bench_rel.get('beta'))}, α = {format_percent(bench_rel.get('jensen_alpha'), signed=True)}, R² = {format_ratio(bench_rel.get('r_squared'))}",
                        ),
                        (
                            "上行 / 下行捕获率",
                            f"{format_percent(bench_rel.get('up_capture_ratio'))} / {format_percent(bench_rel.get('down_capture_ratio'))}",
                        ),
                        (
                            "Memmel Sharpe (策略 vs 基准)",
                            f"{bs_strat} vs {bs_bench} (差 {bs_diff})",
                        ),
                        (
                            "Memmel 检验 (Z / p / 5%显著性)",
                            f"z = {bz_str}, p = {bp_str} ({bsig_str})",
                        ),
                    ),
                )
            )

    backtest_section = _Section(
        heading="4. 组合回测表现与基准归因 (Backtest Performance)",
        blocks=tuple(bt_blocks),
    )
    sections: list[_Section] = [
        summary_section,
        validation_section,
        weights_section,
        backtest_section,
    ]

    # Efficient frontier (optional)
    ef = metrics.get("efficient_frontier")
    if isinstance(ef, Mapping):
        exact = ef.get("exact_slsqp", {})
        gmv = exact.get("min_variance", {}) if isinstance(exact, Mapping) else {}
        tang = exact.get("max_sharpe", {}) if isinstance(exact, Mapping) else {}
        rand = ef.get("random_sampling")
        ef_rows: list[tuple[str, ...]] = []
        if isinstance(gmv, Mapping) and gmv:
            ef_rows.append(
                (
                    "最小方差组合 (GMV, 精确 SLSQP)",
                    format_percent(gmv.get("expected_return"), signed=True),
                    format_percent(gmv.get("volatility")),
                    format_ratio(gmv.get("sharpe")),
                )
            )
        if isinstance(tang, Mapping) and tang:
            ef_rows.append(
                (
                    "最大夏普切点 (Tangency, 精确 SLSQP)",
                    format_percent(tang.get("expected_return"), signed=True),
                    format_percent(tang.get("volatility")),
                    format_ratio(tang.get("sharpe")),
                )
            )
        if isinstance(rand, Mapping) and isinstance(rand.get("best_sampled_sharpe"), Mapping):
            rs = rand["best_sampled_sharpe"]
            ef_rows.append(
                (
                    f"Dirichlet 随机采样最优夏普 ({rand.get('num_portfolios', 0)} 次模拟参考)",
                    format_percent(rs.get("expected_return"), signed=True),
                    format_percent(rs.get("volatility")),
                    format_ratio(rs.get("sharpe")),
                )
            )
        ef_section = _Section(
            heading="5. 均值-方差有效前沿与可行域模拟 (Efficient Frontier & Simulation)",
            blocks=(
                _SectionBlock(
                    kind="table",
                    title=f"SLSQP 凸优化精确前沿与随机模拟对照（估计窗口：{ef.get('estimation_start', '')} ~ {ef.get('estimation_end', '')}）",
                    headers=("参考点", "预期年化收益", "年化波动率", "Sharpe 比率"),
                    rows=tuple(ef_rows),
                    numeric_cols=(1, 2, 3),
                ),
            ),
        )
        sections.append(ef_section)

    # Sensitivity tables (optional)
    sens = metrics.get("sensitivity")
    if isinstance(sens, Mapping):
        sens_blocks: list[_SectionBlock] = []
        cov_rows = sens.get("covariance_method") or []
        if cov_rows:
            sens_blocks.append(
                _SectionBlock(
                    kind="table",
                    title="协方差估计方法敏感性",
                    headers=("协方差方法", "条件数", "CAGR", "年化波动", "最大回撤", "Sharpe"),
                    rows=tuple(
                        (
                            str(r.get("cov_method", "")),
                            format_ratio(r.get("condition_number")),
                            format_percent(r.get("cagr"), signed=True),
                            format_percent(r.get("annualized_volatility")),
                            format_drawdown_percent(r.get("max_drawdown")),
                            format_ratio(r.get("sharpe_ratio")),
                        )
                        if r.get("status") == "ok"
                        else (
                            str(r.get("cov_method", "")),
                            f"失败: {r.get('error', '')}",
                            "-",
                            "-",
                            "-",
                            "-",
                        )
                        for r in cov_rows
                    ),
                    numeric_cols=(1, 2, 3, 4, 5),
                )
            )
        tx_rows = sens.get("transaction_cost_bps") or []
        if tx_rows:
            sens_blocks.append(
                _SectionBlock(
                    kind="table",
                    title="双边交易成本敏感性 (bps)",
                    headers=(
                        "单次费率 (bps)",
                        "累计收益",
                        "CAGR",
                        "最大回撤",
                        "Sharpe",
                        "累计成本比",
                    ),
                    rows=tuple(
                        (
                            format_ratio(r.get("transaction_cost_bps"), decimals=1),
                            format_percent(r.get("cumulative_return"), signed=True),
                            format_percent(r.get("cagr"), signed=True),
                            format_drawdown_percent(r.get("max_drawdown")),
                            format_ratio(r.get("sharpe_ratio")),
                            format_percent(r.get("total_cost_ratio")),
                        )
                        if r.get("status") == "ok"
                        else (
                            str(r.get("transaction_cost_bps", "")),
                            f"失败: {r.get('error', '')}",
                            "-",
                            "-",
                            "-",
                            "-",
                        )
                        for r in tx_rows
                    ),
                    numeric_cols=(0, 1, 2, 3, 4, 5),
                )
            )
        reb_rows = sens.get("rebalance_freq") or []
        if reb_rows:
            sens_blocks.append(
                _SectionBlock(
                    kind="table",
                    title="再平衡频率敏感性",
                    headers=("调仓频率", "CAGR", "年化波动", "最大回撤", "Sharpe", "累计换手"),
                    rows=tuple(
                        (
                            str(r.get("rebalance_freq", "")),
                            format_percent(r.get("cagr"), signed=True),
                            format_percent(r.get("annualized_volatility")),
                            format_drawdown_percent(r.get("max_drawdown")),
                            format_ratio(r.get("sharpe_ratio")),
                            format_percent(r.get("total_turnover")),
                        )
                        if r.get("status") == "ok"
                        else (
                            str(r.get("rebalance_freq", "")),
                            f"失败: {r.get('error', '')}",
                            "-",
                            "-",
                            "-",
                            "-",
                        )
                        for r in reb_rows
                    ),
                    numeric_cols=(1, 2, 3, 4, 5),
                )
            )
        if sens_blocks:
            sections.append(
                _Section(
                    heading=f"{len(sections) + 1}. 参数稳健性与敏感性分析 (Sensitivity Analysis)",
                    blocks=tuple(sens_blocks),
                )
            )

    # Underlying assets + covariance + methodology
    raw_counts = ds.get("raw_counts", {})
    dropped_days = ds.get("dropped_non_overlap_days_by_asset", {})
    underlying_rows = tuple(
        (
            leg,
            format_asset_type_label(meta.asset_type),
            meta.canonical_symbol,
            f"{meta.native_currency or '?'} -> {meta.output_currency or meta.native_currency or '?'}",
            meta.return_basis,
            format_integer(
                raw_counts.get(leg, result.quality[leg].total_rows if leg in result.quality else 0)
            ),
            format_integer(dropped_days.get(leg, 0)),
            "异常" if (leg in result.quality and result.quality[leg].has_anomalies) else "正常",
        )
        for leg, meta in result.assets.items()
    )
    cov_info = metrics.get("covariance", {})
    sections.append(
        _Section(
            heading=f"{len(sections) + 1}. 成分资产、协方差诊断与数据质量 (Universe & Quality)",
            blocks=(
                _SectionBlock(
                    kind="table",
                    headers=(
                        "Leg",
                        "类别",
                        "代码",
                        "币种口径",
                        "收益基础",
                        "原始行数",
                        "非重叠丢弃",
                        "质量",
                    ),
                    rows=underlying_rows,
                    numeric_cols=(5, 6),
                ),
                _SectionBlock(
                    kind="kv",
                    title="协方差估计器诊断",
                    items=(
                        ("协方差方法", f"`{cov_info.get('method', meth.get('cov_method', ''))}`"),
                        (
                            "Ledoit-Wolf 收缩强度",
                            format_ratio(cov_info.get("shrinkage"), decimals=4),
                        ),
                        (
                            "条件数 (Condition Number)",
                            format_ratio(cov_info.get("condition_number")),
                        ),
                        ("最小特征值", format_ratio(cov_info.get("min_eigenvalue"), decimals=6)),
                        ("Ridge 正则化修补", _yes_no(cov_info.get("ridge_applied"))),
                    ),
                ),
            ),
        )
    )

    sections.append(
        _Section(
            heading=f"{len(sections) + 1}. 模型方法与关键假设 (Methodology & Assumptions)",
            blocks=(
                _SectionBlock(
                    kind="kv",
                    items=(
                        ("目标策略", f"`{strategy}`"),
                        (
                            "验证模式",
                            f"`{val_mode}` (样本外={_yes_no(is_oos)}, 前视偏差={_yes_no(has_bias)})",
                        ),
                        (
                            "单资产权重边界",
                            f"[{format_percent(meth.get('min_weight', 0.0))}, {format_percent(meth.get('max_weight', 1.0))}]",
                        ),
                        (
                            "再平衡频率与成本",
                            f"`{meth.get('rebalance_freq', '')}` / 单边 {meth.get('transaction_cost_bps', 0)} bps (仅非可交易日跳过再平衡: {_yes_no(meth.get('non_tradable_rebalance_skipped', True))})",
                        ),
                        (
                            "回测收益率口径",
                            f"`{meth.get('backtest_return_convention', 'simple')}`，年化周期 = {format_ratio(meth.get('periods_per_year'), decimals=1)}",
                        ),
                        ("日历与计价货币", f"`{meth.get('calendar_mode', '')}` / `{base_ccy}`"),
                        (
                            "回撤展示约定",
                            "报告及图表统一展示为非正水下回撤 (<= 0%)；result.json 内部 backtest max_drawdown 为非负深度 (>= 0)。",
                        ),
                    ),
                ),
            ),
        )
    )

    return _ReportDocument(
        title=f"投资组合优化与回测研究报告：{strategy} ({val_mode})",
        subtitle=f"资产数 {len(result.asset_order)} | 回测区间 {ds.get('backtest_start', '')} ~ {ds.get('backtest_end', '')} | 计价 {base_ccy}",
        badges=tuple(badges),
        callouts=tuple(callouts),
        sections=tuple(sections),
        chart_groups=_group_charts(charts),
        skipped_charts=dict(skipped_charts),
        warnings=result.warnings,
        disclaimers=result.disclaimers,
        metadata_footer=f"Schema v{result.schema_version} | Software v{result.software_version} | Command: {result.command}",
    )


def _build_document(
    result: AnalysisResult | MultiAssetResult,
    *,
    charts: Sequence[ChartArtifact] = (),
    skipped_charts: Mapping[str, str] | None = None,
) -> _ReportDocument:
    skipped = dict(skipped_charts or {})
    if isinstance(result, AnalysisResult):
        return _build_single_asset_document(result, charts=charts, skipped_charts=skipped)
    if result.command == "compare":
        return _build_compare_document(result, charts=charts, skipped_charts=skipped)
    if result.command == "portfolio":
        return _build_portfolio_document(result, charts=charts, skipped_charts=skipped)
    raise ValueError(f"Reports are not supported for command {result.command!r}.")


def _escape_md_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _render_markdown(doc: _ReportDocument, *, output_dir: Path | None) -> str:
    lines: list[str] = [
        f"# {doc.title}",
        "",
        f"> {doc.subtitle}",
        "",
        " | ".join(f"`{b.label}`" for b in doc.badges),
        "",
    ]

    for c in doc.callouts:
        prefix = "CRITICAL" if c.tone == "danger" else ("WARNING" if c.tone == "warn" else "NOTE")
        lines.append(f"> **[{prefix}] {c.title}**: {c.body}")
        lines.append("")

    for section in doc.sections:
        lines.append(f"## {section.heading}")
        lines.append("")
        for block in section.blocks:
            if block.title:
                lines.append(f"### {block.title}")
                lines.append("")
            if block.kind == "bullets":
                for b in block.bullets:
                    lines.append(f"- {b}")
                lines.append("")
            elif block.kind == "kv":
                lines.append("| 项目 | 值 |")
                lines.append("| :--- | :--- |")
                for k, v in block.items:
                    lines.append(f"| {_escape_md_cell(k)} | {_escape_md_cell(v)} |")
                lines.append("")
            elif block.kind == "table":
                lines.append("| " + " | ".join(_escape_md_cell(h) for h in block.headers) + " |")
                aligns = [
                    "---:" if idx in block.numeric_cols else ":---"
                    for idx in range(len(block.headers))
                ]
                lines.append("| " + " | ".join(aligns) + " |")
                for row in block.rows:
                    lines.append("| " + " | ".join(_escape_md_cell(cell) for cell in row) + " |")
                lines.append("")
            elif block.kind == "paragraph" and block.text:
                lines.append(block.text)
                lines.append("")

    chart_sec_num = len(doc.sections) + 1
    lines.append(f"## {chart_sec_num}. 可视化图表与说明 (Visualizations)")
    lines.append("")
    if doc.chart_groups:
        for grp in doc.chart_groups:
            lines.append(f"### {grp.title} (`{grp.chart_type}`)")
            lines.append("")
            if grp.description:
                lines.append(grp.description)
                lines.append("")
            static_fmt = (
                "png"
                if "png" in grp.files_by_format
                else ("svg" if "svg" in grp.files_by_format else None)
            )
            if static_fmt is not None:
                rel_img = _chart_href(grp.files_by_format[static_fmt], output_dir)
                lines.append(f"![{grp.title}]({rel_img})")
                lines.append("")
            links: list[str] = []
            for fmt in ("html", "png", "svg"):
                if fmt in grp.files_by_format:
                    rel = _chart_href(grp.files_by_format[fmt], output_dir)
                    label = "交互式图表 (HTML)" if fmt == "html" else f"{fmt.upper()} 静态图"
                    links.append(f"[{label}]({rel})")
            if links:
                lines.append("图表文件：" + " | ".join(links))
                lines.append("")
    else:
        lines.append("_本次报告未附带生成图表文件（可通过 `--charts html,png` 按需生成）。_")
        lines.append("")

    if doc.skipped_charts:
        lines.append("### 未生成/已跳过图表说明")
        lines.append("")
        for ct, reason in doc.skipped_charts.items():
            lines.append(f"- `{ct}`: {reason}")
        lines.append("")

    warn_sec_num = chart_sec_num + 1
    lines.append(f"## {warn_sec_num}. 风险提示与研究声明 (Warnings & Disclaimers)")
    lines.append("")
    if doc.warnings:
        lines.append("### 运行期告警 (Warnings)")
        lines.append("")
        for w in doc.warnings:
            lines.append(f"- {w}")
        lines.append("")
    if doc.disclaimers:
        lines.append("### 合规与研究声明 (Disclaimers)")
        lines.append("")
        for d in doc.disclaimers:
            lines.append(f"- {d}")
        lines.append("")
    if not doc.warnings and not doc.disclaimers:
        lines.append("- 本报告仅供量化研究使用，不构成任何投资建议。")
        lines.append("")

    if doc.metadata_footer:
        lines.append("---")
        lines.append("")
        lines.append(f"_{doc.metadata_footer}_")
        lines.append("")

    return "\n".join(lines)


def _format_inline_code_html(text: str) -> str:
    """Escape HTML and convert simple ``**bold**`` & backtick code spans cleanly."""
    escaped = html.escape(text)
    parts = escaped.split("`")
    if len(parts) > 1 and len(parts) % 2 == 1:
        rebuilt: list[str] = []
        for idx, chunk in enumerate(parts):
            if idx % 2 == 1:
                rebuilt.append(f"<code>{chunk}</code>")
            else:
                rebuilt.append(chunk)
        escaped = "".join(rebuilt)

    bold_parts = escaped.split("**")
    if len(bold_parts) > 1 and len(bold_parts) % 2 == 1:
        rebuilt_b: list[str] = []
        for idx, chunk in enumerate(bold_parts):
            rebuilt_b.append(f"<strong>{chunk}</strong>" if idx % 2 == 1 else chunk)
        escaped = "".join(rebuilt_b)
    return escaped


def _render_html(doc: _ReportDocument, *, output_dir: Path | None) -> str:
    badge_html = "\n".join(
        f'<span class="badge badge-{b.tone}">{html.escape(b.label)}</span>' for b in doc.badges
    )
    callout_html = "\n".join(
        (
            f'<div class="callout callout-{c.tone}">'
            f'<div class="callout-title">{html.escape(c.title)}</div>'
            f'<div class="callout-body">{_format_inline_code_html(c.body)}</div>'
            "</div>"
        )
        for c in doc.callouts
    )

    sections_html: list[str] = []
    for sec in doc.sections:
        block_parts: list[str] = []
        for blk in sec.blocks:
            if blk.title:
                block_parts.append(f"<h3>{html.escape(blk.title)}</h3>")
            if blk.kind == "bullets":
                lis = "".join(f"<li>{_format_inline_code_html(b)}</li>" for b in blk.bullets)
                block_parts.append(f'<ul class="clean-list">{lis}</ul>')
            elif blk.kind == "kv":
                trs = "".join(
                    (f"<tr><th>{html.escape(k)}</th><td>{_format_inline_code_html(v)}</td></tr>")
                    for k, v in blk.items
                )
                block_parts.append(
                    f'<div class="table-wrap"><table class="kv-table"><tbody>{trs}</tbody></table></div>'
                )
            elif blk.kind == "table":
                ths = "".join(
                    f'<th class="{"num" if i in blk.numeric_cols else "txt"}">{html.escape(h)}</th>'
                    for i, h in enumerate(blk.headers)
                )
                body_rows: list[str] = []
                for row in blk.rows:
                    tds = "".join(
                        f'<td class="{"num" if i in blk.numeric_cols else "txt"}">{_format_inline_code_html(cell)}</td>'
                        for i, cell in enumerate(row)
                    )
                    body_rows.append(f"<tr>{tds}</tr>")
                block_parts.append(
                    '<div class="table-wrap"><table class="data-table">'
                    f"<thead><tr>{ths}</tr></thead>"
                    f"<tbody>{''.join(body_rows)}</tbody>"
                    "</table></div>"
                )
            elif blk.kind == "paragraph" and blk.text:
                block_parts.append(f"<p>{_format_inline_code_html(blk.text)}</p>")
        sections_html.append(
            f'<section class="card"><h2>{html.escape(sec.heading)}</h2>{"".join(block_parts)}</section>'
        )

    chart_sec_num = len(doc.sections) + 1
    chart_cards: list[str] = []
    if doc.chart_groups:
        for grp in doc.chart_groups:
            media_html = ""
            static_fmt = (
                "png"
                if "png" in grp.files_by_format
                else ("svg" if "svg" in grp.files_by_format else None)
            )
            if static_fmt is not None:
                rel_img = html.escape(_chart_href(grp.files_by_format[static_fmt], output_dir))
                media_html = (
                    f'<figure class="chart-figure">'
                    f'<img src="{rel_img}" alt="{html.escape(grp.title)}" loading="lazy" />'
                    "</figure>"
                )
            elif "html" in grp.files_by_format:
                rel_html = html.escape(_chart_href(grp.files_by_format["html"], output_dir))
                media_html = (
                    '<div class="chart-iframe-wrap">'
                    f'<iframe src="{rel_html}" title="{html.escape(grp.title)}" loading="lazy"></iframe>'
                    "</div>"
                )

            link_pills: list[str] = []
            for fmt in ("html", "png", "svg"):
                if fmt in grp.files_by_format:
                    rel = html.escape(_chart_href(grp.files_by_format[fmt], output_dir))
                    lbl = "交互式 HTML" if fmt == "html" else fmt.upper()
                    link_pills.append(f'<a class="chart-link" href="{rel}">{lbl}</a>')

            chart_cards.append(
                '<div class="chart-card">'
                f'<h3>{html.escape(grp.title)} <span class="chart-tag">{html.escape(grp.chart_type)}</span></h3>'
                f'<p class="chart-desc">{html.escape(grp.description)}</p>'
                f"{media_html}"
                f'<div class="chart-links">{" ".join(link_pills)}</div>'
                "</div>"
            )
    else:
        chart_cards.append(
            '<p class="muted">本次报告未附带生成图表文件（可通过 <code>--charts html,png</code> 按需生成）。</p>'
        )

    skipped_html = ""
    if doc.skipped_charts:
        sk_lis = "".join(
            f"<li><code>{html.escape(ct)}</code>: {html.escape(reason)}</li>"
            for ct, reason in doc.skipped_charts.items()
        )
        skipped_html = f'<div class="skipped-box"><h4>已跳过图表</h4><ul class="clean-list">{sk_lis}</ul></div>'

    charts_section_html = (
        f'<section class="card"><h2>{chart_sec_num}. 可视化图表与说明 (Visualizations)</h2>'
        f"{''.join(chart_cards)}{skipped_html}</section>"
    )

    warn_sec_num = chart_sec_num + 1
    warn_parts: list[str] = []
    if doc.warnings:
        w_lis = "".join(f"<li>{html.escape(w)}</li>" for w in doc.warnings)
        warn_parts.append(
            f'<h3>运行期告警 (Warnings)</h3><ul class="clean-list warn-list">{w_lis}</ul>'
        )
    if doc.disclaimers:
        d_lis = "".join(f"<li>{html.escape(d)}</li>" for d in doc.disclaimers)
        warn_parts.append(
            f'<h3>合规与研究声明 (Disclaimers)</h3><ul class="clean-list">{d_lis}</ul>'
        )
    if not warn_parts:
        warn_parts.append('<p class="muted">本报告仅供量化研究使用，不构成任何投资建议。</p>')

    warnings_section_html = (
        f'<section class="card"><h2>{warn_sec_num}. 风险提示与研究声明 (Warnings & Disclaimers)</h2>'
        f"{''.join(warn_parts)}</section>"
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(doc.title)}</title>
  <style>
    :root {{
      --bg: #f1f5f9;
      --surface: #ffffff;
      --ink: #0f172a;
      --muted: #475569;
      --border: #e2e8f0;
      --accent: #2563eb;
      --ok-bg: #dcfce7;
      --ok-fg: #166534;
      --warn-bg: #fef3c7;
      --warn-fg: #92400e;
      --danger-bg: #fee2e2;
      --danger-fg: #991b1b;
      --info-bg: #dbeafe;
      --info-fg: #1e40af;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 28px 18px 48px;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      line-height: 1.55;
      font-size: 14px;
    }}
    .container {{ max-width: 1120px; margin: 0 auto; }}
    .report-header {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 22px 26px;
      margin-bottom: 18px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    }}
    .report-header h1 {{ margin: 0 0 6px; font-size: 22px; letter-spacing: -0.01em; }}
    .subtitle {{ color: var(--muted); margin: 0 0 14px; font-size: 13.5px; }}
    .badge-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      background: #e2e8f0;
      color: #334155;
    }}
    .badge-ok {{ background: var(--ok-bg); color: var(--ok-fg); }}
    .badge-warn {{ background: var(--warn-bg); color: var(--warn-fg); }}
    .badge-danger {{ background: var(--danger-bg); color: var(--danger-fg); }}
    .badge-info {{ background: var(--info-bg); color: var(--info-fg); }}
    .callout {{
      border-radius: 10px;
      padding: 13px 16px;
      margin-bottom: 14px;
      border-left: 4px solid;
    }}
    .callout-danger {{ background: var(--danger-bg); border-color: #dc2626; color: var(--danger-fg); }}
    .callout-warn {{ background: var(--warn-bg); border-color: #d97706; color: var(--warn-fg); }}
    .callout-info {{ background: var(--info-bg); border-color: var(--accent); color: var(--info-fg); }}
    .callout-title {{ font-weight: 700; margin-bottom: 4px; }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 18px;
      box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }}
    .card h2 {{
      margin: 0 0 14px;
      font-size: 17px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
    }}
    .card h3 {{ margin: 16px 0 8px; font-size: 14.5px; color: #1e293b; }}
    .clean-list {{ margin: 6px 0 12px; padding-left: 20px; }}
    .clean-list li {{ margin-bottom: 5px; }}
    .warn-list li {{ color: var(--warn-fg); }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12.5px;
      background: #f1f5f9;
      padding: 1px 5px;
      border-radius: 4px;
    }}
    .table-wrap {{ overflow-x: auto; margin: 8px 0 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .data-table th, .data-table td, .kv-table th, .kv-table td {{
      padding: 7px 10px;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }}
    .data-table thead th {{
      background: #f8fafc;
      font-weight: 600;
      color: #334155;
      white-space: nowrap;
    }}
    .data-table tbody tr:hover {{ background: #f8fafc; }}
    .data-table th.num, .data-table td.num {{
      text-align: right;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      white-space: nowrap;
    }}
    .kv-table th {{ width: 30%; color: var(--muted); font-weight: 600; background: #f8fafc; }}
    .chart-card {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      margin-bottom: 14px;
      background: #fcfdff;
    }}
    .chart-tag {{
      font-size: 11.5px;
      font-weight: 600;
      color: var(--accent);
      background: var(--info-bg);
      padding: 1px 7px;
      border-radius: 999px;
      margin-left: 6px;
    }}
    .chart-desc {{ color: var(--muted); font-size: 13px; margin: 4px 0 10px; }}
    .chart-figure {{ margin: 0 0 10px; text-align: center; }}
    .chart-figure img {{ max-width: 100%; height: auto; border-radius: 6px; border: 1px solid var(--border); }}
    .chart-iframe-wrap {{ width: 100%; height: 540px; margin-bottom: 10px; }}
    .chart-iframe-wrap iframe {{
      width: 100%;
      height: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #ffffff;
    }}
    .chart-links {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .chart-link {{
      display: inline-block;
      padding: 4px 10px;
      border: 1px solid #bfdbfe;
      border-radius: 6px;
      color: var(--accent);
      text-decoration: none;
      font-size: 12.5px;
      font-weight: 500;
      background: #eff6ff;
    }}
    .chart-link:hover {{ background: #dbeafe; }}
    .skipped-box {{
      margin-top: 12px;
      padding: 10px 14px;
      border-radius: 8px;
      background: #f8fafc;
      border: 1px dashed var(--border);
    }}
    .skipped-box h4 {{ margin: 0 0 6px; font-size: 13.5px; color: var(--muted); }}
    .muted {{ color: var(--muted); }}
    footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 18px; }}
  </style>
</head>
<body>
  <div class="container">
    <header class="report-header">
      <h1>{html.escape(doc.title)}</h1>
      <p class="subtitle">{html.escape(doc.subtitle)}</p>
      <div class="badge-row">{badge_html}</div>
    </header>
    {callout_html}
    {"".join(sections_html)}
    {charts_section_html}
    {warnings_section_html}
    <footer>{html.escape(doc.metadata_footer)}</footer>
  </div>
</body>
</html>
"""


def build_markdown_report(
    result: AnalysisResult | MultiAssetResult,
    *,
    charts: Sequence[ChartArtifact] = (),
    skipped_charts: Mapping[str, str] | None = None,
    output_dir: str | Path | None = None,
) -> str:
    """Build Markdown research report string directly from ``result`` in memory."""
    doc = _build_document(result, charts=charts, skipped_charts=skipped_charts)
    out_path = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    return _render_markdown(doc, output_dir=out_path)


def build_html_report(
    result: AnalysisResult | MultiAssetResult,
    *,
    charts: Sequence[ChartArtifact] = (),
    skipped_charts: Mapping[str, str] | None = None,
    output_dir: str | Path | None = None,
) -> str:
    """Build self-contained HTML research report string directly from ``result`` in memory."""
    doc = _build_document(result, charts=charts, skipped_charts=skipped_charts)
    out_path = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    return _render_html(doc, output_dir=out_path)


def render_report(
    result: AnalysisResult | MultiAssetResult,
    output_dir: str | Path,
    *,
    formats: Sequence[str],
    charts: Sequence[ChartArtifact] = (),
    skipped_charts: Mapping[str, str] | None = None,
    file_writer: Callable[[Path, bytes], Path | None] | None = None,
) -> ReportRenderOutput:
    """Render requested Markdown / HTML reports into ``<output_dir>``.

    Consumes precomputed result envelopes and already-rendered chart metadata without any
    data fetching, recomputation, or plotting library imports.
    """
    norm_formats = parse_report_formats(",".join(formats)) if formats else ()
    if not norm_formats:
        return ReportRenderOutput()

    out_dir = Path(output_dir).expanduser().resolve()
    doc = _build_document(result, charts=charts, skipped_charts=skipped_charts)
    artifacts: list[ReportArtifact] = []

    for fmt in norm_formats:
        if fmt == "md":
            content = _render_markdown(doc, output_dir=out_dir)
            filename = "report.md"
            art_key = "report_md"
        elif fmt == "html":
            content = _render_html(doc, output_dir=out_dir)
            filename = "report.html"
            art_key = "report_html"
        else:
            raise ValueError(f"Unsupported report format: {fmt!r}")

        target = out_dir / filename
        payload = content.encode("utf-8")
        if file_writer is not None:
            written = file_writer(target, payload)
            final_path = Path(written) if written is not None else target.resolve()
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            final_path = target.resolve()

        artifacts.append(
            ReportArtifact(
                format=fmt,
                path=final_path,
                artifact_key=art_key,
                content=content,
            )
        )

    return ReportRenderOutput(reports=tuple(artifacts))


__all__ = [
    "SUPPORTED_REPORT_FORMATS",
    "ReportArtifact",
    "ReportRenderOutput",
    "build_html_report",
    "build_markdown_report",
    "parse_report_formats",
    "render_report",
]
