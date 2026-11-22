"""Structured UI error taxonomy distinguishing market, empty-data, validation, and compute failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from asset_resolver import format_diagnosis_user_message
from data_sources.errors import (
    ChinaEquityPipelineError,
    DataFetchError,
    EmptySymbolDataError,
    NetworkFetchError,
    SnapshotError,
    VendorUpstreamError,
)

ErrorCategory = Literal["validation", "empty_data", "market_data", "computation", "dependency"]


@dataclass(frozen=True)
class UIError:
    """User-facing structured error payload with clear failure classification."""

    category: ErrorCategory
    badge: str
    title: str
    message: str
    hint: str = ""


def make_validation_error(message: str, *, hint: str = "") -> UIError:
    """Build a structured input-validation error."""
    return UIError(
        category="validation",
        badge="输入校验",
        title="参数或配置校验未通过",
        message=message,
        hint=hint,
    )


def classify_exception(exc: BaseException) -> UIError:
    """Classify backend/input exceptions into explicit user-facing UI categories."""
    raw_msg = format_diagnosis_user_message(exc)
    exc_text = str(exc).strip()

    if isinstance(exc, ImportError):
        return UIError(
            category="dependency",
            badge="可选依赖缺失",
            title="当前功能所需的 Python 可选依赖未安装",
            message=exc_text or raw_msg,
            hint="请按提示安装对应可选包后刷新页面重试；核心指标与数据表不受影响。",
        )

    if isinstance(exc, EmptySymbolDataError):
        return UIError(
            category="empty_data",
            badge="空数据 / 无匹配行情",
            title="未检索到有效市场行情数据",
            message=raw_msg,
            hint="请核对资产类别与代码格式是否匹配、所选起止日期内是否已上市/存在交易日。",
        )

    if isinstance(exc, NetworkFetchError):
        return UIError(
            category="market_data",
            badge="行情网络传输失败",
            title="外部行情接口连接或传输中断",
            message=raw_msg,
            hint="请检查网络连接或代理地址配置，或关闭“跳过本地缓存”以复用已有本地缓存。",
        )

    if isinstance(exc, (VendorUpstreamError, ChinaEquityPipelineError, DataFetchError)):
        return UIError(
            category="market_data",
            badge="行情数据源异常",
            title="上游行情服务暂不可用或返回异常",
            message=raw_msg,
            hint="请确认代码有效性与复权/币种支持范围，必要时开启代理或稍后重试。",
        )

    if isinstance(exc, SnapshotError):
        return UIError(
            category="empty_data",
            badge="快照读取失败",
            title="离线输入快照缺失或校验未通过",
            message=exc_text or raw_msg,
            hint="请检查快照路径及完整性，或切换回在线获取模式。",
        )

    if isinstance(exc, ValueError):
        semantic_tokens = (
            "yield_level",
            "利率水平",
            "收益率水平",
            "fx_rate",
            "汇率",
            "index_points",
            "美元指数",
            "USDT",
        )
        is_semantic_or_input = any(tok in exc_text for tok in semantic_tokens) or (
            "权重" in exc_text or "至少" in exc_text or "日期" in exc_text or "资产" in exc_text
        )
        if is_semantic_or_input:
            return UIError(
                category="validation",
                badge="输入 / 金融语义校验",
                title="当前输入或金融口径不符合分析前提",
                message=raw_msg,
                hint="这是正常的风控与金融语义拦截，并非数据缺失；请调整资产组合、基准货币或权重设置。",
            )
        return UIError(
            category="computation",
            badge="计算参数 / 样本约束",
            title="当前样本区间或参数无法完成模型求解",
            message=raw_msg,
            hint="可尝试放宽起止日期增加有效交易日、缩短滚动/训练窗口，或切换协方差收缩/验证协议。",
        )

    return UIError(
        category="computation",
        badge="分析计算失败",
        title="指标或组合引擎执行失败",
        message=raw_msg,
        hint="请检查参数边界（如权重上下限、窗口长度、基准资产重叠区间）后重试。",
    )


def format_ui_error_markdown(err: UIError | str) -> str:
    """Render a ``UIError`` or plain error message into single-banner markdown."""
    if isinstance(err, str):
        return err
    lines = [f"**【{err.badge}】{err.title}**", "", err.message]
    if err.hint:
        lines.extend(["", f"建议：{err.hint}"])
    return "\n".join(lines)
