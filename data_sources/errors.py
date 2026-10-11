"""Shared data-source exceptions."""


class DataFetchError(RuntimeError):
    """数据抓取或清洗失败时抛出的异常。"""


class EmptySymbolDataError(DataFetchError):
    """标的无行情、空表或清洗后无有效 OHLCV 行时抛出，供 Web 端展示统一提示。"""


class NetworkFetchError(DataFetchError):
    """传输层失败：超时、TLS、连接重置、代理错误等。"""


class VendorUpstreamError(DataFetchError):
    """上游 HTTP 非预期状态、JSON 无法解析或接口明确拒绝等。"""


class ChinaEquityPipelineError(DataFetchError):
    """AKShare 主源与东方财富直连回退均未能返回有效 K 线时的汇总错误。"""
