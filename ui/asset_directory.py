"""Friendly asset directory, human-readable names, and query/leg parsing helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from asset_catalog import ASSET_LABELS, normalize_symbol_alias
from asset_inputs import parse_asset_token


@dataclass(frozen=True)
class AssetEntry:
    """Curated popular asset metadata for intuitive search and quick selection."""

    asset_type: str
    symbol: str
    name: str
    category: str
    note: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def token(self) -> str:
        """Canonical ``asset_type:symbol`` token."""
        return f"{self.asset_type}:{self.symbol}"


POPULAR_ASSETS: tuple[AssetEntry, ...] = (
    AssetEntry(
        asset_type="etf",
        symbol="510300",
        name="沪深300 ETF",
        category="A股 ETF",
        note="追踪沪深300指数，境内大盘核心宽基",
        aliases=("沪深300", "300etf", "hs300", "sh510300"),
    ),
    AssetEntry(
        asset_type="etf",
        symbol="510500",
        name="中证500 ETF",
        category="A股 ETF",
        note="追踪中证500指数，境内中盘均衡代表",
        aliases=("中证500", "500etf", "zz500", "sh510500"),
    ),
    AssetEntry(
        asset_type="etf",
        symbol="159915",
        name="创业板 ETF",
        category="A股 ETF",
        note="追踪创业板指，偏成长风格",
        aliases=("创业板", "cyb", "sz159915"),
    ),
    AssetEntry(
        asset_type="etf",
        symbol="511010",
        name="国债 ETF",
        category="A股 ETF",
        note="5年期国债基准利率债防御配置",
        aliases=("国债etf", "十年国债", "5年国债", "sh511010"),
    ),
    AssetEntry(
        asset_type="etf",
        symbol="518880",
        name="黄金 ETF",
        category="A股 ETF",
        note="境内人民币计价黄金现货 ETF",
        aliases=("黄金etf", "华安黄金", "sh518880"),
    ),
    AssetEntry(
        asset_type="etf",
        symbol="510880",
        name="红利 ETF",
        category="A股 ETF",
        note="高股息价值红利风格指数",
        aliases=("红利", "红利etf", "sh510880"),
    ),
    AssetEntry(
        asset_type="ashare",
        symbol="600519",
        name="贵州茅台",
        category="A股个股",
        note="A股消费龙头个股示例",
        aliases=("茅台", "贵州茅台", "sh600519"),
    ),
    AssetEntry(
        asset_type="ashare",
        symbol="300750",
        name="宁德时代",
        category="A股个股",
        note="新能源制造龙头个股示例",
        aliases=("宁德", "宁王", "sz300750"),
    ),
    AssetEntry(
        asset_type="us_stock",
        symbol="SPY",
        name="标普500 ETF (SPY)",
        category="美股",
        note="美国大盘宽基代表，美元计价",
        aliases=("标普500", "标普", "sp500"),
    ),
    AssetEntry(
        asset_type="us_stock",
        symbol="QQQ",
        name="纳斯达克100 ETF (QQQ)",
        category="美股",
        note="美国科技成长核心指数 ETF",
        aliases=("纳斯达克", "纳指100", "nasdaq100"),
    ),
    AssetEntry(
        asset_type="us_stock",
        symbol="AAPL",
        name="苹果 Apple",
        category="美股",
        note="美股核心科技个股",
        aliases=("苹果", "apple"),
    ),
    AssetEntry(
        asset_type="us_stock",
        symbol="MSFT",
        name="微软 Microsoft",
        category="美股",
        note="美股云计算与软件龙头",
        aliases=("微软", "microsoft"),
    ),
    AssetEntry(
        asset_type="us_stock",
        symbol="NVDA",
        name="英伟达 NVIDIA",
        category="美股",
        note="全球 AI 与半导体算力代表",
        aliases=("英伟达", "nvidia"),
    ),
    AssetEntry(
        asset_type="hk_stock",
        symbol="00700",
        name="腾讯控股",
        category="港股",
        note="港股互联网核心资产，港币计价",
        aliases=("腾讯", "tencent", "700"),
    ),
    AssetEntry(
        asset_type="hk_stock",
        symbol="09988",
        name="阿里巴巴-W",
        category="港股",
        note="港股大型电商与云计算公司",
        aliases=("阿里", "阿里巴巴", "baba", "9988"),
    ),
    AssetEntry(
        asset_type="gold",
        symbol="GC",
        name="COMEX 黄金",
        category="全球贵金属",
        note="国际黄金期货主力连续，美元计价",
        aliases=("黄金", "comex黄金", "au9999", "gold"),
    ),
    AssetEntry(
        asset_type="commodity",
        symbol="CL",
        name="WTI 原油",
        category="大宗商品",
        note="美国西德克萨斯轻质原油期货连续",
        aliases=("原油", "wti", "oil"),
    ),
    AssetEntry(
        asset_type="bond_cn",
        symbol="sh019547",
        name="中国国债 16国债19",
        category="中国国债",
        note="沪深交易所挂牌国债日 K 线",
        aliases=("中国国债", "16国债19", "019547"),
    ),
    AssetEntry(
        asset_type="bond_us_yield",
        symbol="10Y",
        name="10年期美债收益率",
        category="美债利率",
        note="水平利率序列（仅作利率走势观察，不参与净值组合）",
        aliases=("美债收益率", "10年美债", "美债10y", "us10y"),
    ),
    AssetEntry(
        asset_type="global_index",
        symbol=".INX",
        name="标普500指数",
        category="全球指数",
        note="美股基准价格指数（新浪美股指数口径）",
        aliases=("标普指数", "spx", "inx"),
    ),
    AssetEntry(
        asset_type="global_index",
        symbol="HSI",
        name="恒生指数",
        category="全球指数",
        note="香港股票市场整体基准指数",
        aliases=("恒生", "恒指"),
    ),
    AssetEntry(
        asset_type="global_index",
        symbol="DXY",
        name="美元指数",
        category="全球指数",
        note="美元对一篮子主要货币强弱点位指数",
        aliases=("美元指数", "udi"),
    ),
    AssetEntry(
        asset_type="mutual_fund",
        symbol="009691",
        name="公募基金示例 (009691)",
        category="公募基金",
        note="开放式公募基金累计净值序列",
        aliases=("公募基金", "基金009691"),
    ),
    AssetEntry(
        asset_type="creit",
        symbol="508097",
        name="中金普洛斯 REIT",
        category="C-REITs",
        note="中国基础设施公募 REITs",
        aliases=("普洛斯", "reits", "creit"),
    ),
    AssetEntry(
        asset_type="fx",
        symbol="USDCNH",
        name="离岸人民币汇率 (USD/CNH)",
        category="外汇即期",
        note="美元兑离岸人民币即期汇率",
        aliases=("人民币汇率", "usd/cnh", "usd/cny", "usdcny"),
    ),
    AssetEntry(
        asset_type="crypto",
        symbol="BTC/USDT",
        name="比特币 BTC",
        category="加密货币",
        note="7×24 小时交易数字资产，高波动",
        aliases=("比特币", "btc", "bitcoin"),
    ),
    AssetEntry(
        asset_type="crypto",
        symbol="ETH/USDT",
        name="以太坊 ETH",
        category="加密货币",
        note="主流智能合约加密资产，7×24 小时交易",
        aliases=("以太坊", "eth", "ethereum"),
    ),
)

QUICK_START_ASSETS: tuple[AssetEntry, ...] = (
    POPULAR_ASSETS[0],  # 沪深300 ETF (510300)
    POPULAR_ASSETS[1],  # 中证500 ETF (510500)
    POPULAR_ASSETS[10],  # 苹果 Apple (AAPL)
    POPULAR_ASSETS[15],  # COMEX 黄金 (GC)
    POPULAR_ASSETS[8],  # 标普500 ETF (SPY)
)

_ENTRY_BY_KEY: dict[tuple[str, str], AssetEntry] = {
    (
        entry.asset_type,
        normalize_symbol_alias(entry.asset_type, entry.symbol).upper(),
    ): entry
    for entry in POPULAR_ASSETS
}


def lookup_asset_entry(asset_type: str, symbol: str) -> AssetEntry | None:
    """Look up curated friendly metadata for an ``(asset_type, symbol)`` pair."""
    a_type = str(asset_type or "").strip().lower()
    raw_sym = str(symbol or "").strip()
    if not a_type or not raw_sym:
        return None
    norm = normalize_symbol_alias(a_type, raw_sym).upper()
    entry = _ENTRY_BY_KEY.get((a_type, norm))
    if entry is not None:
        return entry
    return _ENTRY_BY_KEY.get((a_type, raw_sym.upper()))


def format_asset_display(asset_type: str, symbol: str) -> str:
    """Format an asset for user-facing cards/pills (e.g. ``沪深300 ETF (510300)``)."""
    entry = lookup_asset_entry(asset_type, symbol)
    if entry is not None:
        if entry.symbol in entry.name:
            return entry.name
        return f"{entry.name} ({entry.symbol})"
    clean_sym = str(symbol or "").strip() or "10Y"
    cat = ASSET_LABELS.get(str(asset_type).strip().lower(), str(asset_type))
    short_cat = cat.split("（", 1)[0].split("(", 1)[0].strip()
    return f"{clean_sym} · {short_cat}"


def format_asset_picker_option(entry: AssetEntry) -> str:
    """Format an ``AssetEntry`` for searchable selectbox options."""
    return f"{entry.name} · {entry.symbol} · {entry.category}"


def resolve_quick_query(query: str) -> tuple[str, str] | None:
    """Resolve user search text (Chinese name, ticker, or ``type:symbol``) into ``(type, symbol)``."""
    cleaned = str(query or "").strip()
    if not cleaned:
        return None

    for sep in (":", ",", "|"):
        if sep in cleaned:
            try:
                spec = parse_asset_token(cleaned, syntax="web")
            except ValueError:
                return None
            sym = spec.symbol if spec.asset_type == "global_index" else spec.canonical_symbol
            return spec.asset_type, sym

    lowered = cleaned.lower()
    # 1) Exact picker option string match
    for entry in POPULAR_ASSETS:
        if lowered == format_asset_picker_option(entry).lower():
            return entry.asset_type, entry.symbol

    # 2) Exact symbol, name, or alias match
    for entry in POPULAR_ASSETS:
        candidates = (
            entry.symbol.lower(),
            entry.name.lower(),
            *(alias.lower() for alias in entry.aliases),
        )
        if lowered in candidates:
            return entry.asset_type, entry.symbol

    # 3) Prefix or substring match on name / aliases / symbol
    for entry in POPULAR_ASSETS:
        haystack = " ".join((entry.name, entry.symbol, *entry.aliases)).lower()
        if lowered in haystack:
            return entry.asset_type, entry.symbol

    return None


def parse_legs_loose(text: str) -> list[tuple[str, str]]:
    """Parse ``text`` into deduplicated ``(asset_type, symbol)`` pairs without requiring >= 2 items."""
    raw = str(text or "")
    items: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in raw.splitlines():
        for part in line.split(";"):
            token = part.strip()
            if not token:
                continue
            try:
                spec = parse_asset_token(token, syntax="web")
            except ValueError:
                continue
            sym = spec.symbol if spec.asset_type == "global_index" else spec.canonical_symbol
            canon_key = (spec.asset_type, normalize_symbol_alias(spec.asset_type, sym).upper())
            if canon_key not in seen:
                seen.add(canon_key)
                items.append((spec.asset_type, sym))
    return items


def serialize_legs(items: Sequence[tuple[str, str]]) -> str:
    """Serialize deduplicated ``(asset_type, symbol)`` pairs into canonical newline tokens."""
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for asset_type, symbol in items:
        a_type = str(asset_type or "").strip().lower()
        raw_sym = str(symbol or "").strip()
        if not a_type or not raw_sym:
            continue
        sym = (
            raw_sym.upper() if a_type == "global_index" else normalize_symbol_alias(a_type, raw_sym)
        )
        key = (a_type, sym.upper())
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{a_type}:{sym}")
    return "\n".join(lines)
