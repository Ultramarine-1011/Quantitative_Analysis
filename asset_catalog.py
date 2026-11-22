"""Stable asset keys, UI labels, defaults and explanations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

CHINA_EQUITY_ASSET_TYPES: tuple[str, ...] = ("etf", "ashare")
# 这些映射把程序内部使用的短 key 转成界面/报错里更容易理解的中文名称。
CHINA_EQUITY_LABELS: dict[str, str] = {
    "etf": "A股 ETF",
    "ashare": "A股个股 (A-Share Stock)",
}
DEFAULT_CHINA_EQUITY_SYMBOLS: dict[str, str] = {
    "etf": "510300",
    "ashare": "600519",
}

SUPPORTED_ASSET_TYPES: tuple[str, ...] = (
    "etf",
    "ashare",
    "mutual_fund",
    "gold",
    "crypto",
    "us_stock",
    "hk_stock",
    "global_index",
    "bond_cn",
    "bond_us_yield",
    "commodity",
    "creit",
    "fx",
)

# 资产默认交易日年化因子：加密货币 7x24 为 365，外汇连续交易周 5 日为 260，其余默认 252。
ASSET_TRADING_DAYS: dict[str, int] = {
    "etf": 252,
    "ashare": 252,
    "mutual_fund": 252,
    "gold": 252,
    "crypto": 365,
    "us_stock": 252,
    "hk_stock": 252,
    "global_index": 252,
    "bond_cn": 252,
    "bond_us_yield": 252,
    "commodity": 252,
    "creit": 252,
    "fx": 260,
}

# 资产默认计价币种；crypto 由交易对后缀动态解析，global_index/fx 按具体代码解析。
ASSET_DEFAULT_CURRENCY: dict[str, str] = {
    "etf": "CNY",
    "ashare": "CNY",
    "mutual_fund": "CNY",
    "gold": "USD",
    "crypto": "QUOTE",
    "us_stock": "USD",
    "hk_stock": "HKD",
    "global_index": "INDEX",
    "bond_cn": "CNY",
    "bond_us_yield": "USD",
    "commodity": "USD",
    "creit": "CNY",
    "fx": "FX",
}

# 收益性质标签：明确区分复权股价、累计净值、期货连续价格、纯价格指数与收益率水平。
ASSET_RETURN_BASIS: dict[str, str] = {
    "etf": "adjusted_price",
    "ashare": "adjusted_price",
    "mutual_fund": "cumulative_nav",
    "gold": "futures_continuous",
    "crypto": "spot_price",
    "us_stock": "adjusted_or_price",
    "hk_stock": "adjusted_or_price",
    "global_index": "price_index",
    "bond_cn": "exchange_bond_price",
    "bond_us_yield": "yield_level",
    "commodity": "futures_continuous",
    "creit": "adjusted_or_price",
    "fx": "fx_rate",
}

ASSET_RETURN_BASIS_NOTES: dict[str, str] = {
    "adjusted_price": "复权价格序列（除权除息按上游复权因子调整，税费与股息再投口径与全收益指数略有差异）。",
    "unadjusted_price": "不复权历史行情序列（遇拆股、分红送转将出现价格跳空，不能直接作为连续复权总回报计算）。",
    "cumulative_nav": "基金累计净值序列（含分红口径，但无日内高低价与成交量）。",
    "futures_continuous": "期货主力连续价格（不含展期收益 Roll Yield 与保证金利息，不能直接等价于可投资总回报）。",
    "spot_price": "现货价格序列（不含质押/借贷收益；若以 USDT 计价，需单独处理 USDT/USD 汇率）。",
    "adjusted_or_price": "股票历史行情（是否复权取决于 adjust 参数；不复权序列遇拆股/派息将出现价格跳空，且复权价与含息净总回报存在税费差异）。",
    "price_index": "市场价格指数（多数不含股息再投资，低估含息总回报）。",
    "exchange_bond_price": "交易所债券价格 K 线（未拆算票息再投、久期与凸性）。",
    "yield_level": "收益率水平序列（%），并非债券持有期回报，禁止直接纳入资产净值组合优化。",
    "fx_rate": "即期汇率比价序列（非生息投资净值，仅反映货币相对波动）。",
}

US_TREASURY_YIELD_10Y_CN_COL = "美国国债收益率10年"

FX_PAIR_ALIASES: dict[str, str] = {
    "USD/CNY": "USDCNH",
    "USD/CNH": "USDCNH",
    "USDCNY": "USDCNH",
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD",
}

GOLD_SYMBOL_ALIASES: dict[str, str] = {"GC": "GC", "AU9999": "GC"}


GLOBAL_INDEX_US_SINA_ALIASES: dict[str, str] = {
    "SPX": ".INX",
    "INX": ".INX",
    "GSPC": ".INX",
    "标普500": ".INX",
    "NDX": ".NDX",
    "IXIC": ".IXIC",
    "纳斯达克": ".IXIC",
    "DJI": ".DJI",
    "DJIA": ".DJI",
    "道琼斯": ".DJI",
}
GLOBAL_INDEX_HK_SINA_ALIASES: dict[str, str] = {
    "HSI": "HSI",
    "恒生指数": "HSI",
    "HSCEI": "HSCEI",
    "国企指数": "HSCEI",
}
GLOBAL_INDEX_GLOBAL_SINA_ALIASES: dict[str, str] = {
    "FTSE": "英国富时100指数",
    "英国富时100": "英国富时100指数",
    "GDAXI": "德国DAX 30种股价指数",
    "德国DAX30": "德国DAX 30种股价指数",
    "FCHI": "法CAC40指数",
    "法国CAC40": "法CAC40指数",
    "SX5E": "欧洲Stoxx50指数",
    "欧洲斯托克50": "欧洲Stoxx50指数",
    "N225": "日经225指数",
    "日经225": "日经225指数",
}


DOLLAR_INDEX_ALIASES = frozenset({"DXY", "UDI", "美元指数", "DINIW"})


_INDEX_SYMBOLS = {
    alias.upper(): canonical
    for routes in (
        GLOBAL_INDEX_US_SINA_ALIASES,
        GLOBAL_INDEX_HK_SINA_ALIASES,
        GLOBAL_INDEX_GLOBAL_SINA_ALIASES,
    )
    for alias, canonical in routes.items()
}
_INDEX_SYMBOLS.update({canonical.upper(): canonical for canonical in _INDEX_SYMBOLS.values()})
_INDEX_SYMBOLS.update(dict.fromkeys(DOLLAR_INDEX_ALIASES, "DXY"))


def check_index_conversion(asset_type: str, symbol: str, base_currency: str) -> None:
    if (
        asset_type == "global_index"
        and symbol.upper() in DOLLAR_INDEX_ALIASES
        and base_currency.upper() != "LOCAL"
    ):
        raise ValueError(
            "index_points_not_currency: 美元指数为相对一篮子货币的指数点位，请使用 LOCAL。"
        )


def normalize_symbol_alias(asset_type: str, symbol: object) -> str:
    """Normalize asset symbol aliases to canonical cache and metadata identifiers."""
    asset_key = str(asset_type).strip().lower()
    raw = "" if symbol is None else str(symbol).strip()
    if asset_key == "global_index":
        return _INDEX_SYMBOLS.get(raw.upper(), raw.upper())
    if asset_key == "bond_us_yield":
        if (
            not raw
            or raw.upper() in {"10Y", "默认", "DEFAULT"}
            or raw == US_TREASURY_YIELD_10Y_CN_COL
        ):
            return "10Y"
        return raw
    if asset_key in {"ashare", "etf", "creit"}:
        upper = raw.upper()
        for prefix in ("SH", "SZ", "BJ"):
            if upper.startswith(prefix) and len(upper) == 8 and upper[2:].isdigit():
                return upper[2:]
            if upper.endswith(f".{prefix}") and len(upper) == 9 and upper[:6].isdigit():
                return upper[:6]
        return upper
    if asset_key == "hk_stock":
        if raw.isdigit():
            return raw.zfill(5)
        return raw.upper()
    if asset_key == "fx":
        cleaned = raw.upper().replace(" ", "")
        return FX_PAIR_ALIASES.get(cleaned, cleaned.replace("/", ""))
    if asset_key == "gold":
        cleaned = raw.upper().replace(".", "")
        return GOLD_SYMBOL_ALIASES.get(cleaned, raw.upper())
    if asset_key == "crypto":
        return raw.upper().replace(" ", "")
    return raw.upper()


def resolve_return_basis(
    asset_type: str,
    adjust: str | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> str:
    """Resolve granular 9-value return basis, accounting for dynamic adjust flags."""
    asset_key = str(asset_type).strip().lower()
    attr_map = attrs if isinstance(attrs, Mapping) else {}
    existing = str(attr_map.get("return_basis", "") or "").strip()
    if existing and existing != "adjusted_or_price":
        return existing

    if asset_key in {"etf", "ashare", "us_stock", "hk_stock"}:
        if adjust is not None:
            eff_adjust = resolve_adjustment(asset_key, adjust)
        elif "adjust" in attr_map and attr_map.get("adjust") is not None:
            eff_adjust = str(attr_map.get("adjust") or "").strip()
        else:
            eff_adjust = DEFAULT_ADJUSTMENT[asset_key]
        return "unadjusted_price" if eff_adjust == "" else "adjusted_price"

    return ASSET_RETURN_BASIS.get(asset_key, "adjusted_or_price")


def resolve_semantic_series_kind(
    asset_type: str,
    return_basis: str | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> str:
    """Resolve semantically aligned series_kind from return_basis and attrs."""
    asset_key = str(asset_type).strip().lower()
    attr_map = attrs if isinstance(attrs, Mapping) else {}
    raw_kind = str(attr_map.get("series_kind", "") or "").strip().lower()
    basis = (
        str(
            return_basis
            or attr_map.get("return_basis")
            or ASSET_RETURN_BASIS.get(asset_key, "")
            or ""
        )
        .strip()
        .lower()
    )

    if raw_kind == "yield_level" or basis == "yield_level" or asset_key == "bond_us_yield":
        return "yield_level"
    if raw_kind == "fx_rate" or basis == "fx_rate" or asset_key == "fx":
        return "fx_rate"
    if raw_kind == "cumulative_nav" or basis == "cumulative_nav" or asset_key == "mutual_fund":
        return "cumulative_nav"
    if basis in {"futures_continuous", "price_index", "exchange_bond_price"}:
        return basis
    return raw_kind or "price"


SUPPORTED_ADJUSTMENTS: dict[str, tuple[str, ...]] = {
    "etf": ("hfq", "qfq", ""),
    "ashare": ("hfq", "qfq", ""),
    "hk_stock": ("hfq", "qfq", ""),
    # 新浪美股日线 stock_us_daily 仅实现前复权 ('qfq') 与不复权 ('')，不支持后复权 ('hfq')。
    "us_stock": ("qfq", ""),
}

DEFAULT_ADJUSTMENT: dict[str, str] = {
    "etf": "hfq",
    "ashare": "hfq",
    "hk_stock": "hfq",
    "us_stock": "qfq",
}

ADJUSTMENT_LABELS: dict[str, str] = {
    "hfq": "后复权 (hfq)",
    "qfq": "前复权 (qfq)",
    "": "不复权 (raw)",
}


def get_supported_adjustments(asset_type: str | None) -> tuple[str, ...]:
    """返回指定资产类别实际支持的历史行情复权方式元组。"""
    if asset_type is None:
        return ()
    return SUPPORTED_ADJUSTMENTS.get(str(asset_type).strip().lower(), ())


def resolve_adjustment(asset_type: str, adjust: str | None) -> str:
    """校验并解析资产支持的复权方式（严格按底层接口能力报错）。"""
    key = str(asset_type).strip().lower()
    supported = get_supported_adjustments(key)
    if not supported:
        if adjust in (None, ""):
            return ""
        raise ValueError(f"资产类别 {key!r} 不支持复权参数 adjust={adjust!r}。")
    selected = DEFAULT_ADJUSTMENT[key] if adjust is None else str(adjust).strip()
    if selected not in supported:
        labels = ", ".join(repr(x) for x in supported)
        raise ValueError(
            f"资产类别 {key!r} 当前数据源不支持复权方式 {selected!r}；仅支持: {labels}。"
        )
    return selected


def get_asset_trading_days(asset_type: str | None) -> int:
    """返回资产类别对应的默认年化交易日数（未知类别默认 252）。"""
    if asset_type is None:
        return 252
    return ASSET_TRADING_DAYS.get(str(asset_type).strip().lower(), 252)


# UI 显示中文标签，内部仍使用稳定的英文 key 进行路由和缓存。
ASSET_LABELS: dict[str, str] = {
    **CHINA_EQUITY_LABELS,
    "mutual_fund": "公募基金 (Mutual Fund)",
    "gold": "全球贵金属",
    "crypto": "加密货币",
    "us_stock": "美股 (US Stock)",
    "hk_stock": "港股 (HK Stock)",
    "global_index": "全球指数（含美股指数/东财国际指数）",
    "bond_cn": "中国国债 K 线",
    "bond_us_yield": "美债收益率曲线（水平序列）",
    "commodity": "外盘大宗商品期货",
    "creit": "C-REITs（基础设施）",
    "fx": "外汇即期",
}
DEFAULT_SYMBOLS: dict[str, str] = {
    **DEFAULT_CHINA_EQUITY_SYMBOLS,
    "mutual_fund": "009691",
    "gold": "GC",
    "crypto": "BTC/USDT",
    "us_stock": "AAPL",
    "hk_stock": "00700",
    "global_index": ".INX",
    "bond_cn": "sh019547",
    "bond_us_yield": "10Y",
    "commodity": "CL",
    "creit": "508097",
    "fx": "USDCNH",
}
# 资产种类说明：类型界定、社会意义、在本应用中的数学处理、代表代码（供用户对照输入）。
ASSET_TYPE_GUIDES: dict[str, dict[str, str]] = {
    "etf": {
        "kind": "在交易所上市交易的开放式指数基金或行业/主题 ETF，份额可像股票一样买卖，净值与标的指数挂钩。",
        "social": "为散户提供分散化、低门槛的指数化投资工具，反映市场对行业、风格与宏观主题的预期，流动性集中体现机构与散户配置行为。",
        "math": "本应用使用**后复权**日线 OHLCV，与 A 股个股同一套收益率、波动率、布林带与回撤公式；成交量为真实成交。",
        "examples": "`510300`（沪深300ETF）、`510500`（中证500ETF）、`159915`（创业板ETF）",
    },
    "ashare": {
        "kind": "在上海或深圳证券交易所挂牌的人民币普通股（A 股），公司所有权凭证，价格由二级市场供需决定。",
        "social": "连接实体经济融资与居民财富管理，个股走势反映行业景气、政策预期与公司治理等信息，是境内权益风险定价的核心载体之一。",
        "math": "后复权日 K 线；指标基于收盘价序列的几何/统计性质（对数收益、滚动波动、极值带），**不**包含基本面因子。",
        "examples": "`600519`、`000001`、`688981`",
    },
    "mutual_fund": {
        "kind": "公募开放式基金，按净值申购赎回；本应用拉取的是**累计净值**序列，已折算为伪 OHLCV（开高低收相同）。",
        "social": "专业管理、分散投资，服务长期储蓄与养老等需求；净值曲线体现基金经理资产配置与申赎压力的综合结果。",
        "math": "无真实日内高低价，OHLC 均为净值；**成交量恒为 0**，图表以折线展示；夏普/CAGR 等仍按净值变化率计算，与 ETF 市价口径不同。",
        "examples": "`009691`、`161725`、`110011`",
    },
    "gold": {
        "kind": "通过外盘期货主力连续合约报价的贵金属（如 COMEX 黄金），美元计价，反映避险与真实利率预期。",
        "social": "传统避险与储备资产锚，与美元信用、地缘政治和通胀预期相关，常作为多元资产配置的分散项。",
        "math": "标准 OHLCV 日频；可存在换月导致的跳空，指标对跳变敏感；与 A 股不同币种与交易时段。",
        "examples": "`GC`（黄金）、`AU9999`（映射至 GC）",
    },
    "crypto": {
        "kind": "去中心化或交易所报价的加密资产，本应用默认通过 CCXT 拉取 **Binance** 现货日线（如 `BTC/USDT`）。",
        "social": "高风险高波动的另类资产，反映全球流动性、监管叙事与技术采用预期；7×24 交易，与股市日历不完全对齐。",
        "math": "UTC 日线合成 OHLCV；可设代理访问；夏普与回撤对极端日收益敏感，不宜与股票直接横比。",
        "examples": "`BTC/USDT`、`ETH/USDT`、`SOL/USDT`",
    },
    "us_stock": {
        "kind": "美国主要交易所上市的公司普通股，美元计价，本应用使用 AKShare 美股日线。",
        "social": "全球科技与消费龙头集中地，股价反映全球增长预期与美元流动性；与 A 股存在时差与制度差异。",
        "math": "日 OHLCV；与境内股相同的特征工程管线；注意财报季跳空对波动率估计的影响。",
        "examples": "`AAPL`、`MSFT`、`NVDA`",
    },
    "hk_stock": {
        "kind": "在香港联合交易所上市的股票，港币计价；代码常为 5 位数字，本应用会自动补零。",
        "social": "连接内地资产与全球资本的中枢之一，大量中资股与美元利率、南向资金流量密切相关。",
        "math": "日 OHLCV；收益率统计与 A 股同构；节假日安排与沪深不同，对齐指数时需注意交易日历。",
        "examples": "`00700`、`09988`、`01810`",
    },
    "global_index": {
        "kind": "表征一组证券整体表现的统计量。以 `.` 开头（如 `.INX`）走**新浪美股指数**；`HSI`/`SPX` 等简码走**东财全球指数**中文名映射。",
        "social": "宏观与风险情绪的晴雨表，用于观察市场宽度、区域轮动与系统性风险，是资产配置与业绩基准的参照。",
        "math": "点位序列视为“价格”；部分来源**无成交量**（Volume=0），图表可能为单面板 K 线；与成分股加总并非线性关系。",
        "examples": "`.INX`（标普500新浪）、`HSI`（恒生）、`SPX`（标普500东财）、`DXY`（美元指数）",
    },
    "bond_cn": {
        "kind": "在沪深挂牌交易的**国债或债券**日 K（新浪源），代码形如 `sh019547`。",
        "social": "无风险利率与债市情绪的表征，影响全市场贴现率；机构配置与货币政策预期会体现在价格波动中。",
        "math": "标准 OHLCV；部分行情源无成交量字段时已补 0；久期与凸性等债项指标**未**在本应用中计算。",
        "examples": "`sh019547`（16国债19，持续交易中；历史示例 `sh010107` 已于 2021-07 到期）",
    },
    "bond_us_yield": {
        "kind": "中债登口径的**美国国债收益率曲线**日度数据（`bond_zh_us_rate`），`Close` 为**收益率水平（%）**，非债券价格。",
        "social": "全球资产定价的锚之一，影响汇率、房贷与企业债利差；10 年期常被视为“无风险长期利率”的代理变量。",
        "math": "伪 OHLCV，水平序列；看板中 CAGR/夏普等解释为**水平变化统计**，**不是**持有美债的持有期回报；默认列可用 `10Y`。",
        "examples": "`10Y`（默认美债10年列）、或粘贴表头完整列名如 `美国国债收益率10年`",
    },
    "commodity": {
        "kind": "外盘**大宗商品期货**主力连续（如原油、铜），与贵金属同源 `futures_foreign_hist`，美元计价为主。",
        "social": "实体经济成本与周期波动的晴雨表，连接通胀、制造业与地缘供给；常作为股票组合的商品敞口代理。",
        "math": "连续合约有换月跳空；波动率与趋势指标对供给冲击敏感；与股票相关性随周期变化。",
        "examples": "`CL`（原油）、`HG`（铜）、`NG`（天然气，视数据源是否提供）",
    },
    "creit": {
        "kind": "中国**基础设施公募 REITs**，在沪深交易，份额代表对底层基础设施收益权的证券化权益。",
        "social": "盘活存量基建资产、拓宽投融资渠道；价格反映分红预期、利率环境与流动性溢价。",
        "math": "东财日 K 线 OHLCV；与股票共用技术指标；分红除权规则与 ETF/股票不同，复权口径以数据源为准。",
        "examples": "`508097`、`180101`、`508000`（示例，以交易所挂牌为准）",
    },
    "fx": {
        "kind": "**外汇即期**汇率对（如离岸人民币 `USDCNH`），本应用经东财多节点拉取日 K，表示一单位基准货币兑另一货币的比价。",
        "social": "国际贸易与资本流动的价格纽带，影响出口竞争力与跨境资金成本；央行预期与利差是主要驱动。",
        "math": "汇率水平为正值序列，宜用 K 线观察；对数收益近似刻画日相对变化；与股票指数联合分析时需注意共线性与因果方向。",
        "examples": "`USDCNH`、`EURUSD`、`USD/JPY`（别名映射为数据源代码时以 `EURUSD` 等形式为准）",
    },
}


def format_asset_guide_markdown(asset_type: str) -> str:
    """生成当前资产类型的 Markdown 说明（侧栏与主区复用）。"""
    key = str(asset_type).strip().lower()
    g = ASSET_TYPE_GUIDES.get(key)
    if g is None:
        return ""
    title = ASSET_LABELS.get(key, key)
    return (
        "#### %s\n\n"
        "**类型**　%s\n\n"
        "**社会意义**　%s\n\n"
        "**数学与诊断口径**　%s\n\n"
        "**代表代码示例**　%s\n"
    ) % (title, g["kind"], g["social"], g["math"], g["examples"])


def build_ef_multi_asset_codes_reference_df() -> pd.DataFrame:
    """Tab3 有效前沿多标的：类型键、默认代码与示例，供前端表格展示。"""
    rows: list[dict[str, str]] = []
    for key in SUPPORTED_ASSET_TYPES:
        default_sym = str(DEFAULT_SYMBOLS.get(key, "") or "").strip()
        g = ASSET_TYPE_GUIDES.get(key, {})
        examples_raw = str(g.get("examples", "") or "")
        examples_plain = examples_raw.replace("`", "")
        wide_example = ""
        if default_sym:
            wide_example = "%s,%s" % (key, default_sym)
        rows.append(
            {
                "类型键（小写）": key,
                "资产名称": ASSET_LABELS.get(key, key),
                "默认代码": default_sym or "—",
                "宽表列名示例": wide_example or "—",
                "代码示例与说明": examples_plain or "—",
            }
        )
    return pd.DataFrame(rows)


PRESET_PORTFOLIOS: dict[str, tuple[str, ...]] = {
    "自定义（默认跨市示例）": ("etf,510300", "etf,510500", "us_stock,AAPL"),
    "境内大类股债均衡": ("etf,510300", "etf,510500", "etf,511010", "etf,518880"),
    "中美权益核心资产": ("etf,510300", "us_stock,AAPL", "us_stock,MSFT", "hk_stock,00700"),
    "全天候跨市场多资产": ("etf,510300", "us_stock,SPY", "gold,GC", "bond_cn,sh019547"),
    "高波动数字与科技": ("us_stock,NVDA", "us_stock,AAPL", "gold,GC", "crypto,BTC/USDT"),
}


_TAB2_SYSTEM_MATH_MARKDOWN = r"""
### 1. 收益与风险度量

设 $P_t$ 为第 $t$ 个交易日收盘价（或复权价），简单收益率与对数收益率常分别写为：

$$R_t=\frac{P_t-P_{t-1}}{P_{t-1}}, \qquad r_t=\ln\frac{P_t}{P_{t-1}}$$

在样本区间 $[0,T]$ 内，若共 $n$ 个交易日，年化复合收益率（CAGR）可表示为：

$$\mathrm{CAGR}=\left(\frac{P_T}{P_0}\right)^{365.25/d}-1$$

其中 $d$ 为起止日期间的自然日数，与诊断看板及 CLI 的实际实现一致。

最大回撤（Maximum Drawdown）衡量自历史峰值以来的最深回落比例。设净值曲线为 $W_t$（例如从 1 起算的累计净值），则：

$$\mathrm{MDD}=\max_{\tau\in[0,T]}\left(\frac{\max_{s\le\tau}W_s-W_\tau}{\max_{s\le\tau}W_s}\right)$$

内部计算返回非负回撤深度；诊断面板和 CLI 以负百分比显示该深度。

夏普比率（Sharpe Ratio）在经典定义下使用超额收益相对波动率的比值，并按 $\sqrt{252}$ 做年化缩放：

$$S=\frac{E[r_t-\ln(1+r_f)/252]}{\sigma(r_t)}\sqrt{252}$$

其中 $r_f=1.6\%$ 为年化无风险利率，$r_t$ 为日对数收益率，$\sigma$ 为样本标准差（ddof=1）。进阶页的有效前沿采用日简单收益率均值与协方差年化。

---

### 2. 累计收益与信号展示

累计收益率可将各期收益链式复合，例如简单收益口径下：

$$1+R_{\mathrm{cum}}=\prod_{t=1}^{n}(1+R_t)$$

看板中的「最新信号」由特征工程与规则化输出共同决定，其底层是对价格、均线与波动包络等构造量的序贯更新；在阅读图表时，可将均线视为对 $P_t$ 的线性平滑，将布林带可视作对条件方差结构的启发式刻画。

---

### 3. 可视化与数据契约

K–线（OHLC）与成交量在同一时间轴上联合展示，横轴为交易日期。清洗后的数据表遵循统一列名契约（`Open` / `High` / `Low` / `Close` / `Volume`），以便跨境、跨资产类别复用同一套指标与绘图管线。

> **说明**：上述记号与年化习惯用于帮助理解面板指标；实际实现细节（如对数/简单收益选用、缺失值处理、复权口径等）以代码与拉取数据源为准。
"""
