"""Asset input adapters shared by command-line, textarea and historical leg APIs."""

import math
from collections.abc import Iterable, Iterator
from typing import Literal

from asset_catalog import SUPPORTED_ASSET_TYPES
from contracts import AssetSpec

Syntax = Literal["cli", "web", "legacy"]


def parse_asset_token(
    token: str,
    *,
    syntax: Syntax = "cli",
    asset_type: str | None = None,
    adjust: str | None = None,
) -> AssetSpec:
    cleaned = str(token or "").strip()
    separators = {"cli": ":|", "web": ",|:", "legacy": ",|"}[syntax]
    if not cleaned:
        raise ValueError(
            "Symbol must be a non-empty string." if syntax == "cli" else "空的标的行。"
        )
    for sep in separators:
        if sep not in cleaned:
            continue
        prefix, symbol = (piece.strip() for piece in cleaned.split(sep, 1))
        key = prefix.lower()
        if not key or not symbol:
            if syntax != "cli":
                raise ValueError(f"{'代码' if key else '资产类型'}不能为空：{token!r}")
            raise ValueError(
                f"Malformed asset identifier {cleaned!r}: expected TYPE{sep}SYMBOL with non-empty parts."
            )
        if key not in SUPPORTED_ASSET_TYPES:
            supported = ", ".join(SUPPORTED_ASSET_TYPES)
            if syntax != "cli":
                raise ValueError(f"不支持的资产类型 {prefix!r}；可选：{supported}")
            raise ValueError(
                f"Unknown asset type prefix {key!r} in {cleaned!r}; supported: {supported}."
            )
        if asset_type is not None and key != asset_type:
            raise ValueError(
                f"Conflicting asset_type: --asset-type={asset_type!r} vs prefix {key!r}."
            )
        break
    else:
        if syntax == "legacy":
            raise ValueError(
                f"每行需为 `资产类型,代码` 或 `资产类型|代码`，例如 `etf,510300`；无效行：{token!r}"
            )
        if syntax == "web":
            raise ValueError(
                f"每行需为 `资产类型,代码`、`资产类型|代码` 或 `资产类型:代码`。 无效输入：{token!r}"
            )
        if asset_type is None:
            raise ValueError(
                f"Ambiguous bare symbol {cleaned!r}: specify -t/--asset-type or prefix as TYPE:SYMBOL (e.g. etf:{cleaned}, us_stock:{cleaned})."
            )
        key, symbol = asset_type, cleaned
    spec = AssetSpec(key, symbol, adjust=adjust)
    if syntax == "web" and symbol != spec.canonical_symbol:
        return AssetSpec(key, spec.canonical_symbol, adjust=adjust)
    return spec


def _segments(items: Iterable[str]) -> Iterator[str]:
    for item in items:
        for line in item.splitlines():
            for part in line.split(";"):
                if part.strip():
                    yield part.strip()


def _cli_tokens(segment: str) -> list[str]:
    parts = [p.strip() for p in segment.split(",")]
    if not all(parts):
        raise ValueError(f"Empty token in comma-separated asset list: {segment!r}")
    if (
        len(parts) == 2
        and not any(c in segment for c in ":|")
        and parts[0].lower() in SUPPORTED_ASSET_TYPES
        and parts[1].lower() not in SUPPORTED_ASSET_TYPES
    ):
        return [":".join(parts)]
    return parts


def parse_asset_list(
    items: Iterable[str],
    *,
    syntax: Syntax = "cli",
    asset_type: str | None = None,
    adjust: str | None = None,
) -> tuple[AssetSpec, ...]:
    specs = []
    seen = set()
    for segment in _segments(items):
        for token in _cli_tokens(segment) if syntax == "cli" else [segment]:
            spec = parse_asset_token(token, syntax=syntax, asset_type=asset_type, adjust=adjust)
            key = (spec.asset_type, spec.canonical_symbol)
            if syntax == "cli" or key not in seen:
                specs.append(spec)
            seen.add(key)
    # CLI request contracts reject duplicate legs. The textarea historically folds them.
    if syntax == "cli" and not specs:
        raise ValueError(
            "At least 2 asset specs are required (e.g. etf:510300 us_stock:SPY or --assets ...)."
        )
    if syntax != "cli" and len(specs) < 2:
        raise ValueError(
            "有效前沿至少需要 2 个不重复标的（`类型,代码` 每行一条，或用 `;` 分隔）。"
            if syntax == "legacy"
            else "至少需要 2 个不重复标的（支持 `类型,代码` 或 `类型:代码`）。"
        )
    return tuple(specs)


def parse_weights(raw: str | None, *, web: bool = False) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if not raw.strip():
        raise ValueError(
            "自定义权重策略 (custom) 需要输入与标的数一致的权重，例如 `0.4, 0.3, 0.3`。"
            if web
            else "--weights cannot be empty when specified."
        )
    weights = []
    for token in raw.replace(";", ",").replace(" ", ",").split(","):
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(
                f"权重包含无效数字：{token!r}"
                if web
                else f"Invalid numeric weight {token!r} in --weights."
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"权重不能为 NaN 或 Inf：{token!r}"
                if web
                else f"Non-finite weight {token!r} in --weights."
            )
        weights.append(value)
    if not weights:
        raise ValueError(
            "请输入有效的数字权重，例如 `0.5, 0.5`。"
            if web
            else "--weights must contain numeric weights matching --assets."
        )
    return tuple(weights)
