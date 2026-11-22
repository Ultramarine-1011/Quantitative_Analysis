import pytest

from asset_inputs import parse_asset_list, parse_weights
from contracts import CompareRequest


@pytest.mark.parametrize("syntax", ["cli", "web", "legacy"])
def test_alias_order_and_duplicate_policy(syntax):
    specs = parse_asset_list(["ETF,sh510300;hk_stock|700\netf,510300.SH"], syntax=syntax)
    ids = [s.canonical_symbol for s in specs]
    if syntax == "cli":
        assert ids == ["510300", "00700", "510300"]
        with pytest.raises(ValueError, match="重复"):
            CompareRequest(specs, "2024-01-01", "2024-02-01")
    else:
        assert ids == ["510300", "00700"]


@pytest.mark.parametrize("web", [True, False])
def test_weight_separators_and_finite_numbers(web):
    assert parse_weights(".1; .2, .7", web=web) == (0.1, 0.2, 0.7)
    with pytest.raises(ValueError):
        parse_weights("1 inf", web=web)


@pytest.mark.parametrize(
    "alias, canonical", [("SPX", ".INX"), ("N225", "日经225指数"), ("美元指数", "DXY")]
)
def test_vendor_index_aliases_share_identity_without_changing_the_requested_route(alias, canonical):
    from data_sources.cache import build_cache_key

    items = [f"global_index:{alias};global_index:{canonical};etf:510300"]
    web = parse_asset_list(items, syntax="web")
    assert len(web) == 2 and web[0].symbol == alias
    assert web[0].canonical_symbol == canonical
    assert build_cache_key("global_index", alias) == build_cache_key("global_index", canonical)
    with pytest.raises(ValueError, match="重复"):
        CompareRequest(parse_asset_list(items), "2024-01-02", "2024-05-31")
