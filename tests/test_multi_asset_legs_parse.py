"""parse_multi_asset_legs / multi_asset_leg_column_name：解析与边界（不拉网、不启动 Streamlit）。"""

from __future__ import annotations

import pytest

from app import SUPPORTED_ASSET_TYPES, multi_asset_leg_column_name, parse_multi_asset_legs


def test_parse_basic_comma_and_pipe() -> None:
    text = "etf,510300\nus_stock|AAPL"
    legs = parse_multi_asset_legs(text)
    assert legs == (("etf", "510300"), ("us_stock", "AAPL"))


def test_parse_semicolon_same_line() -> None:
    text = "etf,510300; etf,510500 ; hk_stock,00700"
    legs = parse_multi_asset_legs(text)
    assert legs == (("etf", "510300"), ("etf", "510500"), ("hk_stock", "00700"))


def test_parse_ignores_empty_lines() -> None:
    text = "\n\netf,510300\n\n\netf,510500\n"
    legs = parse_multi_asset_legs(text)
    assert legs == (("etf", "510300"), ("etf", "510500"))


def test_parse_dedupes_same_type_symbol_preserves_order() -> None:
    text = "etf,510300\netf,510500\netf,510300"
    legs = parse_multi_asset_legs(text)
    assert legs == (("etf", "510300"), ("etf", "510500"))


def test_parse_crypto_column_name_has_slash() -> None:
    text = "crypto,BTC/USDT\ncrypto,ETH/USDT"
    legs = parse_multi_asset_legs(text)
    assert multi_asset_leg_column_name(legs[0][0], legs[0][1]) == "crypto:BTC/USDT"


def test_parse_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="不支持的资产类型"):
        parse_multi_asset_legs("etf,510300\nnope,FOO")


def test_parse_rejects_fewer_than_two_after_dedupe() -> None:
    with pytest.raises(ValueError, match="至少需要 2"):
        parse_multi_asset_legs("etf,510300\netf,510300")


def test_parse_rejects_single_leg() -> None:
    with pytest.raises(ValueError, match="至少需要 2"):
        parse_multi_asset_legs("etf,510300")


def test_parse_rejects_bad_line_format() -> None:
    with pytest.raises(ValueError, match="每行需为"):
        parse_multi_asset_legs("etf 510300\netf,510500")


def test_parse_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="代码不能为空"):
        parse_multi_asset_legs("etf,\netf,510500")


def test_parse_normalizes_asset_type_case() -> None:
    legs = parse_multi_asset_legs("ETF,510300\nEtf,510500")
    assert legs[0][0] == "etf" and legs[1][0] == "etf"


def test_supported_types_cover_expected_keys() -> None:
    assert "etf" in SUPPORTED_ASSET_TYPES
    assert "crypto" in SUPPORTED_ASSET_TYPES
    assert "bond_us_yield" in SUPPORTED_ASSET_TYPES
