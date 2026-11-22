"""Compatibility entry points for multi-asset research."""

from services.comparison import compare_assets
from services.portfolio import analyze_portfolio

__all__ = ["analyze_portfolio", "compare_assets"]
