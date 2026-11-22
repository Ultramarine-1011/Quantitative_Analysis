"""Check the implementation modules as well as the historical import facades."""

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("package", ["models", "services", "data_sources"])
def test_research_layers_have_no_frontend_or_cli_dependencies(package):
    root = Path(__file__).resolve().parents[1]
    forbidden = {"app", "main", "quant_cli", "reporting", "streamlit", "charts", "visualizer"}
    if package == "models":
        forbidden.update({"services", "data_sources", "asset_loader", "data_fetcher", "artifacts"})
    for path in (root / package).glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports = [name.name for name in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
                if package in {"services", "models"} and imports[0].startswith(package + "."):
                    assert all(not name.name.startswith("_") for name in node.names), path
            else:
                continue
            assert not {name.split(".")[0] for name in imports} & forbidden, path
