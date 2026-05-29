"""实验记录 JSON 保存与读取。"""

from __future__ import annotations

import json
from pathlib import Path

from experiment_store import save_experiment


def test_save_experiment_writes_required_fields() -> None:
    output_dir = Path(".pytest_cache") / "experiment_store_tests"
    path = save_experiment(
        {"strategy": "equal_weight"},
        {"sharpe": 1.2},
        {"data_range": "2024"},
        output_dir=output_dir,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["config"]["strategy"] == "equal_weight"
    assert payload["metrics"]["sharpe"] == 1.2
    assert "created_at" in payload
    assert "code_version" in payload
