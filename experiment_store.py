"""本地实验记录保存。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def save_experiment(
    config: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    *,
    output_dir: str | Path = "outputs/experiments",
) -> Path:
    """保存一次实验记录为 JSON，返回文件路径。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload = {
        "created_at": created_at,
        "code_version": _code_version(),
        "config": dict(config),
        "metrics": dict(metrics),
        "artifacts": dict(artifacts or {}),
    }
    safe_ts = created_at.replace(":", "").replace("+", "Z")
    path = out_dir / ("experiment_%s.json" % safe_ts)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return path


__all__ = ["save_experiment"]

