"""Bundle cleanup has a closed set of owned output names."""

import json

from quant_cli import _collect_stale_managed_artifacts


def test_prior_manifest_cannot_claim_user_files(tmp_path):
    (tmp_path / "charts").mkdir()
    names = ["notes.md", "charts/review.png", "custom.csv"]
    for name in names:
        (tmp_path / name).write_bytes(b"user owned")
    (tmp_path / "result.json").write_text(
        json.dumps({"artifacts": dict(zip(("report_md", "chart_price_png", "old_csv"), names))})
    )
    stale = tmp_path / "report.md"
    stale.touch()
    assert _collect_stale_managed_artifacts(
        tmp_path, new_target_resolved={tmp_path / "result.json"}
    ) == [stale]
