"""M6-1c tests: portable artifact storage, SHA-256 manifests, v0.1.0->v0.2.0 migration, security, and round-trips."""

from __future__ import annotations

import ast
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import quant_cli
from artifacts import (
    ArtifactIntegrityError,
    ArtifactSecurityError,
    SchemaMigrationError,
    SchemaMigrationRecord,
    UnsupportedSchemaVersionError,
    load_result_from_artifacts,
    load_result_with_migration_info,
    migrate_result_payload,
    resolve_safe_artifact_path,
    save_result_bundle,
    verify_artifact_manifest,
)
from contracts import (
    SCHEMA_VERSION,
    AnalysisRequest,
    AssetSpec,
    CompareRequest,
    MultiAssetResult,
    PortfolioRequest,
)
from ohlcv import compute_ohlcv_quality
from reporting import build_html_report, build_markdown_report
from services.analysis_service import analyze_single_asset, fetch_single_asset
from services.multi_asset_service import analyze_portfolio, compare_assets

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_ohlcv(
    dates: pd.DatetimeIndex,
    closes: np.ndarray,
    *,
    asset_type: str = "etf",
    symbol: str = "510300",
    adjust: str = "hfq",
    currency: str | None = "CNY",
    volume: float = 10_000.0,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Open": closes * 0.997,
            "High": closes * 1.006,
            "Low": closes * 0.993,
            "Close": closes.astype(float),
            "Volume": np.full(len(dates), volume, dtype=float),
        },
        index=dates.as_unit("ns"),
    )
    df.index.name = "Date"
    df.attrs["asset_type"] = asset_type
    df.attrs["symbol"] = symbol
    df.attrs["canonical_symbol"] = symbol
    df.attrs["adjust"] = adjust
    if currency is not None:
        df.attrs["currency"] = currency
    df.attrs["data_quality"] = compute_ohlcv_quality(df, missing_policy="drop")
    return df


def _multi_asset_loaders() -> tuple[Any, Any]:
    dates = pd.bdate_range("2024-01-02", periods=95)
    t = np.linspace(0.0, 1.0, len(dates))
    frames = {
        "510300": _make_ohlcv(
            dates,
            100.0 + 14.0 * np.sin(5.0 * t) + 12.0 * t,
            asset_type="etf",
            symbol="510300",
            adjust="hfq",
            currency="CNY",
        ),
        "SPY": _make_ohlcv(
            dates,
            400.0 + 35.0 * np.cos(4.0 * t) + 45.0 * t,
            asset_type="us_stock",
            symbol="SPY",
            adjust="qfq",
            currency="USD",
        ),
        "TLT": _make_ohlcv(
            dates,
            95.0 + 5.0 * np.sin(3.5 * t) - 2.0 * t,
            asset_type="bond_cn",
            symbol="TLT",
            adjust="none",
            currency="CNY",
        ),
        "000300": _make_ohlcv(
            dates,
            3500.0 + 180.0 * np.sin(3.0 * t) + 220.0 * t,
            asset_type="global_index",
            symbol="000300",
            adjust="none",
            currency="CNY",
            volume=0.0,
        ),
    }

    def loader(asset_type: str, symbol: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return frames[symbol].copy()

    def fx_loader(pair: str, start_date: str, end_date: str) -> pd.DataFrame:
        return _make_ohlcv(
            dates,
            np.linspace(7.15, 7.22, len(dates)),
            asset_type="fx",
            symbol=pair,
            adjust="none",
            currency=pair[3:],
            volume=0.0,
        )

    return loader, fx_loader


def test_quant_cli_emits_relative_posix_paths_and_verified_sha256_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader, fx_loader = _multi_asset_loaders()

    for cmd_name, argv in (
        (
            "fetch",
            [
                "fetch",
                "etf:510300",
                "--start",
                "2024-01-02",
                "--end",
                "2024-05-15",
                "-o",
                str(tmp_path / "out_fetch"),
                "--csv",
            ],
        ),
        (
            "analyze",
            [
                "analyze",
                "etf:510300",
                "--start",
                "2024-01-02",
                "--end",
                "2024-05-15",
                "-o",
                str(tmp_path / "out_analyze"),
                "--report",
                "md,html",
                "--charts",
                "html,png",
            ],
        ),
        (
            "compare",
            [
                "compare",
                "etf:510300",
                "us_stock:SPY",
                "--start",
                "2024-01-02",
                "--end",
                "2024-05-15",
                "-o",
                str(tmp_path / "out_compare"),
                "--report",
                "md",
                "--charts",
                "png",
            ],
        ),
        (
            "portfolio",
            [
                "portfolio",
                "etf:510300",
                "us_stock:SPY",
                "--strategy",
                "risk_parity",
                "--validation-mode",
                "train_test",
                "--start",
                "2024-01-02",
                "--end",
                "2024-05-15",
                "-o",
                str(tmp_path / "out_portfolio"),
                "--report",
                "html",
                "--charts",
                "html",
            ],
        ),
    ):
        out_dir = tmp_path / f"out_{cmd_name}"
        code, _ = quant_cli.run_cli_command(argv, loader=loader, fx_loader=fx_loader)
        capsys.readouterr()
        assert code == quant_cli.EXIT_OK

        payload = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
        assert payload["schema_version"] == SCHEMA_VERSION
        artifacts = payload["artifacts"]
        assert artifacts["output_dir"] == "."
        assert artifacts["result_json"] == "result.json"
        for k, val in artifacts.items():
            assert "\\" not in val, f"{cmd_name} artifact {k} contains backslash: {val!r}"
            assert not Path(val).is_absolute(), f"{cmd_name} artifact {k} is absolute: {val!r}"
            if k != "output_dir":
                assert (out_dir / val).is_file()

        manifest = payload["artifact_manifest"]
        assert manifest["algorithm"] == "sha256"
        verify_artifact_manifest(out_dir, artifacts, manifest)
        reloaded, record = load_result_with_migration_info(out_dir)
        assert isinstance(record, SchemaMigrationRecord)
        assert not record.applied
        assert reloaded.schema_version == SCHEMA_VERSION


def test_copied_and_relocated_bundle_ignores_deleted_source_and_replays_offline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader, fx_loader = _multi_asset_loaders()
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "moved" / "run_b"

    code, _ = quant_cli.run_cli_command(
        [
            "compare",
            "etf:510300",
            "us_stock:SPY",
            "--benchmark",
            "global_index:000300",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-15",
            "-o",
            str(dir_a),
            "--csv",
        ],
        loader=loader,
        fx_loader=fx_loader,
    )
    capsys.readouterr()
    assert code == quant_cli.EXIT_OK

    shutil.copytree(dir_a, dir_b)
    shutil.rmtree(dir_a)

    reloaded = load_result_from_artifacts(dir_b)
    assert isinstance(reloaded, MultiAssetResult)
    assert reloaded.data_frame is not None
    assert "prices" in reloaded.frames
    md_text = build_markdown_report(reloaded)
    html_text = build_html_report(reloaded)
    assert "510300" in md_text and "<!DOCTYPE html>" in html_text


def test_legacy_v010_migration_rules_by_schema_and_command(
    tmp_path: Path,
) -> None:
    loader, fx_loader = _multi_asset_loaders()

    # 1. Compare v0.1.0 -> v0.2.0 converts non-positive underwater max_drawdown to non-negative depth
    cmp_res = compare_assets(
        CompareRequest(
            assets=(
                AssetSpec("etf", "510300", adjust="hfq"),
                AssetSpec("us_stock", "SPY", adjust="qfq"),
            ),
            benchmark=AssetSpec("global_index", "000300"),
            start_date="2024-01-02",
            end_date="2024-05-15",
        ),
        loader=loader,
        fx_loader=fx_loader,
    )
    expected_etf_dd = cmp_res.metrics["per_asset"]["etf:510300"]["max_drawdown"]
    assert expected_etf_dd > 0.0

    legacy_dir = tmp_path / "legacy_cmp_copy"
    legacy_dir.mkdir()
    cmp_res.data_frame.to_parquet(legacy_dir / "data.parquet", engine="pyarrow", index=True)  # type: ignore[union-attr]
    for name, frame in cmp_res.frames.items():
        frame.to_parquet(legacy_dir / f"{name}.parquet", engine="pyarrow", index=True)

    # Also write mutated data in an unrelated old path that legacy absolute paths point to
    mutated_old_dir = tmp_path / "mutated_old_machine"
    mutated_old_dir.mkdir()
    mutated_df = cmp_res.data_frame.copy() * 999.0  # type: ignore[union-attr]
    mutated_df.to_parquet(mutated_old_dir / "data.parquet", engine="pyarrow", index=True)

    v01_payload = cmp_res.to_dict(include_data=False)
    v01_payload["schema_version"] = "0.1.0"
    v01_payload.pop("artifact_manifest", None)
    for leg_m in v01_payload["metrics"]["per_asset"].values():
        leg_m["max_drawdown"] = -float(leg_m["max_drawdown"])
    v01_payload["benchmark"]["metrics"]["max_drawdown"] = -float(
        v01_payload["benchmark"]["metrics"]["max_drawdown"]
    )
    v01_payload["methodology"].pop("max_drawdown_convention", None)
    v01_payload["artifacts"] = {
        "output_dir": str(mutated_old_dir),
        "result_json": str(mutated_old_dir / "result.json"),
        "data_parquet": str(mutated_old_dir / "data.parquet"),
        "prices_parquet": r"C:\legacy\win_box\prices.parquet",
    }
    (legacy_dir / "result.json").write_text(json.dumps(v01_payload), encoding="utf-8")

    migrated_res, rec = load_result_with_migration_info(legacy_dir)
    assert isinstance(migrated_res, MultiAssetResult)
    assert rec.applied is True
    assert rec.source_schema_version == "0.1.0"
    assert rec.target_schema_version == "0.2.0"
    assert rec.command == "compare"
    assert migrated_res.schema_version == "0.2.0"
    assert migrated_res.metrics["per_asset"]["etf:510300"]["max_drawdown"] == pytest.approx(
        expected_etf_dd
    )
    assert migrated_res.benchmark is not None
    assert migrated_res.benchmark["metrics"]["max_drawdown"] >= 0.0
    assert migrated_res.methodology["max_drawdown_convention"] == "non_negative_depth"
    assert migrated_res.artifacts["output_dir"] == "."
    assert migrated_res.artifacts["data_parquet"] == "data.parquet"
    assert migrated_res.artifacts["prices_parquet"] == "prices.parquet"
    assert migrated_res.data_frame is not None
    pd.testing.assert_frame_equal(
        migrated_res.data_frame,
        cmp_res.data_frame,
        check_freq=False,  # type: ignore[arg-type]
    )

    # 2. Legacy v0.1.0 compare with unexpected positive max_drawdown must NOT blindly abs()
    bad_cmp = json.loads((legacy_dir / "result.json").read_text(encoding="utf-8"))
    bad_cmp["metrics"]["per_asset"]["etf:510300"]["max_drawdown"] = 0.25
    with pytest.raises(SchemaMigrationError, match="underwater return"):
        migrate_result_payload(bad_cmp)

    # 3. Legacy v0.1.0 analyze & portfolio accept >= 0 and reject < 0
    an_res = analyze_single_asset(
        AnalysisRequest("etf", "510300", "2024-01-02", "2024-05-15", adjust="hfq"),
        loader=loader,
    )
    bad_an = an_res.to_dict(include_data=False)
    bad_an["schema_version"] = "0.1.0"
    bad_an["metrics"]["max_drawdown"] = -0.15
    with pytest.raises(SchemaMigrationError, match="expects max_drawdown >= 0"):
        migrate_result_payload(bad_an)

    port_res = analyze_portfolio(
        PortfolioRequest(
            assets=(
                AssetSpec("etf", "510300", adjust="hfq"),
                AssetSpec("us_stock", "SPY", adjust="qfq"),
            ),
            start_date="2024-01-02",
            end_date="2024-05-15",
            strategy="risk_parity",
            validation_mode="in_sample",
            frontier_points=0,
            include_sensitivity=False,
        ),
        loader=loader,
        fx_loader=fx_loader,
    )
    ok_port = port_res.to_dict(include_data=False)
    ok_port["schema_version"] = "0.1.0"
    migrated_port, port_rec = migrate_result_payload(ok_port)
    assert port_rec.applied is True
    assert migrated_port["methodology"]["max_drawdown_convention"] == "non_negative_depth"

    bad_port = port_res.to_dict(include_data=False)
    bad_port["schema_version"] = "0.1.0"
    bad_port["metrics"]["backtest"]["Portfolio"]["max_drawdown"] = -0.12
    with pytest.raises(SchemaMigrationError, match="expects max_drawdown >= 0"):
        migrate_result_payload(bad_port)


def test_security_checks_block_traversal_absolute_v02_and_symlinks(tmp_path: Path) -> None:
    loader, _ = _multi_asset_loaders()
    res = analyze_single_asset(
        AnalysisRequest("etf", "510300", "2024-01-02", "2024-05-15", adjust="hfq"),
        loader=loader,
    )
    bundle_dir = tmp_path / "secure_bundle"
    save_result_bundle(res, bundle_dir)

    outside_file = tmp_path / "secret.parquet"
    res.data_frame.to_parquet(outside_file, engine="pyarrow", index=True)  # type: ignore[union-attr]

    # Direct resolver traversal & absolute path checks
    with pytest.raises(ArtifactSecurityError):
        resolve_safe_artifact_path(bundle_dir, "../secret.parquet", allow_legacy_absolute=False)
    with pytest.raises(ArtifactSecurityError):
        resolve_safe_artifact_path(bundle_dir, "../secret.parquet", allow_legacy_absolute=True)
    with pytest.raises(ArtifactSecurityError):
        resolve_safe_artifact_path(
            bundle_dir, r"C:\old\..\secret.parquet", allow_legacy_absolute=True
        )
    with pytest.raises(ArtifactSecurityError):
        resolve_safe_artifact_path(bundle_dir, "/tmp/data.parquet", allow_legacy_absolute=False)
    with pytest.raises(ArtifactSecurityError):
        resolve_safe_artifact_path(bundle_dir, r"C:\tmp\data.parquet", allow_legacy_absolute=False)

    # Legacy Windows charts absolute path maps safely inside bundle_dir/charts
    mapped_chart = resolve_safe_artifact_path(
        bundle_dir, r"C:\old\run\charts\price.png", allow_legacy_absolute=True
    )
    assert mapped_chart == (bundle_dir.resolve() / "charts" / "price.png")

    orig_json_bytes = (bundle_dir / "result.json").read_text(encoding="utf-8")

    # 1. Path traversal in artifacts
    tampered = json.loads(orig_json_bytes)
    tampered["artifacts"]["data_parquet"] = "../secret.parquet"
    (bundle_dir / "result.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ArtifactSecurityError):
        load_result_from_artifacts(bundle_dir)

    # 2. Absolute path in v0.2.0 artifacts
    tampered = json.loads(orig_json_bytes)
    tampered["artifacts"]["data_parquet"] = str(outside_file.resolve())
    (bundle_dir / "result.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ArtifactSecurityError):
        load_result_from_artifacts(bundle_dir)

    # 3. Tampered output_dir in v0.2.0
    tampered = json.loads(orig_json_bytes)
    tampered["artifacts"]["output_dir"] = "../other"
    (bundle_dir / "result.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ArtifactSecurityError):
        load_result_from_artifacts(bundle_dir)

    # 4. Symlink escape on data.parquet
    (bundle_dir / "result.json").write_text(orig_json_bytes, encoding="utf-8")
    (bundle_dir / "data.parquet").unlink()
    os.symlink(outside_file, bundle_dir / "data.parquet")
    with pytest.raises(ArtifactSecurityError):
        load_result_from_artifacts(bundle_dir)

    # 5. Symlink escape on result.json
    outside_result = tmp_path / "outside_result.json"
    outside_result.write_text(orig_json_bytes, encoding="utf-8")
    sym_dir = tmp_path / "sym_result_dir"
    sym_dir.mkdir()
    os.symlink(outside_result, sym_dir / "result.json")
    with pytest.raises(ArtifactSecurityError):
        load_result_from_artifacts(sym_dir)

    # 6. Symlinked charts/ subdirectory pointing outside output directory
    outside_charts = tmp_path / "outside_charts"
    outside_charts.mkdir()
    (outside_charts / "price.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    sym_charts_dir = tmp_path / "sym_charts_bundle"
    save_result_bundle(res, sym_charts_dir)
    os.symlink(outside_charts, sym_charts_dir / "charts")
    payload_with_chart = json.loads((sym_charts_dir / "result.json").read_text(encoding="utf-8"))
    payload_with_chart["artifacts"]["chart_price_png"] = "charts/price.png"
    payload_with_chart.pop("artifact_manifest", None)
    (sym_charts_dir / "result.json").write_text(json.dumps(payload_with_chart), encoding="utf-8")
    with pytest.raises(ArtifactSecurityError):
        load_result_from_artifacts(sym_charts_dir)


def test_integrity_and_schema_version_errors(tmp_path: Path) -> None:
    loader, _ = _multi_asset_loaders()
    res = analyze_single_asset(
        AnalysisRequest("etf", "510300", "2024-01-02", "2024-05-15", adjust="hfq"),
        loader=loader,
    )
    bundle_dir = tmp_path / "integrity_bundle"
    save_result_bundle(res, bundle_dir)
    orig_json_bytes = (bundle_dir / "result.json").read_text(encoding="utf-8")
    orig_pq_bytes = (bundle_dir / "data.parquet").read_bytes()

    # Missing result.json -> FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_result_from_artifacts(tmp_path / "does_not_exist")

    # 1. Corrupted / tampered data.parquet -> SHA-256 mismatch
    (bundle_dir / "data.parquet").write_bytes(orig_pq_bytes + b"corrupt")
    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        load_result_from_artifacts(bundle_dir)
    # Even with verify_integrity=False, corrupted Parquet fails clearly
    with pytest.raises(ArtifactIntegrityError, match="corrupted Parquet"):
        load_result_from_artifacts(bundle_dir, verify_integrity=False)

    # Restore valid parquet
    (bundle_dir / "data.parquet").write_bytes(orig_pq_bytes)

    # 2. Path substitution inside output_dir
    shutil.copy2(bundle_dir / "data.parquet", bundle_dir / "alt.parquet")
    tampered = json.loads(orig_json_bytes)
    tampered["artifacts"]["data_parquet"] = "alt.parquet"
    (bundle_dir / "result.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="Path substitution"):
        load_result_from_artifacts(bundle_dir)

    # 3. Manifest entry omitted
    tampered = json.loads(orig_json_bytes)
    tampered["artifact_manifest"]["files"].pop("data_parquet")
    (bundle_dir / "result.json").write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="missing from artifact_manifest"):
        load_result_from_artifacts(bundle_dir)

    # 4. Missing declared data.parquet on disk
    (bundle_dir / "result.json").write_text(orig_json_bytes, encoding="utf-8")
    (bundle_dir / "data.parquet").unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing on disk"):
        load_result_from_artifacts(bundle_dir)
    (bundle_dir / "data.parquet").write_bytes(orig_pq_bytes)

    # 5. Unsupported or missing schema_version
    for bad_ver in ("0.0.1", "0.3.0", "2.0.0", "invalid", None):
        bad_payload = json.loads(orig_json_bytes)
        if bad_ver is None:
            bad_payload.pop("schema_version", None)
        else:
            bad_payload["schema_version"] = bad_ver
        (bundle_dir / "result.json").write_text(json.dumps(bad_payload), encoding="utf-8")
        with pytest.raises(UnsupportedSchemaVersionError):
            load_result_from_artifacts(bundle_dir)

    # 6. Corrupted JSON syntax
    (bundle_dir / "result.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="Failed parsing"):
        load_result_from_artifacts(bundle_dir)


def test_v020_forbids_undeclared_fallback_and_requires_manifest_while_v010_migrates(
    tmp_path: Path,
) -> None:
    from artifacts.store import _load_declared_or_fallback_frame

    loader, fx_loader = _multi_asset_loaders()
    an_res = analyze_single_asset(
        AnalysisRequest("etf", "510300", "2024-01-02", "2024-05-15", adjust="hfq"),
        loader=loader,
    )
    cmp_res = compare_assets(
        CompareRequest(
            assets=(
                AssetSpec("etf", "510300", adjust="hfq"),
                AssetSpec("us_stock", "SPY", adjust="qfq"),
            ),
            benchmark=AssetSpec("global_index", "000300"),
            start_date="2024-01-02",
            end_date="2024-05-15",
        ),
        loader=loader,
        fx_loader=fx_loader,
    )

    # 1. Missing artifact_manifest on v0.2.0 must fail integrity verification (no silent bypass)
    no_manifest_dir = tmp_path / "v02_missing_manifest"
    save_result_bundle(an_res, no_manifest_dir)
    orig_bytes = (no_manifest_dir / "data.parquet").read_bytes()
    # Even a syntactically valid substituted Parquet cannot bypass verification when manifest is removed
    (an_res.data_frame * 2.0).to_parquet(  # type: ignore[operator]
        no_manifest_dir / "data.parquet", engine="pyarrow", index=True
    )
    no_manifest_payload = json.loads((no_manifest_dir / "result.json").read_text(encoding="utf-8"))
    no_manifest_payload.pop("artifact_manifest", None)
    (no_manifest_dir / "result.json").write_text(json.dumps(no_manifest_payload), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="artifact_manifest"):
        load_result_from_artifacts(no_manifest_dir)

    # Explicit verify_integrity=False skips missing-manifest check for valid declared files
    (no_manifest_dir / "data.parquet").write_bytes(orig_bytes)
    loaded_noverify = load_result_from_artifacts(no_manifest_dir, verify_integrity=False)
    assert loaded_noverify.data_frame is not None

    # Even with valid data.parquet and empty artifacts, missing artifact_manifest in v0.2.0 fails
    no_manifest_payload["artifacts"] = {"output_dir": ".", "result_json": "result.json"}
    (no_manifest_dir / "result.json").write_text(json.dumps(no_manifest_payload), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="artifact_manifest"):
        load_result_from_artifacts(no_manifest_dir)

    # Non-mapping artifact_manifest fails regardless of verify_integrity
    bad_manifest_payload = dict(no_manifest_payload, artifact_manifest="not-a-mapping")
    (no_manifest_dir / "result.json").write_text(json.dumps(bad_manifest_payload), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="artifact_manifest"):
        load_result_from_artifacts(no_manifest_dir, verify_integrity=False)

    # 2. v0.2.0 must NEVER load undeclared leftover Parquet/CSV files on disk
    undeclared_dir = tmp_path / "v02_undeclared_leftover"
    save_result_bundle(an_res, undeclared_dir, write_csv=True)
    # Remove data_parquet/data_csv declarations while leaving data.parquet & data.csv on disk
    undeclared_payload = json.loads((undeclared_dir / "result.json").read_text(encoding="utf-8"))
    undeclared_payload["artifacts"] = {"output_dir": ".", "result_json": "result.json"}
    undeclared_payload["artifact_manifest"] = {"algorithm": "sha256", "files": {}}
    (undeclared_dir / "result.json").write_text(json.dumps(undeclared_payload), encoding="utf-8")
    assert (undeclared_dir / "data.parquet").is_file()
    assert (undeclared_dir / "data.csv").is_file()
    assert (
        _load_declared_or_fallback_frame(
            undeclared_dir,
            {},
            parquet_key="data_parquet",
            csv_key="data_csv",
            default_parquet="data.parquet",
            default_csv="data.csv",
        )
        is None
    )
    loaded_undeclared = load_result_from_artifacts(undeclared_dir)
    assert loaded_undeclared.data_frame is None
    loaded_undeclared_noverify = load_result_from_artifacts(undeclared_dir, verify_integrity=False)
    assert loaded_undeclared_noverify.data_frame is None

    # 3. v0.2.0 compare bundle ignores leftover portfolio Parquet/CSV files in the same directory
    cmp_dir = tmp_path / "v02_compare_with_stale_portfolio"
    save_result_bundle(cmp_res, cmp_dir)
    # Write leftover undeclared frames & CSV into cmp_dir (simulating prior portfolio artifacts)
    pd.DataFrame({"Asset": ["etf:510300"], "Weight": [0.5]}).to_parquet(
        cmp_dir / "weights.parquet", engine="pyarrow", index=False
    )
    pd.DataFrame({"Volatility": [0.1], "Return": [0.05]}).to_parquet(
        cmp_dir / "frontier.parquet", engine="pyarrow", index=False
    )
    pd.DataFrame({"Portfolio": [1.0, 1.01]}).to_parquet(
        cmp_dir / "backtest.parquet", engine="pyarrow", index=False
    )
    (cmp_dir / "data.csv").write_text("corrupted,csv\n1,2\n", encoding="utf-8")

    reloaded_cmp = load_result_from_artifacts(cmp_dir)
    assert isinstance(reloaded_cmp, MultiAssetResult)
    assert "weights" not in reloaded_cmp.frames
    assert "frontier" not in reloaded_cmp.frames
    assert "backtest" not in reloaded_cmp.frames
    assert set(reloaded_cmp.frames.keys()) == set(cmp_res.frames.keys())

    # Declared data_parquet missing with verify_integrity=False must NOT fall back to undeclared data.csv
    (cmp_dir / "data.parquet").unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing on disk"):
        load_result_from_artifacts(cmp_dir, verify_integrity=False)

    # 4. v0.1.0 retains safe migration compatibility (manifest optional + local directory fallback)
    v01_dir = tmp_path / "v01_undeclared_fallback"
    v01_dir.mkdir()
    an_res.data_frame.to_parquet(v01_dir / "data.parquet", engine="pyarrow", index=True)  # type: ignore[union-attr]
    v01_payload = an_res.to_dict(include_data=False)
    v01_payload["schema_version"] = "0.1.0"
    v01_payload.pop("artifact_manifest", None)
    v01_payload["artifacts"] = {
        "output_dir": "/legacy/old_machine/run",
        "result_json": "/legacy/old_machine/run/result.json",
    }
    (v01_dir / "result.json").write_text(json.dumps(v01_payload), encoding="utf-8")

    migrated_v01, rec_v01 = load_result_with_migration_info(v01_dir)
    assert rec_v01.applied is True
    assert rec_v01.source_schema_version == "0.1.0"
    assert rec_v01.target_schema_version == "0.2.0"
    assert migrated_v01.data_frame is not None
    assert len(migrated_v01.data_frame) == len(an_res.data_frame)  # type: ignore[arg-type]

    # v0.1.0 fallback still enforces symlink escape checks
    outside_pq = tmp_path / "outside_v01.parquet"
    an_res.data_frame.to_parquet(outside_pq, engine="pyarrow", index=True)  # type: ignore[union-attr]
    (v01_dir / "data.parquet").unlink()
    os.symlink(outside_pq, v01_dir / "data.parquet")
    with pytest.raises(ArtifactSecurityError):
        load_result_from_artifacts(v01_dir)


def test_encode_decode_encode_semantic_roundtrip_and_offline_reports(tmp_path: Path) -> None:
    loader, fx_loader = _multi_asset_loaders()

    fetch_res = fetch_single_asset(
        AnalysisRequest("etf", "510300", "2024-01-02", "2024-05-15", adjust="hfq"),
        loader=loader,
    )
    an_res = analyze_single_asset(
        AnalysisRequest("etf", "510300", "2024-01-02", "2024-05-15", adjust="hfq"),
        loader=loader,
    )
    cmp_res = compare_assets(
        CompareRequest(
            assets=(
                AssetSpec("etf", "510300", adjust="hfq"),
                AssetSpec("us_stock", "SPY", adjust="qfq"),
            ),
            benchmark=AssetSpec("global_index", "000300"),
            start_date="2024-01-02",
            end_date="2024-05-15",
        ),
        loader=loader,
        fx_loader=fx_loader,
    )
    port_res = analyze_portfolio(
        PortfolioRequest(
            assets=(
                AssetSpec("etf", "510300", adjust="hfq"),
                AssetSpec("us_stock", "SPY", adjust="qfq"),
            ),
            benchmark=AssetSpec("global_index", "000300"),
            start_date="2024-01-02",
            end_date="2024-05-15",
            strategy="risk_parity",
            validation_mode="train_test",
            frontier_points=6,
            num_random_portfolios=20,
            include_sensitivity=True,
        ),
        loader=loader,
        fx_loader=fx_loader,
    )

    for label, original in (
        ("fetch", fetch_res),
        ("analyze", an_res),
        ("compare", cmp_res),
        ("portfolio", port_res),
    ):
        dir1 = tmp_path / f"{label}_1"
        dir2 = tmp_path / f"{label}_2"

        saved1 = save_result_bundle(original, dir1, write_csv=True)
        decoded = load_result_from_artifacts(dir1)
        saved2 = save_result_bundle(decoded, dir2, write_csv=True)

        assert json.loads((dir1 / "result.json").read_text(encoding="utf-8")) == json.loads(
            (dir2 / "result.json").read_text(encoding="utf-8")
        )
        assert saved1.artifact_manifest == saved2.artifact_manifest
        assert decoded.data_frame is not None and saved1.data_frame is not None
        pd.testing.assert_frame_equal(decoded.data_frame, saved1.data_frame, check_freq=False)

        if label != "fetch":
            assert build_markdown_report(saved1) == build_markdown_report(decoded)
            assert build_html_report(saved1) == build_html_report(decoded)

    # Legacy v0.1.0 compare -> migrate -> save v0.2.0 -> decode -> save fixed-point roundtrip
    v01_dir = tmp_path / "cmp_v01_seed"
    v02_a = tmp_path / "cmp_v02_a"
    v02_b = tmp_path / "cmp_v02_b"
    v01_dir.mkdir()
    cmp_res.data_frame.to_parquet(v01_dir / "data.parquet", engine="pyarrow", index=True)  # type: ignore[union-attr]
    for name, frame in cmp_res.frames.items():
        frame.to_parquet(v01_dir / f"{name}.parquet", engine="pyarrow", index=True)
    legacy_payload = cmp_res.to_dict(include_data=False)
    legacy_payload["schema_version"] = "0.1.0"
    legacy_payload.pop("artifact_manifest", None)
    for leg_m in legacy_payload["metrics"]["per_asset"].values():
        leg_m["max_drawdown"] = -float(leg_m["max_drawdown"])
    if legacy_payload.get("benchmark"):
        legacy_payload["benchmark"]["metrics"]["max_drawdown"] = -float(
            legacy_payload["benchmark"]["metrics"]["max_drawdown"]
        )
    legacy_payload["methodology"].pop("max_drawdown_convention", None)
    legacy_payload["artifacts"] = {
        "output_dir": "/legacy/source/dir",
        "result_json": "/legacy/source/dir/result.json",
        "data_parquet": "/legacy/source/dir/data.parquet",
        **{f"{name}_parquet": f"/legacy/source/dir/{name}.parquet" for name in cmp_res.frames},
    }
    (v01_dir / "result.json").write_text(json.dumps(legacy_payload), encoding="utf-8")

    migrated_decoded, rec = load_result_with_migration_info(v01_dir)
    assert rec.applied is True
    v02_saved_a = save_result_bundle(migrated_decoded, v02_a)
    v02_reloaded = load_result_from_artifacts(v02_a)
    v02_saved_b = save_result_bundle(v02_reloaded, v02_b)
    assert json.loads((v02_a / "result.json").read_text(encoding="utf-8")) == json.loads(
        (v02_b / "result.json").read_text(encoding="utf-8")
    )
    assert v02_saved_a.artifact_manifest == v02_saved_b.artifact_manifest


def test_quant_cli_stale_cleanup_ignores_traversal_and_external_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader, _ = _multi_asset_loaders()
    outside_sentinel = tmp_path / "do_not_delete.txt"
    outside_sentinel.write_text("keep me", encoding="utf-8")

    out_dir = tmp_path / "cli_safe_cleanup"
    out_dir.mkdir()
    prior_payload = {
        "schema_version": "0.1.0",
        "command": "analyze",
        "artifacts": {
            "output_dir": str(out_dir),
            "result_json": str(out_dir / "result.json"),
            "report_md": "../do_not_delete.txt",
            "report_html": str(outside_sentinel),
        },
    }
    (out_dir / "result.json").write_text(json.dumps(prior_payload), encoding="utf-8")

    code, _ = quant_cli.run_cli_command(
        [
            "analyze",
            "etf:510300",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-15",
            "-o",
            str(out_dir),
        ],
        loader=loader,
    )
    capsys.readouterr()
    assert code == quant_cli.EXIT_OK
    assert outside_sentinel.read_text(encoding="utf-8") == "keep me"


def test_quant_cli_stale_cleanup_never_follows_symlinks_outside_output_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loader, _ = _multi_asset_loaders()
    outside_sentinel = tmp_path / "outside_sentinel.md"
    outside_sentinel.write_text("external report sentinel", encoding="utf-8")

    outside_charts = tmp_path / "outside_charts"
    outside_charts.mkdir()
    outside_chart_price = outside_charts / "price.png"
    outside_chart_corr = outside_charts / "correlation.png"
    outside_chart_price.write_bytes(b"external-price-png")
    outside_chart_corr.write_bytes(b"external-corr-png")

    outside_result = tmp_path / "outside_result.json"
    outside_result.write_text(
        json.dumps(
            {
                "schema_version": "0.2.0",
                "command": "analyze",
                "artifacts": {"report_md": "report.md", "chart_price_png": "charts/price.png"},
            }
        ),
        encoding="utf-8",
    )

    # 1. Direct _collect_stale_managed_artifacts check with report.md symlink & charts/ symlink
    out_dir = tmp_path / "cli_symlink_cleanup"
    out_dir.mkdir()
    os.symlink(outside_sentinel, out_dir / "report.md")
    os.symlink(outside_charts, out_dir / "charts")
    stale_regular = out_dir / "weights.parquet"
    stale_regular.write_bytes(b"stale-regular")

    new_targets = {
        (out_dir / "data.parquet").resolve(),
        (out_dir / "result.json").resolve(),
    }
    collected = quant_cli._collect_stale_managed_artifacts(out_dir, new_target_resolved=new_targets)
    assert stale_regular.resolve() in collected
    assert outside_sentinel.resolve() not in collected
    assert outside_chart_price.resolve() not in collected
    assert outside_chart_corr.resolve() not in collected
    for item in collected:
        assert item.is_relative_to(out_dir.resolve())
        assert not item.is_symlink()

    # 2. End-to-end CLI cleanup preserves external targets while purging genuine stale files
    code, _ = quant_cli.run_cli_command(
        [
            "analyze",
            "etf:510300",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-15",
            "-o",
            str(out_dir),
        ],
        loader=loader,
    )
    capsys.readouterr()
    assert code == quant_cli.EXIT_OK
    assert not stale_regular.exists()
    assert outside_sentinel.read_text(encoding="utf-8") == "external report sentinel"
    assert outside_chart_price.read_bytes() == b"external-price-png"
    assert outside_chart_corr.read_bytes() == b"external-corr-png"

    # 3. Also guard symlinked chart files inside a real charts/ dir and symlinked result.json
    real_charts_out = tmp_path / "cli_nested_chart_symlink"
    (real_charts_out / "charts").mkdir(parents=True)
    os.symlink(outside_chart_price, real_charts_out / "charts" / "price.png")
    os.symlink(outside_result, real_charts_out / "result.json")
    nested_collected = quant_cli._collect_stale_managed_artifacts(
        real_charts_out,
        new_target_resolved={
            (real_charts_out / "data.parquet").resolve(),
            (real_charts_out / "result.json").resolve(),
        },
    )
    assert nested_collected == []

    code2, _ = quant_cli.run_cli_command(
        [
            "analyze",
            "etf:510300",
            "--start",
            "2024-01-02",
            "--end",
            "2024-05-15",
            "-o",
            str(real_charts_out),
        ],
        loader=loader,
    )
    capsys.readouterr()
    assert code2 == quant_cli.EXIT_OK
    assert outside_chart_price.read_bytes() == b"external-price-png"
    assert outside_result.exists()


def test_load_artifacts_validates_raw_frame_attrs_before_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import artifacts.store as store_mod

    loader, _ = _multi_asset_loaders()
    an_res = analyze_single_asset(
        AnalysisRequest("etf", "510300", "2024-01-02", "2024-05-15", adjust="hfq"),
        loader=loader,
    )
    bundle_dir = tmp_path / "attrs_mismatch_bundle"
    save_result_bundle(an_res, bundle_dir)

    real_reader = store_mod._read_frame_from_disk

    def _reader_with_conflicting_attrs(
        parquet_path: Path | None, csv_path: Path | None = None
    ) -> pd.DataFrame | None:
        loaded_df = real_reader(parquet_path, csv_path)
        assert loaded_df is not None
        loaded_df.attrs["currency"] = "USD"
        loaded_df.attrs["asset_type"] = "us_stock"
        return loaded_df

    monkeypatch.setattr(store_mod, "_read_frame_from_disk", _reader_with_conflicting_attrs)
    with pytest.raises(ValueError, match="不一致"):
        load_result_from_artifacts(bundle_dir)


def test_artifacts_package_dependency_boundaries() -> None:
    forbidden = {
        "services",
        "quant_cli",
        "app",
        "main",
        "reporting",
        "data_sources",
        "data_fetcher",
        "asset_loader",
        "streamlit",
        "matplotlib",
        "plotly",
    }
    for py_file in (REPO_ROOT / "artifacts").glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden, (
                        f"{py_file.name} imports forbidden module {alias.name!r}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, (
                    f"{py_file.name} imports forbidden module {node.module!r}"
                )
