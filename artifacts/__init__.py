"""Portable research artifact serialization, SHA-256 integrity checking, and schema migration."""

from __future__ import annotations

from artifacts.migration import (
    ArtifactError,
    SchemaMigrationError,
    SchemaMigrationRecord,
    UnsupportedSchemaVersionError,
    migrate_result_payload,
    parse_schema_version,
)
from artifacts.store import (
    ArtifactFileDigest,
    ArtifactIntegrityError,
    ArtifactManifest,
    ArtifactSecurityError,
    build_artifact_manifest,
    compute_file_sha256,
    load_result_from_artifacts,
    load_result_with_migration_info,
    prepare_export_frame,
    resolve_safe_artifact_path,
    save_result_bundle,
    verify_artifact_manifest,
)

__all__ = [
    "ArtifactError",
    "ArtifactFileDigest",
    "ArtifactIntegrityError",
    "ArtifactManifest",
    "ArtifactSecurityError",
    "SchemaMigrationError",
    "SchemaMigrationRecord",
    "UnsupportedSchemaVersionError",
    "build_artifact_manifest",
    "compute_file_sha256",
    "load_result_from_artifacts",
    "load_result_with_migration_info",
    "migrate_result_payload",
    "parse_schema_version",
    "prepare_export_frame",
    "resolve_safe_artifact_path",
    "save_result_bundle",
    "verify_artifact_manifest",
]
