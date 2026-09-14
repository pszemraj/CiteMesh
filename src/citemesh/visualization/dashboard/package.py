"""On-disk dashboard package format: filenames, validation, and persistence.

Owns the ``dashboard.citemesh.json`` collection package — its filenames and
lock, the staged-artifact commit protocol, schema validation for graph payloads
and result entries, legacy manifest migration, and the viewer snapshot rendered
from the latest collection state.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
from filelock import FileLock, Timeout

from citemesh.data.cache import atomic_write_json, path_exists

from .contracts import (
    DASHBOARD_COLLECTION_KIND,
    DASHBOARD_COLLECTION_SCHEMA_VERSION,
    GRAPH_PAYLOAD_KIND,
    GRAPH_PAYLOAD_SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)


DASHBOARD_PACKAGE_LOCK_TIMEOUT_SECONDS = 60.0

DASHBOARD_COLLECTION_FILENAME = "dashboard.html"
DASHBOARD_PACKAGE_FILENAME = "dashboard.citemesh.json"
LEGACY_DASHBOARD_MANIFEST_FILENAME = "dashboard.manifest.json"


class DashboardPackageError(ValueError):
    """Raised when an existing dashboard package violates its format contract."""


def _dashboard_package_lock_path(package_path: Path) -> Path:
    """Return a shared lock beside the resolved dashboard package.

    Writers must coordinate even when their cache roots differ.

    :param Path package_path: Dashboard package path being coordinated.
    :return Path: Hidden lock path in the package directory.
    """
    resolved_path = package_path.expanduser().resolve()
    return resolved_path.with_name(f".{resolved_path.name}.lock")


def _commit_staged_dashboard_artifacts(
    package_path: Path,
    package: dict[str, Any],
    *,
    staged_result_files: dict[Path, Path],
    obsolete_result_paths: set[Path],
) -> None:
    """Publish one collection result and restore prior files on failure.

    The package is the final commit marker. Result files are backed up before
    promotion so an ordinary export or filesystem exception restores the prior
    completed bundle. Unknown files and other strategy basenames are untouched.

    :param Path package_path: Authoritative collection package destination.
    :param Dict[str, Any] package: Validated package payload to publish last.
    :param Dict[Path, Path] staged_result_files: Final paths mapped to complete
        same-filesystem staging files.
    :param Set[Path] obsolete_result_paths: Exact known-format files omitted by
        the new result and removed on successful publication.
    :return None: Publishes the complete result bundle and package.
    """
    obsolete_paths = set(obsolete_result_paths) - set(staged_result_files)
    target_paths = sorted(
        set(staged_result_files) | obsolete_paths,
        key=lambda path: str(path),
    )
    promoted_paths: set[Path] = set()
    with tempfile.TemporaryDirectory(
        prefix=".citemesh-dashboard-backup-",
        dir=package_path.parent,
        ignore_cleanup_errors=True,
    ) as backup_name:
        backup_dir = Path(backup_name)
        backups: dict[Path, Path] = {}
        try:
            for index, target_path in enumerate(target_paths):
                if not path_exists(target_path):
                    continue
                backup_path = backup_dir / f"{index}-{target_path.name}"
                try:
                    os.link(target_path, backup_path)
                except OSError:
                    target_path.replace(backup_path)
                backups[target_path] = backup_path

            for target_path, staged_path in sorted(
                staged_result_files.items(), key=lambda item: str(item[0])
            ):
                backup_path = backups.get(target_path)
                if backup_path is not None:
                    staged_path.chmod(backup_path.stat().st_mode & 0o7777)
                staged_path.replace(target_path)
                promoted_paths.add(target_path)

            for obsolete_path in sorted(obsolete_paths, key=str):
                obsolete_path.unlink(missing_ok=True)

            atomic_write_json(package_path, package, indent=2)
        except BaseException:
            for target_path in reversed(target_paths):
                backup_path = backups.get(target_path)
                if backup_path is not None:
                    backup_path.replace(target_path)
                elif target_path in promoted_paths:
                    target_path.unlink(missing_ok=True)
            raise


def _load_package_object(path: Path, *, label: str) -> dict[str, Any]:
    """Read one UTF-8 JSON object with a context-rich package error.

    :param Path path: JSON file to read.
    :param str label: Human-readable artifact label for errors.
    :return Dict[str, Any]: Parsed JSON object.
    :raises DashboardPackageError: If the file is unreadable, malformed, or not an object.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise DashboardPackageError(f"Could not read {label} at {path}: {exc}") from exc
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise DashboardPackageError(f"Malformed {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DashboardPackageError(
            f"{label.capitalize()} at {path} must be a JSON object."
        )
    return payload


def _validated_non_negative_count(raw: object, *, field: str) -> int:
    """Validate a non-negative integer package summary count.

    :param object raw: Candidate count value.
    :param str field: Field label for validation errors.
    :return int: Validated count.
    :raises DashboardPackageError: If ``raw`` is not a non-negative integer.
    """
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise DashboardPackageError(
            f"Dashboard package {field} must be a non-negative integer."
        )
    return raw


def _validated_dashboard_token(raw: object, *, field: str) -> str:
    """Validate a non-empty dashboard identity token without hidden whitespace.

    :param object raw: Candidate seed, strategy, node, or edge token.
    :param str field: Field label for validation errors.
    :return str: Canonical token.
    :raises DashboardPackageError: If the token is empty or padded with whitespace.
    """
    token = str(raw or "")
    if not token or token != token.strip():
        raise DashboardPackageError(
            f"Dashboard package {field} must be a non-empty canonical token."
        )
    return token


def _validate_dashboard_graph_payload(
    raw_payload: object, *, result_id: str
) -> dict[str, Any]:
    """Validate one canonical graph payload embedded in a dashboard package.

    :param object raw_payload: Candidate graph payload.
    :param str result_id: Owning result identifier for contextual errors.
    :return Dict[str, Any]: Shallow normalized graph payload copy.
    :raises DashboardPackageError: If the graph payload contract is invalid.
    """
    if not isinstance(raw_payload, dict):
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} payload must be an object."
        )
    if raw_payload.get("kind") != GRAPH_PAYLOAD_KIND:
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} has unsupported graph kind."
        )
    graph_schema = raw_payload.get("schema_version")
    if type(graph_schema) is not int or graph_schema != GRAPH_PAYLOAD_SCHEMA_VERSION:
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} has unsupported graph schema "
            f"version {graph_schema!r}."
        )
    seed_id = _validated_dashboard_token(
        raw_payload.get("seed_id"), field=f"result {result_id!r} payload seed_id"
    )
    meta = raw_payload.get("meta")
    summary = raw_payload.get("summary")
    if not isinstance(meta, dict) or not isinstance(summary, dict):
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} has incomplete graph metadata."
        )
    payload_strategy = _validated_dashboard_token(
        meta.get("strategy"), field=f"result {result_id!r} payload strategy"
    )
    nodes = raw_payload.get("nodes")
    edges = raw_payload.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} nodes and edges must be arrays."
        )
    node_count = _validated_non_negative_count(
        summary.get("nodes"), field=f"result {result_id!r} payload summary.nodes"
    )
    edge_count = _validated_non_negative_count(
        summary.get("edges"), field=f"result {result_id!r} payload summary.edges"
    )
    if node_count != len(nodes) or edge_count != len(edges):
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} summary does not match its "
            "node and edge arrays."
        )
    node_ids: list[str] = []
    for node in nodes:
        if not isinstance(node, dict):
            raise DashboardPackageError(
                f"Dashboard package result {result_id!r} nodes must be objects."
            )
        node_id = _validated_dashboard_token(
            node.get("id"), field=f"result {result_id!r} node ID"
        )
        node_ids.append(node_id)
    node_id_set = set(node_ids)
    if len(node_id_set) != len(node_ids) or seed_id not in node_id_set:
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} requires unique node IDs "
            "including its seed."
        )
    for edge in edges:
        if not isinstance(edge, dict):
            raise DashboardPackageError(
                f"Dashboard package result {result_id!r} edges must be objects."
            )
        source_id = _validated_dashboard_token(
            edge.get("source"), field=f"result {result_id!r} edge source"
        )
        target_id = _validated_dashboard_token(
            edge.get("target"), field=f"result {result_id!r} edge target"
        )
        if source_id not in node_id_set or target_id not in node_id_set:
            raise DashboardPackageError(
                f"Dashboard package result {result_id!r} has an edge with an "
                "unknown endpoint."
            )

    dashboard = raw_payload.get("dashboard")
    dashboard_meta = dashboard.get("meta") if isinstance(dashboard, dict) else None
    if not isinstance(dashboard_meta, dict):
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} is missing dashboard geometry."
        )
    dashboard_seed_id = _validated_dashboard_token(
        dashboard_meta.get("seed_id"),
        field=f"result {result_id!r} dashboard seed_id",
    )
    dashboard_strategy = _validated_dashboard_token(
        dashboard_meta.get("strategy"),
        field=f"result {result_id!r} dashboard strategy",
    )
    dashboard_summary = dashboard_meta.get("summary")
    dashboard_node_count = (
        _validated_non_negative_count(
            dashboard_summary.get("nodes"),
            field=f"result {result_id!r} dashboard summary.nodes",
        )
        if isinstance(dashboard_summary, dict)
        else None
    )
    dashboard_edge_count = (
        _validated_non_negative_count(
            dashboard_summary.get("edges"),
            field=f"result {result_id!r} dashboard summary.edges",
        )
        if isinstance(dashboard_summary, dict)
        else None
    )
    if (
        dashboard_seed_id != seed_id
        or dashboard_strategy != payload_strategy
        or dashboard_node_count != node_count
        or dashboard_edge_count != edge_count
    ):
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} has inconsistent dashboard metadata."
        )
    raw_order = dashboard_meta.get("plotly_node_order")
    raw_positions = dashboard_meta.get("plotly_positions")
    raw_sizes = dashboard_meta.get("plotly_node_sizes")
    if not all(
        isinstance(value, list) for value in (raw_order, raw_positions, raw_sizes)
    ):
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} has incomplete dashboard geometry."
        )
    geometry_order = [
        _validated_dashboard_token(
            node_id, field=f"result {result_id!r} geometry node ID"
        )
        for node_id in raw_order
    ]
    if (
        len(geometry_order) != len(node_ids)
        or len(set(geometry_order)) != len(geometry_order)
        or set(geometry_order) != node_id_set
    ):
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} geometry must cover each node exactly once."
        )
    if len(raw_positions) != len(node_ids) or len(raw_sizes) != len(node_ids):
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} geometry arrays must align with its nodes."
        )
    for position in raw_positions:
        if (
            not isinstance(position, list)
            or len(position) != 2
            or any(
                isinstance(coordinate, bool)
                or not isinstance(coordinate, (int, float))
                or not math.isfinite(float(coordinate))
                for coordinate in position
            )
        ):
            raise DashboardPackageError(
                f"Dashboard package result {result_id!r} has an invalid layout position."
            )
    if any(
        isinstance(size, bool)
        or not isinstance(size, (int, float))
        or not math.isfinite(float(size))
        or float(size) <= 0.0
        for size in raw_sizes
    ):
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} has invalid node sizes."
        )
    return dict(raw_payload)


def _validate_dashboard_result_entry(raw_entry: object) -> dict[str, Any]:
    """Validate and normalize one dashboard collection result entry.

    :param object raw_entry: Candidate result entry.
    :return Dict[str, Any]: Normalized result entry.
    :raises DashboardPackageError: If required descriptor or payload fields are invalid.
    """
    if not isinstance(raw_entry, dict):
        raise DashboardPackageError("Dashboard package result entries must be objects.")
    result_id = str(raw_entry.get("result_id") or "").strip()
    seed_id = str(raw_entry.get("seed_id") or "").strip()
    strategy = str(raw_entry.get("strategy") or "").strip()
    title = str(raw_entry.get("title") or "").strip()
    updated_at = str(raw_entry.get("updated_at") or "").strip()
    if not all((result_id, seed_id, strategy, title, updated_at)):
        raise DashboardPackageError(
            "Dashboard package result entries require result_id, seed_id, title, "
            "strategy, and updated_at."
        )
    expected_result_id = f"{strategy}:{seed_id}"
    if result_id != expected_result_id:
        raise DashboardPackageError(
            f"Dashboard package result_id {result_id!r} does not match "
            f"{expected_result_id!r}."
        )

    summary = raw_entry.get("summary")
    if not isinstance(summary, dict):
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} summary must be an object."
        )
    normalized_summary = {
        "nodes": _validated_non_negative_count(
            summary.get("nodes"), field=f"result {result_id!r} summary.nodes"
        ),
        "edges": _validated_non_negative_count(
            summary.get("edges"), field=f"result {result_id!r} summary.edges"
        ),
    }
    payload = _validate_dashboard_graph_payload(
        raw_entry.get("payload"), result_id=result_id
    )
    payload_meta = payload["meta"]
    payload_summary = payload["summary"]
    if (
        str(payload.get("seed_id") or "").strip() != seed_id
        or str(payload_meta.get("strategy") or "").strip() != strategy
    ):
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} descriptor does not match its payload."
        )
    if normalized_summary != {
        "nodes": payload_summary.get("nodes"),
        "edges": payload_summary.get("edges"),
    }:
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} summary does not match its payload."
        )
    build = raw_entry.get("build", {})
    if not isinstance(build, dict):
        raise DashboardPackageError(
            f"Dashboard package result {result_id!r} build settings must be an object."
        )
    return {
        "result_id": result_id,
        "seed_id": seed_id,
        "title": title,
        "strategy": strategy,
        "summary": normalized_summary,
        "updated_at": updated_at,
        "payload": payload,
        "build": dict(build),
    }


def _validate_dashboard_package(raw_package: object) -> dict[str, Any]:
    """Validate a dashboard collection package and deterministically deduplicate it.

    When duplicate result identifiers are present, the first entry wins. Package
    upserts always place the newest entry first, so this also repairs duplicate
    state deterministically without inventing another recency policy.

    :param object raw_package: Candidate package object.
    :return Dict[str, Any]: Canonical package object.
    :raises DashboardPackageError: If the top-level or entry contract is invalid.
    """
    if not isinstance(raw_package, dict):
        raise DashboardPackageError("Dashboard package must be a JSON object.")
    try:
        json.dumps(raw_package, allow_nan=False)
    except ValueError as exc:
        raise DashboardPackageError(
            "Dashboard package contains non-finite numeric values."
        ) from exc
    if raw_package.get("kind") != DASHBOARD_COLLECTION_KIND:
        raise DashboardPackageError(
            f"Unsupported dashboard package kind {raw_package.get('kind')!r}."
        )
    schema_version = raw_package.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != DASHBOARD_COLLECTION_SCHEMA_VERSION
    ):
        raise DashboardPackageError(
            f"Unsupported dashboard package schema version {schema_version!r}."
        )
    raw_results = raw_package.get("results")
    if not isinstance(raw_results, list):
        raise DashboardPackageError("Dashboard package results must be an array.")

    results: list[dict[str, Any]] = []
    seen_result_ids: set[str] = set()
    for raw_entry in raw_results:
        entry = _validate_dashboard_result_entry(raw_entry)
        result_id = entry["result_id"]
        if result_id in seen_result_ids:
            continue
        seen_result_ids.add(result_id)
        results.append(entry)

    raw_current_result_id = raw_package.get("current_result_id")
    current_result_id = (
        str(raw_current_result_id).strip()
        if raw_current_result_id is not None
        else None
    )
    if current_result_id == "":
        current_result_id = None
    if current_result_id is not None and current_result_id not in seen_result_ids:
        raise DashboardPackageError(
            "Dashboard package current_result_id does not reference a package result."
        )
    return {
        "kind": DASHBOARD_COLLECTION_KIND,
        "schema_version": DASHBOARD_COLLECTION_SCHEMA_VERSION,
        "current_result_id": current_result_id,
        "results": results,
    }


def load_dashboard_package(package_path: Path) -> dict[str, Any]:
    """Load and strictly validate an existing dashboard collection package.

    :param Path package_path: Package file to load.
    :return Dict[str, Any]: Canonical validated package.
    :raises DashboardPackageError: If the package cannot be decoded or validated.
    """
    return _validate_dashboard_package(
        _load_package_object(package_path, label="dashboard package")
    )


def _resolve_collection_artifact_path(
    collection_root: Path, relative_path: object
) -> Path | None:
    """Resolve a legacy manifest artifact path confined to its collection root.

    :param Path collection_root: Legacy collection directory.
    :param object relative_path: Manifest-provided relative artifact path.
    :return Optional[Path]: Confined resolved path, or ``None`` when unsafe.
    """
    candidate = Path(str(relative_path or "").strip())
    if not str(candidate) or candidate.is_absolute():
        return None
    resolved_root = collection_root.resolve()
    resolved_candidate = (resolved_root / candidate).resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved_candidate


def _load_legacy_dashboard_results(collection_root: Path) -> list[dict[str, Any]]:
    """Load valid entries from a legacy manifest without modifying legacy files.

    Invalid legacy entries are reported and skipped. Their source files remain
    untouched, allowing manual recovery while safe entries migrate forward.

    :param Path collection_root: Collection directory containing a legacy manifest.
    :return list[Dict[str, Any]]: Valid normalized package entries in manifest order.
    """
    manifest_path = collection_root / LEGACY_DASHBOARD_MANIFEST_FILENAME
    if not path_exists(manifest_path):
        return []
    try:
        manifest = _load_package_object(
            manifest_path, label="legacy dashboard manifest"
        )
    except DashboardPackageError as exc:
        logger.warning("Skipping invalid legacy dashboard manifest: %s", exc)
        return []
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("results"), list
    ):
        logger.warning(
            "Skipping unsupported legacy dashboard manifest at %s.", manifest_path
        )
        return []

    results: list[dict[str, Any]] = []
    seen_result_ids: set[str] = set()
    for index, raw_entry in enumerate(manifest["results"]):
        if not isinstance(raw_entry, dict):
            logger.warning(
                "Skipping invalid legacy dashboard result at index %d.", index
            )
            continue
        json_path = _resolve_collection_artifact_path(
            collection_root, raw_entry.get("json_path")
        )
        config_path = _resolve_collection_artifact_path(
            collection_root, raw_entry.get("config_path")
        )
        if json_path is None or config_path is None:
            logger.warning(
                "Skipping legacy dashboard result %r with an unsafe artifact path.",
                raw_entry.get("result_id"),
            )
            continue
        try:
            graph_payload = _load_package_object(
                json_path, label="legacy graph payload"
            )
            config_payload = _load_package_object(
                config_path, label="legacy graph config"
            )
            graph_payload = dict(graph_payload)
            graph_payload.setdefault("kind", GRAPH_PAYLOAD_KIND)
            graph_payload.setdefault("schema_version", GRAPH_PAYLOAD_SCHEMA_VERSION)
            build = config_payload.get("build", {})
            summary = graph_payload.get("summary", raw_entry.get("summary"))
            candidate = _validate_dashboard_result_entry(
                {
                    "result_id": raw_entry.get("result_id"),
                    "seed_id": raw_entry.get("seed_id"),
                    "title": raw_entry.get("title"),
                    "strategy": raw_entry.get("strategy"),
                    "summary": summary,
                    "updated_at": raw_entry.get("updated_at"),
                    "payload": graph_payload,
                    "build": build,
                }
            )
        except DashboardPackageError as exc:
            logger.warning(
                "Skipping invalid legacy dashboard result %r: %s",
                raw_entry.get("result_id"),
                exc,
            )
            continue
        result_id = candidate["result_id"]
        if result_id in seen_result_ids:
            continue
        seen_result_ids.add(result_id)
        results.append(candidate)
    return results


def update_dashboard_package(
    package_path: Path,
    *,
    graph: nx.Graph,
    seed_id: str,
    strategy: str,
    payload: dict[str, Any],
    build: dict[str, Any],
    staged_result_files: dict[Path, Path] | None = None,
    obsolete_result_paths: set[Path] | None = None,
) -> dict[str, Any]:
    """Atomically create or update a portable dashboard collection package.

    One slot is retained per ``(strategy, seed_id)`` pair. The package lock covers
    existing-package validation, optional legacy migration, merge, and atomic
    replacement. Complete per-result exports may be staged before locking and
    published with rollback here, so concurrent or failed builds cannot leave a
    mixed successful result bundle.

    :param Path package_path: Portable collection package path.
    :param nx.Graph graph: Built graph used for seed metadata.
    :param str seed_id: Seed node identifier.
    :param str strategy: Active strategy name.
    :param Dict[str, Any] payload: Canonical graph payload from the exporter.
    :param Dict[str, Any] build: Portable resolved build settings.
    :param Optional[Dict[Path, Path]] staged_result_files: Final result paths mapped
        to fully written same-filesystem staging files.
    :param Optional[Set[Path]] obsolete_result_paths: Exact prior optional exports
        to remove if this result is published successfully.
    :return Dict[str, Any]: Canonical package written to disk.
    :raises DashboardPackageError: If an existing package is invalid or unsupported.
    """
    result_id = f"{strategy}:{seed_id}"
    validated_payload = _validate_dashboard_graph_payload(payload, result_id=result_id)
    payload_summary = validated_payload["summary"]
    seed_payload = next(
        node
        for node in validated_payload["nodes"]
        if str(node.get("id")) == str(validated_payload["seed_id"])
    )
    seed_title = str(seed_payload.get("title") or seed_id)
    entry = {
        "result_id": result_id,
        "seed_id": seed_id,
        "title": seed_title,
        "strategy": strategy,
        "summary": {
            "nodes": payload_summary["nodes"],
            "edges": payload_summary["edges"],
        },
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "payload": validated_payload,
        "build": dict(build),
    }
    entry = _validate_dashboard_result_entry(entry)
    package_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _dashboard_package_lock_path(package_path)
    lock = FileLock(str(lock_path), timeout=DASHBOARD_PACKAGE_LOCK_TIMEOUT_SECONDS)
    try:
        with lock:
            if path_exists(package_path):
                existing_package = load_dashboard_package(package_path)
                existing_results = existing_package["results"]
            else:
                existing_results = _load_legacy_dashboard_results(package_path.parent)
            filtered = [
                item for item in existing_results if item.get("result_id") != result_id
            ]
            package = _validate_dashboard_package(
                {
                    "kind": DASHBOARD_COLLECTION_KIND,
                    "schema_version": DASHBOARD_COLLECTION_SCHEMA_VERSION,
                    "current_result_id": result_id,
                    "results": [entry, *filtered],
                }
            )
            if staged_result_files is not None:
                _commit_staged_dashboard_artifacts(
                    package_path,
                    package,
                    staged_result_files=staged_result_files,
                    obsolete_result_paths=obsolete_result_paths or set(),
                )
            else:
                atomic_write_json(package_path, package, indent=2)
            return package
    except Timeout as exc:
        raise RuntimeError(
            "Timed out waiting for dashboard package lock "
            f"at {lock_path} after {DASHBOARD_PACKAGE_LOCK_TIMEOUT_SECONDS:.1f}s."
        ) from exc


def render_dashboard_collection_snapshot(
    package_path: Path,
    *,
    dashboard_path: Path,
    exporter: Any,
    metadata: dict[str, Any],
    theme: str,
) -> dict[str, Any]:
    """Render an atomic dashboard snapshot from the latest locked package state.

    The package and its embedded HTML snapshot must be serialized by the same
    lock. Otherwise two successful builds can write their HTML snapshots out of
    order even though their package upserts were individually atomic.

    :param Path package_path: Authoritative dashboard collection package.
    :param Path dashboard_path: HTML viewer path to refresh.
    :param Any exporter: Graph exporter for the build's current graph.
    :param Dict[str, Any] metadata: Mutable exporter metadata mapping.
    :param str theme: Requested dashboard theme.
    :return Dict[str, Any]: Latest package embedded in the rendered viewer.
    """
    lock_path = _dashboard_package_lock_path(package_path)
    lock = FileLock(str(lock_path), timeout=DASHBOARD_PACKAGE_LOCK_TIMEOUT_SECONDS)
    try:
        with lock:
            package = load_dashboard_package(package_path)
            metadata["dashboard_collection"] = package
            exporter.to_dashboard_html(dashboard_path, theme=theme)
            return package
    except Timeout as exc:
        raise RuntimeError(
            "Timed out waiting to refresh dashboard snapshot "
            f"at {lock_path} after {DASHBOARD_PACKAGE_LOCK_TIMEOUT_SECONDS:.1f}s."
        ) from exc
