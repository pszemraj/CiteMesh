"""``citemesh build``: construct a graph and write the requested export artifacts.

The run is a pipeline: plan the exports, build the graph, resolve output paths,
assemble metadata, render the artifacts, then write the sidecar and -- in
collection mode -- upsert the dashboard package. Each stage is a function here
so the names tests patch on this module (``GraphExporter``, ``compute_layout``,
``visualize_graph``, ``_build_strategy_graph``, ``update_dashboard_package``)
stay module globals resolved at call time.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import networkx as nx

from citemesh.data.cache import atomic_write_json, path_exists
from citemesh.data.user_config import UserConfig
from citemesh.services import SemanticScholarUnavailableError, get_client
from citemesh.strategies.candidates import CandidateAcquisitionError
from citemesh.visualization import (
    GraphExporter,
    compute_layout,
    generate_output_path,
    visualize_graph,
)
from citemesh.visualization.dashboard.package import (
    DASHBOARD_PACKAGE_FILENAME,
    DashboardPackageError,
    load_dashboard_package,
    render_dashboard_collection_snapshot,
    update_dashboard_package,
)

from ..build_contract import (
    _build_strategy_graph,
    _log_build_side_effect_contract,
    _validate_build_cli_contract,
)
from ..build_options import (
    _apply_user_config_defaults,
    _embedding_branch_enabled,
    _embedding_export_metadata,
    _plot_overlay_metadata,
    _strategy_score_contract,
)
from ..cache_ops import _confirm_force_rebuild_cache
from ..console import logger
from ..graph_config import (
    _build_graph_config_payload,
    canonicalize_paper_id_for_metadata,
)
from ..outputs import (
    _EXPORTER_METHOD,
    _THEME_AWARE_FORMATS,
    EXPORT_FORMATS,
    _dashboard_optional_result_paths,
    _is_standalone_dashboard_output,
    _resolve_dashboard_collection_root,
    resolve_dashboard_collection_outputs,
    resolve_graph_config_path,
    resolve_output_paths,
)

#: Formats whose layout must be computed once and shared across exporters.
_LAYOUT_DEPENDENT_FORMATS = ("png", "plotly", "dashboard", "json")


@dataclass(frozen=True)
class _BuildPlan:
    """Export selection and dashboard-collection decisions for one build run.

    :param list[str] selected_formats: Requested export formats, deduplicated.
    :param bool explicit_output: Whether ``--output`` was supplied.
    :param bool standalone_dashboard: Whether the run writes a single dashboard file.
    :param bool dashboard_collection_mode: Whether the run upserts a collection package.
    :param Path | None preflight_package_path: Collection package path validated
        before the graph is built, or ``None`` outside collection mode.
    """

    selected_formats: list[str]
    explicit_output: bool
    standalone_dashboard: bool
    dashboard_collection_mode: bool
    preflight_package_path: Path | None


def _resolve_build_plan(args: argparse.Namespace) -> _BuildPlan:
    """Decide which formats to export and whether this run owns a collection.

    Any existing collection package is loaded here, before the graph is built,
    so a corrupt package fails the run before expensive work happens.

    :param argparse.Namespace args: Parsed CLI namespace for ``build``.
    :return _BuildPlan: Resolved export selection and dashboard mode.
    """
    raw_exports = args.export or ["png"]
    if "all" in raw_exports:
        selected_formats = list(EXPORT_FORMATS)
    else:
        selected_formats = list(dict.fromkeys(raw_exports))

    explicit_output = bool(args.output)
    requested_base_output_path = Path(args.output or "out")
    standalone_dashboard = _is_standalone_dashboard_output(
        base_output_path=requested_base_output_path,
        selected_formats=selected_formats,
        explicit_output=explicit_output,
    )
    dashboard_collection_mode = (
        "dashboard" in selected_formats and not standalone_dashboard
    )
    preflight_package_path: Path | None = None
    if dashboard_collection_mode:
        collection_root = _resolve_dashboard_collection_root(
            requested_base_output_path,
            explicit_output=explicit_output,
        )
        preflight_package_path = collection_root / DASHBOARD_PACKAGE_FILENAME
        if path_exists(preflight_package_path):
            load_dashboard_package(preflight_package_path)

    return _BuildPlan(
        selected_formats=selected_formats,
        explicit_output=explicit_output,
        standalone_dashboard=standalone_dashboard,
        dashboard_collection_mode=dashboard_collection_mode,
        preflight_package_path=preflight_package_path,
    )


def _resolve_run_output_paths(
    args: argparse.Namespace,
    plan: _BuildPlan,
    graph: nx.Graph,
    seed_id: str,
) -> tuple[dict[str, Path], Path | None]:
    """Resolve every artifact path for this run and create their directories.

    :param argparse.Namespace args: Parsed CLI namespace for ``build``.
    :param _BuildPlan plan: Export selection resolved before the graph was built.
    :param nx.Graph graph: Constructed graph, used to derive a default path.
    :param str seed_id: Canonical seed identifier for the run.
    :return tuple[dict[str, Path], Path | None]: Format-keyed output paths and
        the collection package path, or ``None`` outside collection mode.
    :raises RuntimeError: If collection planning disagrees with the preflight path.
    """
    if args.output:
        base_output_path = Path(args.output)
    elif plan.dashboard_collection_mode:
        # The collection resolver places shared and per-paper artifacts.
        base_output_path = Path("out")
    else:
        base_output_path = generate_output_path(graph, seed_id, strategy=args.strategy)

    dashboard_package_path: Path | None = None
    if plan.dashboard_collection_mode:
        output_paths, dashboard_package_path = resolve_dashboard_collection_outputs(
            base_output_path=base_output_path,
            selected_formats=plan.selected_formats,
            explicit_output=plan.explicit_output,
            strategy=args.strategy,
            graph=graph,
            seed_id=seed_id,
        )
        if dashboard_package_path != plan.preflight_package_path:
            raise RuntimeError(
                "Dashboard package planning changed after graph construction."
            )
    else:
        output_paths = resolve_output_paths(
            base_output_path=base_output_path,
            selected_formats=plan.selected_formats,
            explicit_output=plan.explicit_output,
            strategy=args.strategy,
        )
    if dashboard_package_path is not None:
        logger.debug(
            "Dashboard collection mode: viewer=%s package=%s graph=%s.",
            output_paths["dashboard"],
            dashboard_package_path,
            output_paths["json"],
        )
    planned_paths = list(output_paths.values())
    if dashboard_package_path is not None:
        planned_paths.append(dashboard_package_path)
    for parent in {path.parent for path in planned_paths}:
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
    return output_paths, dashboard_package_path


def _build_export_metadata(
    args: argparse.Namespace, graph: nx.Graph, seed_id: str
) -> dict[str, Any]:
    """Assemble the run metadata embedded in every export artifact.

    :param argparse.Namespace args: Parsed CLI namespace for ``build``.
    :param nx.Graph graph: Constructed graph carrying runtime annotations.
    :param str seed_id: Canonical seed identifier for the run.
    :return dict[str, Any]: Metadata passed to the exporter and sidecar payload.
    """
    metadata: dict[str, Any] = {
        "paper_id": canonicalize_paper_id_for_metadata(args.paper_id),
        "seed_id": seed_id,
        "strategy": args.strategy,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "theme": args.theme,
        "score_contract": _strategy_score_contract(args.strategy),
    }
    raw_source_status = graph.graph.get("candidate_source_status")
    if isinstance(raw_source_status, dict):
        metadata["candidate_source_status"] = {
            str(source): str(status)
            for source, status in sorted(
                raw_source_status.items(), key=lambda item: str(item[0])
            )
        }
    include_embedding_metadata = _embedding_branch_enabled(args)
    if include_embedding_metadata:
        runtime_embedding_metadata: dict[str, Any] | None = None
        raw_runtime_metadata = graph.graph.get("embedding_runtime")
        if isinstance(raw_runtime_metadata, dict):
            runtime_embedding_metadata = raw_runtime_metadata
        metadata["embedding"] = _embedding_export_metadata(
            args, runtime_embedding_metadata
        )
    if args.include_timestamp:
        metadata["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return metadata


def _create_graph_exporter(
    args: argparse.Namespace,
    graph: nx.Graph,
    seed_id: str,
    metadata: dict[str, Any],
    output_paths: dict[str, Path],
) -> tuple[Any, Any]:
    """Compute the run's shared layout and build the exporter bound to it.

    JSON embeds dashboard geometry too, so it shares the run's layout (honoring
    ``--spring-iterations``/``--seed``) instead of a default one.

    :param argparse.Namespace args: Parsed CLI namespace for ``build``.
    :param nx.Graph graph: Constructed graph to export.
    :param str seed_id: Canonical seed identifier for the run.
    :param dict[str, Any] metadata: Run metadata embedded in exports.
    :param dict[str, Path] output_paths: Resolved artifact paths for the run.
    :return tuple[Any, Any]: The exporter and the shared layout, which is
        ``None`` when no selected format needs one.
    """
    layout_required = any(fmt in output_paths for fmt in _LAYOUT_DEPENDENT_FORMATS)
    shared_layout = (
        compute_layout(
            graph,
            iterations=args.spring_iterations,
            layout_seed=args.seed,
        )
        if layout_required
        else None
    )

    exporter = GraphExporter(
        graph,
        seed_id,
        metadata=metadata,
        theme_name=args.theme,
        layout=shared_layout,
    )
    return exporter, shared_layout


def _write_export_artifacts(
    export_paths: dict[str, Path],
    *,
    args: argparse.Namespace,
    graph: nx.Graph,
    seed_id: str,
    exporter: Any,
    plot_metadata: dict[str, Any],
    shared_layout: Any,
    defer_collection_formats: bool,
) -> None:
    """Write selected graph exports to their supplied paths.

    :param dict[str, Path] export_paths: Destination paths for this write.
    :param argparse.Namespace args: Parsed CLI namespace for ``build``.
    :param nx.Graph graph: Constructed graph to render.
    :param str seed_id: Canonical seed identifier for the run.
    :param Any exporter: Exporter bound to the run's graph and layout.
    :param dict[str, Any] plot_metadata: Compact metadata overlaid on the plot.
    :param Any shared_layout: Layout shared by every geometry-bearing format.
    :param bool defer_collection_formats: Whether the collection stages JSON
        separately and renders the viewer from the package afterward.
    :return None: Writes the selected artifacts to their resolved paths.
    """
    if "png" in export_paths:
        visualize_graph(
            graph,
            seed_id,
            export_paths["png"],
            iterations=args.spring_iterations,
            dpi=args.dpi,
            metadata=plot_metadata,
            theme_name=args.theme,
            layout=shared_layout,
        )

    for fmt, method_name in _EXPORTER_METHOD.items():
        if fmt not in export_paths:
            continue
        if defer_collection_formats and fmt in ("dashboard", "json"):
            # Collection JSON is staged separately; the viewer is rendered
            # afterward from the latest collection snapshot.
            continue
        method = getattr(exporter, method_name)
        if fmt in _THEME_AWARE_FORMATS:
            method(export_paths[fmt], theme=args.theme)
        else:
            method(export_paths[fmt])


def _write_run_sidecar(
    args: argparse.Namespace,
    plan: _BuildPlan,
    *,
    output_paths: dict[str, Path],
    graph_config_payload: dict[str, Any],
    dashboard_package_path: Path | None,
) -> Path | None:
    """Write the run's ``*.config.json`` sidecar when this run owns one.

    Collection runs stage their sidecar with the rest of the result bundle, and
    a standalone dashboard export has nowhere stable to put one.

    :param argparse.Namespace args: Parsed CLI namespace for ``build``.
    :param _BuildPlan plan: Export selection resolved for this run.
    :param dict[str, Path] output_paths: Resolved artifact paths for the run.
    :param dict[str, Any] graph_config_payload: Reproducible build payload.
    :param Path | None dashboard_package_path: Collection package path, if any.
    :return Path | None: The sidecar path written, or ``None`` if skipped.
    """
    standalone_dashboard_only = plan.standalone_dashboard and plan.selected_formats == [
        "dashboard"
    ]
    if dashboard_package_path is not None or standalone_dashboard_only:
        return None
    graph_config_path = resolve_graph_config_path(
        output_paths=output_paths,
        strategy=args.strategy,
    )
    atomic_write_json(graph_config_path, graph_config_payload, indent=2)
    return graph_config_path


def _stage_dashboard_result(
    args: argparse.Namespace,
    *,
    dashboard_package_path: Path,
    output_paths: dict[str, Path],
    graph: nx.Graph,
    seed_id: str,
    exporter: Any,
    graph_config_payload: dict[str, Any],
    write_export_artifacts: Callable[[dict[str, Path]], None],
) -> Path:
    """Stage this run's result bundle and upsert it into the collection package.

    Every artifact is written into a sibling temporary directory first, so the
    package update either promotes the whole bundle or leaves the previous one
    untouched.

    :param argparse.Namespace args: Parsed CLI namespace for ``build``.
    :param Path dashboard_package_path: Collection package to update.
    :param dict[str, Path] output_paths: Final artifact paths for the run.
    :param nx.Graph graph: Constructed graph being recorded.
    :param str seed_id: Canonical seed identifier for the run.
    :param Any exporter: Exporter bound to the run's graph and layout.
    :param dict[str, Any] graph_config_payload: Reproducible build payload.
    :param Callable[[dict[str, Path]], None] write_export_artifacts: Writer that
        renders the selected formats into the supplied paths.
    :return Path: Final sidecar path promoted into the result directory.
    """
    graph_payload = exporter.graph_payload()
    result_directory = output_paths["json"].parent
    graph_config_path = resolve_graph_config_path(
        output_paths=output_paths,
        strategy=args.strategy,
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{result_directory.name}.stage-",
        dir=result_directory.parent,
        ignore_cleanup_errors=True,
    ) as staging_name:
        staging_directory = Path(staging_name)
        staged_output_paths = {
            fmt: staging_directory / path.name
            for fmt, path in output_paths.items()
            if fmt != "dashboard"
        }
        write_export_artifacts(staged_output_paths)
        atomic_write_json(staged_output_paths["json"], graph_payload, indent=2)
        staged_config_path = staging_directory / graph_config_path.name
        atomic_write_json(staged_config_path, graph_config_payload, indent=2)
        staged_result_files = {
            output_paths[fmt]: staged_path
            for fmt, staged_path in staged_output_paths.items()
        }
        staged_result_files[graph_config_path] = staged_config_path
        update_dashboard_package(
            dashboard_package_path,
            graph=graph,
            seed_id=seed_id,
            strategy=args.strategy,
            payload=graph_payload,
            build=dict(graph_config_payload.get("build", {})),
            staged_result_files=staged_result_files,
            obsolete_result_paths=(
                _dashboard_optional_result_paths(output_paths)
                - set(staged_result_files)
            ),
        )
    return graph_config_path


def _render_dashboard_viewer(
    args: argparse.Namespace,
    *,
    dashboard_package_path: Path,
    output_paths: dict[str, Path],
    exporter: Any,
    metadata: dict[str, Any],
) -> bool:
    """Refresh the collection viewer from the just-committed package.

    The package is already durable at this point, so a renderer failure is
    reported with recovery instructions instead of rolling anything back.

    :param argparse.Namespace args: Parsed CLI namespace for ``build``.
    :param Path dashboard_package_path: Committed collection package.
    :param dict[str, Path] output_paths: Resolved artifact paths for the run.
    :param Any exporter: Exporter bound to the run's graph and layout.
    :param dict[str, Any] metadata: Run metadata embedded in the viewer.
    :return bool: ``True`` if the viewer was refreshed, ``False`` on failure.
    """
    try:
        render_dashboard_collection_snapshot(
            dashboard_package_path,
            dashboard_path=output_paths["dashboard"],
            exporter=exporter,
            metadata=metadata,
            theme=args.theme,
        )
    except Exception as exc:
        logger.error(
            "Dashboard data was saved safely at %s, but the viewer "
            "refresh failed: %s. Recover by opening an existing "
            "dashboard.html, choosing Add Results, and selecting this "
            "package, or rerun after fixing the renderer.",
            dashboard_package_path,
            exc,
            exc_info=logging.getLogger().level == logging.DEBUG,
        )
        return False
    return True


def _log_run_summary(
    graph: nx.Graph,
    *,
    output_paths: dict[str, Path],
    graph_config_path: Path | None,
    dashboard_package_path: Path | None,
) -> None:
    """Log one user-facing completion result and auxiliary paths for debugging.

    :param nx.Graph graph: Constructed graph being summarized.
    :param dict[str, Path] output_paths: Resolved artifact paths for the run.
    :param Path | None graph_config_path: Sidecar path, if one was written.
    :param Path | None dashboard_package_path: Collection package path, if any.
    :return None: Emits the closing informational and debug log lines.
    """
    if set(output_paths) == set(EXPORT_FORMATS):
        output_directories = sorted(
            {str(path.parent) for path in output_paths.values()}
        )
        if len(output_directories) == 1:
            logger.info(
                "Build complete: nodes=%d, edges=%d; %d artifacts saved to %s",
                graph.number_of_nodes(),
                graph.number_of_edges(),
                len(output_paths),
                output_directories[0],
            )
        else:
            logger.info(
                "Build complete: nodes=%d, edges=%d; %d artifacts saved across "
                "%d directories: %s",
                graph.number_of_nodes(),
                graph.number_of_edges(),
                len(output_paths),
                len(output_directories),
                ", ".join(output_directories),
            )
    else:
        primary_outputs = ", ".join(
            f"{format_name}={path}"
            for format_name, path in sorted(output_paths.items())
        )
        logger.info(
            "Build complete: nodes=%d, edges=%d; outputs: %s",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            primary_outputs,
        )

    auxiliary_paths: dict[str, Path] = {}
    if graph_config_path is not None:
        auxiliary_paths["config"] = graph_config_path
    if dashboard_package_path is not None:
        auxiliary_paths["dashboard_package"] = dashboard_package_path
    if auxiliary_paths:
        logger.debug(
            "Build auxiliary artifacts: %s",
            ", ".join(
                f"{artifact_name}={path}"
                for artifact_name, path in sorted(auxiliary_paths.items())
            ),
        )


def run_build_command(
    args: argparse.Namespace,
    build_parser: argparse.ArgumentParser,
    *,
    provided_build_options: set[str],
    user_config: UserConfig,
) -> int:
    """Run the ``build`` subcommand end to end.

    :param argparse.Namespace args: Parsed CLI namespace for ``build``.
    :param argparse.ArgumentParser build_parser: Parser used for contract errors.
    :param set[str] provided_build_options: Destinations explicitly supplied on the CLI.
    :param UserConfig user_config: Loaded user configuration supplying defaults.
    :return int: Process-style exit code.
    """
    config_default_dests = _apply_user_config_defaults(
        args, provided_build_options, user_config
    )
    _validate_build_cli_contract(
        args,
        build_parser,
        provided_build_options,
        config_defaults=config_default_dests,
        config_path=user_config.path,
    )
    try:
        plan = _resolve_build_plan(args)

        if not _confirm_force_rebuild_cache(args):
            logger.info("Build aborted.")
            return 1
        _log_build_side_effect_contract(args)
        # Build graph based on strategy
        logger.info(f"Building graph using {args.strategy} strategy...")
        graph, seed_id = _build_strategy_graph(args, args.strategy)

        output_paths, dashboard_package_path = _resolve_run_output_paths(
            args, plan, graph, seed_id
        )

        # Visualize / export
        metadata = _build_export_metadata(args, graph, seed_id)
        exporter, shared_layout = _create_graph_exporter(
            args, graph, seed_id, metadata, output_paths
        )
        write_export_artifacts = partial(
            _write_export_artifacts,
            args=args,
            graph=graph,
            seed_id=seed_id,
            exporter=exporter,
            plot_metadata=_plot_overlay_metadata(metadata),
            shared_layout=shared_layout,
            defer_collection_formats=dashboard_package_path is not None,
        )

        if dashboard_package_path is None:
            write_export_artifacts(output_paths)

        config_output_paths = dict(output_paths)
        if dashboard_package_path is not None:
            config_output_paths["dashboard_package"] = dashboard_package_path
        graph_config_payload = _build_graph_config_payload(
            cli_args=args,
            seed_id=seed_id,
            metadata=metadata,
            selected_formats=plan.selected_formats,
            output_paths=config_output_paths,
            s2_retry_budget=(
                getattr(args, "_s2_client", None) or get_client()
            ).retry_budget_seconds,
        )
        graph_config_path = _write_run_sidecar(
            args,
            plan,
            output_paths=output_paths,
            graph_config_payload=graph_config_payload,
            dashboard_package_path=dashboard_package_path,
        )

        if dashboard_package_path is not None:
            graph_config_path = _stage_dashboard_result(
                args,
                dashboard_package_path=dashboard_package_path,
                output_paths=output_paths,
                graph=graph,
                seed_id=seed_id,
                exporter=exporter,
                graph_config_payload=graph_config_payload,
                write_export_artifacts=write_export_artifacts,
            )
            if not _render_dashboard_viewer(
                args,
                dashboard_package_path=dashboard_package_path,
                output_paths=output_paths,
                exporter=exporter,
                metadata=metadata,
            ):
                return 1

        _log_run_summary(
            graph,
            output_paths=output_paths,
            graph_config_path=graph_config_path,
            dashboard_package_path=dashboard_package_path,
        )

    except DashboardPackageError as e:
        logger.error(
            "Failed to prepare dashboard collection: %s",
            e,
            exc_info=logging.getLogger().level == logging.DEBUG,
        )
        return 1
    except (CandidateAcquisitionError, SemanticScholarUnavailableError) as e:
        logger.error(
            "Build incomplete: current Semantic Scholar discovery could not be "
            "acquired. %s",
            e,
            exc_info=logging.getLogger().level == logging.DEBUG,
        )
        return 1
    except Exception as e:
        logger.error(
            "Failed to build graph: %s",
            e,
            exc_info=logging.getLogger().level == logging.DEBUG,
        )
        return 1

    return 0
