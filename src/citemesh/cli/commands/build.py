"""``citemesh build``: construct a graph and write the requested export artifacts."""

from __future__ import annotations

import argparse
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from citemesh.core.user_config import UserConfig
from citemesh.data.cache import atomic_write_json, path_exists
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

        if not _confirm_force_rebuild_cache(args):
            logger.info("Build aborted.")
            return 1
        _log_build_side_effect_contract(args)
        # Build graph based on strategy
        logger.info(f"Building graph using {args.strategy} strategy...")
        graph, seed_id = _build_strategy_graph(
            args, args.strategy, validate_contract=False
        )

        if args.output:
            base_output_path = Path(args.output)
        elif dashboard_collection_mode:
            # The collection resolver places shared and per-paper artifacts.
            base_output_path = Path("out")
        else:
            base_output_path = generate_output_path(
                graph, seed_id, strategy=args.strategy
            )

        dashboard_package_path: Path | None = None
        if dashboard_collection_mode:
            output_paths, dashboard_package_path = resolve_dashboard_collection_outputs(
                base_output_path=base_output_path,
                selected_formats=selected_formats,
                explicit_output=explicit_output,
                strategy=args.strategy,
                graph=graph,
                seed_id=seed_id,
            )
            if dashboard_package_path != preflight_package_path:
                raise RuntimeError(
                    "Dashboard package planning changed after graph construction."
                )
        else:
            output_paths = resolve_output_paths(
                base_output_path=base_output_path,
                selected_formats=selected_formats,
                explicit_output=explicit_output,
                strategy=args.strategy,
            )
        if dashboard_package_path is not None:
            logger.info(
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

        # Visualize / export
        metadata = {
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
        plot_metadata = _plot_overlay_metadata(metadata)
        # JSON embeds dashboard geometry too, so it shares the run's layout
        # (honoring --spring-iterations/--seed) instead of a default one.
        layout_required = any(
            fmt in output_paths for fmt in ("png", "plotly", "dashboard", "json")
        )
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

        def write_export_artifacts(export_paths: dict[str, Path]) -> None:
            """Write selected graph exports to their supplied paths.

            :param Dict[str, Path] export_paths: Destination paths for this write.
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
                if dashboard_package_path is not None and fmt in (
                    "dashboard",
                    "json",
                ):
                    # Collection JSON is staged separately; the viewer is
                    # rendered afterward from the latest collection snapshot.
                    continue
                method = getattr(exporter, method_name)
                if fmt in _THEME_AWARE_FORMATS:
                    method(export_paths[fmt], theme=args.theme)
                else:
                    method(export_paths[fmt])

        if dashboard_package_path is None:
            write_export_artifacts(output_paths)

        config_output_paths = dict(output_paths)
        if dashboard_package_path is not None:
            config_output_paths["dashboard_package"] = dashboard_package_path
        graph_config_payload = _build_graph_config_payload(
            cli_args=args,
            seed_id=seed_id,
            metadata=metadata,
            selected_formats=selected_formats,
            output_paths=config_output_paths,
        )
        standalone_dashboard_only = standalone_dashboard and selected_formats == [
            "dashboard"
        ]
        graph_config_path: Path | None = None
        if dashboard_package_path is None and not standalone_dashboard_only:
            graph_config_path = resolve_graph_config_path(
                output_paths=output_paths,
                strategy=args.strategy,
            )
            atomic_write_json(graph_config_path, graph_config_payload, indent=2)

        if dashboard_package_path is not None:
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
                return 1

        artifact_paths = dict(output_paths)
        if graph_config_path is not None:
            artifact_paths["config"] = graph_config_path
        if dashboard_package_path is not None:
            artifact_paths["dashboard_package"] = dashboard_package_path
        saved_artifact_count = len(artifact_paths)
        output_dirs = sorted({str(path.parent) for path in artifact_paths.values()})
        if saved_artifact_count:
            if len(output_dirs) == 1:
                logger.info(
                    "%d export artifacts saved to:\t%s",
                    saved_artifact_count,
                    output_dirs[0],
                )
            else:
                logger.info(
                    "%d export artifacts saved across %d directories: %s",
                    saved_artifact_count,
                    len(output_dirs),
                    ", ".join(output_dirs),
                )

        logger.info(
            "Graph summary: nodes=%d, edges=%d",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

    except DashboardPackageError as e:
        logger.error(
            "Failed to prepare dashboard collection: %s",
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
