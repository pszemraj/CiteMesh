"""Dashboard metadata, payload, collection bundle, and HTML template rendering."""

from __future__ import annotations

import json
import re
from collections.abc import Hashable
from functools import lru_cache
from importlib import resources
from typing import Any

from citemesh.visualization.export.geometry import (
    _UI_PALETTES,
    DASHBOARD_AXIS_MIN_PADDING,
    DASHBOARD_AXIS_X_PADDING,
    DASHBOARD_LABEL_CAP,
    DASHBOARD_LABEL_MIN_DISTANCE,
    DASHBOARD_MAX_NODE_DIAMETER,
    DASHBOARD_SELECTION_HALO_SCALE,
    UI_FONT_FAMILY,
    _rgb_tuple_to_hex,
    _theme_color_scheme,
    theme_hover_label,
)
from citemesh.visualization.export.nodes import _sorted_edges, _strategy

from ..themes import Theme
from ..years import optional_publication_year_bounds
from .contracts import (
    DASHBOARD_COLLECTION_KIND,
    DASHBOARD_COLLECTION_SCHEMA_VERSION,
    GRAPH_PAYLOAD_KIND,
    GRAPH_PAYLOAD_SCHEMA_VERSION,
)

_DASHBOARD_ASSET_PACKAGE = "citemesh.visualization.dashboard"
_DASHBOARD_CSS_SLOT = "__DASHBOARD_CSS__\n"
_DASHBOARD_JS_SLOT = "__DASHBOARD_JS__\n"


def _read_dashboard_asset(name: str) -> str:
    """Read a packaged dashboard asset as UTF-8 text.

    :param str name: Asset file name under the dashboard ``assets`` directory.
    :return str: Decoded asset contents.
    """
    asset = resources.files(_DASHBOARD_ASSET_PACKAGE) / "assets" / name
    return asset.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _dashboard_template_source() -> str:
    """Assemble the dashboard HTML template from its packaged assets.

    The CSS and JS slots are filled here rather than through the caller's
    token substitution: both assets themselves contain substitution tokens
    (for example ``__BODY_BG__`` and ``__PLOTLY_DIV_ID__``), which a
    single-pass substitution over the markup alone would never visit.

    :return str: Template markup with the stylesheet and script inlined.
    """
    template = _read_dashboard_asset("template.html")
    template = template.replace(
        _DASHBOARD_CSS_SLOT, _read_dashboard_asset("dashboard.css")
    )
    return template.replace(_DASHBOARD_JS_SLOT, _read_dashboard_asset("dashboard.js"))


def _dashboard_template(
    *,
    theme_obj: Theme,
    div_id: str,
    plotly_js: str,
    payload_json: str,
    figure_json: str,
    collection_json: str,
) -> str:
    """Render standalone dashboard HTML template.

    :param Theme theme_obj: Active theme.
    :param str div_id: Plotly mount div ID.
    :param str plotly_js: Inline Plotly runtime JS.
    :param str payload_json: Serialized dashboard payload JSON.
    :param str figure_json: Serialized Plotly figure JSON.
    :param str collection_json: Serialized collection bundle JSON.
    :return str: Dashboard HTML content.
    """
    color_scheme = _theme_color_scheme(theme_obj)
    palette = _UI_PALETTES[color_scheme]
    vars_map = {
        "__COLOR_SCHEME__": color_scheme,
        "__BODY_BG__": palette["body_bg"],
        "__PANEL_BG__": palette["panel_bg"],
        "__PANEL_BORDER__": palette["panel_border"],
        "__TEXT_PRIMARY__": palette["text_primary"],
        "__TEXT_MUTED__": palette["text_muted"],
        "__ACCENT__": palette["accent"],
        "__ACCENT_SOFT__": palette["accent_soft"],
        "__UI_FONT_FAMILY__": UI_FONT_FAMILY,
        # The viewer rebuilds the figure client-side, so the hoverlabel it
        # applies is the very dict the Python figure carries rather than a
        # hand-copied twin.
        "__HOVER_LABEL_JSON__": json.dumps(
            theme_hover_label(theme_obj), sort_keys=True
        ),
        "__GRAPH_BG__": theme_obj.background,
        "__NODE_COLOR_OLD__": _rgb_tuple_to_hex(theme_obj.node_color_old),
        "__NODE_COLOR_NEW__": _rgb_tuple_to_hex(theme_obj.node_color_new),
        "__SEED_RING__": _rgb_tuple_to_hex(theme_obj.seed_color),
        "__DASHBOARD_EDGE_COLOR__": _rgb_tuple_to_hex(theme_obj.edge_color),
        "__DASHBOARD_AXIS_MIN_PADDING__": str(DASHBOARD_AXIS_MIN_PADDING),
        "__DASHBOARD_AXIS_X_PADDING__": str(DASHBOARD_AXIS_X_PADDING),
        "__DASHBOARD_LABEL_CAP__": str(DASHBOARD_LABEL_CAP),
        "__DASHBOARD_LABEL_MIN_DISTANCE__": str(DASHBOARD_LABEL_MIN_DISTANCE),
        "__DASHBOARD_MAX_NODE_DIAMETER__": str(DASHBOARD_MAX_NODE_DIAMETER),
        "__DASHBOARD_SELECTION_HALO_SCALE__": str(DASHBOARD_SELECTION_HALO_SCALE),
        "__GRAPH_PAYLOAD_KIND_JSON__": json.dumps(GRAPH_PAYLOAD_KIND),
        "__GRAPH_PAYLOAD_SCHEMA_VERSION__": str(GRAPH_PAYLOAD_SCHEMA_VERSION),
        "__COLLECTION_KIND_JSON__": json.dumps(DASHBOARD_COLLECTION_KIND),
        "__COLLECTION_SCHEMA_VERSION__": str(DASHBOARD_COLLECTION_SCHEMA_VERSION),
        "__PLOTLY_JS__": plotly_js,
        "__PAYLOAD_JSON__": payload_json,
        "__FIGURE_JSON__": figure_json,
        "__COLLECTION_JSON__": collection_json,
        "__PLOTLY_DIV_ID__": div_id,
    }
    template = _dashboard_template_source()
    # Single-pass substitution over the template only: sequential
    # str.replace would rescan already-injected values, letting a token
    # such as __PLOTLY_DIV_ID__ inside paper metadata get rewritten (or,
    # for the JSON tokens, corrupt the embedded payloads).
    token_pattern = re.compile(
        "|".join(re.escape(token) for token in sorted(vars_map, key=len, reverse=True))
    )
    return token_pattern.sub(lambda match: vars_map[match.group(0)], template)


class DashboardPayloadMixin:
    """Exporter-state dashboard metadata, payload, and collection assembly."""

    def _dashboard_meta(
        self,
        *,
        theme_obj: Theme,
        node_ids: list[Hashable],
        node_payloads: list[dict[str, Any]],
        sorted_edges: list[tuple[Hashable, Hashable, dict[str, Any]]],
        include_plotly_geometry: bool,
    ) -> dict[str, Any]:
        """Build dashboard metadata shared by dashboard HTML and JSON exports.

        :param Theme theme_obj: Active visualization theme.
        :param list[Hashable] node_ids: Node order used by Plotly points.
        :param list[Dict[str, Any]] node_payloads: Enriched node payloads.
        :param list[tuple[Hashable, Hashable, Dict[str, Any]]] sorted_edges:
            Deterministically ordered edge payloads.
        :param bool include_plotly_geometry: Whether layout-backed Plotly geometry
            should be embedded in the metadata.
        :return Dict[str, Any]: Dashboard metadata payload.
        """
        # The dashboard import contract requires a non-empty strategy token;
        # graphs built outside the CLI/builders may carry none.
        strategy = _strategy(self.graph, self.metadata) or "unknown"
        # Null rather than the color-scale sentinel: the timeline renders "-" for
        # a missing range and would otherwise show years no paper carries.
        bounds = optional_publication_year_bounds(
            node.get("year") for node in node_payloads
        )
        year_range = {"min": bounds[0], "max": bounds[1]} if bounds else None

        meta: dict[str, Any] = {
            "seed_id": str(self.seed_id),
            "strategy": strategy,
            "theme": theme_obj.name,
            "summary": {
                "nodes": len(node_payloads),
                "edges": len(sorted_edges),
            },
            "year_range": year_range,
        }
        raw_source_status = self.metadata.get("candidate_source_status")
        if isinstance(raw_source_status, dict):
            meta["candidate_source_status"] = {
                str(source): str(status)
                for source, status in sorted(
                    raw_source_status.items(), key=lambda item: str(item[0])
                )
            }
        if include_plotly_geometry:
            positions = self._get_layout()
            meta["plotly_node_order"] = [str(node_id) for node_id in node_ids]
            meta["plotly_positions"] = [
                [float(positions[node_id][0]), float(positions[node_id][1])]
                for node_id in node_ids
            ]
            meta["plotly_node_sizes"] = [
                max(6.0, self._node_size(node_id) / 50.0) for node_id in node_ids
            ]
        return meta

    def _dashboard_payload(
        self, *, theme_obj: Theme, node_ids: list[Hashable]
    ) -> dict[str, Any]:
        """Build deterministic dashboard payload from graph metadata.

        :param Theme theme_obj: Active visualization theme.
        :param list[Hashable] node_ids: Node order used by Plotly points.
        :return Dict[str, Any]: JSON payload consumed by dashboard JS.
        """
        node_payloads = self._enriched_nodes()
        sorted_edges = _sorted_edges(self.graph)
        payload: dict[str, Any] = {
            "meta": self._dashboard_meta(
                theme_obj=theme_obj,
                node_ids=node_ids,
                node_payloads=node_payloads,
                sorted_edges=sorted_edges,
                include_plotly_geometry=True,
            ),
            "nodes": node_payloads,
            "edges": [
                {
                    "source": str(left),
                    "target": str(right),
                    "weight": float(data.get("weight", 0.0)),
                }
                for left, right, data in sorted_edges
            ],
        }
        return payload

    def _dashboard_collection_bundle(self) -> dict[str, Any]:
        """Normalize optional collection metadata for shared dashboard shells.

        :return Dict[str, Any]: Collection result descriptors and embedded payloads.
        """
        empty_bundle: dict[str, Any] = {
            "kind": DASHBOARD_COLLECTION_KIND,
            "schema_version": DASHBOARD_COLLECTION_SCHEMA_VERSION,
            "current_result_id": None,
            "results": [],
        }
        raw_bundle = self.metadata.get("dashboard_collection")
        if not isinstance(raw_bundle, dict):
            return empty_bundle

        declared_kind = str(raw_bundle.get("kind") or "").strip()
        raw_results = raw_bundle.get("results")
        raw_payloads = raw_bundle.get("payloads")
        legacy_payloads = raw_payloads if isinstance(raw_payloads, dict) else {}
        results: list[dict[str, Any]] = []
        if isinstance(raw_results, list):
            for raw_entry in raw_results:
                if not isinstance(raw_entry, dict):
                    continue
                result_id = str(raw_entry.get("result_id") or "").strip()
                payload = raw_entry.get("payload")
                if not isinstance(payload, dict):
                    payload = legacy_payloads.get(result_id)
                if not result_id or not isinstance(payload, dict):
                    continue
                entry: dict[str, Any] = {
                    key: raw_entry[key]
                    for key in (
                        "result_id",
                        "seed_id",
                        "title",
                        "strategy",
                        "summary",
                        "updated_at",
                    )
                    if key in raw_entry
                }
                entry["result_id"] = result_id
                entry["payload"] = payload
                build = raw_entry.get("build")
                if isinstance(build, dict):
                    entry["build"] = build
                elif declared_kind == DASHBOARD_COLLECTION_KIND:
                    # Versioned entries must carry build metadata; the viewer
                    # rejects the whole bundle when the key is missing.
                    entry["build"] = {}
                results.append(entry)
        raw_current_result_id = raw_bundle.get("current_result_id")
        current_result_id = (
            str(raw_current_result_id).strip()
            if raw_current_result_id is not None
            else None
        )
        if current_result_id not in {entry["result_id"] for entry in results}:
            # Malformed entries are dropped above, and the viewer rejects a bundle
            # whose current_result_id names no included result.
            current_result_id = results[0]["result_id"] if results else None
        bundle: dict[str, Any] = {
            "current_result_id": current_result_id,
            "results": results,
        }
        if declared_kind:
            if (
                declared_kind != DASHBOARD_COLLECTION_KIND
                or raw_bundle.get("schema_version")
                != DASHBOARD_COLLECTION_SCHEMA_VERSION
            ):
                raise ValueError(
                    "Unsupported dashboard collection metadata kind or schema version."
                )
            bundle.update(
                {
                    "kind": DASHBOARD_COLLECTION_KIND,
                    "schema_version": DASHBOARD_COLLECTION_SCHEMA_VERSION,
                }
            )
        return bundle
