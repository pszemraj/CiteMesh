    const GRAPH_PAYLOAD_KIND = __GRAPH_PAYLOAD_KIND_JSON__;
    const GRAPH_PAYLOAD_SCHEMA_VERSION = __GRAPH_PAYLOAD_SCHEMA_VERSION__;
    const COLLECTION_KIND = __COLLECTION_KIND_JSON__;
    const COLLECTION_SCHEMA_VERSION = __COLLECTION_SCHEMA_VERSION__;
    // Injected from the Python theme so a client-rebuilt figure keeps the same
    // themed hover card as the server-rendered one.
    const HOVER_LABEL = __HOVER_LABEL_JSON__;
    let payload = JSON.parse(document.getElementById("citemesh-dashboard-data").textContent);
    const baseFigureTemplate = JSON.parse(document.getElementById("citemesh-dashboard-figure").textContent);
    let figureSpec = JSON.parse(document.getElementById("citemesh-dashboard-figure").textContent);
    // Embed the collection bundle directly in the shell so saved-result browsing
    // still works when the dashboard is opened from the local filesystem.
    const embeddedCollectionBundle = JSON.parse(
      document.getElementById("citemesh-dashboard-collection").textContent
    );
    let collectionBundle = emptyCollectionPackage();
    let initialEntry = null;
    let bootstrapFailureMessage = "";
    let bootstrapStatusMessage = "";
    try {
      initialEntry = collectionEntryFromGraphPayload(payload, "Current graph", true);
    } catch (err) {
      bootstrapFailureMessage =
        "Dashboard graph data is invalid: " + String((err && err.message) || err);
    }
    if (initialEntry) {
      try {
        collectionBundle = normalizeCollectionPackage(
          embeddedCollectionBundle,
          "this dashboard",
          true
        );
      } catch (err) {
        bootstrapStatusMessage =
          "Embedded graph collection was ignored: " + String((err && err.message) || err);
        collectionBundle = emptyCollectionPackage();
      }
      if (!collectionBundle.results.some((entry) => entry.result_id === initialEntry.result_id)) {
        collectionBundle.results.unshift(initialEntry);
      }
      if (!collectionBundle.current_result_id) {
        collectionBundle.current_result_id = initialEntry.result_id;
      }
    }
    const graphDiv = document.getElementById("__PLOTLY_DIV_ID__");
    const plotConfig = {
      displaylogo: false,
      responsive: true,
    };

    let nodes = [];
    let nodeOrder = [];
    let nodeById = new Map();
    let nodeIndexById = new Map();
    let yearRange = {};
    let seedNode = null;
    let seedYear = null;
    let traceSpecs = [];
    let nodeTraceIndex = 0;
    let haloTraceIndex = -1;
    let neighborhoodTraceIndex = -1;
    let defaultNodeSizes = [];
    let defaultNodeX = [];
    let defaultNodeY = [];
    let adjacency = new Map();
    let collectionResultId = String(collectionBundle.current_result_id || "") || null;
    let runtimeStatusMessage = "";
    let runtimeStatusTone = "warning";

    function normalizeArray(rawValue, length, fallbackValue) {
      if (Array.isArray(rawValue)) {
        if (rawValue.length >= length) {
          return rawValue.slice(0, length).map((value) => Number(value));
        }
        const expanded = rawValue.map((value) => Number(value));
        while (expanded.length < length) {
          expanded.push(fallbackValue);
        }
        return expanded;
      }
      const scalar = Number(rawValue);
      const safe = Number.isFinite(scalar) ? scalar : fallbackValue;
      return Array.from({ length }, () => safe);
    }

    function deepClone(value) {
      return JSON.parse(JSON.stringify(value));
    }

    function safeFiniteNumber(value, fallbackValue) {
      const numeric = Number(value);
      return Number.isFinite(numeric) ? numeric : fallbackValue;
    }

    function stableHash(value) {
      const text = String(value || "");
      let digest = 0;
      for (const char of text) {
        digest = ((digest * 33) + char.charCodeAt(0)) >>> 0;
      }
      return digest >>> 0;
    }

    function stableCurveDirection(leftId, rightId) {
      const [keyLeft, keyRight] = [String(leftId || ""), String(rightId || "")].sort();
      return stableHash(`${keyLeft}|${keyRight}`) % 2 === 0 ? 1 : -1;
    }

    function selectDashboardLabelIds(order, nodeById, xPairs, yPairs) {
      const rankedIds = order.slice().sort((leftId, rightId) => {
        const left = nodeById.get(leftId) || {};
        const right = nodeById.get(rightId) || {};
        if (!!left.is_seed !== !!right.is_seed) {
          return left.is_seed ? -1 : 1;
        }
        const citationDelta =
          Number(right.citation_count || 0) - Number(left.citation_count || 0);
        if (citationDelta !== 0) {
          return citationDelta;
        }
        const leftYear = Number.isFinite(Number(left.year)) ? Number(left.year) : -1;
        const rightYear = Number.isFinite(Number(right.year)) ? Number(right.year) : -1;
        if (leftYear !== rightYear) {
          return rightYear - leftYear;
        }
        return String(leftId).localeCompare(String(rightId));
      });
      const indexById = new Map(order.map((nodeId, idx) => [nodeId, idx]));
      const selectedIds = [];
      for (const nodeId of rankedIds) {
        const node = nodeById.get(nodeId) || {};
        const nodeIdx = indexById.get(nodeId);
        const isCrowded = !node.is_seed && selectedIds.some((selectedId) => {
          const selectedIdx = indexById.get(selectedId);
          return Math.hypot(
            Number(xPairs[nodeIdx] || 0) - Number(xPairs[selectedIdx] || 0),
            Number(yPairs[nodeIdx] || 0) - Number(yPairs[selectedIdx] || 0)
          ) < __DASHBOARD_LABEL_MIN_DISTANCE__;
        });
        if (isCrowded) {
          continue;
        }
        selectedIds.push(nodeId);
        if (selectedIds.length >= __DASHBOARD_LABEL_CAP__) {
          break;
        }
      }
      return new Set(selectedIds);
    }

    function dashboardNodeLabel(node, nodeId) {
      const authors = Array.isArray(node.authors) ? node.authors : [];
      const firstAuthor = String(authors[0] || "").trim();
      if (firstAuthor) {
        const surname = firstAuthor.split(/\s+/).pop();
        return escapeHtml(`${surname}, ${node.year || "n.d."}`);
      }
      const title = String(node.title || nodeId || "Unknown");
      return escapeHtml(title.length <= 26 ? title : `${title.slice(0, 23)}...`);
    }

    function currentSeedRingColor() {
      const styles = getComputedStyle(document.documentElement);
      const color = String(styles.getPropertyValue("--seed-ring") || "").trim();
      return color || "__SEED_RING__";
    }

    function colorWithAlpha(hexColor, alpha) {
      const match = /^#([0-9a-f]{6})$/i.exec(String(hexColor || "").trim());
      const clampedAlpha = Math.max(0, Math.min(1, Number(alpha) || 0));
      if (!match) {
        return `rgba(255,255,255,${clampedAlpha.toFixed(3)})`;
      }
      const value = match[1];
      const red = parseInt(value.slice(0, 2), 16);
      const green = parseInt(value.slice(2, 4), 16);
      const blue = parseInt(value.slice(4, 6), 16);
      return `rgba(${red},${green},${blue},${clampedAlpha.toFixed(3)})`;
    }

    function normalizeDashboardEdgeStrengths(edges) {
      // This mirrors Python's _edge_strength_scale because imported offline
      // payloads rebuild in the browser without running Python.
      const weights = edges.map((edge) =>
        Math.max(safeFiniteNumber(edge && edge.weight, 0), 0)
      );
      if (!weights.length) {
        return [];
      }
      const minimum = Math.min(...weights);
      const maximum = Math.max(...weights);
      const span = maximum - minimum;
      if (span <= 1e-9) {
        return weights.map(() => 0.5);
      }
      return weights.map((weight) => (weight - minimum) / span);
    }

    function wrapDashboardHoverTitle(value, width) {
      const words = String(value || "").trim().split(/\s+/).filter(Boolean);
      const lines = [];
      for (const word of words) {
        const current = lines.length ? lines[lines.length - 1] : "";
        if (!current || current.length + 1 + word.length > width) {
          lines.push(word);
        } else {
          lines[lines.length - 1] = `${current} ${word}`;
        }
      }
      return lines;
    }

    function dashboardHoverText(node, nodeId) {
      const titleLines = wrapDashboardHoverTitle(node.title || nodeId, 58);
      const titleHtml = titleLines.map(escapeHtml).join("<br>") || escapeHtml(nodeId);
      const lines = [`<b>${titleHtml}</b>`];
      const authorNames = Array.isArray(node.authors)
        ? node.authors.map((author) => String(author || "").trim()).filter(Boolean)
        : [];
      let authors = authorNames.slice(0, 3).join(", ") || "Unknown";
      if (authorNames.length > 3) {
        authors += ` +${authorNames.length - 3}`;
      }
      lines.push(escapeHtml(authors));

      const citationCount = Math.max(
        Math.trunc(safeFiniteNumber(node.citation_count, 0)),
        0
      );
      const factBits = [
        hasYear(node) ? String(Number(node.year)) : "n.d.",
        `${citationCount.toLocaleString("en-US")} citations`,
      ];
      const venue = String(node.venue || "").trim().split(/\s+/).filter(Boolean).join(" ");
      if (venue) {
        factBits.push(venue.length <= 44 ? venue : `${venue.slice(0, 41)}...`);
      }
      lines.push(escapeHtml(factBits.join(" | ")));

      const relationLabels = {
        seed: "seed paper",
        referenced_by_seed: "referenced by seed",
        cites_seed: "cites seed",
        overlap: "prior + derivative work",
        semantic_only: "semantic match",
        citation: "citation graph",
        semantic: "semantic match",
        both: "citations + semantic match",
      };
      const relationKey = node.is_seed
        ? "seed"
        : String(node.seed_relation || node.provenance || "");
      const relationLabel = relationLabels[relationKey] || "";
      if (relationLabel) {
        lines.push(`<i>${escapeHtml(relationLabel)}</i>`);
      }
      return lines.join("<br>");
    }

    function embeddedScriptJson(parsedDocument, scriptId, required) {
      // Parse imported dashboard HTML as a document instead of regex-matching script
      // tags. This avoids brittle parsing and keeps the inline runtime free of raw
      // script-closing sequences that would terminate the surrounding HTML script tag.
      const scriptElement = parsedDocument.getElementById(String(scriptId || ""));
      const isJsonScript =
        scriptElement &&
        String(scriptElement.tagName || "").toLowerCase() === "script" &&
        String(scriptElement.getAttribute("type") || "").toLowerCase() === "application/json";
      if (!isJsonScript) {
        if (!required) {
          return null;
        }
        throw new Error(`Imported dashboard file is missing ${scriptId}.`);
      }
      const rawJson = String(scriptElement.textContent || "").trim();
      if (!rawJson) {
        if (!required) {
          return null;
        }
        throw new Error(`Imported dashboard file is missing ${scriptId}.`);
      }
      return JSON.parse(rawJson);
    }

    function extractEmbeddedScriptJson(text, scriptId) {
      const parsedDocument = new DOMParser().parseFromString(
        String(text || ""),
        "text/html"
      );
      return embeddedScriptJson(parsedDocument, scriptId, true);
    }

    function parseImportedResultSetFromText(fileText, filename) {
      const text = String(fileText || "");
      const lowerName = String(filename || "").toLowerCase();
      const looksLikeDashboardHtml =
        lowerName.endsWith(".html") || text.includes('id="citemesh-dashboard-data"');
      if (looksLikeDashboardHtml) {
        const parsedDocument = new DOMParser().parseFromString(text, "text/html");
        const importedGraph = embeddedScriptJson(
          parsedDocument,
          "citemesh-dashboard-data",
          true
        );
        const importedCollection = embeddedScriptJson(
          parsedDocument,
          "citemesh-dashboard-collection",
          false
        );
        const resultSet = importedCollection
          ? normalizeCollectionPackage(importedCollection, filename, true)
          : emptyCollectionPackage();
        const packageCurrentResultId = String(
          (importedCollection && importedCollection.current_result_id) || ""
        ).trim();
        const currentEntry = collectionEntryFromGraphPayload(
          importedGraph,
          filename,
          true
        );
        if (!resultSet.results.some((entry) => entry.result_id === currentEntry.result_id)) {
          resultSet.results.push(currentEntry);
        }
        if (!packageCurrentResultId) {
          resultSet.current_result_id = currentEntry.result_id;
        }
        return resultSet;
      }

      const imported = JSON.parse(text);
      if (imported && imported.kind === COLLECTION_KIND) {
        return normalizeCollectionPackage(imported, filename, false);
      }
      const entry = collectionEntryFromGraphPayload(imported, filename, true);
      const resultSet = emptyCollectionPackage();
      resultSet.current_result_id = entry.result_id;
      resultSet.results.push(entry);
      return resultSet;
    }

    function hasCompleteDashboardGeometry(meta) {
      if (!meta || typeof meta !== "object") {
        return false;
      }
      const order = Array.isArray(meta.plotly_node_order)
        ? meta.plotly_node_order.map((nodeId) => String(nodeId || ""))
        : [];
      const positions = Array.isArray(meta.plotly_positions) ? meta.plotly_positions : [];
      const sizes = Array.isArray(meta.plotly_node_sizes) ? meta.plotly_node_sizes : [];
      if (
        !order.length
        || order.some((nodeId) => !nodeId)
        || order.some((nodeId) => nodeId !== nodeId.trim())
        || new Set(order).size !== order.length
        || positions.length !== order.length
        || sizes.length !== order.length
      ) {
        return false;
      }
      const positionsAreFinite = positions.every((position) => (
        Array.isArray(position)
        && position.length === 2
        && position.every((coordinate) => Number.isFinite(coordinate))
      ));
      const sizesAreFinite = sizes.every(
        (size) => Number.isFinite(size) && size > 0
      );
      return positionsAreFinite && sizesAreFinite;
    }

    function buildFigureSpecFromPayload(nextPayload) {
      const meta = (nextPayload && nextPayload.meta) || {};
      const order = Array.isArray(meta.plotly_node_order)
        ? meta.plotly_node_order.map((nodeId) => String(nodeId || ""))
        : [];
      const positions = Array.isArray(meta.plotly_positions) ? meta.plotly_positions : [];
      const alignedNodeSizes = normalizeArray(meta.plotly_node_sizes, order.length, 8);
      const maxAlignedNodeSize = alignedNodeSizes.length
        ? Math.max(...alignedNodeSizes)
        : 1.0;
      const nextMarkerSizeRef = Math.max(
        (2.0 * maxAlignedNodeSize) / (__DASHBOARD_MAX_NODE_DIAMETER__ ** 2),
        1e-6
      );
      if (!order.length || positions.length !== order.length) {
        throw new Error(
          "JSON is missing dashboard layout positions. Re-export results with a newer CiteMesh build."
        );
      }

      const template = deepClone(baseFigureTemplate);
      const nextNodes = Array.isArray(nextPayload.nodes) ? nextPayload.nodes : [];
      const nextNodeById = new Map(
        nextNodes.map((node) => [String(node.id || ""), node])
      );
      const nextEdges = Array.isArray(nextPayload.edges) ? nextPayload.edges : [];
      const nextYearRange = meta.year_range || {};
      const nextTraceSpecs = Array.isArray(template.data) ? template.data : [];
      const nextNodeTraceIndex = (() => {
        const namedIdx = nextTraceSpecs.findIndex(
          (trace) => String(trace.name || "") === "nodes"
        );
        if (namedIdx >= 0) {
          return namedIdx;
        }
        const fallbackIdx = nextTraceSpecs.findIndex((trace) =>
          String(trace.mode || "").includes("markers+text")
        );
        return fallbackIdx >= 0 ? fallbackIdx : 0;
      })();
      const nextHaloTraceIndex = nextTraceSpecs.findIndex(
        (trace) => String(trace.name || "") === "selection-halo"
      );
      const nextNeighborhoodTraceIndex = nextTraceSpecs.findIndex(
        (trace) => String(trace.name || "") === "neighborhood-edges"
      );
      const templateNodeTrace = nextTraceSpecs[nextNodeTraceIndex] || {};
      const templateMarker = templateNodeTrace.marker || {};
      const templateLayout = template.layout || {};
      const xPairs = normalizeArray(
        positions.map((position) => Array.isArray(position) ? position[0] : 0),
        order.length,
        0
      );
      const yPairs = normalizeArray(
        positions.map((position) => Array.isArray(position) ? position[1] : 0),
        order.length,
        0
      );
      const yearMin = Number(nextYearRange.min || 0);
      const yearMax = Number(nextYearRange.max || 0);
      const safeYearMin = Number.isFinite(yearMin) ? yearMin : 0;
      const rawSafeYearMax = Number.isFinite(yearMax) ? yearMax : safeYearMin;
      const safeYearMax = rawSafeYearMax > safeYearMin
        ? rawSafeYearMax
        : safeYearMin + 1.0;
      const missingYear = (safeYearMin + safeYearMax) / 2.0;
      const labelIds = selectDashboardLabelIds(
        order,
        nextNodeById,
        xPairs,
        yPairs
      );
      const seedId = String(meta.seed_id || "");
      const seedRingColor = currentSeedRingColor();
      const edgeStrengths = normalizeDashboardEdgeStrengths(nextEdges);

      const nodeTexts = [];
      const hoverTexts = [];
      const nodeYears = [];
      const lineWidths = [];
      const lineColors = [];
      for (let idx = 0; idx < order.length; idx += 1) {
        const nodeId = order[idx];
        const node = nextNodeById.get(nodeId) || {};
        const label = labelIds.has(nodeId) ? dashboardNodeLabel(node, nodeId) : "";
        nodeTexts.push(label);
        const nodeYear = Number.isFinite(Number(node.year)) && Number(node.year) > 0
          ? Number(node.year)
          : missingYear;
        nodeYears.push(nodeYear);
        hoverTexts.push(dashboardHoverText(node, nodeId));
        if (node.is_seed) {
          lineWidths.push(4.0);
          lineColors.push(seedRingColor);
        } else {
          lineWidths.push(0.0);
          lineColors.push("rgba(0,0,0,0)");
        }
      }

      const xMin = Math.min(...xPairs);
      const xMax = Math.max(...xPairs);
      const yMin = Math.min(...yPairs);
      const yMax = Math.max(...yPairs);
      const xSpan = Math.max(xMax - xMin, 1e-6);
      const ySpan = Math.max(yMax - yMin, 1e-6);
      const xPad = Math.max(__DASHBOARD_AXIS_X_PADDING__, xSpan * 0.1);
      const yPad = Math.max(__DASHBOARD_AXIS_MIN_PADDING__, ySpan * 0.08);
      const dashboardEdgeColor = "__DASHBOARD_EDGE_COLOR__";
      const edgeShapes = [];
      nextEdges.forEach((edge, edgeIndex) => {
        const leftId = String(edge.source || "");
        const rightId = String(edge.target || "");
        const leftIdx = order.indexOf(leftId);
        const rightIdx = order.indexOf(rightId);
        if (leftIdx < 0 || rightIdx < 0) {
          return;
        }
        const x0 = xPairs[leftIdx];
        const y0 = yPairs[leftIdx];
        const x1 = xPairs[rightIdx];
        const y1 = yPairs[rightIdx];
        const midX = (x0 + x1) / 2.0;
        const midY = (y0 + y1) / 2.0;
        const dx = x1 - x0;
        const dy = y1 - y0;
        const direction = stableCurveDirection(leftId, rightId);
        const cx = midX - (dy * 0.15 * direction);
        const cy = midY + (dx * 0.15 * direction);
        const strength = edgeStrengths[edgeIndex];
        edgeShapes.push({
          type: "path",
          path: `M ${x0},${y0} Q ${cx},${cy} ${x1},${y1}`,
          line: {
            color: colorWithAlpha(dashboardEdgeColor, 0.07 + (0.25 * strength)),
            width: 0.45 + (1.2 * strength),
          },
          layer: "below",
        });
      });

      const nodeTrace = templateNodeTrace;
      nodeTrace.mode = "markers";
      nodeTrace.x = xPairs;
      nodeTrace.y = yPairs;
      nodeTrace.text = nodeTexts;
      nodeTrace.hovertext = hoverTexts;
      nodeTrace.hoverlabel = deepClone(HOVER_LABEL);
      nodeTrace.marker = Object.assign({}, templateMarker, {
        size: alignedNodeSizes,
        sizeref: nextMarkerSizeRef,
        color: nodeYears,
        cmin: safeYearMin,
        cmax: safeYearMax,
        line: {
          width: lineWidths,
          color: lineColors,
        },
      });
      nextTraceSpecs[nextNodeTraceIndex] = nodeTrace;

      if (nextHaloTraceIndex >= 0) {
        const seedIdx = order.indexOf(seedId);
        const haloTrace = nextTraceSpecs[nextHaloTraceIndex] || {};
        haloTrace.x = seedIdx >= 0 ? [xPairs[seedIdx]] : [];
        haloTrace.y = seedIdx >= 0 ? [yPairs[seedIdx]] : [];
        haloTrace.marker = Object.assign({}, haloTrace.marker || {}, {
          size: seedIdx >= 0 ? [alignedNodeSizes[seedIdx] * 2.05] : [],
          sizeref: nextMarkerSizeRef,
          color: seedIdx >= 0 ? [colorWithAlpha(seedRingColor, 0.26)] : [],
        });
        nextTraceSpecs[nextHaloTraceIndex] = haloTrace;
      }

      if (nextNeighborhoodTraceIndex >= 0) {
        const neighborhoodTrace = nextTraceSpecs[nextNeighborhoodTraceIndex] || {};
        neighborhoodTrace.x = [];
        neighborhoodTrace.y = [];
        nextTraceSpecs[nextNeighborhoodTraceIndex] = neighborhoodTrace;
      }

      const layout = Object.assign({}, templateLayout);
      layout.xaxis = Object.assign({}, layout.xaxis || {}, {
        autorange: false,
        range: [xMin - xPad, xMax + xPad],
      });
      layout.yaxis = Object.assign({}, layout.yaxis || {}, {
        autorange: false,
        range: [yMin - yPad, yMax + yPad],
      });
      layout.hoverlabel = deepClone(HOVER_LABEL);
      layout.shapes = edgeShapes;
      layout.annotations = nodeTexts.flatMap((text, idx) => text ? [{
        x: xPairs[idx],
        y: yPairs[idx],
        xref: "x",
        yref: "y",
        text,
        font: nodeTrace.textfont,
        showarrow: false,
        xanchor: "center",
        yanchor: "bottom",
        borderpad: 0,
        yshift: Math.max(
          4.0,
          Math.sqrt(alignedNodeSizes[idx] * __DASHBOARD_SELECTION_HALO_SCALE__ / (2.0 * nextMarkerSizeRef)),
          Math.max(4.0, Math.sqrt(alignedNodeSizes[idx] / (2.0 * nextMarkerSizeRef))) + lineWidths[idx] / 2.0
        ) + 3.0,
      }] : []);
      layout.uirevision = `citemesh-dashboard-static-layout-v1:${String(meta.strategy || "")}:${String(meta.seed_id || "")}`;

      template.data = nextTraceSpecs;
      template.layout = layout;
      return template;
    }

    function rebuildDerivedData() {
      nodes = Array.isArray(payload.nodes) ? payload.nodes : [];
      nodeOrder = Array.isArray(payload.meta && payload.meta.plotly_node_order)
        ? payload.meta.plotly_node_order.map((nodeId) => String(nodeId || ""))
        : [];
      nodeById = new Map(nodes.map((node) => [String(node.id || ""), node]));
      nodeIndexById = new Map(nodeOrder.map((nodeId, idx) => [nodeId, idx]));
      yearRange = (payload.meta && payload.meta.year_range) || {};
      seedNode = nodeById.get((payload.meta && payload.meta.seed_id) || "") || null;
      seedYear = seedNode && Number.isFinite(Number(seedNode.year)) && Number(seedNode.year) > 0
        ? Number(seedNode.year)
        : null;

      traceSpecs = figureSpec.data || [];
      nodeTraceIndex = (() => {
        const namedIdx = traceSpecs.findIndex((trace) => String(trace.name || "") === "nodes");
        if (namedIdx >= 0) {
          return namedIdx;
        }
        const fallbackIdx = traceSpecs.findIndex((trace) => String(trace.mode || "").includes("markers+text"));
        return fallbackIdx >= 0 ? fallbackIdx : 0;
      })();
      haloTraceIndex = traceSpecs.findIndex(
        (trace) => String(trace.name || "") === "selection-halo"
      );
      neighborhoodTraceIndex = traceSpecs.findIndex(
        (trace) => String(trace.name || "") === "neighborhood-edges"
      );
      const nodeTraceSource = (traceSpecs[nodeTraceIndex] || {});
      const markerSource = nodeTraceSource.marker || {};
      defaultNodeSizes = normalizeArray(markerSource.size, nodeOrder.length, 8);
      defaultNodeX = normalizeArray(nodeTraceSource.x, nodeOrder.length, 0);
      defaultNodeY = normalizeArray(nodeTraceSource.y, nodeOrder.length, 0);

      adjacency = new Map();
      (payload.edges || []).forEach((edge) => {
        const left = String(edge.source || "");
        const right = String(edge.target || "");
        const weight = Number(edge.weight || 0);
        if (!left || !right) {
          return;
        }
        if (!adjacency.has(left)) {
          adjacency.set(left, []);
        }
        if (!adjacency.has(right)) {
          adjacency.set(right, []);
        }
        adjacency.get(left).push({ id: right, weight });
        adjacency.get(right).push({ id: left, weight });
      });
    }

    rebuildDerivedData();

    // Full persisted reading list for the active result key, including IDs the
    // current graph no longer contains: the key survives rebuilds, so persisting
    // only the displayable subset would erase saves whenever a node drops out.
    let persistedSavedIds = new Set();

    const state = {
      selectedId: (payload.meta && payload.meta.seed_id) || null,
      hoverId: null,
      filters: { citation: true, semantic: true, both: true },
      searchText: "",
      sortKey: "relevance",
      scopeMode: "all",
      yearMin: null,
      yearMax: null,
      visibleIds: new Set(nodeOrder),
      savedOnly: false,
      savedIds: loadSavedIdSet(),
    };
    const overlayState = {
      neighborhoodKey: "",
      haloKey: "",
    };

    const controls = {
      list: document.getElementById("paper-list"),
      count: document.getElementById("paper-count"),
      search: document.getElementById("search-input"),
      sort: document.getElementById("sort-select"),
      yearMin: document.getElementById("year-min"),
      yearMax: document.getElementById("year-max"),
      clearSelection: document.getElementById("clear-selection"),
      chips: Array.from(document.querySelectorAll("#provenance-filters .chip")),
      scopeButtons: Array.from(document.querySelectorAll("#scope-nav [data-scope]")),
      filtersToggle: document.getElementById("filters-toggle"),
      listViewBtn: document.getElementById("list-view-btn"),
      moreBtn: document.getElementById("more-btn"),
      toolbar: document.getElementById("dashboard-toolbar"),
      detailMode: document.getElementById("detail-mode"),
      detailTitle: document.getElementById("detail-title"),
      detailSubtitle: document.getElementById("detail-subtitle"),
      detailMetrics: document.getElementById("detail-metrics"),
      detailCategories: document.getElementById("detail-categories"),
      detailLinks: document.getElementById("detail-links"),
      detailWhy: document.getElementById("detail-why-lines"),
      detailActions: document.getElementById("detail-actions"),
      detailAbstract: document.getElementById("detail-abstract"),
      graphHint: document.getElementById("graph-hint"),
      timelineYearMin: document.getElementById("timeline-year-min"),
      timelineYearMax: document.getElementById("timeline-year-max"),
      resultSelect: document.getElementById("result-select"),
      statusBanner: document.getElementById("dashboard-status"),
      savedChip: document.getElementById("saved-filter"),
      savedBibBtn: document.getElementById("export-saved-bib-btn"),
      copySavedBtn: document.getElementById("copy-saved-links-btn"),
    };

    function savedStorageKey() {
      const meta = (payload && payload.meta) || {};
      const strategy = String(meta.strategy || "default");
      const seedId = String(meta.seed_id || "default");
      return `citemesh-saved:${strategy}:${seedId}`;
    }

    function pruneSavedIdsForPayload(savedIds) {
      const availableIds = new Set(
        (payload.nodes || []).map((node) => String(node.id || "")).filter(Boolean)
      );
      return new Set(
        Array.from(savedIds).filter((nodeId) => availableIds.has(String(nodeId)))
      );
    }

    function loadPersistedSavedIds() {
      try {
        const raw = window.localStorage.getItem(savedStorageKey());
        const parsed = raw ? JSON.parse(raw) : [];
        return new Set(Array.isArray(parsed) ? parsed.map(String) : []);
      } catch (err) {
        return new Set();
      }
    }

    function loadSavedIdSet() {
      persistedSavedIds = loadPersistedSavedIds();
      return pruneSavedIdsForPayload(persistedSavedIds);
    }

    function persistSavedIds() {
      try {
        window.localStorage.setItem(savedStorageKey(), JSON.stringify(Array.from(persistedSavedIds)));
      } catch (err) {
        // Storage unavailable (strict privacy mode, some file:// contexts):
        // the reading list still works for the current session.
      }
    }

    function isSaved(nodeId) {
      return state.savedIds.has(String(nodeId || ""));
    }

    function savedNodes() {
      return (payload.nodes || []).filter((node) => state.savedIds.has(String(node.id || "")));
    }

    function updateSavedUi() {
      const count = state.savedIds.size;
      if (controls.savedChip) {
        controls.savedChip.textContent = count ? `Saved (${count})` : "Saved";
        controls.savedChip.classList.toggle("active", state.savedOnly);
      }
      const showExports = count > 0;
      if (controls.savedBibBtn) {
        controls.savedBibBtn.style.display = showExports ? "" : "none";
      }
      if (controls.copySavedBtn) {
        controls.copySavedBtn.style.display = showExports ? "" : "none";
      }
    }

    function refreshSavedState() {
      state.savedIds = loadSavedIdSet();
      state.savedOnly = false;
      updateSavedUi();
    }

    function toggleSaved(nodeId) {
      const key = String(nodeId || "");
      if (!key) {
        return;
      }
      if (state.savedIds.has(key)) {
        state.savedIds.delete(key);
        persistedSavedIds.delete(key);
      } else {
        state.savedIds.add(key);
        persistedSavedIds.add(key);
      }
      persistSavedIds();
      if (!state.savedIds.size) {
        state.savedOnly = false;
      }
      updateSavedUi();
      renderList();
      if (state.hoverId === key) {
        renderDetail(key, true);
      } else if (state.selectedId === key && !state.hoverId) {
        renderDetail(key, false);
      }
    }

    function clearDashboardStatus() {
      runtimeStatusMessage = "";
      runtimeStatusTone = "warning";
      if (!controls.statusBanner) {
        return;
      }
      controls.statusBanner.textContent = "";
      controls.statusBanner.classList.remove("visible", "warning", "info");
    }

    function setDashboardStatus(message, tone) {
      runtimeStatusMessage = String(message || "").trim();
      runtimeStatusTone = tone === "info" ? "info" : "warning";
      if (!controls.statusBanner) {
        return;
      }
      controls.statusBanner.textContent = runtimeStatusMessage;
      controls.statusBanner.classList.toggle("visible", !!runtimeStatusMessage);
      controls.statusBanner.classList.toggle(
        "warning",
        !!runtimeStatusMessage && runtimeStatusTone === "warning"
      );
      controls.statusBanner.classList.toggle(
        "info",
        !!runtimeStatusMessage && runtimeStatusTone === "info"
      );
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    function markdownLinkText(value) {
      // An unescaped bracket in a title closes the Markdown link label early.
      return String(value ?? "").replace(/([\[\]])/g, "\\$1");
    }

    function hasYear(node) {
      return Number.isFinite(Number(node.year)) && Number(node.year) > 0;
    }

    function nodeFilterClass(node) {
      const base = node.provenance_base || node.provenance || "citation";
      return state.filters[base] === true;
    }

    // Scope is a citation-direction question, not a chronology one: a paper the
    // seed references is prior work whatever year it carries, and a paper citing
    // the seed is derivative work even when it predates the seed. Publication
    // year only decides nodes the citation graph places in neither direction.
    const PRIOR_SEED_RELATIONS = new Set(["referenced_by_seed", "overlap"]);
    const DERIVATIVE_SEED_RELATIONS = new Set(["cites_seed", "overlap"]);

    function nodeScopeMatches(node) {
      if (state.scopeMode !== "prior" && state.scopeMode !== "derivative") {
        return true;
      }
      const relation = String(node.seed_relation || "").trim().toLowerCase();
      if (node.is_seed || relation === "seed") {
        return true;
      }
      const priorRelation = PRIOR_SEED_RELATIONS.has(relation);
      const derivativeRelation = DERIVATIVE_SEED_RELATIONS.has(relation);
      if (priorRelation || derivativeRelation) {
        return state.scopeMode === "prior" ? priorRelation : derivativeRelation;
      }
      // "semantic_only", the direction-less "citation" token, and an absent
      // relation all fall back to the seed year; an unusable year stays visible.
      if (seedYear === null || !hasYear(node)) {
        return true;
      }
      const nodeYear = Number(node.year);
      return state.scopeMode === "prior" ? nodeYear <= seedYear : nodeYear >= seedYear;
    }

    function nodeMatches(node) {
      if (state.savedOnly && !state.savedIds.has(String(node.id || ""))) {
        return false;
      }
      if (!nodeFilterClass(node)) {
        return false;
      }
      if (!nodeScopeMatches(node)) {
        return false;
      }
      if (state.yearMin !== null && (!hasYear(node) || Number(node.year) < state.yearMin)) {
        return false;
      }
      if (state.yearMax !== null && (!hasYear(node) || Number(node.year) > state.yearMax)) {
        return false;
      }
      if (!state.searchText) {
        return true;
      }
      const haystack = [
        node.title || "",
        Array.isArray(node.authors) ? node.authors.join(" ") : "",
        node.abstract || "",
      ].join(" ").toLowerCase();
      return haystack.includes(state.searchText);
    }

    function tieBreak(nodeA, nodeB) {
      const citationsA = Number(nodeA.citation_count || 0);
      const citationsB = Number(nodeB.citation_count || 0);
      if (citationsA !== citationsB) {
        return citationsB - citationsA;
      }
      const yearA = hasYear(nodeA) ? Number(nodeA.year) : -1;
      const yearB = hasYear(nodeB) ? Number(nodeB.year) : -1;
      if (yearA !== yearB) {
        return yearB - yearA;
      }
      return String(nodeA.id).localeCompare(String(nodeB.id));
    }

    function compareNodes(nodeA, nodeB) {
      if (!!nodeA.is_seed !== !!nodeB.is_seed) {
        return nodeA.is_seed ? -1 : 1;
      }

      if (state.sortKey === "title") {
        const titleCmp = String(nodeA.title || "").localeCompare(String(nodeB.title || ""));
        return titleCmp || tieBreak(nodeA, nodeB);
      }
      if (state.sortKey === "year") {
        const yearA = hasYear(nodeA) ? Number(nodeA.year) : -1;
        const yearB = hasYear(nodeB) ? Number(nodeB.year) : -1;
        if (yearA !== yearB) {
          return yearB - yearA;
        }
        return tieBreak(nodeA, nodeB);
      }
      if (state.sortKey === "citation_count") {
        const citationsA = Number(nodeA.citation_count || 0);
        const citationsB = Number(nodeB.citation_count || 0);
        if (citationsA !== citationsB) {
          return citationsB - citationsA;
        }
        return tieBreak(nodeA, nodeB);
      }

      const relevanceA = Number(nodeA.seed_relevance || 0);
      const relevanceB = Number(nodeB.seed_relevance || 0);
      if (relevanceA !== relevanceB) {
        return relevanceB - relevanceA;
      }
      return tieBreak(nodeA, nodeB);
    }

    function relationBadgeLabel(node) {
      const relation = String(node.seed_relation || "").trim();
      if (relation === "referenced_by_seed") {
        return "referenced by seed";
      }
      if (relation === "cites_seed") {
        return "cites seed";
      }
      if (relation === "semantic_only") {
        return "semantic-only";
      }
      if (relation === "overlap") {
        return "prior+derivative";
      }
      if (relation === "seed") {
        return "origin";
      }
      return String(node.provenance_base || node.provenance || "citation");
    }

    function filteredNodes() {
      const selected = nodes.filter(nodeMatches);
      selected.sort(compareNodes);
      return selected;
    }

    function safeExternalUrl(value) {
      const candidate = String(value || "").trim();
      if (!candidate) {
        return "";
      }
      try {
        const parsed = new URL(candidate);
        return parsed.protocol === "https:" || parsed.protocol === "http:"
          ? parsed.href
          : "";
      } catch (err) {
        return "";
      }
    }

    function detailLinkEntries(links) {
      const entries = [];
      const pdfUrl = safeExternalUrl(links && links.arxiv_pdf);
      const arxivUrl = safeExternalUrl(links && links.arxiv_abs);
      const doiUrl = safeExternalUrl(links && links.doi);
      const semanticScholarUrl = safeExternalUrl(links && links.semantic_scholar);
      if (pdfUrl) {
        entries.push({ kind: "pdf", title: "Open PDF", href: pdfUrl });
      }
      if (arxivUrl) {
        entries.push({ kind: "arxiv", title: "Open arXiv page", href: arxivUrl });
      }
      if (doiUrl) {
        entries.push({ kind: "doi", title: "Open DOI", href: doiUrl });
      }
      if (semanticScholarUrl) {
        entries.push({ kind: "s2", title: "Open Semantic Scholar", href: semanticScholarUrl });
      }
      return entries;
    }

    function linkIconSvg(kind) {
      if (kind === "pdf") {
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7z"></path><polyline points="14 2 14 7 19 7"></polyline><line x1="8" y1="12" x2="16" y2="12"></line><line x1="8" y1="16" x2="13" y2="16"></line></svg>';
      }
      if (kind === "arxiv") {
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 18L10 6l2 6 2-4 6 10"></path><circle cx="10" cy="6" r="1.2"></circle><circle cx="12" cy="12" r="1.2"></circle><circle cx="14" cy="8" r="1.2"></circle></svg>';
      }
      if (kind === "doi") {
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 7h6"></path><path d="M9 12h6"></path><path d="M9 17h6"></path><circle cx="6.5" cy="7" r="1"></circle><circle cx="6.5" cy="12" r="1"></circle><circle cx="6.5" cy="17" r="1"></circle><path d="M17.5 7a2.5 2.5 0 0 1 0 5"></path><path d="M17.5 12a2.5 2.5 0 0 0 0 5"></path></svg>';
      }
      if (kind === "s2") {
        return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h12"></path><path d="M6 12h9"></path><path d="M6 19h12"></path><path d="M17 10l2 2-2 2"></path></svg>';
      }
      return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4h16v16H4z"></path><path d="M8 8h8v8H8z"></path></svg>';
    }

    function detailLinksHtml(links) {
      const entries = detailLinkEntries(links);
      if (!entries.length) {
        return "";
      }
      return entries
        .map((entry) => `<a class="icon-link" href="${escapeHtml(entry.href)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(entry.title)}" aria-label="${escapeHtml(entry.title)}">${linkIconSvg(entry.kind)}</a>`)
        .join("");
    }

    function copyText(value) {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        return navigator.clipboard.writeText(value);
      }
      const fallback = document.createElement("textarea");
      fallback.value = value;
      document.body.appendChild(fallback);
      fallback.focus();
      fallback.select();
      try {
        document.execCommand("copy");
      } finally {
        fallback.remove();
      }
      return Promise.resolve();
    }

    function compactNodeLabel(nodeId) {
      const node = nodeById.get(nodeId);
      if (!node) {
        return String(nodeId);
      }
      if (Array.isArray(node.authors) && node.authors.length) {
        const surname = String(node.authors[0]).split(" ").filter(Boolean).slice(-1)[0] || "Unknown";
        const year = hasYear(node) ? String(node.year) : "n.d.";
        return `${surname}, ${year}`;
      }
      return String(node.title || node.id || nodeId);
    }

    function portableGraphPayload(nextPayload) {
      const nextNodes = Array.isArray(nextPayload.nodes) ? nextPayload.nodes : [];
      const nextEdges = Array.isArray(nextPayload.edges) ? nextPayload.edges : [];
      const nextMeta = (nextPayload && nextPayload.meta) || {};
      const nextNodeById = new Map(
        nextNodes.map((node) => [String(node.id || ""), node])
      );
      const portableNodeLabel = (nodeId) => {
        const node = nextNodeById.get(String(nodeId || ""));
        if (!node) {
          return String(nodeId || "");
        }
        if (Array.isArray(node.authors) && node.authors.length) {
          const surname = String(node.authors[0]).split(" ").filter(Boolean).slice(-1)[0] || "Unknown";
          const year = hasYear(node) ? String(node.year) : "n.d.";
          return `${surname}, ${year}`;
        }
        return String(node.title || node.id || nodeId);
      };
      const summary = {
        nodes: nextNodes.length,
        edges: nextEdges.length,
      };
      const dashboardMeta = Object.assign({}, nextMeta, {
        summary,
      });
      const edges = nextEdges.map((edge) => {
        const sourceId = String(edge.source || "");
        const targetId = String(edge.target || "");
        const sourceNode = nextNodeById.get(sourceId);
        const targetNode = nextNodeById.get(targetId);
        return {
          source: sourceId,
          target: targetId,
          source_title: sourceNode ? String(sourceNode.title || sourceId) : sourceId,
          target_title: targetNode ? String(targetNode.title || targetId) : targetId,
          source_label: portableNodeLabel(sourceId),
          target_label: portableNodeLabel(targetId),
          weight: Number(edge.weight || 0),
        };
      });
      return {
        kind: GRAPH_PAYLOAD_KIND,
        schema_version: GRAPH_PAYLOAD_SCHEMA_VERSION,
        seed_id: nextMeta.seed_id || "",
        meta: {
          strategy: nextMeta.strategy || "",
          year_range: nextMeta.year_range ?? null,
          candidate_source_status: nextMeta.candidate_source_status || {},
        },
        summary,
        dashboard: {
          meta: dashboardMeta,
        },
        nodes: nextNodes,
        edges,
      };
    }

    function buildPortableJsonPayload() {
      return portableGraphPayload(payload);
    }

    function currentResultIdForPayload(nextPayload) {
      const meta = (nextPayload && nextPayload.meta) || {};
      const seedId = String(meta.seed_id || "");
      const strategy = String(meta.strategy || "");
      if (!seedId || !strategy) {
        return null;
      }
      return `${strategy}:${seedId}`;
    }

    function emptyCollectionPackage() {
      return {
        kind: COLLECTION_KIND,
        schema_version: COLLECTION_SCHEMA_VERSION,
        current_result_id: null,
        results: [],
      };
    }

    function isObjectRecord(value) {
      return !!value && typeof value === "object" && !Array.isArray(value);
    }

    function isNonNegativeInteger(value) {
      return Number.isInteger(value) && value >= 0;
    }

    function normalizeImportedDashboardPayload(imported, label, allowLegacy) {
      if (!isObjectRecord(imported)) {
        throw new Error("Expected a CiteMesh graph object.");
      }
      const declaredKind = String(imported.kind || "");
      if (declaredKind) {
        if (declaredKind !== GRAPH_PAYLOAD_KIND) {
          throw new Error(`Unsupported result kind: ${declaredKind}.`);
        }
        if (imported.schema_version !== GRAPH_PAYLOAD_SCHEMA_VERSION) {
          throw new Error(
            `Unsupported ${GRAPH_PAYLOAD_KIND} schema version: ${String(imported.schema_version)}.`
          );
        }
      } else if (!allowLegacy) {
        throw new Error(`Result payload is missing kind=${GRAPH_PAYLOAD_KIND}.`);
      }

      if (!Array.isArray(imported.nodes) || !imported.nodes.length) {
        throw new Error("No nodes found in graph results.");
      }
      if (!Array.isArray(imported.edges)) {
        throw new Error("Graph results must contain an edges array.");
      }
      const importedNodes = imported.nodes;
      const importedEdges = imported.edges;
      const nodeIds = importedNodes.map((node) => String((node && node.id) || ""));
      if (nodeIds.some((nodeId) => !nodeId) || new Set(nodeIds).size !== nodeIds.length) {
        throw new Error("Graph result nodes must have unique, non-empty IDs.");
      }
      if (declaredKind && nodeIds.some((nodeId) => nodeId !== nodeId.trim())) {
        throw new Error("Versioned graph node IDs cannot contain surrounding whitespace.");
      }
      const nodeIdSet = new Set(nodeIds);
      if (importedEdges.some((edge) => (
        !isObjectRecord(edge)
        || !nodeIdSet.has(String(edge.source || ""))
        || !nodeIdSet.has(String(edge.target || ""))
      ))) {
        throw new Error("Graph result edges must reference included node IDs.");
      }
      if (declaredKind && importedEdges.some((edge) => (
        String(edge.source || "") !== String(edge.source || "").trim()
        || String(edge.target || "") !== String(edge.target || "").trim()
      ))) {
        throw new Error("Versioned graph edge IDs cannot contain surrounding whitespace.");
      }
      if (declaredKind) {
        if (!isObjectRecord(imported.summary)) {
          throw new Error("Versioned graph results must contain a summary object.");
        }
        if (
          !isNonNegativeInteger(imported.summary.nodes)
          || !isNonNegativeInteger(imported.summary.edges)
          || imported.summary.nodes !== importedNodes.length
          || imported.summary.edges !== importedEdges.length
        ) {
          throw new Error("Graph result summary does not match its node and edge arrays.");
        }
      }
      const importedDashboardMeta =
        imported && imported.dashboard && imported.dashboard.meta
          ? imported.dashboard.meta
          : {};
      const importedMeta =
        imported && imported.meta && typeof imported.meta === "object"
          ? imported.meta
          : {};
      const declaredSeedId = String(imported.seed_id || "");
      const declaredStrategy = String(importedMeta.strategy || "");
      if (declaredKind && (
        !declaredSeedId
        || declaredSeedId !== declaredSeedId.trim()
        || !isObjectRecord(imported.meta)
        || !declaredStrategy
        || declaredStrategy !== declaredStrategy.trim()
      )) {
        throw new Error(
          "Versioned graph results require canonical top-level seed_id and meta.strategy."
        );
      }
      if (declaredKind && (
        !isObjectRecord(importedDashboardMeta)
        || String(importedDashboardMeta.seed_id || "") !== declaredSeedId
        || String(importedDashboardMeta.strategy || "") !== declaredStrategy
        || !isObjectRecord(importedDashboardMeta.summary)
        || importedDashboardMeta.summary.nodes !== importedNodes.length
        || importedDashboardMeta.summary.edges !== importedEdges.length
      )) {
        throw new Error(
          "Versioned graph dashboard metadata must match its top-level identity and summary."
        );
      }
      const seedId = String(
        declaredSeedId
        || importedDashboardMeta.seed_id
        || importedMeta.seed_id
        || ((importedNodes.find((node) => !!(node && node.is_seed)) || {}).id || "")
      );
      const strategy = declaredKind
        ? declaredStrategy
        : String(importedDashboardMeta.strategy || importedMeta.strategy || "");
      if (!seedId || !nodeIds.includes(seedId)) {
        throw new Error("Graph results must identify a seed node present in nodes.");
      }
      if (!strategy) {
        throw new Error("Graph results must identify the build strategy.");
      }
      const baseMeta = {
        seed_id: seedId,
        strategy,
        theme: String(
          (payload.meta && payload.meta.theme)
          || importedDashboardMeta.theme
          || importedMeta.theme
          || "light"
        ),
        summary: {
          nodes: importedNodes.length,
          edges: importedEdges.length,
        },
        year_range: importedDashboardMeta.year_range ?? importedMeta.year_range ?? null,
        candidate_source_status:
          importedDashboardMeta.candidate_source_status
          || importedMeta.candidate_source_status
          || {},
        plotly_node_order:
          importedDashboardMeta.plotly_node_order || importedMeta.plotly_node_order || [],
        plotly_positions:
          importedDashboardMeta.plotly_positions || importedMeta.plotly_positions || [],
        plotly_node_sizes:
          importedDashboardMeta.plotly_node_sizes || importedMeta.plotly_node_sizes || [],
      };
      if (!hasCompleteDashboardGeometry(baseMeta)) {
        throw new Error(
          `Imported results from ${String(label || "the selected file")} are missing stored dashboard geometry. Export dashboard-compatible CiteMesh results first.`
        );
      }
      const geometryOrder = baseMeta.plotly_node_order.map((nodeId) => String(nodeId || ""));
      if (
        geometryOrder.length !== nodeIds.length
        || new Set(geometryOrder).size !== geometryOrder.length
        || geometryOrder.some((nodeId) => !nodeIds.includes(nodeId))
      ) {
        throw new Error("Dashboard geometry must cover each graph node exactly once.");
      }

      return {
        payload: {
          meta: baseMeta,
          nodes: importedNodes,
          edges: imported.edges,
        },
      };
    }

    function collectionEntryFromGraphPayload(imported, label, allowLegacy) {
      const normalized = normalizeImportedDashboardPayload(imported, label, allowLegacy);
      const normalizedPayload = normalized.payload;
      const resultId = currentResultIdForPayload(normalizedPayload);
      if (!resultId) {
        throw new Error("Graph results do not provide a stable strategy and seed ID.");
      }
      const seedId = String(normalizedPayload.meta.seed_id || "");
      const seedNode = normalizedPayload.nodes.find(
        (node) => String((node && node.id) || "") === seedId
      );
      return {
        result_id: resultId,
        seed_id: seedId,
        title: String((seedNode && seedNode.title) || seedId || label || "Graph result"),
        strategy: String(normalizedPayload.meta.strategy || ""),
        summary: {
          nodes: normalizedPayload.nodes.length,
          edges: normalizedPayload.edges.length,
        },
        payload: normalizedPayload,
        updated_at: new Date().toISOString(),
        build: {},
      };
    }

    function normalizeCollectionPackage(imported, label, allowLegacy) {
      if (!isObjectRecord(imported)) {
        throw new Error("Expected a CiteMesh dashboard collection object.");
      }
      const declaredKind = String(imported.kind || "");
      const isVersionedCollection = declaredKind === COLLECTION_KIND;
      if (declaredKind && !isVersionedCollection) {
        throw new Error(`Unsupported collection kind: ${declaredKind}.`);
      }
      if (isVersionedCollection && imported.schema_version !== COLLECTION_SCHEMA_VERSION) {
        throw new Error(
          `Unsupported ${COLLECTION_KIND} schema version: ${String(imported.schema_version)}.`
        );
      }
      if (!isVersionedCollection && !allowLegacy) {
        throw new Error(`Collection package is missing kind=${COLLECTION_KIND}.`);
      }
      if (!Array.isArray(imported.results)) {
        throw new Error("Collection package must contain a results array.");
      }

      const normalized = emptyCollectionPackage();
      const legacyPayloads = isObjectRecord(imported.payloads) ? imported.payloads : {};
      imported.results.forEach((rawEntry, index) => {
        if (!isObjectRecord(rawEntry)) {
          throw new Error(`Collection result ${index + 1} must be an object.`);
        }
        const declaredResultId = String(rawEntry.result_id || "").trim();
        const rawPayload = isObjectRecord(rawEntry.payload)
          ? rawEntry.payload
          : legacyPayloads[declaredResultId];
        if (!isObjectRecord(rawPayload)) {
          throw new Error(`Collection result ${index + 1} is missing its graph payload.`);
        }
        const normalizedEntry = collectionEntryFromGraphPayload(
          rawPayload,
          `${label || "collection"} result ${index + 1}`,
          allowLegacy
        );
        if (declaredResultId && declaredResultId !== normalizedEntry.result_id) {
          throw new Error(
            `Collection result ${index + 1} ID does not match its graph strategy and seed.`
          );
        }
        if (!declaredResultId && isVersionedCollection) {
          throw new Error(`Collection result ${index + 1} is missing result_id.`);
        }
        if (isVersionedCollection) {
          if (!String(rawEntry.title || "").trim()) {
            throw new Error(`Collection result ${index + 1} is missing title.`);
          }
          if (!String(rawEntry.updated_at || "").trim()) {
            throw new Error(`Collection result ${index + 1} is missing updated_at.`);
          }
          if (!isObjectRecord(rawEntry.summary)) {
            throw new Error(`Collection result ${index + 1} is missing summary.`);
          }
          if (
            !isNonNegativeInteger(rawEntry.summary.nodes)
            || !isNonNegativeInteger(rawEntry.summary.edges)
            || rawEntry.summary.nodes !== normalizedEntry.summary.nodes
            || rawEntry.summary.edges !== normalizedEntry.summary.edges
          ) {
            throw new Error(`Collection result ${index + 1} has an inconsistent summary.`);
          }
          if (!isObjectRecord(rawEntry.build)) {
            throw new Error(`Collection result ${index + 1} is missing build metadata.`);
          }
        }
        for (const field of ["seed_id", "strategy"]) {
          if (
            rawEntry[field] !== undefined
            && String(rawEntry[field]) !== String(normalizedEntry[field])
          ) {
            throw new Error(`Collection result ${index + 1} has inconsistent ${field}.`);
          }
        }
        if (rawEntry.title !== undefined) {
          normalizedEntry.title = String(rawEntry.title || normalizedEntry.title);
        }
        if (rawEntry.updated_at !== undefined) {
          normalizedEntry.updated_at = String(rawEntry.updated_at || "");
        }
        if (rawEntry.build !== undefined) {
          if (!isObjectRecord(rawEntry.build)) {
            throw new Error(`Collection result ${index + 1} build metadata must be an object.`);
          }
          normalizedEntry.build = rawEntry.build;
        }
        const duplicateIndex = normalized.results.findIndex(
          (entry) => entry.result_id === normalizedEntry.result_id
        );
        if (duplicateIndex < 0) {
          normalized.results.push(normalizedEntry);
        }
      });

      const currentId = String(imported.current_result_id || "").trim();
      if (currentId && !normalized.results.some((entry) => entry.result_id === currentId)) {
        throw new Error("Collection current_result_id does not name an included result.");
      }
      normalized.current_result_id = currentId || (
        normalized.results.length ? normalized.results[0].result_id : null
      );
      return normalized;
    }

    function upsertCollectionEntries(targetCollection, incomingEntries) {
      const incomingUnique = [];
      const incomingIds = new Set();
      incomingEntries.forEach((incomingEntry) => {
        const resultId = String(incomingEntry.result_id || "");
        if (resultId && !incomingIds.has(resultId)) {
          incomingIds.add(resultId);
          incomingUnique.push(incomingEntry);
        }
      });
      const retained = targetCollection.results.filter(
        (entry) => !incomingIds.has(String(entry.result_id || ""))
      );
      targetCollection.results = incomingUnique.concat(retained);
    }

    function portableCollectionPackage() {
      return {
        kind: COLLECTION_KIND,
        schema_version: COLLECTION_SCHEMA_VERSION,
        current_result_id: collectionResultId || currentResultIdForPayload(payload),
        results: collectionEntries().map((entry) => {
          const portableEntry = {
            result_id: entry.result_id,
            seed_id: entry.seed_id,
            title: entry.title,
            strategy: entry.strategy,
            summary: entry.summary,
            payload: portableGraphPayload(entry.payload),
            updated_at: entry.updated_at || new Date().toISOString(),
            build: isObjectRecord(entry.build) ? entry.build : {},
          };
          return portableEntry;
        }),
      };
    }

    function collectionEntryLabel(entry) {
      const title = String(entry.title || entry.seed_id || entry.result_id || "Graph result");
      const strategy = String(entry.strategy || "");
      const summary = entry.summary || {};
      const nodeCount = Number(summary.nodes || 0);
      const edgeCount = Number(summary.edges || 0);
      const strategyLabel = strategy ? ` [${strategy}]` : "";
      return `${title}${strategyLabel} • ${nodeCount} papers / ${edgeCount} links`;
    }

    function collectionEntries() {
      return Array.isArray(collectionBundle.results) ? collectionBundle.results : [];
    }

    function populateCollectionSelector() {
      const select = controls.resultSelect;
      if (!select) {
        return;
      }
      const results = collectionEntries();
      const currentId = collectionResultId || currentResultIdForPayload(payload);
      select.innerHTML = "";
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "Select a graph";
      placeholder.disabled = true;
      select.appendChild(placeholder);

      results.forEach((entry) => {
        const option = document.createElement("option");
        option.value = String(entry.result_id || "");
        option.textContent = collectionEntryLabel(entry);
        select.appendChild(option);
      });

      if (currentId && results.some((entry) => String(entry.result_id || "") === currentId)) {
        select.value = currentId;
      } else {
        select.value = "";
      }
      select.disabled = results.length <= 1;
    }

    function updateYearPlaceholders() {
      const validYears = nodes
        .map((node) => (hasYear(node) ? Number(node.year) : null))
        .filter((year) => year !== null);
      if (!validYears.length) {
        controls.yearMin.placeholder = "Year min";
        controls.yearMax.placeholder = "Year max";
        return;
      }
      const minYear = Math.min(...validYears);
      const maxYear = Math.max(...validYears);
      controls.yearMin.placeholder = `Year min (${minYear})`;
      controls.yearMax.placeholder = `Year max (${maxYear})`;
    }

    function applyImportedPayload(imported, label) {
      const normalizedImport = normalizeImportedDashboardPayload(imported, label, true);
      const nextPayload = normalizedImport.payload;
      const nextFigureSpec = buildFigureSpecFromPayload(nextPayload);
      payload = nextPayload;
      figureSpec = nextFigureSpec;
      clearDashboardStatus();
      collectionResultId = currentResultIdForPayload(nextPayload);
      rebuildDerivedData();
      refreshSavedState();
      overlayState.neighborhoodKey = "";
      overlayState.haloKey = "";
      state.selectedId = (payload.meta && payload.meta.seed_id) || null;
      state.hoverId = null;
      state.searchText = "";
      state.sortKey = "relevance";
      state.scopeMode = "all";
      state.yearMin = null;
      state.yearMax = null;
      state.visibleIds = new Set(nodeOrder);
      controls.search.value = "";
      controls.sort.value = "relevance";
      controls.yearMin.value = "";
      controls.yearMax.value = "";
      controls.scopeButtons.forEach((entry) => entry.classList.remove("active"));
      controls.chips.forEach((chip) => {
        const filterKey = chip.getAttribute("data-filter");
        if (!filterKey) {
          return;
        }
        state.filters[filterKey] = true;
        chip.classList.add("active");
      });
      populateCollectionSelector();
      updateYearPlaceholders();
      renderTimeline();
      return Plotly.react(graphDiv, figureSpec.data, figureSpec.layout, plotConfig).then(() => {
        setupGraphInteractions();
        if (state.selectedId && nodeById.has(state.selectedId)) {
          renderDetail(state.selectedId, false);
        } else {
          renderDetail(null, false);
        }
        renderList();
        const loadedTitle = nodeById.get(state.selectedId || "");
        controls.graphHint.textContent = "Loaded: " + (loadedTitle ? loadedTitle.title : label);
      });
    }

    function loadCollectionResult(resultId) {
      const normalizedId = String(resultId || "").trim();
      if (!normalizedId) {
        return Promise.resolve();
      }
      const entry = collectionEntries().find(
        (candidate) => String(candidate.result_id || "") === normalizedId
      );
      if (!entry || !entry.payload) {
        setDashboardStatus(
          "That graph is unavailable in the current result set. Add its package again.",
          "warning"
        );
        return Promise.resolve();
      }
      collectionResultId = normalizedId;
      clearDashboardStatus();
      return applyImportedPayload(
        entry.payload,
        collectionEntryLabel(entry)
      );
    }

    function shortestPathIds(sourceId, targetId) {
      if (!sourceId || !targetId || sourceId === targetId) {
        return sourceId && targetId ? [sourceId] : [];
      }
      const queue = [sourceId];
      const visited = new Set([sourceId]);
      const previous = new Map();

      while (queue.length) {
        const current = queue.shift();
        const neighbors = adjacency.get(current) || [];
        for (const entry of neighbors) {
          const nextId = entry.id;
          if (visited.has(nextId)) {
            continue;
          }
          visited.add(nextId);
          previous.set(nextId, current);
          if (nextId === targetId) {
            const path = [targetId];
            let cursor = targetId;
            while (previous.has(cursor)) {
              cursor = previous.get(cursor);
              path.push(cursor);
              if (cursor === sourceId) {
                break;
              }
            }
            path.reverse();
            return path;
          }
          queue.push(nextId);
        }
      }
      return [];
    }

    function renderWhyLines(nodeId) {
      const node = nodeId ? nodeById.get(nodeId) : null;
      if (!node) {
        controls.detailWhy.textContent = "Select a paper to inspect neighborhood evidence.";
        controls.detailWhy.classList.add("muted");
        return;
      }

      // Share the list badge's label map so one paper never reads
      // "semantic-only" in the list and "semantic_only" in this card.
      const relationLabel = relationBadgeLabel(node);
      const neighbors = (adjacency.get(node.id) || [])
        .slice()
        .sort((left, right) => Number(right.weight || 0) - Number(left.weight || 0))
        .slice(0, 3);
      const neighborLine = neighbors.length
        ? `Top links: ${neighbors.map((entry) => `${compactNodeLabel(entry.id)} (w=${Number(entry.weight || 0).toFixed(2)})`).join("; ")}`
        : "Top links: none";
      if (node.is_seed) {
        controls.detailWhy.classList.remove("muted");
        controls.detailWhy.innerHTML = [
          `<div>${escapeHtml("Seed paper - every other node in this graph was gathered around it.")}</div>`,
          `<div>${escapeHtml(neighborLine)}</div>`,
        ].join("");
        return;
      }
      const seedId = (payload.meta && payload.meta.seed_id) || null;
      const path = seedId ? shortestPathIds(seedId, node.id) : [];
      const pathLine = path.length
        ? `Shortest path to seed: ${path.map((pid) => compactNodeLabel(pid)).join(" -> ")}`
        : "Shortest path to seed: unavailable";
      const relevanceLine = `Seed relevance: ${Number(node.seed_relevance || 0).toFixed(4)} | Relation: ${relationLabel}`;

      controls.detailWhy.classList.remove("muted");
      controls.detailWhy.innerHTML = [
        `<div>${escapeHtml(relevanceLine)}</div>`,
        `<div>${escapeHtml(neighborLine)}</div>`,
        `<div>${escapeHtml(pathLine)}</div>`,
      ].join("");
    }

    function renderDetail(nodeId, previewOnly) {
      const node = nodeId ? nodeById.get(nodeId) : null;
      if (!node) {
        controls.detailMode.textContent = "No selection";
        controls.detailTitle.textContent = "Select a paper";
        controls.detailSubtitle.textContent = "";
        controls.detailMetrics.innerHTML = "";
        controls.detailCategories.innerHTML = "";
        controls.detailLinks.innerHTML = "";
        controls.detailWhy.textContent = "Select a paper to inspect neighborhood evidence.";
        controls.detailWhy.classList.add("muted");
        controls.detailActions.innerHTML = "";
        controls.detailAbstract.textContent = "Hover or click a paper to inspect abstract and metadata.";
        controls.detailAbstract.classList.add("muted");
        controls.graphHint.textContent = "Hover to preview, click to lock";
        return;
      }

      controls.detailMode.textContent = previewOnly ? "Preview" : "Selected";
      if (previewOnly && state.selectedId && state.selectedId !== node.id) {
        controls.graphHint.textContent = "Previewing node (selection locked)";
      } else if (previewOnly) {
        controls.graphHint.textContent = "Previewing node • arcs show strongest direct links";
      } else {
        controls.graphHint.textContent = "Selection locked • red arcs are strongest direct links";
      }
      controls.detailTitle.textContent = node.title || node.id;
      let authorText = "Unknown authors";
      if (Array.isArray(node.authors) && node.authors.length > 0) {
        if (node.authors.length > 2) {
          authorText = `${node.authors[0]} + ${node.authors.length - 1} authors`;
        } else {
          authorText = node.authors.join(", ");
        }
      }
      const yearText = hasYear(node) ? String(node.year) : "n.d.";
      const venueText = node.venue ? `, ${node.venue}` : "";
      controls.detailSubtitle.textContent = `${authorText} | ${yearText}${venueText}`;

      const provenance = node.provenance || "unknown";
      const provenanceLabel = provenance === "seed" ? `seed (${node.provenance_base || "citation"})` : provenance;
      const metrics = [
        `Citations: ${Number(node.citation_count || 0).toLocaleString()}`,
        `Source: ${provenanceLabel}`,
        `Year: ${yearText}`,
      ];
      controls.detailMetrics.innerHTML = metrics
        .map((metric) => `<span class="metric-pill">${escapeHtml(metric)}</span>`)
        .join("");

      const categories = Array.isArray(node.categories) ? node.categories.filter(Boolean).slice(0, 8) : [];
      controls.detailCategories.innerHTML = categories
        .map((category) => `<span class="category-chip">${escapeHtml(category)}</span>`)
        .join("");

      controls.detailLinks.innerHTML = detailLinksHtml(node.links || {});
      renderWhyLines(node.id);
      controls.detailAbstract.textContent = node.abstract || "No abstract available for this record.";
      controls.detailAbstract.classList.toggle("muted", !node.abstract);

      controls.detailActions.innerHTML = "";
      const saveBtn = document.createElement("button");
      saveBtn.type = "button";
      saveBtn.textContent = isSaved(node.id) ? "★ Saved" : "☆ Save";
      saveBtn.title = isSaved(node.id)
        ? "Remove from reading list"
        : "Save to reading list";
      saveBtn.addEventListener("click", () => {
        toggleSaved(node.id);
      });
      controls.detailActions.appendChild(saveBtn);

      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.textContent = "Copy BibTeX";
      copyBtn.addEventListener("click", () => {
        copyText(node.bibtex || "").then(() => {
          copyBtn.textContent = "Copied";
          window.setTimeout(() => {
            copyBtn.textContent = "Copy BibTeX";
          }, 1000);
        });
      });

      const downloadBtn = document.createElement("button");
      downloadBtn.type = "button";
      downloadBtn.textContent = "Download BibTeX";
      downloadBtn.addEventListener("click", () => {
        const blob = new Blob([node.bibtex || ""], { type: "text/plain;charset=utf-8" });
        const anchor = document.createElement("a");
        anchor.href = URL.createObjectURL(blob);
        anchor.download = `${String(node.id || "paper").replace(/[^a-zA-Z0-9._-]+/g, "_")}.bib`;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(anchor.href);
      });

      controls.detailActions.appendChild(copyBtn);
      controls.detailActions.appendChild(downloadBtn);
    }

    function syncRowHighlights() {
      const rows = Array.from(document.querySelectorAll(".paper-row"));
      rows.forEach((row) => {
        const rowId = row.getAttribute("data-node-id");
        row.classList.toggle("is-hover", !!state.hoverId && rowId === state.hoverId);
        row.classList.toggle("is-selected", !!state.selectedId && rowId === state.selectedId);
      });
    }

    function getNodePointPaths() {
      const traceGroups = graphDiv.querySelectorAll(".scatterlayer .trace");
      if (!traceGroups || !traceGroups.length) {
        return [];
      }
      const traceGroup = traceGroups[nodeTraceIndex] || traceGroups[traceGroups.length - 1];
      return Array.from(traceGroup.querySelectorAll("path.point"));
    }

    function syncGraphHighlights() {
      if (!(window.Plotly && graphDiv && graphDiv.data && graphDiv.data.length > nodeTraceIndex)) {
        return;
      }
      const graphNodePaths = getNodePointPaths();
      const hoverIndex = state.hoverId && nodeIndexById.has(state.hoverId)
        ? nodeIndexById.get(state.hoverId)
        : -1;
      const selectedIndex = state.selectedId && nodeIndexById.has(state.selectedId)
        ? nodeIndexById.get(state.selectedId)
        : -1;
      const hasActiveFocus = hoverIndex !== -1 || selectedIndex !== -1;
      const focusId = state.hoverId || state.selectedId;
      const neighborIds = new Set(
        focusId && adjacency.has(focusId)
          ? (adjacency.get(focusId) || []).map((entry) => entry.id)
          : []
      );

      graphNodePaths.forEach((path, idx) => {
        const rawPointIndex = path.getAttribute("data-point-number");
        const pointIndex = Number.parseInt(rawPointIndex ?? "", 10);
        const stableIdx = Number.isInteger(pointIndex) && pointIndex >= 0 ? pointIndex : idx;
        const nodeId = nodeOrder[stableIdx];
        const isVisible = !!nodeId && state.visibleIds.has(nodeId);
        const isTarget = stableIdx === hoverIndex || stableIdx === selectedIndex;
        const isNeighbor = !!nodeId && neighborIds.has(nodeId) && !isTarget;
        path.classList.toggle("is-filter-hidden", !isVisible);
        path.classList.toggle("is-neighbor", hasActiveFocus && isNeighbor);
        path.classList.toggle("is-dimmed", hasActiveFocus && !isTarget && !isNeighbor);
        path.classList.toggle("is-glowing", isTarget);
      });

      if (neighborhoodTraceIndex >= 0) {
        let neighborhoodX = [];
        let neighborhoodY = [];
        let topNeighbors = [];
        if (focusId && adjacency.has(focusId)) {
          topNeighbors = (adjacency.get(focusId) || [])
            .filter((entry) => state.visibleIds.has(entry.id))
            .sort((left, right) => Number(right.weight || 0) - Number(left.weight || 0))
            .slice(0, 6);
          for (const entry of topNeighbors) {
            if (!nodeIndexById.has(entry.id) || !nodeIndexById.has(focusId)) {
              continue;
            }
            const leftIdx = nodeIndexById.get(focusId);
            const rightIdx = nodeIndexById.get(entry.id);
            neighborhoodX.push(defaultNodeX[leftIdx], defaultNodeX[rightIdx], null);
            neighborhoodY.push(defaultNodeY[leftIdx], defaultNodeY[rightIdx], null);
          }
        }
        const neighborhoodKey = `${focusId || ""}|${topNeighbors.map((entry) => entry.id).join(",")}`;
        if (overlayState.neighborhoodKey !== neighborhoodKey) {
          overlayState.neighborhoodKey = neighborhoodKey;
          Plotly.restyle(
            graphDiv,
            {
              x: [neighborhoodX],
              y: [neighborhoodY],
            },
            [neighborhoodTraceIndex]
          );
        }
      }

      if (haloTraceIndex >= 0) {
        let haloX = [];
        let haloY = [];
        let haloSize = [];
        let haloColor = [];
        if (focusId && nodeIndexById.has(focusId)) {
          const idx = nodeIndexById.get(focusId);
          haloX = [defaultNodeX[idx]];
          haloY = [defaultNodeY[idx]];
          haloSize = [defaultNodeSizes[idx] * (state.selectedId ? __DASHBOARD_SELECTION_HALO_SCALE__ : 1.88)];
          haloColor = [colorWithAlpha(currentSeedRingColor(), state.selectedId ? 0.34 : 0.26)];
        }
        const haloKey = `${focusId || ""}|${state.selectedId ? "selected" : "hover"}`;
        if (overlayState.haloKey !== haloKey) {
          overlayState.haloKey = haloKey;
          Plotly.restyle(
            graphDiv,
            {
              x: [haloX],
              y: [haloY],
              "marker.size": [haloSize],
              "marker.color": [haloColor],
            },
            [haloTraceIndex]
          );
        }
      }
    }

    function syncHighlights() {
      syncRowHighlights();
      syncGraphHighlights();
    }

    function semanticScholarTarget() {
      const focusId = state.selectedId || ((payload.meta && payload.meta.seed_id) || null);
      const focusNode = focusId ? nodeById.get(focusId) : null;
      return safeExternalUrl(
        focusNode && focusNode.links && focusNode.links.semantic_scholar
      );
    }

    function syncSemanticScholarButton() {
      if (!controls.moreBtn) {
        return;
      }
      if (semanticScholarTarget()) {
        controls.moreBtn.removeAttribute("disabled");
      } else {
        controls.moreBtn.setAttribute("disabled", "");
      }
    }

    // Rebuilding the list drops the row under the cursor and inserts a fresh one,
    // which fires mouseenter with the pointer stationary. Previews stay disarmed
    // until the pointer really moves, so a click leaves the panel on "Selected".
    let listPreviewArmed = false;

    function armListPreview() {
      listPreviewArmed = true;
    }

    function previewRowOnHover(nodeId) {
      if (!listPreviewArmed) {
        return;
      }
      state.hoverId = nodeId;
      // Hovering the locked row must never demote the panel back to a preview.
      renderDetail(nodeId, nodeId !== state.selectedId);
      syncHighlights();
    }

    function renderList() {
      const listNodes = filteredNodes();
      listPreviewArmed = false;
      syncSemanticScholarButton();
      controls.count.textContent = `${listNodes.length.toLocaleString()} ${listNodes.length === 1 ? "paper" : "papers"}`;
      controls.list.innerHTML = "";
      state.visibleIds = new Set(listNodes.map((node) => node.id));
      if (state.selectedId && nodeById.has(state.selectedId)) {
        state.visibleIds.add(state.selectedId);
      }

      if (!listNodes.length) {
        const empty = document.createElement("li");
        empty.className = "paper-row";
        empty.innerHTML = '<div class="paper-title">No papers match current filters.</div>';
        controls.list.appendChild(empty);
        syncHighlights();
        return;
      }

      listNodes.forEach((node) => {
        const row = document.createElement("li");
        row.className = "paper-row";
        row.setAttribute("data-node-id", node.id);

        const yearText = hasYear(node) ? String(node.year) : "n.d.";
        const authors = Array.isArray(node.authors) && node.authors.length
          ? node.authors.slice(0, 4).join(", ")
          : "Unknown authors";
        const provenance = relationBadgeLabel(node);
        const provenanceClass = node.is_seed ? "meta-origin" : "";
        const provenanceLabel = provenance;

        const saved = isSaved(node.id);
        row.innerHTML = `
          <div class="paper-row-head">
            <div class="paper-title">${escapeHtml(node.title || node.id)}</div>
            <button class="star-btn${saved ? " saved" : ""}" type="button" title="${saved ? "Remove from reading list" : "Save to reading list"}" aria-label="${saved ? "Remove from reading list" : "Save to reading list"}" aria-pressed="${saved ? "true" : "false"}">${saved ? "&#9733;" : "&#9734;"}</button>
            <div class="paper-year">${escapeHtml(yearText)}</div>
          </div>
          <div class="paper-subline">${escapeHtml(authors)}</div>
          <div class="paper-meta">
            <span>${Number(node.citation_count || 0).toLocaleString()} citations</span>
            <span class="meta-dot"></span>
            <span class="${provenanceClass}">${escapeHtml(provenanceLabel)}</span>
          </div>
        `;

        const starBtn = row.querySelector(".star-btn");
        if (starBtn) {
          starBtn.addEventListener("click", (event) => {
            event.stopPropagation();
            toggleSaved(node.id);
          });
        }

        row.addEventListener("mouseenter", () => {
          previewRowOnHover(node.id);
        });
        row.addEventListener("mouseleave", () => {
          state.hoverId = null;
          if (state.selectedId && nodeById.has(state.selectedId)) {
            renderDetail(state.selectedId, false);
          } else {
            renderDetail(null, false);
          }
          syncHighlights();
        });
        row.addEventListener("click", () => {
          state.selectedId = node.id;
          renderDetail(node.id, false);
          syncHighlights();
          renderList();
        });
        controls.list.appendChild(row);
      });

      syncHighlights();
    }

    function setControlsCollapsed(collapsed) {
      controls.toolbar.classList.toggle("collapsed", collapsed);
      controls.filtersToggle.classList.toggle("active", !collapsed);
    }

    function setupControls() {
      // The list element survives every rebuild, so one listener re-arms previews
      // as soon as the pointer moves after a re-render.
      controls.list.addEventListener("mousemove", armListPreview);
      controls.search.addEventListener("input", (event) => {
        state.searchText = String(event.target.value || "").trim().toLowerCase();
        renderList();
      });
      controls.sort.addEventListener("change", (event) => {
        state.sortKey = String(event.target.value || "relevance");
        renderList();
      });
      controls.yearMin.addEventListener("input", (event) => {
        const value = String(event.target.value || "").trim();
        const parsed = Number(value);
        state.yearMin = value === "" || !Number.isFinite(parsed) ? null : parsed;
        renderList();
      });
      controls.yearMax.addEventListener("input", (event) => {
        const value = String(event.target.value || "").trim();
        const parsed = Number(value);
        state.yearMax = value === "" || !Number.isFinite(parsed) ? null : parsed;
        renderList();
      });
      controls.clearSelection.addEventListener("click", () => {
        state.selectedId = null;
        if (state.hoverId && nodeById.has(state.hoverId)) {
          renderDetail(state.hoverId, true);
        } else {
          renderDetail(null, false);
        }
        syncHighlights();
        renderList();
      });

      controls.chips.forEach((chip) => {
        chip.addEventListener("click", () => {
          const filterKey = chip.getAttribute("data-filter");
          if (!filterKey) {
            return;
          }
          state.filters[filterKey] = !state.filters[filterKey];
          chip.classList.toggle("active", state.filters[filterKey]);
          renderList();
        });
      });

      controls.scopeButtons.forEach((button) => {
        button.addEventListener("click", () => {
          const scope = button.getAttribute("data-scope");
          if (!scope) {
            return;
          }
          state.scopeMode = state.scopeMode === scope ? "all" : scope;
          controls.scopeButtons.forEach((entry) => {
            const entryScope = entry.getAttribute("data-scope");
            entry.classList.toggle("active", !!entryScope && entryScope === state.scopeMode);
          });
          renderList();
        });
      });

      controls.filtersToggle.addEventListener("click", () => {
        const collapsed = !controls.toolbar.classList.contains("collapsed");
        setControlsCollapsed(collapsed);
      });
      controls.listViewBtn.addEventListener("click", () => {
        const listPane = document.getElementById("paper-list-pane");
        if (listPane) {
          listPane.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      });
      controls.moreBtn.addEventListener("click", () => {
        const target = semanticScholarTarget();
        if (target) {
          window.open(target, "_blank", "noopener,noreferrer");
        }
      });
      syncSemanticScholarButton();

      function downloadBlob(content, filename, mime) {
        const blob = new Blob([content], { type: mime });
        const anchor = document.createElement("a");
        anchor.href = URL.createObjectURL(blob);
        anchor.download = filename;
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(anchor.href);
      }

      function seedSlug() {
        const seedNode = nodeById.get((payload.meta && payload.meta.seed_id) || "");
        const slug = seedNode && seedNode.title
          ? seedNode.title.replace(/[^a-zA-Z0-9]+/g, "_").substring(0, 40).replace(/_+$/, "").toLowerCase()
          : "";
        // Titles without ASCII alphanumerics slug to "", which would name the
        // download ".json" and give the browser no stem to disambiguate.
        return slug || "citemesh";
      }

      document.getElementById("export-json-btn").addEventListener("click", () => {
        const exportPayload = buildPortableJsonPayload();
        downloadBlob(JSON.stringify(exportPayload, null, 2), seedSlug() + ".json", "application/json");
      });

      document.getElementById("export-collection-btn").addEventListener("click", () => {
        const exportPayload = portableCollectionPackage();
        downloadBlob(
          JSON.stringify(exportPayload, null, 2),
          "dashboard.citemesh.json",
          "application/json"
        );
        setDashboardStatus(
          `Exported ${exportPayload.results.length} ${exportPayload.results.length === 1 ? "graph" : "graphs"} as one collection package.`,
          "info"
        );
      });

      document.getElementById("export-csv-btn").addEventListener("click", () => {
        const cols = ["id","title","year","authors","citation_count","venue","arxiv_id","doi","categories","is_seed","provenance","seed_relation","seed_relevance","arxiv_url","doi_url","semantic_scholar_url","abstract"];
        // Mirror the CLI CSV writer: neutralize formula-leading cells (CWE-1236).
        function csvGuard(v) { const s = String(v == null ? "" : v); return /^[=+\-@\t\r]/.test(s) ? "'" + s : s; }
        function csvEscape(v) { const s = csvGuard(v); return s.includes(",") || s.includes('"') || s.includes("\n") || s.includes("\r") ? '"' + s.replace(/"/g, '""') + '"' : s; }
        const rows = [cols.join(",")];
        for (const n of (payload.nodes || [])) {
          const links = n.links || {};
          rows.push([
            n.id, n.title, n.year, (n.authors||[]).join("; "), n.citation_count, n.venue||"", n.arxiv_id||"", n.doi||"",
            (n.categories||[]).join("; "), (n.is_seed ? "true" : "false"), n.provenance||"", n.seed_relation||"",
            Number(n.seed_relevance||0).toFixed(6), links.arxiv_abs||"", links.doi||"", links.semantic_scholar||"", n.abstract||""
          ].map(csvEscape).join(","));
        }
        downloadBlob(rows.join("\n"), seedSlug() + ".csv", "text/csv;charset=utf-8");
      });

      document.getElementById("export-bib-btn").addEventListener("click", () => {
        const entries = (payload.nodes || []).map((n) => (n.bibtex || "").trim()).filter(Boolean);
        downloadBlob(entries.join("\n\n") + "\n", seedSlug() + ".bib", "text/plain;charset=utf-8");
      });

      controls.savedBibBtn.addEventListener("click", () => {
        const entries = savedNodes().map((n) => (n.bibtex || "").trim()).filter(Boolean);
        if (!entries.length) {
          return;
        }
        downloadBlob(entries.join("\n\n") + "\n", seedSlug() + "-saved.bib", "text/plain;charset=utf-8");
      });

      controls.copySavedBtn.addEventListener("click", () => {
        const lines = savedNodes().map((n) => {
          const links = n.links || {};
          const href = safeExternalUrl(
            links.arxiv_abs || links.doi || links.semantic_scholar
          );
          const yearText = hasYear(n) ? ` (${n.year})` : "";
          const title = String(n.title || n.id);
          return href
            ? `- [${markdownLinkText(title)}](${href})${yearText}`
            : `- ${title}${yearText}`;
        });
        if (!lines.length) {
          return;
        }
        copyText(lines.join("\n")).then(() => {
          controls.copySavedBtn.textContent = "Copied";
          window.setTimeout(() => {
            controls.copySavedBtn.textContent = "Copy Saved Links";
          }, 1000);
        });
      });

      controls.savedChip.addEventListener("click", () => {
        state.savedOnly = !state.savedOnly;
        updateSavedUi();
        renderList();
      });

      function readFileText(file) {
        return new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = (event) => resolve(String(event.target.result || ""));
          reader.onerror = () => reject(
            new Error(reader.error ? reader.error.message : "Browser could not read the file.")
          );
          reader.readAsText(file);
        });
      }

      async function addResultFiles(files) {
        let importedCount = 0;
        let desiredResultId = null;
        const failures = [];
        for (const file of files) {
          try {
            const fileText = await readFileText(file);
            const importedCollection = parseImportedResultSetFromText(fileText, file.name);
            upsertCollectionEntries(collectionBundle, importedCollection.results);
            importedCount += importedCollection.results.length;
            desiredResultId = importedCollection.current_result_id || desiredResultId;
          } catch (err) {
            failures.push(`${file.name}: ${err.message}`);
          }
        }

        if (importedCount > 0) {
          const nextResultId = desiredResultId || collectionBundle.results[0].result_id;
          collectionBundle.current_result_id = nextResultId;
          collectionResultId = nextResultId;
          populateCollectionSelector();
          try {
            await loadCollectionResult(nextResultId);
          } catch (err) {
            failures.push(`Display: ${err.message}`);
          }
        }

        const uniqueGraphCount = collectionBundle.results.length;
        const importedMessage = importedCount > 0
          ? `Merged ${importedCount} ${importedCount === 1 ? "graph entry" : "graph entries"}; ${uniqueGraphCount} unique ${uniqueGraphCount === 1 ? "graph" : "graphs"} in this session.`
          : "No graphs were added.";
        const failureMessage = failures.length
          ? ` Skipped ${failures.length} ${failures.length === 1 ? "file" : "files"}: ${failures.join(" | ")}`
          : "";
        setDashboardStatus(
          importedMessage + failureMessage,
          failures.length ? "warning" : "info"
        );
      }

      const addResultsInput = document.getElementById("add-results-input");
      document.getElementById("add-results-btn").addEventListener("click", () => {
        addResultsInput.click();
      });
      addResultsInput.addEventListener("change", (event) => {
        const files = Array.from((event.target && event.target.files) || []);
        addResultsInput.value = "";
        if (!files.length) {
          return;
        }
        addResultFiles(files).catch((err) => {
          setDashboardStatus("Failed to add results: " + err.message, "warning");
        });
      });
      controls.resultSelect.addEventListener("change", (event) => {
        const resultId = String(event.target.value || "").trim();
        if (!resultId) {
          return;
        }
        loadCollectionResult(resultId).catch((err) => {
          setDashboardStatus("Failed to switch graphs: " + err.message, "warning");
        });
      });

      setControlsCollapsed(true);
      populateCollectionSelector();
      updateYearPlaceholders();
      updateSavedUi();
    }

    function setupGraphInteractions() {
      if (typeof graphDiv.removeAllListeners === "function") {
        graphDiv.removeAllListeners("plotly_hover");
        graphDiv.removeAllListeners("plotly_unhover");
        graphDiv.removeAllListeners("plotly_click");
      }

      graphDiv.on("plotly_hover", (event) => {
        if (!event || !Array.isArray(event.points)) {
          return;
        }
        const nodePoint = event.points.find((point) => point.curveNumber === nodeTraceIndex);
        if (!nodePoint) {
          return;
        }
        const nodeId = nodeOrder[nodePoint.pointIndex];
        if (!nodeId) {
          return;
        }
        state.hoverId = nodeId;
        renderDetail(nodeId, true);
        syncHighlights();
      });

      graphDiv.on("plotly_unhover", () => {
        state.hoverId = null;
        if (state.selectedId && nodeById.has(state.selectedId)) {
          renderDetail(state.selectedId, false);
        } else {
          renderDetail(null, false);
        }
        syncHighlights();
      });

      graphDiv.on("plotly_click", (event) => {
        if (!event || !Array.isArray(event.points)) {
          return;
        }
        const nodePoint = event.points.find((point) => point.curveNumber === nodeTraceIndex);
        if (!nodePoint) {
          return;
        }
        const nodeId = nodeOrder[nodePoint.pointIndex];
        if (!nodeId) {
          return;
        }
        state.selectedId = nodeId;
        renderDetail(nodeId, false);
        syncHighlights();
        renderList();
      });
    }

    function renderTimeline() {
      const minYear = Number(yearRange.min || 0);
      const maxYear = Number(yearRange.max || 0);
      controls.timelineYearMin.textContent = minYear > 0 ? String(minYear) : "-";
      controls.timelineYearMax.textContent = maxYear > 0 ? String(maxYear) : "-";
    }

    function initialize() {
      if (bootstrapFailureMessage) {
        setDashboardStatus(bootstrapFailureMessage, "warning");
        return;
      }
      if (bootstrapStatusMessage) {
        setDashboardStatus(bootstrapStatusMessage, "warning");
      }
      setupControls();
      // Toolbar changes resize the pane without triggering a window resize.
      const graphResizeObserver = new ResizeObserver(() => Plotly.Plots.resize(graphDiv));
      graphResizeObserver.observe(graphDiv);
      renderTimeline();
      // Result IDs survive rebuilds, so the saved payload can be newer than the
      // initial graph even when both identify the same seed and strategy.
      if (
        collectionResultId
        && !bootstrapStatusMessage
        && embeddedCollectionBundle.results.length > 0
      ) {
        loadCollectionResult(collectionResultId).catch((err) => {
          setDashboardStatus("Failed to load the selected graph: " + err.message, "warning");
        });
        return;
      }
      Plotly.react(graphDiv, figureSpec.data, figureSpec.layout, plotConfig).then(() => {
        setupGraphInteractions();
        if (state.selectedId && nodeById.has(state.selectedId)) {
          renderDetail(state.selectedId, false);
        } else {
          renderDetail(null, false);
        }
        renderList();
      });
    }

    initialize();
