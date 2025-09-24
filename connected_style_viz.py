#!/usr/bin/env python3
"""
reference tool-style visualization with proper similarity-based layout.

Uses a combination of force-directed layout and similarity metrics to create
an elegant, readable graph even with many papers.
"""

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from semanticscholar import SemanticScholar
from sklearn.manifold import TSNE
from tqdm import tqdm


@dataclass
class PaperNode:
    """Paper with citation network information."""
    id: str
    title: str
    year: Optional[int] = None
    citation_count: int = 0
    authors: List[str] = field(default_factory=list)
    abstract: Optional[str] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    references: List[str] = field(default_factory=list)  # Papers this cites
    citations: List[str] = field(default_factory=list)   # Papers citing this


class ConnectedStyleBuilder:
    """Build reference tool-style visualizations with proper similarity layout."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = SemanticScholar(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        self._paper_cache: Dict[str, PaperNode] = {}
        
    def fetch_paper_network(
        self, 
        root_id: str,
        max_citations: int = 30,
        max_references: int = 30,
    ) -> Dict[str, PaperNode]:
        """Fetch the paper and its citation network."""
        
        # Fetch root paper
        try:
            root_paper = self.client.get_paper(root_id)
            if not root_paper:
                raise ValueError(f"Paper not found: {root_id}")
                
            root_node = self._paper_to_node(root_paper)
            papers = {root_node.id: root_node}
            
            # Fetch citations (papers citing the root)
            print("Fetching citations (papers citing this paper)...")
            citations = self.client.get_paper_citations(root_paper.paperId, limit=max_citations)
            for i, cit in enumerate(tqdm(list(citations), desc="Citations", leave=False)):
                try:
                    if cit["citingPaper"] and cit["citingPaper"].get("paperId"):
                        citing_node = self._dict_to_node(cit["citingPaper"])
                        papers[citing_node.id] = citing_node
                        root_node.citations.append(citing_node.id)
                except (KeyError, TypeError):
                    continue
                    
            # Fetch references (papers the root cites)
            print("Fetching references (papers this paper cites)...")
            references = self.client.get_paper_references(root_paper.paperId, limit=max_references)
            for ref in tqdm(list(references), desc="References", leave=False):
                try:
                    if ref["citedPaper"] and ref["citedPaper"].get("paperId"):
                        cited_node = self._dict_to_node(ref["citedPaper"])
                        papers[cited_node.id] = cited_node
                        root_node.references.append(cited_node.id)
                except (KeyError, TypeError):
                    continue
                    
            return papers
            
        except Exception as e:
            self.logger.error(f"Error fetching network: {e}")
            raise
            
    def _paper_to_node(self, paper) -> PaperNode:
        """Convert Paper object to PaperNode."""
        authors = []
        if paper.authors:
            authors = [a.name for a in paper.authors[:5] if a.name]
            
        doi = None
        if hasattr(paper, "externalIds") and paper.externalIds:
            doi = paper.externalIds.get("DOI")
            
        return PaperNode(
            id=paper.paperId,
            title=paper.title or "Unknown",
            year=paper.year,
            citation_count=paper.citationCount or 0,
            authors=authors,
            abstract=paper.abstract[:500] if paper.abstract else None,
            venue=paper.venue,
            doi=doi,
            url=paper.url,
        )
        
    def _dict_to_node(self, paper_dict: dict) -> PaperNode:
        """Convert paper dictionary to PaperNode."""
        authors = []
        if "authors" in paper_dict and paper_dict["authors"]:
            for a in paper_dict["authors"][:5]:
                if isinstance(a, dict) and a.get("name"):
                    authors.append(a["name"])
                    
        doi = None
        if "externalIds" in paper_dict and paper_dict["externalIds"]:
            doi = paper_dict["externalIds"].get("DOI")
            
        return PaperNode(
            id=paper_dict.get("paperId"),
            title=paper_dict.get("title", "Unknown"),
            year=paper_dict.get("year"),
            citation_count=paper_dict.get("citationCount", 0),
            authors=authors,
            abstract=paper_dict.get("abstract", "")[:500] if paper_dict.get("abstract") else None,
            venue=paper_dict.get("venue"),
            doi=doi,
            url=paper_dict.get("url"),
        )
        
    def compute_similarity_matrix(self, papers: Dict[str, PaperNode], root_id: str) -> np.ndarray:
        """
        Compute pairwise similarity between papers based on citation relationships.
        
        Uses a combination of:
        - Direct citation relationship (strongest)
        - Bibliographic coupling (shared references)
        - Co-citation (cited together)
        - Temporal proximity
        """
        paper_ids = list(papers.keys())
        n = len(paper_ids)
        similarity = np.zeros((n, n))
        
        # Create index mapping
        id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
        
        # Direct relationships with root (strongest signal)
        root_idx = id_to_idx[root_id]
        root_paper = papers[root_id]
        
        for pid in root_paper.references:
            if pid in id_to_idx:
                idx = id_to_idx[pid]
                similarity[root_idx, idx] = 0.8
                similarity[idx, root_idx] = 0.8
                
        for pid in root_paper.citations:
            if pid in id_to_idx:
                idx = id_to_idx[pid]
                similarity[root_idx, idx] = 0.8
                similarity[idx, root_idx] = 0.8
                
        # Bibliographic coupling and co-citation for all papers
        for i, pid1 in enumerate(paper_ids):
            p1 = papers[pid1]
            for j, pid2 in enumerate(paper_ids[i+1:], start=i+1):
                p2 = papers[pid2]
                
                # Bibliographic coupling (shared references)
                if p1.references and p2.references:
                    shared_refs = set(p1.references) & set(p2.references)
                    if shared_refs:
                        bc_score = len(shared_refs) / max(len(p1.references), len(p2.references))
                        similarity[i, j] += bc_score * 0.4
                        similarity[j, i] += bc_score * 0.4
                        
                # Co-citation (cited together)
                if p1.citations and p2.citations:
                    shared_cits = set(p1.citations) & set(p2.citations)
                    if shared_cits:
                        cc_score = len(shared_cits) / max(len(p1.citations), len(p2.citations))
                        similarity[i, j] += cc_score * 0.3
                        similarity[j, i] += cc_score * 0.3
                        
                # Temporal proximity
                if p1.year and p2.year:
                    year_diff = abs(p1.year - p2.year)
                    time_score = 1.0 / (1.0 + year_diff / 5.0)
                    similarity[i, j] += time_score * 0.1
                    similarity[j, i] += time_score * 0.1
                    
        # Normalize to [0, 1]
        np.fill_diagonal(similarity, 1.0)
        similarity = np.clip(similarity, 0, 1)
        
        return similarity
        
    def compute_layout(
        self,
        papers: Dict[str, PaperNode],
        root_id: str,
        similarity_matrix: np.ndarray,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute 2D layout using similarity-based positioning.
        
        Uses t-SNE or MDS to project high-dimensional similarity to 2D,
        ensuring similar papers are positioned close together.
        """
        n = len(papers)
        paper_ids = list(papers.keys())
        
        if n <= 3:
            # Simple layout for very few papers
            positions = {}
            positions[root_id] = (400, 300)
            other_ids = [pid for pid in paper_ids if pid != root_id]
            for i, pid in enumerate(other_ids):
                angle = i * 2 * math.pi / len(other_ids)
                x = 400 + 200 * math.cos(angle)
                y = 300 + 200 * math.sin(angle)
                positions[pid] = (x, y)
            return positions
            
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        # Use t-SNE for dimensionality reduction
        if n > 30:
            # For many papers, use t-SNE with careful parameters
            tsne = TSNE(
                n_components=2,
                metric="precomputed",
                init="random",
                perplexity=min(30, n-1),
                max_iter=1000,
                random_state=42,
            )
            coords = tsne.fit_transform(distance_matrix)
        else:
            # For fewer papers, use MDS which preserves distances better
            from sklearn.manifold import MDS
            mds = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=42,
            )
            coords = mds.fit_transform(distance_matrix)
            
        # Normalize to viewport
        coords -= coords.min(axis=0)
        coords /= coords.max(axis=0) + 1e-10
        
        # Scale to viewport with padding
        viewport_width = 700
        viewport_height = 500
        padding = 50
        
        coords[:, 0] = coords[:, 0] * viewport_width + padding
        coords[:, 1] = coords[:, 1] * viewport_height + padding
        
        # Ensure root is somewhat centered
        root_idx = paper_ids.index(root_id)
        center_x, center_y = 400, 300
        offset_x = center_x - coords[root_idx, 0]
        offset_y = center_y - coords[root_idx, 1]
        
        # Apply partial centering (don't fully center, just nudge)
        coords[:, 0] += offset_x * 0.5
        coords[:, 1] += offset_y * 0.5
        
        # Create position dictionary
        positions = {
            pid: (coords[i, 0], coords[i, 1])
            for i, pid in enumerate(paper_ids)
        }
        
        return positions
        
    def create_visualization(
        self,
        papers: Dict[str, PaperNode],
        root_id: str,
        output_path: Path,
    ):
        """Create the reference tool-style HTML visualization."""
        
        # Compute similarity matrix
        print("Computing paper similarities...")
        similarity_matrix = self.compute_similarity_matrix(papers, root_id)
        
        # Compute layout
        print("Computing optimal layout...")
        positions = self.compute_layout(papers, root_id, similarity_matrix)
        
        # Prepare visualization data
        nodes_data = []
        edges_data = []
        
        # Year and citation scales for visual encoding
        years = [p.year for p in papers.values() if p.year]
        min_year = min(years) if years else 2020
        max_year = max(years) if years else 2024
        
        citations = [p.citation_count for p in papers.values()]
        citation_percentiles = np.percentile(citations, [50, 75, 90, 95]) if citations else [10, 20, 30, 40]
        
        # Process nodes
        for pid, paper in papers.items():
            x, y = positions[pid]
            
            # Size based on citations (logarithmic scale for better distribution)
            if paper.citation_count == 0:
                size = 8
            elif paper.citation_count <= citation_percentiles[0]:
                size = 10
            elif paper.citation_count <= citation_percentiles[1]:
                size = 15
            elif paper.citation_count <= citation_percentiles[2]:
                size = 20
            elif paper.citation_count <= citation_percentiles[3]:
                size = 25
            else:
                size = 30
                
            # Color based on year (gradient from light to dark blue)
            if paper.year:
                year_norm = (paper.year - min_year) / max(max_year - min_year, 1)
                # HSL: hue=200 (blue), saturation decreases with age, lightness increases with age
                saturation = 30 + year_norm * 40  # 30% to 70%
                lightness = 75 - year_norm * 35    # 75% to 40%
                color = f"hsl(200, {saturation}%, {lightness}%)"
            else:
                color = "hsl(200, 20%, 70%)"  # Gray-blue for unknown year
                
            nodes_data.append({
                "id": pid,
                "x": x,
                "y": y,
                "size": size,
                "color": color,
                "title": paper.title[:80] + "..." if len(paper.title) > 80 else paper.title,
                "year": paper.year,
                "citation_count": paper.citation_count,
                "authors": paper.authors,
                "is_root": pid == root_id,
                "has_citations": len(paper.citations) > 0,
                "has_references": len(paper.references) > 0,
            })
            
        # Process edges based on similarity
        paper_ids = list(papers.keys())
        for i, pid1 in enumerate(paper_ids):
            for j, pid2 in enumerate(paper_ids[i+1:], start=i+1):
                sim = similarity_matrix[i, j]
                if sim > 0.15:  # Only show meaningful connections
                    edges_data.append({
                        "source": pid1,
                        "target": pid2,
                        "weight": sim,
                        "opacity": min(0.1 + sim * 0.5, 0.6),
                        "width": 0.5 + sim * 2,
                    })
                    
        # Generate HTML
        html = self._generate_html(nodes_data, edges_data, papers[root_id].title)
        
        with open(output_path, "w") as f:
            f.write(html)
            
        print(f"Visualization saved to {output_path}")
        print(f"Graph has {len(nodes_data)} nodes and {len(edges_data)} edges")
        
    def _generate_html(self, nodes: List[dict], edges: List[dict], root_title: str) -> str:
        """Generate the HTML visualization."""
        
        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(edges)
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>reference tool: {root_title[:50]}</title>
    <meta charset="utf-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            overflow: hidden;
        }}
        #container {{
            display: flex;
            height: 100vh;
            background: #ffffff;
        }}
        #graph-container {{
            flex: 1;
            position: relative;
            background: #fafafa;
        }}
        #graph {{
            width: 100%;
            height: 100%;
        }}
        #info-panel {{
            width: 380px;
            background: white;
            box-shadow: -4px 0 15px rgba(0,0,0,0.08);
            display: flex;
            flex-direction: column;
        }}
        #panel-header {{
            padding: 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        #panel-header h1 {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        #panel-header p {{
            font-size: 14px;
            opacity: 0.9;
        }}
        #panel-content {{
            flex: 1;
            overflow-y: auto;
            padding: 25px;
        }}
        .paper-card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 2px solid transparent;
            transition: all 0.3s;
            cursor: pointer;
        }}
        .paper-card:hover {{
            border-color: #667eea;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.1);
        }}
        .paper-card.selected {{
            border-color: #764ba2;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
        }}
        .paper-title {{
            font-size: 16px;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 8px;
            line-height: 1.4;
        }}
        .paper-authors {{
            font-size: 13px;
            color: #718096;
            margin-bottom: 8px;
            font-style: italic;
        }}
        .paper-meta {{
            display: flex;
            gap: 15px;
            font-size: 13px;
            color: #4a5568;
        }}
        .meta-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .meta-icon {{
            width: 16px;
            height: 16px;
            fill: #718096;
        }}
        .legend {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            min-width: 200px;
        }}
        .legend-title {{
            font-size: 14px;
            font-weight: 600;
            color: #2d3748;
            margin-bottom: 15px;
        }}
        .legend-section {{
            margin-bottom: 15px;
        }}
        .legend-subtitle {{
            font-size: 12px;
            font-weight: 500;
            color: #4a5568;
            margin-bottom: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: #718096;
            margin-bottom: 5px;
        }}
        .legend-circle {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        .legend-line {{
            width: 30px;
            height: 2px;
            background: #cbd5e0;
        }}
        .controls {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: white;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            display: flex;
            gap: 10px;
        }}
        .control-btn {{
            padding: 8px 15px;
            border: 1px solid #e2e8f0;
            background: white;
            border-radius: 8px;
            font-size: 13px;
            color: #4a5568;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .control-btn:hover {{
            background: #f7fafc;
            border-color: #cbd5e0;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(45, 55, 72, 0.95);
            color: white;
            padding: 12px 15px;
            border-radius: 8px;
            font-size: 13px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            max-width: 300px;
            z-index: 1000;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        .tooltip-title {{
            font-weight: 600;
            margin-bottom: 5px;
        }}
        .tooltip-meta {{
            opacity: 0.8;
            font-size: 12px;
        }}
        svg {{
            width: 100%;
            height: 100%;
        }}
        .node {{
            cursor: pointer;
            stroke: white;
            stroke-width: 2px;
            transition: all 0.2s;
        }}
        .node:hover {{
            stroke-width: 3px;
            filter: brightness(1.1);
        }}
        .node.selected {{
            stroke: #764ba2;
            stroke-width: 4px;
        }}
        .node.root {{
            stroke: #ff6b6b;
            stroke-width: 3px;
        }}
        .link {{
            stroke: #cbd5e0;
            pointer-events: none;
            transition: all 0.2s;
        }}
        .link.highlighted {{
            stroke: #667eea;
            stroke-width: 2px;
        }}
        #search-box {{
            padding: 15px 25px;
            border-bottom: 1px solid #e2e8f0;
        }}
        #search-input {{
            width: 100%;
            padding: 10px 15px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.2s;
        }}
        #search-input:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="graph-container">
            <svg id="graph"></svg>
            
            <div class="legend">
                <div class="legend-title">Visual Encoding</div>
                
                <div class="legend-section">
                    <div class="legend-subtitle">Node Size = Citations</div>
                    <div class="legend-item">
                        <div class="legend-circle" style="width: 8px; height: 8px; background: #cbd5e0;"></div>
                        <span>Few citations</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-circle" style="width: 20px; height: 20px; background: #cbd5e0;"></div>
                        <span>Many citations</span>
                    </div>
                </div>
                
                <div class="legend-section">
                    <div class="legend-subtitle">Node Color = Year</div>
                    <div class="legend-item">
                        <div class="legend-circle" style="background: hsl(200, 30%, 75%);"></div>
                        <span>Older papers</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-circle" style="background: hsl(200, 70%, 40%);"></div>
                        <span>Recent papers</span>
                    </div>
                </div>
                
                <div class="legend-section">
                    <div class="legend-subtitle">Position = Similarity</div>
                    <div class="legend-item">
                        <span style="font-size: 11px;">Papers with similar references and citations are positioned closer together</span>
                    </div>
                </div>
            </div>
            
            <div class="controls">
                <button class="control-btn" onclick="resetZoom()">Reset View</button>
                <button class="control-btn" onclick="toggleEdges()">Toggle Edges</button>
            </div>
            
            <div class="tooltip"></div>
        </div>
        
        <div id="info-panel">
            <div id="panel-header">
                <h1>Paper Network</h1>
                <p>{len(nodes)} papers • {len(edges)} connections</p>
            </div>
            
            <div id="search-box">
                <input type="text" id="search-input" placeholder="Search papers by title or author...">
            </div>
            
            <div id="panel-content">
                <div id="paper-list"></div>
            </div>
        </div>
    </div>
    
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
        const nodes = {nodes_json};
        const edges = {edges_json};
        
        // Create node and edge maps
        const nodeMap = new Map(nodes.map(n => [n.id, n]));
        const edgeMap = new Map();
        edges.forEach(e => {{
            if (!edgeMap.has(e.source)) edgeMap.set(e.source, []);
            if (!edgeMap.has(e.target)) edgeMap.set(e.target, []);
            edgeMap.get(e.source).push(e.target);
            edgeMap.get(e.target).push(e.source);
        }});
        
        // SVG setup
        const svg = d3.select("#graph");
        const width = document.getElementById("graph-container").clientWidth;
        const height = document.getElementById("graph-container").clientHeight;
        
        const g = svg.append("g");
        
        // Draw edges
        const link = g.append("g")
            .selectAll("line")
            .data(edges)
            .join("line")
            .attr("class", "link")
            .attr("x1", d => nodeMap.get(d.source).x)
            .attr("y1", d => nodeMap.get(d.source).y)
            .attr("x2", d => nodeMap.get(d.target).x)
            .attr("y2", d => nodeMap.get(d.target).y)
            .attr("stroke-width", d => d.width)
            .attr("stroke-opacity", d => d.opacity);
            
        // Draw nodes
        const node = g.append("g")
            .selectAll("circle")
            .data(nodes)
            .join("circle")
            .attr("class", d => d.is_root ? "node root" : "node")
            .attr("cx", d => d.x)
            .attr("cy", d => d.y)
            .attr("r", d => d.size)
            .attr("fill", d => d.color);
            
        // Tooltip
        const tooltip = d3.select(".tooltip");
        
        // Paper list in sidebar
        const paperList = d3.select("#paper-list");
        nodes.sort((a, b) => b.citation_count - a.citation_count);
        
        const paperCards = paperList.selectAll(".paper-card")
            .data(nodes)
            .join("div")
            .attr("class", "paper-card")
            .html(d => `
                <div class="paper-title">${{d.title}}</div>
                <div class="paper-authors">${{d.authors.slice(0, 3).join(", ") || "No authors"}}</div>
                <div class="paper-meta">
                    <div class="meta-item">📅 ${{d.year || "N/A"}}</div>
                    <div class="meta-item">📚 ${{d.citation_count}} citations</div>
                </div>
            `);
            
        // Node interactions
        let selectedNode = null;
        
        node.on("mouseover", function(event, d) {{
            tooltip
                .style("opacity", 1)
                .html(`
                    <div class="tooltip-title">${{d.title}}</div>
                    <div class="tooltip-meta">
                        ${{d.year || "Unknown year"}} • ${{d.citation_count}} citations<br>
                        ${{d.authors.slice(0, 2).join(", ") || "No authors"}}
                    </div>
                `)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
                
            // Highlight connected edges
            link.classed("highlighted", e => e.source === d.id || e.target === d.id);
        }})
        .on("mouseout", function() {{
            tooltip.style("opacity", 0);
            if (!selectedNode) {{
                link.classed("highlighted", false);
            }}
        }})
        .on("click", function(event, d) {{
            selectedNode = d;
            
            // Update node selection
            node.classed("selected", n => n.id === d.id);
            
            // Update paper cards
            paperCards.classed("selected", p => p.id === d.id);
            
            // Scroll to selected paper
            const selectedCard = paperCards.filter(p => p.id === d.id).node();
            if (selectedCard) {{
                selectedCard.scrollIntoView({{ behavior: "smooth", block: "center" }});
            }}
            
            // Keep edges highlighted
            link.classed("highlighted", e => e.source === d.id || e.target === d.id);
        }});
        
        // Paper card clicks
        paperCards.on("click", function(event, d) {{
            selectedNode = d;
            
            // Update selections
            node.classed("selected", n => n.id === d.id);
            paperCards.classed("selected", p => p.id === d.id);
            
            // Highlight edges
            link.classed("highlighted", e => e.source === d.id || e.target === d.id);
            
            // Pan to node
            const scale = d3.zoomTransform(svg.node()).k;
            const x = -d.x * scale + width / 2;
            const y = -d.y * scale + height / 2;
            
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity.translate(x, y).scale(scale));
        }});
        
        // Search functionality
        d3.select("#search-input").on("input", function() {{
            const query = this.value.toLowerCase();
            
            paperCards.style("display", d => {{
                const matchTitle = d.title.toLowerCase().includes(query);
                const matchAuthors = d.authors.some(a => a.toLowerCase().includes(query));
                return (matchTitle || matchAuthors) ? "block" : "none";
            }});
            
            node.style("opacity", d => {{
                const matchTitle = d.title.toLowerCase().includes(query);
                const matchAuthors = d.authors.some(a => a.toLowerCase().includes(query));
                return (query === "" || matchTitle || matchAuthors) ? 1 : 0.2;
            }});
        }});
        
        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.3, 3])
            .on("zoom", function(event) {{
                g.attr("transform", event.transform);
            }});
            
        svg.call(zoom);
        
        // Control functions
        let edgesVisible = true;
        
        function resetZoom() {{
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity);
        }}
        
        function toggleEdges() {{
            edgesVisible = !edgesVisible;
            link.style("opacity", edgesVisible ? null : 0);
        }}
        
        // Initial zoom to fit
        const nodeXs = nodes.map(n => n.x);
        const nodeYs = nodes.map(n => n.y);
        const xExtent = d3.extent(nodeXs);
        const yExtent = d3.extent(nodeYs);
        const xRange = xExtent[1] - xExtent[0];
        const yRange = yExtent[1] - yExtent[0];
        
        const scale = Math.min(
            width / (xRange + 100),
            height / (yRange + 100),
            1
        );
        
        const xCenter = (xExtent[0] + xExtent[1]) / 2;
        const yCenter = (yExtent[0] + yExtent[1]) / 2;
        
        svg.call(
            zoom.transform,
            d3.zoomIdentity
                .translate(width / 2, height / 2)
                .scale(scale)
                .translate(-xCenter, -yCenter)
        );
    </script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Create reference tool-style visualizations")
    parser.add_argument("paper_id", help="Paper ID (DOI, arXiv, or S2 ID)")
    parser.add_argument("-o", "--output", type=Path, default=Path("connected_viz.html"))
    parser.add_argument("--max-citations", type=int, default=30)
    parser.add_argument("--max-references", type=int, default=30)
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        builder = ConnectedStyleBuilder()
        
        print(f"Fetching paper network for {args.paper_id}...")
        papers = builder.fetch_paper_network(
            args.paper_id,
            max_citations=args.max_citations,
            max_references=args.max_references,
        )
        
        print(f"Building visualization with {len(papers)} papers...")
        # Get the actual root paper ID (might be different from input ID)
        root_id = list(papers.keys())[0]  # First paper is always the root
        builder.create_visualization(papers, root_id, args.output)
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())