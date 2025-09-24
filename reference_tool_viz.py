#!/usr/bin/env python3
"""
reference tool-style Citation Network Visualizer

Creates similarity-based paper networks using bibliographic coupling and co-citation analysis.
Visualizes papers in a force-directed layout where position indicates similarity.

Author: Research Tools
License: MIT
"""

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from semanticscholar import SemanticScholar
from tqdm import tqdm


@dataclass
class PaperNode:
    """Enhanced paper node with citation data for similarity calculation."""

    id: str
    title: str
    year: Optional[int] = None
    citation_count: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    abstract: Optional[str] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    references: Set[str] = field(default_factory=set)  # Paper IDs this paper cites
    citations: Set[str] = field(default_factory=set)  # Paper IDs that cite this paper
    
    def __hash__(self):
        return hash(self.id)


class reference toolBuilder:
    """
    Builds similarity-based citation networks using reference tool methodology.
    
    Unlike traditional citation graphs, this creates a similarity graph where:
    - Papers are positioned by similarity (bibliographic coupling + co-citation)
    - Node size represents citation count
    - Node color represents publication year
    - Edge thickness represents similarity strength
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize with optional Semantic Scholar API key for better rate limits."""
        self.client = SemanticScholar(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        self._paper_cache: Dict[str, PaperNode] = {}
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
        
    def _fetch_paper_details(self, paper_id: str) -> Optional[PaperNode]:
        """Fetch comprehensive paper details including references and citations."""
        if paper_id in self._paper_cache:
            return self._paper_cache[paper_id]
            
        try:
            # Fetch main paper
            paper = self.client.get_paper(paper_id)
            if not paper:
                return None
                
            # Create base node
            authors = []
            if paper.authors:
                authors = [a.name for a in paper.authors[:5] if a.name]
                
            doi = None
            if hasattr(paper, "externalIds") and paper.externalIds:
                doi = paper.externalIds.get("DOI")
                
            node = PaperNode(
                id=paper.paperId,
                title=paper.title or "Unknown Title",
                year=paper.year,
                citation_count=paper.citationCount or 0,
                authors=authors,
                abstract=paper.abstract[:500] if paper.abstract else None,
                venue=paper.venue,
                doi=doi,
                url=paper.url,
            )
            
            # Fetch references (papers this paper cites)
            self.logger.info(f"Fetching references for {node.title[:50]}...")
            references = self.client.get_paper_references(paper.paperId, limit=100)
            for ref in references:
                try:
                    if ref["citedPaper"] and ref["citedPaper"].get("paperId"):
                        node.references.add(ref["citedPaper"]["paperId"])
                except (KeyError, TypeError):
                    continue
                    
            # Fetch citations (papers that cite this paper)
            self.logger.info(f"Fetching citations for {node.title[:50]}...")
            citations = self.client.get_paper_citations(paper.paperId, limit=100)
            for cit in citations:
                try:
                    if cit["citingPaper"] and cit["citingPaper"].get("paperId"):
                        node.citations.add(cit["citingPaper"]["paperId"])
                except (KeyError, TypeError):
                    continue
                    
            self._paper_cache[paper_id] = node
            return node
            
        except Exception as e:
            self.logger.error(f"Error fetching paper {paper_id}: {e}")
            return None
            
    def _calculate_similarity(self, paper1: PaperNode, paper2: PaperNode) -> float:
        """
        Calculate similarity between two papers using bibliographic coupling and co-citation.
        
        Bibliographic Coupling: Papers that cite the same references
        Co-citation: Papers that are cited together by other papers
        """
        # Check cache
        cache_key = tuple(sorted([paper1.id, paper2.id]))
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
            
        # Bibliographic coupling: shared references
        if paper1.references and paper2.references:
            shared_refs = paper1.references & paper2.references
            total_refs = paper1.references | paper2.references
            bc_score = len(shared_refs) / len(total_refs) if total_refs else 0
        else:
            bc_score = 0
            
        # Co-citation: shared citations (papers that cite both)
        if paper1.citations and paper2.citations:
            shared_cits = paper1.citations & paper2.citations
            total_cits = paper1.citations | paper2.citations
            cc_score = len(shared_cits) / len(total_cits) if total_cits else 0
        else:
            cc_score = 0
            
        # Combined similarity with weights (reference tool uses proprietary weights)
        # We'll use 60% bibliographic coupling, 40% co-citation
        similarity = 0.6 * bc_score + 0.4 * cc_score
        
        # Adjust for publication year difference (papers from similar era are more relevant)
        if paper1.year and paper2.year:
            year_diff = abs(paper1.year - paper2.year)
            year_penalty = 1.0 / (1.0 + year_diff / 10.0)  # Smooth decay
            similarity *= year_penalty
            
        self._similarity_cache[cache_key] = similarity
        return similarity
        
    def build_similarity_graph(
        self,
        root_paper_id: str,
        max_papers: int = 40,
        min_similarity: float = 0.1,
    ) -> nx.Graph:
        """
        Build a similarity-based graph starting from a root paper.
        
        Args:
            root_paper_id: Starting paper identifier
            max_papers: Maximum number of related papers to include
            min_similarity: Minimum similarity threshold for connections
            
        Returns:
            NetworkX graph with similarity-based edges
        """
        graph = nx.Graph()
        
        # Fetch root paper
        root = self._fetch_paper_details(root_paper_id)
        if not root:
            raise ValueError(f"Could not fetch root paper: {root_paper_id}")
            
        # Collect related papers (references and citations of root)
        related_ids = set()
        related_ids.update(list(root.references)[:max_papers // 2])
        related_ids.update(list(root.citations)[:max_papers // 2])
        
        # Fetch all related papers
        papers = [root]
        for pid in tqdm(list(related_ids)[:max_papers], desc="Fetching related papers"):
            paper = self._fetch_paper_details(pid)
            if paper:
                papers.append(paper)
                
        self.logger.info(f"Computing similarities for {len(papers)} papers...")
        
        # Add nodes to graph
        for paper in papers:
            graph.add_node(
                paper.id,
                title=paper.title,
                year=paper.year,
                citation_count=paper.citation_count,
                authors=paper.authors,
                abstract=paper.abstract,
                venue=paper.venue,
                doi=paper.doi,
                url=paper.url,
                is_root=(paper.id == root.id),
            )
            
        # Calculate all pairwise similarities and add edges
        for i, p1 in enumerate(papers):
            for p2 in papers[i + 1:]:
                similarity = self._calculate_similarity(p1, p2)
                if similarity >= min_similarity:
                    graph.add_edge(p1.id, p2.id, weight=similarity)
                    
        return graph
        
    def create_visualization(
        self,
        graph: nx.Graph,
        output_path: Path,
        title: str = "reference tool Visualization",
    ) -> None:
        """
        Create an interactive HTML visualization with reference tool styling.
        """
        if graph.number_of_nodes() == 0:
            raise ValueError("Cannot visualize empty graph")
            
        # Calculate layout using spring layout (force-directed)
        # Weight edges by similarity for better clustering
        pos = nx.spring_layout(
            graph,
            weight="weight",
            k=2 / math.sqrt(graph.number_of_nodes()),
            iterations=50,
            seed=42,
        )
        
        # Normalize positions to viewport
        x_values = [p[0] for p in pos.values()]
        y_values = [p[1] for p in pos.values()]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        
        for node_id in pos:
            x, y = pos[node_id]
            # Scale to viewport (800x600 with padding)
            pos[node_id] = [
                50 + 700 * (x - x_min) / (x_max - x_min) if x_max != x_min else 400,
                50 + 500 * (y - y_min) / (y_max - y_min) if y_max != y_min else 300,
            ]
            
        # Prepare data for visualization
        nodes_data = []
        edges_data = []
        
        # Color scale for years (light to dark)
        years = [d.get("year", 2020) for _, d in graph.nodes(data=True)]
        min_year = min(years) if years else 2020
        max_year = max(years) if years else 2024
        
        # Size scale for citations
        citations = [d.get("citation_count", 0) for _, d in graph.nodes(data=True)]
        max_citations = max(citations) if citations else 100
        
        # Process nodes
        for node_id, attrs in graph.nodes(data=True):
            x, y = pos[node_id]
            
            # Calculate node size based on citations (5-30 radius)
            citation_count = attrs.get("citation_count", 0)
            size = 5 + 25 * (citation_count / max(max_citations, 1)) ** 0.5
            
            # Calculate color based on year (older = lighter)
            year = attrs.get("year", min_year)
            year_normalized = (year - min_year) / max(max_year - min_year, 1)
            # Use HSL for smooth color gradient (blue hue)
            lightness = 80 - year_normalized * 40  # 80% to 40% lightness
            color = f"hsl(210, 70%, {lightness}%)"
            
            nodes_data.append({
                "id": node_id,
                "x": x,
                "y": y,
                "size": size,
                "color": color,
                "title": attrs.get("title", "Unknown"),
                "year": year,
                "citation_count": citation_count,
                "authors": attrs.get("authors", []),
                "is_root": attrs.get("is_root", False),
            })
            
        # Process edges
        for source, target, attrs in graph.edges(data=True):
            weight = attrs.get("weight", 0.1)
            edges_data.append({
                "source": source,
                "target": target,
                "weight": weight,
                "opacity": min(0.2 + weight * 0.8, 1.0),  # More similar = more opaque
                "width": 0.5 + weight * 2,  # More similar = thicker line
            })
            
        # Generate HTML with D3.js visualization
        html_content = self._generate_html(nodes_data, edges_data, title)
        
        with open(output_path, "w") as f:
            f.write(html_content)
            
        self.logger.info(f"Visualization saved to {output_path}")
        
    def _generate_html(
        self,
        nodes: List[dict],
        edges: List[dict],
        title: str,
    ) -> str:
        """Generate the HTML/JavaScript for the visualization."""
        
        # Convert to JSON
        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(edges)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }}
        #container {{
            display: flex;
            height: 100vh;
        }}
        #graph {{
            flex: 1;
            background: white;
            position: relative;
        }}
        #sidebar {{
            width: 350px;
            background: white;
            border-left: 1px solid #ddd;
            padding: 20px;
            overflow-y: auto;
            box-shadow: -2px 0 5px rgba(0,0,0,0.1);
        }}
        #paper-info {{
            display: none;
        }}
        #paper-info.active {{
            display: block;
        }}
        .paper-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }}
        .paper-meta {{
            color: #666;
            font-size: 14px;
            margin: 5px 0;
        }}
        .paper-authors {{
            color: #888;
            font-size: 13px;
            font-style: italic;
        }}
        .legend {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255,255,255,0.9);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .legend-title {{
            font-weight: 600;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        .legend-item {{
            font-size: 12px;
            margin: 5px 0;
            display: flex;
            align-items: center;
        }}
        .legend-circle {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            display: inline-block;
        }}
        .tooltip {{
            position: absolute;
            text-align: left;
            padding: 10px;
            font-size: 12px;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            border-radius: 5px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            max-width: 300px;
        }}
        svg {{
            width: 100%;
            height: 100%;
        }}
        .node {{
            cursor: pointer;
            stroke: #fff;
            stroke-width: 2px;
        }}
        .node:hover {{
            stroke: #333;
            stroke-width: 3px;
        }}
        .node.selected {{
            stroke: #ff6b6b;
            stroke-width: 3px;
        }}
        .node.root {{
            stroke: #ff6b6b;
            stroke-width: 3px;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
            pointer-events: none;
        }}
        .link.highlighted {{
            stroke: #ff6b6b;
            stroke-opacity: 1;
        }}
        h3 {{
            margin-top: 0;
            color: #333;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="graph">
            <svg></svg>
            <div class="legend">
                <div class="legend-title">Paper Visualization</div>
                <div class="legend-item">
                    <span class="legend-circle" style="background: hsl(210, 70%, 40%);"></span>
                    Recent papers (darker)
                </div>
                <div class="legend-item">
                    <span class="legend-circle" style="background: hsl(210, 70%, 80%);"></span>
                    Older papers (lighter)
                </div>
                <div class="legend-item">
                    <span class="legend-circle" style="width: 20px; height: 20px; background: #ccc;"></span>
                    More citations (larger)
                </div>
                <div class="legend-item" style="margin-top: 10px;">
                    <strong>Click</strong> a paper to see details →
                </div>
            </div>
            <div class="tooltip"></div>
        </div>
        <div id="sidebar">
            <h3>Paper Details</h3>
            <div id="paper-info">
                <div class="paper-title" id="selected-title">Select a paper to view details</div>
                <div class="paper-meta" id="selected-year"></div>
                <div class="paper-meta" id="selected-citations"></div>
                <div class="paper-authors" id="selected-authors"></div>
            </div>
            <div id="default-message" style="color: #999; margin-top: 20px;">
                Click on any paper node to view its details. Papers are positioned by similarity - 
                closer papers share more references and citations. Node size indicates citation count,
                and darker colors represent more recent publications.
            </div>
        </div>
    </div>
    
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
        const nodes = {nodes_json};
        const edges = {edges_json};
        
        // Create SVG
        const svg = d3.select("svg");
        const width = document.getElementById("graph").clientWidth;
        const height = document.getElementById("graph").clientHeight;
        svg.attr("viewBox", [0, 0, width, height]);
        
        // Create edge map for quick lookup
        const edgeMap = new Map();
        edges.forEach(e => {{
            if (!edgeMap.has(e.source)) edgeMap.set(e.source, []);
            if (!edgeMap.has(e.target)) edgeMap.set(e.target, []);
            edgeMap.get(e.source).push(e.target);
            edgeMap.get(e.target).push(e.source);
        }});
        
        // Create node map
        const nodeMap = new Map(nodes.map(n => [n.id, n]));
        
        // Draw edges
        const link = svg.append("g")
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
        const node = svg.append("g")
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
        
        // Node interactions
        node.on("mouseover", function(event, d) {{
            tooltip
                .style("opacity", 1)
                .html(`<strong>${{d.title}}</strong><br/>
                      Year: ${{d.year || 'Unknown'}}<br/>
                      Citations: ${{d.citation_count}}<br/>
                      ${{d.authors.slice(0, 3).join(", ")}}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 28) + "px");
                
            // Highlight connected edges
            link.classed("highlighted", e => 
                e.source === d.id || e.target === d.id
            );
        }})
        .on("mouseout", function(event, d) {{
            tooltip.style("opacity", 0);
            link.classed("highlighted", false);
        }})
        .on("click", function(event, d) {{
            // Update selection
            node.classed("selected", false);
            d3.select(this).classed("selected", true);
            
            // Update sidebar
            document.getElementById("default-message").style.display = "none";
            document.getElementById("paper-info").classList.add("active");
            document.getElementById("selected-title").textContent = d.title;
            document.getElementById("selected-year").textContent = `Year: ${{d.year || 'Unknown'}}`;
            document.getElementById("selected-citations").textContent = `Citations: ${{d.citation_count}}`;
            document.getElementById("selected-authors").textContent = d.authors.join(", ") || "No authors listed";
        }});
        
        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.5, 5])
            .on("zoom", function(event) {{
                svg.selectAll("g").attr("transform", event.transform);
            }});
            
        svg.call(zoom);
    </script>
</body>
</html>"""
        
        return html


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build reference tool-style similarity visualizations"
    )
    parser.add_argument(
        "paper_id",
        help="Paper identifier (DOI, arXiv ID, or Semantic Scholar ID)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("reference tool.html"),
        help="Output HTML file path"
    )
    parser.add_argument(
        "-n", "--max-papers",
        type=int,
        default=40,
        help="Maximum number of papers to include"
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.05,
        help="Minimum similarity threshold for edges"
    )
    parser.add_argument(
        "--api-key",
        help="Semantic Scholar API key for better rate limits"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        builder = reference toolBuilder(api_key=args.api_key)
        
        print(f"Building similarity graph for {args.paper_id}...")
        print("This may take a few minutes to fetch citation data...")
        
        graph = builder.build_similarity_graph(
            root_paper_id=args.paper_id,
            max_papers=args.max_papers,
            min_similarity=args.min_similarity,
        )
        
        print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        builder.create_visualization(
            graph=graph,
            output_path=args.output,
            title=f"reference tool: {args.paper_id}"
        )
        
        print(f"Visualization saved to {args.output}")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())