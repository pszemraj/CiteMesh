#!/usr/bin/env python
"""Debug script to understand what's going wrong with the visualization."""

import numpy as np
from citation_graph import CitationGraphBuilder, LayoutStyle, GraphConfig
from pathlib import Path

# Create builder
config = GraphConfig(
    max_citations=3,
    max_references=3,
    max_depth=1,
    layout_style=LayoutStyle.SIMILARITY,
)
builder = CitationGraphBuilder(config)

# Build a small test graph
print("Building graph...")
graph = builder.build_graph("10.48550/arXiv.2507.11412")

print(
    f"\nGraph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
)

# Check what's in the nodes
print("\nNode data sample:")
for i, (node_id, data) in enumerate(graph.nodes(data=True)):
    if i < 3:  # Show first 3 nodes
        print(f"\nNode: {node_id[:20]}...")
        print(f"  Title: {data.get('title', 'Unknown')[:50]}...")
        print(f"  Year: {data.get('year', 'N/A')}")
        print(f"  Citations: {data.get('citation_count', 0)}")
        print(f"  Has references list: {'references' in data}")
        print(f"  Has citations list: {'citations' in data}")
        if "references" in data:
            print(f"    References count: {len(data['references'])}")
        if "citations" in data:
            print(f"    Citations count: {len(data['citations'])}")

# Test similarity computation
print("\n\nTesting similarity matrix computation...")
similarity_matrix = builder.compute_similarity_matrix(graph)
print(f"Similarity matrix shape: {similarity_matrix.shape}")
print(
    f"Similarity matrix min: {similarity_matrix.min():.3f}, max: {similarity_matrix.max():.3f}"
)
print(
    f"Non-diagonal non-zero elements: {((similarity_matrix > 0) & (~np.eye(len(similarity_matrix), dtype=bool))).sum()}"
)

# Show some similarity values
nodes = list(graph.nodes())
if len(nodes) > 1:
    print("\nSimilarity between first two nodes:")
    print(f"  {nodes[0][:30]}... vs {nodes[1][:30]}...")
    print(f"  Similarity: {similarity_matrix[0, 1]:.3f}")

# Test layout computation
print("\n\nTesting layout computation...")
positions = builder.compute_layout(graph, LayoutStyle.SIMILARITY)
print(f"Positions computed for {len(positions)} nodes")
if positions:
    pos_sample = list(positions.items())[:3]
    for node_id, (x, y) in pos_sample:
        print(f"  {node_id[:30]}...: ({x:.1f}, {y:.1f})")

# Check color and size calculations
print("\n\nChecking visual encoding...")
years = [d.get("year", 2020) for _, d in graph.nodes(data=True)]
citations = [d.get("citation_count", 0) for _, d in graph.nodes(data=True)]
print(f"Year range: {min(years)} - {max(years)}")
print(f"Citation range: {min(citations)} - {max(citations)}")

import numpy as np

citation_percentiles = (
    np.percentile(citations, [50, 75, 90, 95]) if citations else [10, 20, 30, 40]
)
print(f"Citation percentiles (50,75,90,95): {citation_percentiles}")

# Try to save visualization
print("\n\nSaving visualization...")
try:
    builder.visualize(
        graph, Path("out/debug_test.html"), layout_style=LayoutStyle.SIMILARITY
    )
    print("✓ Visualization saved to out/debug_test.html")
except Exception as e:
    print(f"✗ Error saving visualization: {e}")

print("\n\nDebug complete!")
