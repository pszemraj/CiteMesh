#!/usr/bin/env python
"""Test script to verify visualization works properly."""

from pathlib import Path
from citation_graph import CitationGraphBuilder, LayoutStyle, GraphConfig

print("Building citation graph...")

# Create builder with similarity layout
config = GraphConfig(layout_style=LayoutStyle.SIMILARITY)
builder = CitationGraphBuilder(config)

# Build graph with reasonable limits
graph = builder.build(
    "10.48550/arXiv.2507.11412", depth=1, max_citations=20, max_references=20
)

print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

# Check node data
print("\nSample node data:")
for i, (node_id, data) in enumerate(list(graph.nodes(data=True))[:3]):
    print(f"  {data.get('title', 'Unknown')[:40]}...")
    print(
        f"    Year: {data.get('year', 'N/A')}, Citations: {data.get('citation_count', 0)}"
    )

# Save visualization
output_path = Path("out/test_similarity.html")
print(f"\nSaving visualization to {output_path}...")
builder.visualize(graph, output_path, layout_style=LayoutStyle.SIMILARITY)
print("✓ HTML saved")

# Convert to PNG for comparison
print("\nConverting to PNG...")
import subprocess

result = subprocess.run(
    ["python", "html_to_png.py", str(output_path), "out/test_similarity.png"],
    capture_output=True,
    text=True,
)
if result.returncode == 0:
    print("✓ PNG saved to out/test_similarity.png")
else:
    print(f"✗ PNG conversion failed: {result.stderr}")

print("\nDone! Compare out/test_similarity.png with out/REFERENCE.jpg")
