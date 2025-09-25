from citation_graph import CitationGraphBuilder, LayoutStyle, GraphConfig

config = GraphConfig(layout_style=LayoutStyle.SIMILARITY)
builder = CitationGraphBuilder(config, rate_limit=0)

# Small test
graph = builder.build(
    "10.48550/arXiv.2507.11412", depth=1, max_citations=5, max_references=5
)
positions = builder.compute_layout(graph, LayoutStyle.SIMILARITY)

print(f"Nodes: {graph.number_of_nodes()}")
print(f"Positions computed: {len(positions)}")
print("\nFirst 3 positions:")
for i, (node, (x, y)) in enumerate(list(positions.items())[:3]):
    print(f"  {node[:30]}...: x={x:.1f}, y={y:.1f}")

# Check if positions are valid
x_vals = [p[0] for p in positions.values()]
y_vals = [p[1] for p in positions.values()]
print(f"\nX range: {min(x_vals):.1f} to {max(x_vals):.1f}")
print(f"Y range: {min(y_vals):.1f} to {max(y_vals):.1f}")
