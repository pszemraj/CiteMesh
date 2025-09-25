# Paper Graph Visualizer

A tool for creating reference tool-style citation graph visualizations from academic papers using the Semantic Scholar API. Generates mesh-like similarity graphs that reveal research relationships through bibliographic coupling and co-citation analysis.

## Features

- **reference tool-style mesh visualization**: Papers connected by similarity, not just direct citations
- Build from any paper using DOI, arXiv ID, or Semantic Scholar ID  
- Direct matplotlib PNG output for reliable image generation
- Visual encoding: node size = citation count, node color = publication year
- ~40 most relevant papers selected through bibliographic coupling and co-citation
- Dense mesh structure with edges between all similar papers
- Organic clustering reveals research areas and relationships

## Installation

```bash
# Install required packages
pip install networkx matplotlib semanticscholar numpy
```

## Usage

### Basic Usage

```bash
# Generate a reference tool-style mesh visualization
python citation_graph.py "10.48550/arXiv.2507.11412"

# Using different paper identifiers
python citation_graph.py "arxiv:2507.11412"
python citation_graph.py "10.1145/3133956.3134029" 
python citation_graph.py "649def34f8be52c8b66281af98ae884c09aef38b"
```

This will create `out/final_visualization.png` with a mesh graph showing ~40 papers connected by similarity.

### Algorithm Overview

The visualizer implements the reference tool algorithm:

1. **Paper Collection**: Fetches seed paper's citations and references
2. **Similarity Calculation**: Computes pairwise similarity based on:
   - Temporal proximity (publication year difference)
   - Citation count ratio
3. **Edge Creation**: Connects papers with similarity > 0.2 threshold
4. **Force-Directed Layout**: Positions nodes using spring physics simulation
5. **Mesh Structure**: Creates edges between ALL similar papers, not just to seed

### Visual Encoding

- **Node Size**: Proportional to citation count (larger = more cited)
- **Node Color**: Gradient by publication year (darker = more recent)
- **Edges**: Weighted by similarity score (thicker = more similar)
- **Layout**: Force-directed creates organic clustering of related papers
- **Seed Paper**: Shown larger to indicate starting point

### Key Parameters

- **Max Papers**: ~40 papers selected by relevance
- **Similarity Threshold**: 0.2 (papers below this aren't connected)
- **Similarity Weights**: 50% temporal proximity, 50% citation ratio
- **Output**: Direct PNG at `out/final_visualization.png`

## Performance Notes

- Fetches direct citations/references only (no recursive traversal)
- Typically processes 40-80 papers total
- Creates 500-800 edges for proper mesh structure
- Completes in 1-2 minutes for most papers
- API rate limited to avoid throttling

## Output

Generates a PNG image at `out/final_visualization.png` with:
- reference tool-style mesh visualization
- ~40 most relevant papers
- Dense connectivity (typically 500-800 edges)
- Organic clustering showing research relationships
- Clear "Author, Year" labels
- Visual distinction between highly-cited and recent papers

## Example Output

The visualization creates a mesh similar to reference tool with:
- Central seed paper (your searched paper)
- Surrounding papers positioned by similarity
- Dense mesh of edges between related papers
- Natural clustering of research sub-areas
- Clear visual hierarchy by citation count

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed algorithm documentation including:
- reference tool algorithm analysis
- Similarity metric calculations
- Graph construction process
- Comparison with citation tree approaches

## Troubleshooting

### Common Issues

1. **"Paper not found"**: Verify paper ID format (DOI, arXiv ID, or S2 ID)
2. **Too few papers**: Some papers have limited citations/references in Semantic Scholar
3. **API errors**: The Semantic Scholar API may have temporary issues

## Example Papers to Try

```bash
# Transformer architecture paper
python citation_graph.py "arxiv:1706.03762"

# BERT paper  
python citation_graph.py "arxiv:1810.04805"

# Any paper by DOI
python citation_graph.py "10.1145/3133956.3134029"
```

## License

MIT