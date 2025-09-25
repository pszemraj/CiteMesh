# Paper Graph Visualizer

A tool for creating Connected Papers-style citation graph visualizations from academic papers using the Semantic Scholar API. Generates mesh-like similarity graphs that reveal research relationships through bibliographic coupling and co-citation analysis.

## Features

- **Connected Papers-style mesh visualization**: Papers connected by similarity, not just direct citations
- Build from any paper using DOI, arXiv ID, or Semantic Scholar ID  
- Direct matplotlib PNG output for reliable image generation
- Visual encoding: node size = citation count, node color = publication year
- ~40 most relevant papers selected through bibliographic coupling and co-citation
- Dense mesh structure with edges between all similar papers
- Organic clustering reveals research areas and relationships

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/paper-graph-vis.git
cd paper-graph-vis

# Install dependencies
pip install -r requirements.txt

# Generate your first visualization (auto-names output)
python citation_graph.py "arxiv:1706.03762"  # Creates: out/attention-is-all-you-need.png

# View the output
open out/*.png  # macOS - opens the generated file
# or
xdg-open out/*.png  # Linux
```

## Installation

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install packages directly
pip install networkx matplotlib semanticscholar numpy
```

## Usage

### Basic Usage

```bash
# Generate visualization with auto-named output (recommended)
python citation_graph.py "arxiv:1706.03762"         # Creates: out/attention-is-all-you-need.png
python citation_graph.py "arxiv:1810.04805"         # Creates: out/bert-pre-training-of-deep-bidirectional.png

# Custom output path
python citation_graph.py "10.1038/nature14539" -o my_deep_learning.png

# Quick visualization with fewer papers for faster results
python citation_graph.py "arxiv:2005.14165" -p 20 -i 50

# High-quality visualization with more papers and iterations
python citation_graph.py "arxiv:1706.03762" -p 60 -i 200 -d 300

# Adjust similarity threshold for denser/sparser mesh
python citation_graph.py "10.1145/3133956.3134029" -s 0.3  # Stricter (fewer edges)
python citation_graph.py "10.1145/3133956.3134029" -s 0.1  # Looser (more edges)
```

This will create an auto-named PNG file in the `out/` directory (or your specified path) with a mesh graph showing papers connected by similarity.

### Algorithm Overview

The visualizer implements the Connected Papers algorithm:

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

### Command-Line Options

```bash
python citation_graph.py -h  # Show help with all options
```

| Option | Description | Default |
|--------|-------------|---------|  
| `paper_id` | Paper identifier (DOI, arXiv ID, or S2 ID) | Required |
| `-o, --output` | Output PNG file path | Auto-named from title |
| `-p, --max-papers` | Maximum total papers to include | 40 |
| `-c, --max-citations` | Maximum citations to fetch | 20 |
| `-r, --max-references` | Maximum references to fetch | 20 |
| `-s, --similarity-threshold` | Min similarity for edges (0-1) | 0.2 |
| `-i, --iterations` | Layout iterations (quality) | 100 |
| `-d, --dpi` | Output image resolution | 150 |
| `-h, --help` | Show help message | - |

### Key Parameters

- **Max Papers**: 40 papers selected by relevance (adjustable with `-p`)
- **Similarity Threshold**: 0.2 minimum for edge creation (adjustable with `-s`)
- **Similarity Weights**: 50% temporal proximity, 50% citation ratio
- **Output Format**: PNG image via matplotlib (resolution adjustable with `-d`)
- **Auto-naming**: Output files automatically named from paper title when `-o` not specified

## Performance Notes

- Fetches direct citations/references only (no recursive traversal)
- Typically processes 40-80 papers total
- Creates 500-800 edges for proper mesh structure
- Completes in 1-2 minutes for most papers
- API rate limited to avoid throttling

## Output

Generates a PNG image (auto-named from paper title) with:
- Connected Papers-style mesh visualization
- ~40 most relevant papers
- Dense connectivity (typically 500-800 edges)
- Organic clustering showing research relationships
- Clear "Author, Year" labels
- Visual distinction between highly-cited and recent papers

## Example Output

The visualization creates a mesh similar to Connected Papers with:
- Central seed paper (your searched paper)
- Surrounding papers positioned by similarity
- Dense mesh of edges between related papers
- Natural clustering of research sub-areas
- Clear visual hierarchy by citation count

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed algorithm documentation including:
- Connected Papers algorithm analysis
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