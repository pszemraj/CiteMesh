# Citation Graph Visualizer

An interactive tool for building and visualizing citation networks from academic papers using the Semantic Scholar API. Features both traditional force-directed layouts and Connected Papers-style similarity-based clustering.

## Features

- Build citation networks from any paper using DOI, arXiv ID, or Semantic Scholar ID
- **Two layout styles**: Traditional force-directed or Connected Papers-style similarity clustering
- Interactive HTML visualization with customizable depth and breadth
- Visual encoding: node size = citation count, node color = publication year
- Export to multiple formats (GraphML, GEXF, JSON)
- Export visualizations to PNG using the included utility
- Detailed graph statistics and analysis
- Progress bars for long-running operations
- Caching to avoid redundant API calls

## Installation

```bash
# Install required packages
pip install networkx pyvis semanticscholar tqdm scikit-learn numpy

# For PNG export (optional)
pip install playwright
playwright install chromium
```

## Usage

### Basic Usage

```bash
# Build a citation graph with Connected Papers-style layout (default)
python citation_graph.py "10.48550/arXiv.2507.11412" --output out/my_graph.html

# Use traditional force-directed layout
python citation_graph.py "10.48550/arXiv.2507.11412" --layout force --output out/force_graph.html

# Quick test with minimal data (depth=1, 10 papers each)
python citation_graph.py "10.48550/arXiv.2507.11412" -d 1 -c 10 -r 10 --output out/test.html

# Using different paper identifiers
python citation_graph.py "arxiv:2507.11412" --output out/arxiv_graph.html
python citation_graph.py "10.1145/3133956.3134029" --output out/doi_graph.html
python citation_graph.py "649def34f8be52c8b66281af98ae884c09aef38b" --output out/s2_graph.html
```

### Advanced Options

```bash
# Full example with all options
python citation_graph.py "10.48550/arXiv.2507.11412" \
    --output out/my_graph.html \
    --layout similarity \          # Use Connected Papers-style layout (default)
    --depth 1 \                    # Traverse depth (default: 1, use 2+ with caution - very slow!)
    --max-citations 30 \           # Max citations per paper
    --max-references 30 \          # Max references per paper
    --export-all \                 # Export to GraphML, GEXF, and JSON
    --stats \                      # Show detailed statistics
    --rate-limit 0.5 \            # API rate limiting in seconds
    --verbose                      # Enable detailed logging

# Convert HTML to PNG for viewing/sharing
python html_to_png.py out/my_graph.html out/my_graph.png --width 1600 --height 1200
```

### Layout Styles

The tool supports two visualization styles:

1. **`similarity` (default)**: Connected Papers-style layout using bibliographic coupling and co-citation analysis. Papers with similar references/citations cluster together.
   - Best for understanding research areas and paper relationships
   - Uses t-SNE/MDS for positioning based on citation similarity
   
2. **`force`**: Traditional force-directed graph layout
   - Best for seeing direct citation paths
   - Uses spring physics simulation

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `paper_id` | Paper identifier (DOI, arXiv ID, or S2 ID) | Required |
| `-o`, `--output` | Output HTML file path | `citation_graph.html` |
| `-d`, `--depth` | Maximum traversal depth | `1` (recommended) |
| `-c`, `--max-citations` | Max citations to fetch per paper | `20` |
| `-r`, `--max-references` | Max references to fetch per paper | `20` |
| `--layout` | Layout algorithm: `force` or `similarity` | `similarity` |
| `--export-all` | Export to multiple formats | `False` |
| `--stats` | Print detailed graph statistics | `False` |
| `--rate-limit` | Seconds between API calls | `0.5` |
| `-v`, `--verbose` | Increase verbosity (can repeat) | `0` |

## Performance Notes

- **Depth = 1** (default): Fast, typically completes in under a minute
- **Depth = 2**: Can take 5-10 minutes depending on paper connectivity
- **Depth = 3+**: Not recommended unless you have time - can take hours!

The Semantic Scholar API can be slow, especially for papers with many citations/references. The tool now includes progress bars to track processing.

## Output Formats

### HTML Visualization
The default output is an interactive HTML file that can be opened in any web browser. Features include:
- Zoom and pan controls
- Node hovering for paper details  
- Node size represents citation count (larger = more cited)
- Node color represents publication year (darker = more recent)
- Red edges = references (paper cites another)
- Blue edges = citations (paper is cited by another)
- Physics controls for force-directed layout

### Export Formats
With `--export-all`, the tool also generates:
- **GraphML** (`.graphml`): For use with graph analysis tools
- **GEXF** (`.gexf`): For Gephi visualization
- **JSON** (`.json`): Node-link format for custom processing

## Example Statistics Output

```
============================================================
CITATION GRAPH STATISTICS
============================================================

Graph Structure:
  Nodes: 66
  Edges: 65
  Citations: 3
  References: 62
  Density: 0.0152
  Components: 1
  Largest Component: 66 nodes

Degree Statistics:
  Average Degree: 1.97
  Max In-Degree: 3
  Max Out-Degree: 62
  Clustering Coefficient: 0.0000

Most Cited Papers:
  - Paper Title 1: 3 citations
  - Paper Title 2: 1 citations
  ...
```

## API Rate Limiting

The Semantic Scholar API has rate limits. By default, the tool waits 0.5 seconds between calls. You can adjust this with `--rate-limit`:
- `--rate-limit 0`: No delay (fastest, may hit rate limits)
- `--rate-limit 0.5`: Default, balanced
- `--rate-limit 1`: Conservative, slower but safer

### Converting to PNG

Use the included utility to convert HTML graphs to PNG images:

```bash
# Basic conversion
python html_to_png.py out/my_graph.html out/my_graph.png

# Custom dimensions
python html_to_png.py out/my_graph.html out/large_graph.png --width 2000 --height 1500

# Process all HTML files in a directory
for html in out/*.html; do
    python html_to_png.py "$html" "${html%.html}.png"
done
```

## Troubleshooting

### Common Issues

1. **"Paper not found"**: Check that your paper ID is correct
2. **Rate limiting errors (429)**: Increase `--rate-limit` value
3. **Timeout errors**: The API can be slow; try reducing `--max-citations` and `--max-references`
4. **Missing citations/references**: Some papers may not have complete data in Semantic Scholar

### Tips for Better Performance

- Start with `--depth 1` (default) to test
- Use smaller values for `--max-citations` and `--max-references` for faster results
- Set `--rate-limit 0` for faster processing if not hitting rate limits
- Use `--verbose` to see detailed progress

## Example Workflows

### Compare Layout Styles
```bash
# Generate both layouts for the same paper
python citation_graph.py "10.48550/arXiv.2507.11412" --layout force --output out/force.html
python citation_graph.py "10.48550/arXiv.2507.11412" --layout similarity --output out/similarity.html

# Convert both to PNG for comparison
python html_to_png.py out/force.html out/force.png
python html_to_png.py out/similarity.html out/similarity.png
```

### Analyze a Research Area
```bash
# Get a comprehensive view with depth=2 (warning: slow!)
python citation_graph.py "10.48550/arXiv.2507.11412" \
    --depth 2 \
    --max-citations 15 \
    --max-references 15 \
    --layout similarity \
    --stats \
    --export-all \
    --output out/research_area.html
```

### Quick Paper Overview
```bash
# Fast overview with minimal API calls
python citation_graph.py "10.48550/arXiv.2507.11412" \
    -d 1 -c 5 -r 5 \
    --stats \
    --output out/quick_overview.html
```

## License

MIT