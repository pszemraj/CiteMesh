# Citation Graph Visualizer

An interactive tool for building and visualizing citation networks from academic papers using the Semantic Scholar API.

## Features

- Build citation networks from any paper using DOI, arXiv ID, or Semantic Scholar ID
- Interactive HTML visualization with customizable depth and breadth
- Export to multiple formats (GraphML, GEXF, JSON)
- Detailed graph statistics and analysis
- Progress bars for long-running operations
- Caching to avoid redundant API calls

## Installation

```bash
# Install required packages
pip install networkx pyvis semanticscholar tqdm
```

## Usage

### Basic Usage

```bash
# Build a citation graph for an arXiv paper (depth=1 by default)
python citation_graph.py "arxiv:2507.11412" --output my_graph.html

# Using a DOI
python citation_graph.py "10.1145/3133956.3134029" --output paper_graph.html

# Using a Semantic Scholar paper ID
python citation_graph.py "649def34f8be52c8b66281af98ae884c09aef38b" --output s2_graph.html
```

### Advanced Options

```bash
# Full example with all options
python citation_graph.py "arxiv:2507.11412" \
    --output my_graph.html \
    --depth 1 \                    # Traverse depth (default: 1, use 2+ with caution - very slow!)
    --max-citations 30 \           # Max citations per paper
    --max-references 30 \          # Max references per paper
    --export-all \                 # Export to GraphML, GEXF, and JSON
    --stats \                      # Show detailed statistics
    --rate-limit 0 \              # API rate limiting (0 = no limit)
    --verbose                      # Enable detailed logging
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `paper_id` | Paper identifier (DOI, arXiv ID, or S2 ID) | Required |
| `-o`, `--output` | Output HTML file path | `citation_graph.html` |
| `-d`, `--depth` | Maximum traversal depth | `1` (recommended) |
| `-c`, `--max-citations` | Max citations to fetch per paper | `20` |
| `-r`, `--max-references` | Max references to fetch per paper | `20` |
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
- Physics simulation for layout
- Color coding (gold = highly cited papers)

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

## License

MIT

## Author

Research Tools