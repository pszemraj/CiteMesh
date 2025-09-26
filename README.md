# Paper Graph Visualizer

A tool for creating Connected Papers-style citation graph visualizations from academic papers using multiple approaches: citation networks, semantic embeddings, and hybrid intelligence. Generates mesh-like similarity graphs that reveal research relationships through bibliographic coupling, co-citation analysis, and content similarity.

## Features

- **Three visualization approaches**:
  - `citation_graph.py`: Citation-based similarity with co-citation patterns
  - `embedding_similarity.py`: Semantic similarity using sentence transformers
  - `hybrid_similarity.py`: Intelligent combination of both approaches
- **Connected Papers-style mesh visualization**: Papers connected by multiple similarity metrics
- Build from any paper using DOI, arXiv ID, or Semantic Scholar ID  
- Direct matplotlib PNG output with auto-naming from paper titles
- Visual encoding: 
  - Node size = citation count + importance ranking
  - Node color = smooth gradient by publication year
  - Edge thickness = similarity strength
- ~30-40 most relevant papers with sparse, meaningful connections
- Organic Kamada-Kawai clustering reveals research relationships
- Multi-factor similarity: temporal, categorical, author collaboration, and semantic

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/paper-graph-vis.git
cd paper-graph-vis

# Install dependencies
pip install -r requirements.txt

# Generate visualization with citation-based similarity
python citation_graph.py "arxiv:1706.03762"  # Creates: out/attention-is-all-you-need.png

# Or use semantic embedding similarity
python embedding_similarity.py "arxiv:1706.03762" -p 30

# Or use hybrid approach (combines both)
python hybrid_similarity.py "arxiv:1706.03762" --max-semantic 10

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
pip install networkx matplotlib semanticscholar numpy sentence-transformers torch datasets joblib tqdm
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

The visualizers implement enhanced versions of the Connected Papers algorithm:

#### Citation Graph (`citation_graph.py`)
1. **Paper Collection**: Fetches seed paper's citations and references
2. **Co-citation Analysis**: Papers cited together are considered similar
3. **Bibliographic Coupling**: Papers citing same works are related
4. **Similarity Calculation**: 
   - Temporal proximity with strong penalties (>5 years apart)
   - Citation impact similarity (log scale)
   - Simulated shared references
5. **Sparse Edge Creation**: ~30-40 edges total for clean visualization
6. **Kamada-Kawai Layout**: Organic clustering of related papers

#### Embedding Similarity (`embedding_similarity.py`)
1. **Dataset Loading**: Uses HuggingFace ArXiv datasets with progress bars
2. **Embedding Computation**: Sentence transformers (EmbeddingGemma by default)
3. **Multi-Factor Similarity**:
   - Semantic embedding similarity (50%)
   - Year proximity (20%)
   - Category overlap (20%)  
   - Author collaboration (10%)
4. **Top-k Edge Selection**: Each node connects to 2-3 most similar papers
5. **Citation Integration**: Fetches real citation counts from Semantic Scholar

#### Hybrid Approach (`hybrid_similarity.py`)
1. **Intelligent Paper Selection**: Filters citations by relevance score
2. **Semantic Enrichment**: Finds semantically similar papers from embeddings
3. **Metadata Fetching**: Gets citation counts for top semantic matches
4. **Co-citation Patterns**: Analyzes papers cited/referenced together
5. **Adaptive Similarity**: Different weights for different relationship types
6. **Edge Limiting**: Max 5 connections per node for clarity

### Visual Encoding

- **Node Size**: 
  - Extreme variation (80-2500 pixels) based on importance
  - Combines citation count (log scale) + similarity ranking
  - Seed paper always largest (2500 pixels)
- **Node Color**: 
  - Smooth RGB gradient by year (not discrete bands)
  - Light blue/gray (old) → dark teal (recent)
  - Special colors for seed (red) and relationship types in hybrid
- **Edges**: 
  - Very thin and subtle (0.3-0.6 alpha)
  - Thickness based on similarity strength
  - Sparse connections (~30-40 edges total)
- **Layout**: 
  - Kamada-Kawai for organic clustering
  - Small random perturbations for natural look
  - Papers cluster by actual similarity, not forced positioning
- **Labels**: Author surname + year format

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

- **Citation Graph**: Fetches direct citations/references only
- **Embedding Similarity**: Caches embeddings with joblib for speed
- **Hybrid**: Intelligently fetches metadata for top papers only
- Typically processes 30-40 papers total
- Creates 30-70 edges for clean, readable structure
- Completes in 30-60 seconds for most papers
- Uses tqdm progress bars for long operations
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
- Connected Papers algorithm analysis and implementation
- Co-citation and bibliographic coupling theory
- Multi-factor similarity calculations
- Comparison of three approaches (citation, embedding, hybrid)
- Visual encoding decisions and improvements
- Performance optimizations and caching strategies

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