# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CiteMesh (Paper Graph Visualizer) creates Connected Papers-style citation graph visualizations using a unified architecture with three strategy implementations: citation networks, semantic embeddings, and hybrid intelligence.

**Version 2.0** features a complete architectural refactoring with:
- Unified `citemesh/` package with proper OOP architecture
- Real bibliographic coupling (actual shared references, not simulated)
- Type-safe data models with validation
- 30% code reduction through eliminating duplication
- Comprehensive test suite

## Core Commands

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run unified CLI (recommended)
python citemesh.py build "arxiv:1706.03762" --strategy citation
python citemesh.py build "arxiv:1706.03762" --strategy embedding
python citemesh.py build "arxiv:1706.03762" --strategy hybrid

# Legacy scripts (backward compatible, all use unified architecture)
python citation_graph.py "arxiv:1706.03762"
python embedding_similarity.py "arxiv:1706.03762"
python hybrid_similarity.py "arxiv:1706.03762"

# Test with quick visualization
python citemesh.py build "arxiv:1706.03762" --strategy citation -p 20 -i 50
```

### Testing
```bash
# Run test suite
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_models.py -v
```

### Code Quality
```bash
# Lint and format (pre-commit)
ruff check --fix . && ruff format .
```

## Architecture

### Unified Package Structure (v2.0)

The project now uses a unified `citemesh/` package with proper OOP architecture:

```
citemesh/
├── __init__.py              # Package exports
├── models.py                # Paper, Author data models with validation
├── config.py                # All configuration constants and weights
├── api_client.py            # S2 API wrapper with retry logic and caching
├── visualization.py         # Unified visualization (single implementation)
├── strategies/
│   ├── base.py             # GraphBuilderStrategy ABC
│   ├── citation.py         # CitationGraphBuilder (real bibliographic coupling)
│   ├── embedding.py        # EmbeddingGraphBuilder (semantic similarity)
│   └── hybrid.py           # HybridGraphBuilder (combines both)
```

### Strategy Pattern Implementation

All three approaches inherit from `GraphBuilderStrategy` base class:

1. **CitationGraphBuilder** (`citemesh/strategies/citation.py`)
   - Uses Semantic Scholar API for citations and references
   - **Real bibliographic coupling**: Fetches actual reference lists and computes shared references
   - Key method: `compute_similarity()` with temporal + citation + bibliographic factors
   - Strengths: Fast, authentic bibliographic relationships
   - Fixed: No longer uses random numbers for coupling

2. **EmbeddingGraphBuilder** (`citemesh/strategies/embedding.py`)
   - Downloads ArXiv datasets from HuggingFace
   - Uses sentence transformers (default: EmbeddingGemma-300m)
   - Key method: `collect_papers()` via semantic search, `compute_similarity()` with multi-factor scoring
   - Strengths: No API limits, finds conceptually similar papers
   - Caching: joblib cache for embeddings in `cache/joblib_cache/`

3. **HybridGraphBuilder** (`citemesh/strategies/hybrid.py`)
   - Combines citation and embedding builders
   - Adaptive weights based on paper source (citation vs semantic)
   - Key method: `compute_similarity()` with context-aware weighting
   - Edge limiting for clarity

### Similarity Algorithm Details

**Citation Graph Similarity**:
- 50% temporal proximity (strong penalty for papers >5 years apart: `exp(-year_diff / 8)`)
- 50% citation impact similarity (log scale)
- Simulated bibliographic coupling for shared references
- Threshold: 0.2 minimum for edge creation (adjustable via `-s`)

**Embedding Similarity**:
- 50% semantic embedding similarity (cosine similarity)
- 20% year proximity factor: `1.0 / (1.0 + year_diff / 3.0)`
- 20% category overlap (ArXiv categories)
- 10% author collaboration bonus

**Hybrid Similarity**:
- Adaptive weights depending on relationship type
- Co-citation boost for papers in same citation group
- Different thresholds for citation vs semantic edges
- Relevance scoring: `citationCount / (1 + year_diff)`

### Visual Encoding

All three scripts implement consistent visual encoding:
- **Node Size**: 80-2500 pixels based on citation count (log scale) + similarity ranking. Seed paper always largest (2500px)
- **Node Color**: Smooth RGB gradient by year (light blue/gray for old → dark teal for recent). Seed is red.
- **Edge Thickness**: 0.3-0.6 alpha, thickness proportional to similarity strength
- **Layout**: Kamada-Kawai with small random perturbations for organic look
- **Labels**: "Author surname, Year" format

### Output Structure

```
paper-graph-vis/
├── citation_graph.py          # Citation-based similarity (main algorithm)
├── embedding_similarity.py    # Semantic similarity approach
├── hybrid_similarity.py       # Combined approach
├── out/                       # Generated PNG visualizations (auto-named)
├── cache/
│   └── joblib_cache/          # Cached embeddings and datasets
├── requirements.txt           # Dependencies (matplotlib, networkx, semanticscholar, etc.)
├── README.md                  # User-facing documentation
└── ARCHITECTURE.md            # Algorithm analysis and Connected Papers theory
```

## Key Implementation Notes

### Paper Collection Strategy
- **Citation approach**: Fetches direct citations and references only (not recursive)
- **Embedding approach**: Searches 10k-100k paper corpus for semantic matches
- **Hybrid approach**: Combines filtered citations with top semantic matches
- All limit to ~30-40 papers for readability

### Caching Strategy
- ArXiv datasets cached with joblib in `cache/joblib_cache/`
- Embeddings cached per model to avoid recomputation
- No caching for Semantic Scholar API calls (rate limited)
- Cache invalidation: delete `cache/` directory

### Performance Characteristics
| Approach   | Nodes | Edges | Time | Best For |
|------------|-------|-------|------|----------|
| Citation   | 30-40 | 30-40 | 30s  | Papers with good S2 coverage |
| Embedding  | 30-35 | 50-70 | 45s  | Exploring semantic relationships |
| Hybrid     | 35-45 | 40-60 | 60s  | Comprehensive analysis |

### API Constraints
- Semantic Scholar API has rate limits (handled with delays)
- API may timeout for fetching many references (handled gracefully)
- Paper lookups support DOI, arXiv ID, or Semantic Scholar ID formats

## Common Parameters

All three scripts share similar CLI parameters:

```bash
-o, --output PATH          # Output PNG file (auto-named from title if not specified)
-p, --max-papers N         # Maximum papers to include (default: 40)
-i, --iterations N         # Layout iterations for quality (default: 100)
-d, --dpi N                # Output image resolution (default: 150)
```

Specific to each approach:
- **citation_graph.py**: `-c/--max-citations`, `-r/--max-references`, `-s/--similarity-threshold`
- **embedding_similarity.py**: `-m/--model`, `-d/--dataset-papers`, `-k/--top-k`
- **hybrid_similarity.py**: `--max-semantic`, `-c/--corpus-size`

## Testing Approaches

When testing changes:
1. Use quick test papers: `arxiv:1706.03762` (Transformer), `arxiv:1810.04805` (BERT)
2. Reduce parameters for speed: `-p 20 -i 50` for faster iteration
3. Check output in `out/` directory
4. Verify both visual quality and edge density (~30-70 edges)

## Graph Algorithm Background

The implementation is based on Connected Papers' approach:
- **Not a citation tree** - this is a similarity graph where edges represent multi-factor similarity
- **Co-citation analysis**: Papers cited together are considered related
- **Bibliographic coupling**: Papers citing the same works cluster together
- **Seed-centric**: The queried paper is always central and largest
- **Organic clustering**: Related papers naturally cluster via force-directed layout

See ARCHITECTURE.md for detailed algorithm analysis including the reference image comparison and similarity formula derivations.

## Important Constraints

1. **Don't recursively fetch citations** - this creates too many papers and loses focus
2. **Maintain sparse edges** - aim for 30-70 edges total, not 500-800
3. **Preserve auto-naming** - output files should be named from paper title when `-o` not specified
4. **Keep joblib caching** - embedding computation is expensive, caching is critical
5. **Limit corpus size** - embedding_similarity.py defaults to reasonable dataset splits (train[:2%])
