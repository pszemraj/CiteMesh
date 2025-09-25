# reference tool Architecture Analysis

## Reference Image Analysis (out/REFERENCE.jpg)

### Visual Characteristics
1. **Node Count**: ~40 papers (not 200+)
2. **Central Paper**: One dominant node (Köhler, 2019) - significantly larger than others
3. **Layout Pattern**: Organic, asymmetric clustering - NOT circular, NOT grid-based
4. **Temporal Flow**: Clear left-to-right progression (2018→2021)
5. **Edge Density**: Dense mesh in center, sparse at periphery
6. **Labels**: Visible "Author, Year" format on every node
7. **Color Gradient**: Light (old) to dark (new) papers
8. **Size Variation**: 5-6 distinct size classes based on importance/citations

### Why This Layout Emerges (Not Random!)

reference tool doesn't use traditional citation trees. Instead:

1. **Similarity Graph** (not citation graph):
   - Edges represent similarity, NOT direct citations
   - Similarity = bibliographic coupling + co-citation analysis
   - Papers citing similar papers cluster together
   - Papers cited together cluster together

2. **The Central Paper**:
   - This is the SEED paper the user searched for
   - Gets the largest size and most central position
   - All other papers are selected based on similarity to this seed
   - NOT the most cited paper, but the query paper

3. **Paper Selection Algorithm**:
   - Start with seed paper
   - Find papers with HIGH similarity to seed (not just citations/references)
   - Limited to ~40 most relevant papers
   - Includes both older foundational works AND newer derivative works
   - NOT recursive fetching of citations of citations

4. **Layout Forces**:
   - Papers with high similarity attract each other
   - All papers repel to prevent overlap
   - NO forced temporal positioning - time emerges naturally
   - Seed paper has strong centrality force

## Current Implementation (citation_graph.py)

### Successfully Implemented Features

1. **Correct Paper Fetching** (lines 46-86)
   - Fetches ONLY direct citations and references of seed paper
   - No recursive fetching that would explode to 200+ papers
   - Respects configurable limits via CLI parameters
   - Uses proper Semantic Scholar API attributes (.paper not .citingPaper)

2. **Simulated Similarity-Based Selection** (lines 94-123)
   - Creates mesh connections based on similarity metrics
   - Combines temporal proximity and citation count ratio
   - Configurable similarity threshold (default 0.2)
   - Special handling for seed paper connections

3. **Force-Directed Layout** (line 141)
   - Uses NetworkX spring_layout with similarity weights
   - Natural clustering emerges from similarity-based edges
   - Seed paper gently centered (lines 146-149)
   - Configurable iterations for quality

4. **Proper Visual Encoding** (lines 150-237)
   - Seed paper emphasized with larger size (1200 vs 80-480)
   - Node size based on citation count
   - Color gradient by publication year
   - Edge weight/opacity based on similarity
   - Clear "Author, Year" labels

## Algorithm as Implemented

### Phase 1: Paper Collection
```python
def build_mesh_graph(paper_id, max_papers=40, max_citations=20, max_references=20):
    1. Fetch seed paper via Semantic Scholar API
    2. Get citations (papers citing seed) - up to max_citations
    3. Get references (papers seed cites) - up to max_references
    4. Add papers until reaching max_papers limit
    5. Mark seed with is_seed=True flag
```

### Phase 2: Similarity Mesh Creation
```python
# Simplified similarity for performance (lines 94-123)
For each pair of papers:
  - year_similarity = 1.0 / (1.0 + year_diff / 3.0)
  - citation_ratio = min(cit1, cit2) / max(cit1, cit2)
  - similarity = 0.5 * year_sim + 0.5 * citation_ratio
  - Add random factor (0.5-1.5x) for organic appearance
  - Create edge if similarity > threshold (configurable)
```

### Phase 3: Force-Directed Layout
```python
# Using NetworkX spring_layout (line 141)
pos = nx.spring_layout(graph, k=1.2, iterations=iterations, seed=42, weight="weight")

# Ensure seed stays central (lines 146-149)
if seed_id in pos:
    current = pos[seed_id]
    center = np.array([0.5, 0.5])
    pos[seed_id] = current * 0.4 + center * 0.6  # Blend toward center
```

### Phase 4: Visual Encoding
```python
# Node sizing (lines 151-159)
if is_seed:
    size = 1200
else:
    size = 80 + min(400, citation_count * 3)

# Color by year (lines 161-176)  
if year_norm < 0.33: color = "#b8d4e3"  # Light (old)
elif year_norm < 0.66: color = "#6ba3be"  # Medium
else: color = "#457b9d"  # Dark (recent)

# Edges styled by similarity (lines 178-196)
alpha = min(0.6, weight)
width = max(0.5, weight * 2)

# Labels (lines 213-237)
format: "{LastName}, {Year}"
font_size: 10 for seed, 8 for others
```

## Key Insights

1. **reference tool is NOT a citation tree visualizer** - it's a similarity graph
2. **The layout is data-driven** - clustering emerges from actual paper relationships
3. **Limited scope is intentional** - ~40 papers is optimal for readability
4. **Seed paper is special** - it's the user's query, not just another node
5. **Temporal patterns emerge** - they're not forced, they arise from citation patterns

## CLI Parameters (Restored)

| Option | Description | Default |
|--------|-------------|---------|  
| `-p, --max-papers` | Maximum total papers to include | 40 |
| `-c, --max-citations` | Maximum citations to fetch | 20 |
| `-r, --max-references` | Maximum references to fetch | 20 |
| `-s, --similarity-threshold` | Min similarity for edges (0-1) | 0.2 |
| `-i, --iterations` | Layout iterations (quality) | 100 |
| `-d, --dpi` | Output image resolution | 150 |
| `-o, --output` | Output path (auto-named if not specified) | None |

## Performance Characteristics

- **Typical graph size**: 40 nodes, 600-800 edges
- **API calls**: 3 (seed + citations + references)
- **Processing time**: 10-30 seconds
- **Output**: PNG via matplotlib, auto-named from paper title

## Known Limitations

1. **Simplified similarity**: Uses temporal/citation metrics instead of true bibliographic coupling
2. **No recursive fetching**: Only direct citations/references (by design)
3. **Font rendering**: May warn about missing glyphs for non-Latin characters
4. **API limits**: Semantic Scholar rate limiting may affect large fetches