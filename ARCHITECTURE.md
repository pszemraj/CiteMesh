# Connected Papers Architecture Analysis

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

Connected Papers doesn't use traditional citation trees. Instead:

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

## Current Implementation Problems

### Problem 1: Fetching Too Many Papers
```python
# WRONG - citation_graph.py lines 284-286
if current_depth < depth - 1 and i < 5:
    fetch_and_add(citing_node.id, current_depth + 1)  # Recursive explosion!
```
This recursively fetches citations of citations, leading to 200+ papers.

**Solution**: Don't recursively fetch. Get direct citations/references of ONLY the seed paper.

### Problem 2: Wrong Paper Selection
```python
# WRONG - We're getting ALL citations/references up to a limit
citations = self.client.get_paper_citations(paper.paperId, limit=max_citations)
```
Connected Papers doesn't just take the first N citations. It selects papers based on SIMILARITY.

**Solution**: 
1. Get a larger pool of candidates (citations + references)
2. Calculate similarity to seed for each
3. Select top ~40 by similarity score

### Problem 3: Wrong Layout Algorithm
```python
# WRONG - citation_graph.py lines 484-497
# Start with temporal positioning
year_norm = (years[i] - min_year) / year_range
x_base = 200 + year_norm * 400  # Forced temporal positioning!
```
This forces papers into temporal positions, preventing natural clustering.

**Solution**: 
1. Initialize positions randomly or in a small circle
2. Let force simulation create natural clusters
3. Temporal positioning should EMERGE from the data

### Problem 4: No Seed Paper Emphasis
```python
# WRONG - All papers treated equally in initial fetch
centrality = nx.degree_centrality(graph)
seed_node = max(centrality, key=centrality.get)  # This finds most connected, not seed!
```

**Solution**: Track which paper is the seed from the beginning.

## Correct Algorithm

### Phase 1: Paper Collection
```
1. Fetch seed paper
2. Get seed's citations (papers citing seed) - up to 100
3. Get seed's references (papers seed cites) - up to 100  
4. For each paper in this pool:
   - Calculate similarity to seed
   - Similarity = shared_refs/total_refs + shared_citations/total_citations
5. Sort by similarity, take top 40
6. Mark seed paper specially
```

### Phase 2: Similarity Matrix
```
For each pair of selected papers:
  - Bibliographic coupling = |shared_references| / |union_references|
  - Co-citation = |shared_citations| / |union_citations|  
  - Similarity = 0.6 * coupling + 0.4 * cocitation
  - Apply temporal decay factor
```

### Phase 3: Force-Directed Layout
```
Initialize:
  - Seed at center (0, 0)
  - Others in small random cloud around seed
  
Forces:
  - Repulsion: charge = k * sqrt(citations) for all pairs
  - Attraction: spring force based on similarity (only if sim > threshold)
  - Seed anchor: gentle force keeping seed near center
  
Run simulation until convergence
```

### Phase 4: Visual Encoding
```
Node size:
  - Seed: largest (size = 100)
  - Others: size = 10 + sqrt(citations) * scale_factor
  
Node color:
  - Gradient based on year (light=old, dark=new)
  
Edges:
  - Only show if similarity > 0.15
  - Width and opacity based on similarity strength
  
Labels:
  - "LastName, Year" format
  - Must be readable (min font size 8pt)
```

## Key Insights

1. **Connected Papers is NOT a citation tree visualizer** - it's a similarity graph
2. **The layout is data-driven** - clustering emerges from actual paper relationships
3. **Limited scope is intentional** - ~40 papers is optimal for readability
4. **Seed paper is special** - it's the user's query, not just another node
5. **Temporal patterns emerge** - they're not forced, they arise from citation patterns

## Implementation Plan

1. Create new `connected_papers_viz.py` implementing the correct algorithm
2. Strictly limit to ~40 papers using similarity selection
3. Properly track and emphasize seed paper
4. Use pure force-directed layout without temporal forcing
5. Test with same paper as reference to verify equivalence