# Potential Improvements to Match Connected Papers

## Critical Missing Features

### 1. Expand Candidate Pool
**Current**: Only fetch 20-40 direct citations/references
**Needed**: Fetch citations-of-citations and references-of-references to build ~1000+ candidate pool
```python
# Pseudocode for expanded fetching
candidates = set()
# Level 1: Direct citations/references
for paper in seed.citations + seed.references:
    candidates.add(paper)
    # Level 2: Their citations/references (limited)
    for p2 in paper.citations[:10] + paper.references[:10]:
        candidates.add(p2)
```

### 2. True Bibliographic Coupling
**Current**: Simulated with year/citation similarity
**Needed**: Actually compare reference lists
```python
def bibliographic_coupling(paper1, paper2):
    refs1 = set(paper1.references)
    refs2 = set(paper2.references)
    intersection = refs1 & refs2
    if not refs1 or not refs2:
        return 0
    return len(intersection) / math.sqrt(len(refs1) * len(refs2))
```

### 3. Co-citation Analysis
**Current**: Not implemented
**Needed**: Find papers frequently cited together
```python
def cocitation_similarity(paper1, paper2):
    # Find papers that cite both paper1 and paper2
    citers1 = set(paper1.citations)
    citers2 = set(paper2.citations)
    shared_citers = citers1 & citers2
    if not citers1 or not citers2:
        return 0
    return len(shared_citers) / math.sqrt(len(citers1) * len(citers2))
```

### 4. Prior and Derivative Works
**Current**: Not implemented
**Needed**: Identify common ancestors/descendants
```python
def find_prior_works(graph_papers):
    # Papers cited by many nodes in the graph
    citation_counts = Counter()
    for paper in graph_papers:
        for ref in paper.references:
            citation_counts[ref] += 1
    return citation_counts.most_common(10)

def find_derivative_works(graph_papers):
    # Papers that cite many nodes in the graph
    citing_counts = Counter()
    for paper in graph_papers:
        for citer in paper.citations:
            citing_counts[citer] += 1
    return citing_counts.most_common(10)
```

## Implementation Challenges

1. **API Limitations**: Semantic Scholar rate limits prevent fetching 50,000 papers
   - Solution: Implement caching and batch processing
   - Alternative: Focus on most promising candidates first

2. **Performance**: Computing pairwise similarity for thousands of papers is expensive
   - Solution: Use sparse matrices and vectorized operations
   - Alternative: Progressive refinement - start with rough similarity, refine top candidates

3. **Memory**: Storing reference/citation lists for thousands of papers
   - Solution: Use efficient data structures (sets, sparse matrices)
   - Alternative: Stream processing with database backend

## Recommended Improvements (Feasible)

### Phase 1: Better Similarity (Quick Win)
- Fetch full reference lists for all papers (not just metadata)
- Implement true bibliographic coupling
- Cache paper data to avoid repeated API calls

### Phase 2: Expanded Coverage
- Implement 2-hop fetching (citations of citations)
- Add co-citation analysis
- Increase candidate pool to 200-500 papers

### Phase 3: Additional Features  
- Add prior/derivative works sidebar
- Implement year-based penalties for cross-generation connections
- Add discipline filtering based on paper fields

## Current Strengths to Preserve

✅ Clean, working implementation
✅ Good CLI interface with customization options
✅ Auto-naming from paper titles
✅ Correct force-directed layout
✅ Proper visual encoding
✅ Fast execution (10-30 seconds)

## Conclusion

Our implementation captures the visual style and basic concept of Connected Papers but uses simplified similarity metrics. The main gap is the candidate pool size and true bibliographic coupling. These could be added incrementally without breaking the current working system.