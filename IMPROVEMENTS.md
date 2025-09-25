# Improvements to Match Connected Papers

## Current Status (improved-alg branch)

### ✅ Completed Improvements

#### 1. True Bibliographic Coupling (Partially Complete)
**Implemented**: Papers now use actual shared references for similarity calculation
- Using Connected Papers formula: `intersection / sqrt(|A| * |B|)`
- Fetching reference lists for seed and first 10 papers
- 70% bibliographic coupling weight, 30% temporal similarity

**Limitation**: Only fetching refs for first 10 papers due to API timeout issues

#### 2. Temporal Penalties
**Implemented**: Exponential decay for cross-generation connections
- Formula: `math.exp(-year_diff / 8)`
- Prevents connecting papers from vastly different eras

#### 3. Improved Seed Centrality
**Implemented**: Seed node properly centered and emphasized
- Force seed to exact center
- Larger size (2000 vs 150-800 for others)
- Surrounding nodes kept at reasonable distance

#### 4. Year Diversity
**Implemented**: Fetching both references (older) and citations (newer)
- References first (up to half of max_papers)
- Then citations to fill remaining slots
- Results in better temporal spread

### ⚠️ Partially Implemented

#### 1. Bibliographic Coupling Coverage
**Issue**: Can only fetch references for ~10 papers before timeout
**Impact**: Most papers have empty reference sets, reducing coupling effectiveness

#### 2. Recommendations API
**Issue**: Returns no results for most papers (requires specific S2 ID format)
**Fallback**: Using citations/references as before

### ❌ Not Yet Implemented

#### 1. Co-citation Analysis
**Current**: Not tracking papers cited together
**Needed**: Find papers that are frequently cited alongside the seed

#### 2. Smart Candidate Selection
**Current**: Just taking direct citations/references
**Needed**: Papers that cite the same references as seed (true bibliographic coupling candidates)

#### 3. Prior and Derivative Works Lists
**Current**: Not showing common ancestors/descendants
**Needed**: Identify papers referenced by many in graph (prior) and papers citing many in graph (derivative)
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

## Comparison with Reference Image

### Reference (out/REFERENCE.jpg)
- **Central node**: Köhler, 2019 (large, prominent)
- **Layout**: Organic, natural spread across canvas
- **Density**: Dense mesh throughout, ~40 nodes, hundreds of edges
- **Years**: 2018-2021 (tight 3-year span)
- **Clustering**: Natural groups emerge from similarity

### Our Current Output (improved-alg branch)
- **Central node**: Seed properly centered and large ✅
- **Layout**: Better than before but still somewhat lopsided
- **Density**: Good (~600-700 edges for 40 nodes) ✅
- **Years**: Split between old refs (2016-2017) and new citations (2025)
- **Clustering**: Some clustering but not as organic as reference

## Key Remaining Gaps

1. **API Constraints**: Can't fetch 50,000 papers like real Connected Papers
2. **Reference Fetching**: Timeouts prevent getting refs for all papers
3. **Co-citation**: Not implemented due to API limits
4. **Year Bias**: Getting mostly 2025 papers from citations (this is correct - newer papers citing the 2017 Transformer)

## Next Steps (Priority Order)

### 1. Optimize Reference Fetching
- Add caching to avoid re-fetching
- Batch requests more efficiently
- Try to get refs for at least 20-30 papers

### 2. Improve Candidate Selection
- For each seed reference, get papers that also cite it
- This finds true bibliographically coupled papers
- Even with API limits, should improve quality

### 3. Add Prior/Derivative Works Display
- Simple analysis of what graph papers commonly cite/are cited by
- Just print to console, don't need UI

## Current Strengths

✅ True bibliographic coupling formula implemented
✅ Proper seed centrality
✅ Good edge density (~600-700 for 40 nodes)
✅ CLI with all key parameters
✅ Auto-naming from paper titles
✅ Works within API constraints

## Bottom Line

We've successfully implemented the core Connected Papers algorithm with true bibliographic coupling. The main limitation is API constraints preventing us from analyzing 50,000 papers. Within the ~100 paper limit, we're achieving reasonable results that capture the essence of Connected Papers' approach.