"""
Tests for similarity calculations.
"""

from citemesh.models import Paper
from citemesh.strategies.base import GraphBuilderStrategy


class TestSimilarityFunctions:
    """Test similarity computation functions."""

    def test_temporal_similarity_same_year(self):
        """Test temporal similarity for papers from same year."""
        paper1 = Paper(paper_id="p1", title="Test 1", year=2020)
        paper2 = Paper(paper_id="p2", title="Test 2", year=2020)

        sim = GraphBuilderStrategy.temporal_similarity(paper1, paper2)
        assert sim == 1.0

    def test_temporal_similarity_close_years(self):
        """Test temporal similarity for papers 2 years apart."""
        paper1 = Paper(paper_id="p1", title="Test 1", year=2020)
        paper2 = Paper(paper_id="p2", title="Test 2", year=2022)

        sim = GraphBuilderStrategy.temporal_similarity(paper1, paper2)
        # Should be high but not 1.0
        assert 0.5 < sim < 1.0

    def test_temporal_similarity_distant_years(self):
        """Test temporal similarity for papers >5 years apart."""
        paper1 = Paper(paper_id="p1", title="Test 1", year=2010)
        paper2 = Paper(paper_id="p2", title="Test 2", year=2020)

        sim = GraphBuilderStrategy.temporal_similarity(paper1, paper2)
        # Should have strong penalty
        assert sim == 0.1

    def test_temporal_similarity_with_unknown_year(self):
        """Unknown year should return neutral temporal similarity."""
        paper1 = Paper(paper_id="p1", title="Test 1", year=None)
        paper2 = Paper(paper_id="p2", title="Test 2", year=2020)
        sim = GraphBuilderStrategy.temporal_similarity(paper1, paper2)
        assert sim == 0.5

    def test_citation_similarity_similar_counts(self):
        """Test citation similarity for papers with similar citation counts."""
        paper1 = Paper(paper_id="p1", title="Test 1", year=2020, citation_count=100)
        paper2 = Paper(paper_id="p2", title="Test 2", year=2020, citation_count=105)

        sim = GraphBuilderStrategy.citation_similarity(paper1, paper2)
        # Should be very high (log scale makes them nearly identical)
        assert sim > 0.9

    def test_citation_similarity_different_magnitudes(self):
        """Test citation similarity for papers with very different citation counts."""
        paper1 = Paper(paper_id="p1", title="Test 1", year=2020, citation_count=10)
        paper2 = Paper(paper_id="p2", title="Test 2", year=2020, citation_count=1000)

        sim = GraphBuilderStrategy.citation_similarity(paper1, paper2)
        # Should be lower due to magnitude difference
        assert 0.1 < sim < 0.7

    def test_citation_similarity_zero_citations(self):
        """Test citation similarity when one paper has no citations."""
        paper1 = Paper(paper_id="p1", title="Test 1", year=2020, citation_count=0)
        paper2 = Paper(paper_id="p2", title="Test 2", year=2020, citation_count=100)

        sim = GraphBuilderStrategy.citation_similarity(paper1, paper2)
        # Should return default value
        assert sim == 0.3

    def test_bibliographic_coupling_identical_refs(self):
        """Test bibliographic coupling for papers with identical references."""
        paper1 = Paper(
            paper_id="p1",
            title="Test 1",
            year=2020,
            references=["ref1", "ref2", "ref3"],
        )
        paper2 = Paper(
            paper_id="p2",
            title="Test 2",
            year=2020,
            references=["ref1", "ref2", "ref3"],
        )

        coupling = GraphBuilderStrategy.bibliographic_coupling(paper1, paper2)
        assert coupling == 1.0

    def test_bibliographic_coupling_no_overlap(self):
        """Test bibliographic coupling for papers with no shared references."""
        paper1 = Paper(
            paper_id="p1",
            title="Test 1",
            year=2020,
            references=["ref1", "ref2"],
        )
        paper2 = Paper(
            paper_id="p2",
            title="Test 2",
            year=2020,
            references=["ref3", "ref4"],
        )

        coupling = GraphBuilderStrategy.bibliographic_coupling(paper1, paper2)
        assert coupling == 0.0

    def test_exponential_temporal_decay(self):
        """Test exponential temporal decay function."""
        paper1 = Paper(paper_id="p1", title="Test 1", year=2020)
        paper2 = Paper(paper_id="p2", title="Test 2", year=2020)

        # Same year should be 1.0
        decay = GraphBuilderStrategy.exponential_temporal_decay(paper1, paper2)
        assert decay == 1.0

        # 8 years apart (default decay factor) should be ~0.368
        paper3 = Paper(paper_id="p3", title="Test 3", year=2012)
        decay = GraphBuilderStrategy.exponential_temporal_decay(
            paper1, paper3, decay_factor=8.0
        )
        assert abs(decay - 0.368) < 0.01

    def test_exponential_temporal_decay_with_unknown_year(self):
        """Unknown year should return neutral decay value."""
        paper1 = Paper(paper_id="p1", title="Test 1", year=None)
        paper2 = Paper(paper_id="p2", title="Test 2", year=2020)
        decay = GraphBuilderStrategy.exponential_temporal_decay(paper1, paper2)
        assert decay == 0.5
