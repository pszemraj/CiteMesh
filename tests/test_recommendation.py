"""Tests for recommendation-based graph strategy."""

from unittest.mock import MagicMock, patch

from citemesh.core import Paper
from citemesh.strategies.recommendation import RecommendationGraphBuilder


class TestRecommendationGraphBuilder:
    """Tests for S2 recommendation graph behavior."""

    def test_tfidf_recommendation_prefers_related_papers(self):
        """Related papers should score higher than unrelated ones."""
        papers = {
            "seed": Paper(
                paper_id="seed",
                title="Attention Is All You Need",
                year=2017,
                abstract=(
                    "We propose a new architecture based on attention mechanisms, "
                    "dispensing with recurrence and convolutions entirely."
                ),
            ),
            "related": Paper(
                paper_id="related",
                title="BERT: Pre-training of Deep Bidirectional Transformers",
                year=2018,
                abstract=(
                    "We introduce BERT, a new language representation model based "
                    "on bidirectional transformers for pre-training."
                ),
            ),
            "unrelated": Paper(
                paper_id="unrelated",
                title="Crystallographic Analysis of Protein Folding",
                year=2018,
                abstract=(
                    "X-ray crystallography results showing protein folding patterns "
                    "in thermophilic bacteria."
                ),
            ),
        }

        builder = RecommendationGraphBuilder()
        builder._abstract_index.build(papers)

        related_similarity = builder._abstract_index.similarity("seed", "related")
        unrelated_similarity = builder._abstract_index.similarity("seed", "unrelated")

        assert related_similarity > unrelated_similarity

    def test_should_create_edge_deterministic_thresholds(self):
        """Edges are now deterministic and should not call random logic."""
        seed = Paper(paper_id="seed", title="Seed", year=2020, abstract="seed abstract")
        related = Paper(
            paper_id="related",
            title="Related",
            year=2020,
            abstract="related abstract text",
        )
        weak = Paper(
            paper_id="weak",
            title="Weak",
            year=2020,
            abstract="something else",
        )
        seed.is_seed = True

        builder = RecommendationGraphBuilder(similarity_threshold=0.2)
        builder._abstract_index.build(
            {
                seed.paper_id: seed,
                related.paper_id: related,
                weak.paper_id: weak,
            }
        )

        assert builder.should_create_edge(seed, related, 0.99)
        assert builder.should_create_edge(seed, related, 0.20)
        assert not builder.should_create_edge(seed, weak, 0.19)
        assert not builder.should_create_edge(weak, related, 0.12)
        assert not builder.should_create_edge(weak, related, 0.25)

    @patch("citemesh.strategies.recommendation.get_client")
    def test_collect_papers_filters_missing_abstract(self, mock_get_client):
        """Recommendations without abstract text should be filtered out."""
        mock_client = MagicMock()
        mock_client.get_paper.return_value = Paper(
            paper_id="seed", title="Seed", year=2020, abstract="seed abstract"
        )
        mock_client.get_recommended_papers.side_effect = [
            [
                Paper(
                    paper_id="valid",
                    title="Valid",
                    year=2021,
                    abstract="valid abstract",
                ),
                Paper(
                    paper_id="missing_abstract",
                    title="Missing",
                    year=2021,
                    abstract="",
                ),
            ]
        ]
        mock_get_client.return_value = mock_client

        builder = RecommendationGraphBuilder(max_papers=2)
        papers = builder.collect_papers("seed")

        assert "valid" in papers
        assert "missing_abstract" not in papers
