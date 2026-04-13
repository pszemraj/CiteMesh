"""Text similarity helpers for paper ranking."""

from __future__ import annotations

from typing import Dict

from citemesh.core import Paper
from citemesh.data.model_profiles import compose_title_abstract_text


class AbstractSimilarityIndex:
    """Simple TF-IDF cosine similarity index over paper abstracts."""

    def __init__(self, max_features: int = 5000):
        """Create a simple TF-IDF cosine similarity index.

        :param int max_features: Maximum number of TF-IDF features.
        """
        self.max_features = max_features
        self._matrix = None
        self._id_to_idx: Dict[str, int] = {}

    def build(self, papers: Dict[str, Paper]) -> None:
        """Build cosine index from normalized paper titles + abstracts."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        ids = []
        texts = []
        for paper_id, paper in papers.items():
            text = compose_title_abstract_text(
                {"title": paper.title, "abstract": paper.abstract}
            )
            if text:
                ids.append(paper_id)
                texts.append(text)

        if len(texts) < 2:
            self._matrix = None
            self._id_to_idx = {}
            return

        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)

        self._matrix = matrix
        self._id_to_idx = {paper_id: idx for idx, paper_id in enumerate(ids)}

    def similarity(self, paper_id_a: str, paper_id_b: str) -> float:
        """Return cosine similarity between two indexed papers.

        :param str paper_id_a: First paper identifier.
        :param str paper_id_b: Second paper identifier.
        :return float: Cosine similarity in [0.0, 1.0].
        """
        if self._matrix is None:
            return 0.0

        idx_a = self._id_to_idx.get(paper_id_a)
        idx_b = self._id_to_idx.get(paper_id_b)
        if idx_a is None or idx_b is None:
            return 0.0

        return float((self._matrix[idx_a] @ self._matrix[idx_b].T).toarray()[0, 0])
