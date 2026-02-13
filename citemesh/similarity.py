"""Text similarity helpers for paper ranking."""

from __future__ import annotations

from typing import Dict, List

from citemesh.core import Paper


class AbstractSimilarityIndex:
    """Simple TF-IDF cosine similarity index over paper abstracts."""

    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self._matrix = None
        self._ids: List[str] = []
        self._id_to_idx: Dict[str, int] = {}

    def build(self, papers: Dict[str, Paper]) -> None:
        """Build cosine index from normalized paper titles + abstracts."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        ids = []
        texts = []
        for paper_id, paper in papers.items():
            text = f"{paper.title}. {paper.abstract}".strip()
            if text:
                ids.append(paper_id)
                texts.append(text)

        if len(texts) < 2:
            self._matrix = None
            self._ids = []
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
        self._ids = ids
        self._id_to_idx = {paper_id: idx for idx, paper_id in enumerate(ids)}

    def similarity(self, paper_id_a: str, paper_id_b: str) -> float:
        """Return cosine similarity between two indexed papers."""
        if self._matrix is None:
            return 0.0

        idx_a = self._id_to_idx.get(paper_id_a)
        idx_b = self._id_to_idx.get(paper_id_b)
        if idx_a is None or idx_b is None:
            return 0.0

        return float((self._matrix[idx_a] @ self._matrix[idx_b].T).toarray()[0, 0])

    @property
    def is_ready(self) -> bool:
        """Whether the index has enough papers to compute similarities."""
        return self._matrix is not None
