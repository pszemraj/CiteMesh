"""Handwritten topical pairs for a bounded semantic-quality evaluation."""

from itertools import combinations
from unittest.mock import MagicMock

import huggingface_hub.constants
import pytest

from citemesh.core import EMBEDDING_CONFIG, Paper
from citemesh.strategies.embedding import (
    EmbeddingGraphBuilder,
    EmbeddingTask,
    format_paper_for_embedding,
)
from citemesh.strategies.hybrid import HybridGraphBuilder

pytestmark = pytest.mark.slow

# The first six topics select the boundary; the last six are held out.
# Within each topic the two descriptions address the same research problem.
# Across topics they address different problems, including neighboring ML tasks.
CASES = [
    (
        "translation",
        "self attention for machine translation",
        (
            (
                "Attention Is All You Need",
                "The Transformer replaces recurrent neural networks with multi-head self-attention for machine translation.",
            ),
            (
                "Attention-Based Sequence Translation",
                "An encoder-decoder built from self-attention translates sentences between languages without recurrent layers.",
            ),
        ),
    ),
    (
        "graph",
        "message passing for graph node classification",
        (
            (
                "Graph Convolutional Node Classification",
                "Neighborhood aggregation learns node representations for semi-supervised classification on citation networks.",
            ),
            (
                "Learning Labels on Networks",
                "A message-passing neural network combines adjacent node features to predict missing node labels in a graph.",
            ),
        ),
    ),
    (
        "protein",
        "predicting protein three dimensional structure from amino acid sequences",
        (
            (
                "Protein Folding from Sequence",
                "A neural model predicts three-dimensional protein structure from amino acid sequences and evolutionary alignments.",
            ),
            (
                "Learning Protein Structures",
                "Residue representations and geometric constraints reconstruct the folded coordinates of proteins from their sequences.",
            ),
        ),
    ),
    (
        "coral",
        "ocean warming causes coral bleaching",
        (
            (
                "Coral Reef Bleaching",
                "Rising ocean temperatures disrupt coral symbiosis with algae, causing bleaching and marine biodiversity loss.",
            ),
            (
                "Thermal Stress in Reef Corals",
                "Heat exposure causes corals to expel symbiotic algae; field observations measure bleaching and subsequent reef mortality.",
            ),
        ),
    ),
    (
        "wheat",
        "wheat root traits and yield under drought",
        (
            (
                "Wheat Drought Tolerance",
                "Field trials examine root depth, soil moisture, and crop yield under water stress in wheat cultivation.",
            ),
            (
                "Root Architecture in Drought-Stressed Wheat",
                "Deep-rooted wheat varieties extract soil water and maintain grain yield during periods of limited rainfall.",
            ),
        ),
    ),
    (
        "pottery",
        "medieval pottery kiln clay and glaze analysis",
        (
            (
                "Medieval Ceramic Production",
                "Archaeological analysis of kiln temperatures, clay sources, and glaze composition in medieval pottery workshops.",
            ),
            (
                "Characterizing Historical Pottery Workshops",
                "Chemical analysis of ceramic glazes and clay minerals reconstructs firing practices in medieval kilns.",
            ),
        ),
    ),
    (
        "retrieval",
        "dense passage retrieval for question answering",
        (
            (
                "Dense Passage Retrieval",
                "Dual encoders retrieve relevant passages for open-domain question answering by matching question and document embeddings.",
            ),
            (
                "Neural Evidence Retrieval",
                "A question encoder and a passage encoder select textual evidence from a large collection to answer factual questions.",
            ),
        ),
    ),
    (
        "recommendation",
        "collaborative filtering for personalized product recommendations",
        (
            (
                "Collaborative Product Recommendation",
                "User-item interaction embeddings predict shopping preferences for personalized product recommendations.",
            ),
            (
                "Learning Shopper Preferences",
                "Collaborative filtering factors a customer purchase matrix to rank unseen products by individual user interest.",
            ),
        ),
    ),
    (
        "image-diffusion",
        "image generation with denoising diffusion models",
        (
            (
                "Denoising Diffusion Image Generation",
                "A generative neural network reverses a gradual noise process to synthesize realistic images from Gaussian noise.",
            ),
            (
                "Score-Based Image Synthesis",
                "Learning to remove noise at successive timesteps enables a diffusion model to sample high-quality natural images.",
            ),
        ),
    ),
    (
        "molecular-diffusion",
        "molecular diffusion of solutes in liquids",
        (
            (
                "Molecular Diffusion in Liquids",
                "Brownian motion and concentration gradients determine solute transport and diffusion coefficients in aqueous solutions.",
            ),
            (
                "Measuring Solute Transport",
                "Tracer experiments quantify how dissolved molecules spread through water and estimate temperature-dependent diffusion constants.",
            ),
        ),
    ),
    (
        "planet",
        "detecting exoplanets from stellar transit light curves",
        (
            (
                "Exoplanet Transit Detection",
                "Periodic decreases in stellar brightness reveal orbiting planets; light-curve fitting estimates planetary radii and orbital periods.",
            ),
            (
                "Planet Discovery with Transit Photometry",
                "Repeated dips in a star's light curve identify transiting exoplanets and constrain their sizes and orbital geometry.",
            ),
        ),
    ),
    (
        "cancer",
        "chemotherapy and hormone therapy for metastatic breast cancer",
        (
            (
                "Breast Cancer Treatment",
                "A clinical trial evaluates chemotherapy and hormone therapy outcomes in patients with metastatic breast cancer.",
            ),
            (
                "Systemic Therapy for Advanced Breast Tumors",
                "Comparing endocrine treatment with chemotherapy measures survival and disease progression in metastatic breast carcinoma.",
            ),
        ),
    ),
]


def _quality_builder(monkeypatch: pytest.MonkeyPatch) -> EmbeddingGraphBuilder:
    """Load the designated cached model for a network-free CPU evaluation.

    :param pytest.MonkeyPatch monkeypatch: Restores offline settings after the test.
    :return EmbeddingGraphBuilder: Real encoder with the default task profile.
    """
    pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    builder = EmbeddingGraphBuilder(device="cpu", client=MagicMock())
    builder._load_model()
    return builder


def _quality_papers() -> list[Paper]:
    """Build paper records with metadata that previously admitted false edges.

    :return list[Paper]: Two papers per labeled research topic in fixture order.
    """
    return [
        Paper(f"{topic}-{index}", title, 2024, abstract=abstract, citation_count=100)
        for topic, _, descriptions in CASES
        for index, (title, abstract) in enumerate(descriptions)
    ]


def test_real_model_semantic_edge_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject cross-topic pairs while retaining related pairs on held-out topics.

    :param pytest.MonkeyPatch monkeypatch: Keeps model resolution offline.
    :return None: Both graph scorers satisfy the labeled edge decisions.
    """
    builder = _quality_builder(monkeypatch)
    papers = _quality_papers()
    vectors = builder._encode_texts(
        [
            format_paper_for_embedding(
                profile=builder.model_profile,
                paper=paper,
                task=EmbeddingTask.GRAPH_SIMILARITY,
            )
            for paper in papers
        ]
    )
    builder.embeddings = {
        paper.paper_id: vector for paper, vector in zip(papers, vectors)
    }
    hybrid = HybridGraphBuilder(client=MagicMock())
    hybrid.embedding_builder = builder

    # Select once on development topics; held-out topics never set the threshold.
    development = [
        (float(vectors[i] @ vectors[j]), i // 2 == j // 2)
        for i, j in combinations(range(12), 2)
    ]
    positive_floor = min(score for score, related in development if related)
    negative_ceiling = max(score for score, related in development if not related)
    assert negative_ceiling < EMBEDDING_CONFIG.min_semantic_similarity < positive_floor
    assert (
        abs(
            EMBEDDING_CONFIG.min_semantic_similarity
            - (positive_floor + negative_ceiling) / 2
        )
        < 0.02
    )

    for offset in (0, 12):
        for i, j in combinations(range(offset, offset + 12), 2):
            left, right = papers[i], papers[j]
            related = i // 2 == j // 2
            score = builder.compute_similarity(left, right)
            assert bool(builder.should_create_edge(left, right, score)) == related, (
                left.paper_id,
                right.paper_id,
                score,
            )
            for source in ("citation", "semantic", "both"):
                hybrid.paper_sources = {left.paper_id: source, right.paper_id: source}
                score = hybrid.compute_similarity(left, right)
                assert bool(hybrid.should_create_edge(left, right, score)) == related, (
                    source,
                    left.paper_id,
                    right.paper_id,
                    score,
                )
