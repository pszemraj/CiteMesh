"""Hybrid discovery flows against a fake Semantic Scholar and a tiny real corpus."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from unittest.mock import MagicMock
from urllib.parse import unquote

import networkx as nx
import numpy as np
import pytest
from numpy.typing import NDArray

from citemesh.core.models import Paper
from citemesh.services import SemanticScholarUnavailableError
from citemesh.services.semantic_scholar import SemanticScholarClient
from citemesh.services.semantic_scholar.endpoints import (
    PAPER_BASE_URL,
    RECOMMENDATION_BASE_URL,
)
from citemesh.strategies import hybrid as hybrid_module
from citemesh.strategies.candidates import CandidateAcquisitionError
from citemesh.strategies.embedding import EmbeddingGraphBuilder
from citemesh.strategies.hybrid import HybridGraphBuilder
from citemesh.visualization.export import GraphExporter
from tests._helpers import WordHashEncodeModel, disable_embedding_dep_checks

pytestmark = pytest.mark.usefixtures("arxiv_unavailable")

_DATASET_SOURCE = "fixture/hybrid-discovery-corpus"
_FINGERPRINT = "artifact-hybrid-discovery"
_SEED_ARXIV = "2401.00000"
_SEED_ID = f"arxiv:{_SEED_ARXIV}"
_MODES = ("candidates", "arxiv-corpus", "both")

CorpusRow = dict[str, object]


@pytest.fixture(autouse=True)
def _disable_embedding_optional_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch optional embedding dependency guards for hybrid flow tests."""
    disable_embedding_dep_checks(monkeypatch)


def install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    builder: EmbeddingGraphBuilder,
    model: WordHashEncodeModel,
    corpus_rows: Sequence[CorpusRow] = (),
) -> None:
    """Bind a fake model, a pinned fingerprint and a fixture corpus to a builder.

    :param pytest.MonkeyPatch monkeypatch: Runtime boundary patch fixture.
    :param EmbeddingGraphBuilder builder: Embedding child of the hybrid builder.
    :param WordHashEncodeModel model: Encoder shared by every builder in a test.
    :param Sequence[CorpusRow] corpus_rows: Dataset rows for hydration.
    :return None: Patches the builder in place.
    """

    def load_model() -> None:
        """Bind the fake model as a fresh load would."""
        if builder.model is not None:
            return
        builder.model = model
        builder._active_model_name = builder.model_name
        builder._bind_embedding_cache_to_active_model()

    monkeypatch.setattr(builder, "_load_model", load_model)
    monkeypatch.setattr(builder, "_resolve_model_fingerprint", lambda: _FINGERPRINT)
    monkeypatch.setattr(
        builder,
        "_load_dataset_for_hydration",
        lambda **_kwargs: (_DATASET_SOURCE, [dict(row) for row in corpus_rows]),
    )


def _corpus_row(arxiv_id: str, title: str, *, year: int = 2023) -> CorpusRow:
    """Build one corpus dataset row.

    :param str arxiv_id: Bare arXiv identifier.
    :param str title: Title, also the start of the abstract.
    :param int year: Publication year.
    :return CorpusRow: Row in the corpus dataset layout.
    """
    return {
        "id": f"arxiv:{arxiv_id}",
        "title": title,
        "abstract": f"{title} abstract",
        "authors": "Ada Example",
        "year": year,
    }


def _response(payload: object) -> MagicMock:
    """Wrap a JSON payload as a successful transport response.

    :param object payload: Decoded JSON body.
    :return MagicMock: Response with status 200.
    """
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _not_found() -> MagicMock:
    """Return an HTTP 404 transport response.

    :return MagicMock: Response with status 404.
    """
    response = MagicMock()
    response.status_code = 404
    response.headers = {}
    return response


@dataclass
class FakeSemanticScholar:
    """In-memory provider serving one seed's relations and every paper's references.

    ``requests`` counts every HTTP call by kind: ``paper``, ``references`` and
    ``citations`` (inline relation listings), ``reference_ids`` (ID-only
    reference listings used for bibliographic coupling), ``recommendations`` and
    ``batch``.
    """

    papers: dict[str, dict[str, object]] = field(default_factory=dict)
    references: dict[str, list[str]] = field(default_factory=dict)
    citations: dict[str, list[str]] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)
    requests: Counter[str] = field(default_factory=Counter)
    recommendation_params: list[dict[str, object]] = field(default_factory=list)

    def add(
        self,
        paper_id: str,
        title: str,
        *,
        arxiv_id: str = "",
        doi: str = "",
        year: int = 2023,
    ) -> str:
        """Register one provider record under its S2 ID and external aliases.

        :param str paper_id: Semantic Scholar paper ID.
        :param str title: Title, also the start of the abstract.
        :param str arxiv_id: Optional bare arXiv identifier.
        :param str doi: Optional DOI.
        :param int year: Publication year.
        :return str: The record's S2 ID.
        """
        external: dict[str, str] = {}
        if arxiv_id:
            external["ArXiv"] = arxiv_id
            self.aliases[f"arxiv:{arxiv_id}"] = paper_id
        if doi:
            external["DOI"] = doi
            self.aliases[doi.lower()] = paper_id
        self.papers[paper_id] = {
            "paperId": paper_id,
            "title": title,
            "abstract": f"{title} abstract",
            "year": year,
            "citationCount": 10,
            "fieldsOfStudy": [],
            "authors": [{"name": "Ada Example"}],
            "externalIds": external,
        }
        return paper_id

    def _resolve(self, raw_id: str) -> str | None:
        """Map a requested identifier to a known S2 ID.

        :param str raw_id: Path or batch identifier, possibly prefixed.
        :return str | None: Known S2 ID, if any.
        """
        identifier = unquote(raw_id)
        if identifier in self.papers:
            return identifier
        return self.aliases.get(identifier.lower().removeprefix("doi:"))

    def _page(
        self, ids: list[str], key: str, params: dict[str, object]
    ) -> dict[str, object]:
        """Serve one relation page honoring limit and offset.

        :param list[str] ids: Related S2 IDs in provider order.
        :param str key: Nested paper key (``citedPaper`` or ``citingPaper``).
        :param dict[str, object] params: Request parameters.
        :return dict[str, object]: Relation page payload.
        """
        offset = int(str(params.get("offset", 0)))
        limit = int(str(params.get("limit", 1000)))
        window = ids[offset : offset + limit]
        return {
            "data": [{key: self.papers[paper_id]} for paper_id in window],
            "next": offset + limit if offset + limit < len(ids) else None,
        }

    def get(
        self, url: str, params: dict[str, object] | None = None, **_kwargs: object
    ) -> MagicMock:
        """Answer one GET request.

        :param str url: Requested endpoint.
        :param dict[str, object] | None params: Query parameters.
        :return MagicMock: Transport response.
        """
        params = params or {}
        if url.startswith(RECOMMENDATION_BASE_URL):
            self.requests["recommendations"] += 1
            self.recommendation_params.append(dict(params))
            return _response(
                {
                    "recommendedPapers": [
                        self.papers[paper_id] for paper_id in self.recommendations
                    ]
                }
            )
        path = url.removeprefix(f"{PAPER_BASE_URL}/")
        relation = path.rsplit("/", 1)[-1] if "/" in path else ""
        if relation in {"references", "citations"}:
            paper_id = self._resolve(path.rsplit("/", 1)[0])
            if paper_id is None:
                return _not_found()
            if relation == "citations":
                self.requests["citations"] += 1
                ids = self.citations.get(paper_id, [])
                return _response(self._page(ids, "citingPaper", params))
            kind = (
                "reference_ids" if params.get("fields") == "paperId" else "references"
            )
            self.requests[kind] += 1
            ids = self.references.get(paper_id, [])
            return _response(self._page(ids, "citedPaper", params))
        self.requests["paper"] += 1
        paper_id = self._resolve(path)
        return _not_found() if paper_id is None else _response(self.papers[paper_id])

    def post(
        self,
        url: str,
        json: dict[str, list[str]] | None = None,
        **_kwargs: object,
    ) -> MagicMock:
        """Answer one batch request.

        :param str url: Requested endpoint.
        :param dict[str, list[str]] | None json: Batch body.
        :return MagicMock: Transport response.
        """
        assert url == f"{PAPER_BASE_URL}/batch"
        self.requests["batch"] += 1
        rows = []
        for raw_id in (json or {}).get("ids", []):
            paper_id = self._resolve(raw_id)
            rows.append(None if paper_id is None else self.papers[paper_id])
        return _response(rows)


@contextmanager
def fake_client(world: FakeSemanticScholar) -> Iterator[SemanticScholarClient]:
    """Open a real client whose transport is the fake provider.

    :param FakeSemanticScholar world: Provider state and request counter.
    :return Iterator[SemanticScholarClient]: Client without request pacing.
    """
    with SemanticScholarClient(timeout=1) as client:
        client._rate_limit = MagicMock()
        client._session.get = MagicMock(side_effect=world.get)
        client._session.post = MagicMock(side_effect=world.post)
        yield client


@dataclass
class Scenario:
    """A seed's provider world, its corpus, and the S2 IDs of each source list."""

    world: FakeSemanticScholar
    corpus: list[CorpusRow]
    references: list[str]
    citers: list[str]
    recommendations: list[str]
    corpus_only: list[str]


def _scenario(*, citers: int = 60) -> Scenario:
    """Build a seed with indexed references, unindexed citers and mixed recommendations.

    :param int citers: Number of papers citing the seed.
    :return Scenario: Provider world and corpus rows.
    """
    world = FakeSemanticScholar()
    seed_title = "learning rate schedules seed"
    seed_s2 = world.add("f" * 40, seed_title, arxiv_id=_SEED_ARXIV, year=2024)
    corpus = [_corpus_row(_SEED_ARXIV, seed_title, year=2024)]
    references = []
    for index in range(30):
        arxiv_id = f"2301.{index:05d}"
        title = f"learning rate warmup reference {index}"
        references.append(world.add(f"{index + 1:040x}", title, arxiv_id=arxiv_id))
        corpus.append(_corpus_row(arxiv_id, title))
    citing = [
        world.add(f"{index + 10_001:040x}", f"schedule study citing paper {index}")
        for index in range(citers)
    ]
    recommendations = []
    for index in range(40):
        arxiv_id = f"2302.{index:05d}"
        title = f"optimizer recommendation {index}"
        recommendations.append(
            world.add(f"{index + 5_001:040x}", title, arxiv_id=arxiv_id)
        )
        if index % 2 == 0:
            corpus.append(_corpus_row(arxiv_id, title))
    corpus_only = []
    for index in range(60):
        arxiv_id = f"2303.{index:05d}"
        corpus.append(_corpus_row(arxiv_id, f"corpus only neighbour learning {index}"))
        corpus_only.append(f"arxiv:{arxiv_id}")
    world.references[seed_s2] = references
    world.citations[seed_s2] = citing
    world.recommendations = recommendations
    for paper_id in [*references, *citing, *recommendations]:
        world.references[paper_id] = references[:3]
    return Scenario(world, corpus, references, citing, recommendations, corpus_only)


def _builder(
    client: SemanticScholarClient,
    scenario: Scenario,
    mode: str,
    **options: int,
) -> HybridGraphBuilder:
    """Create a hybrid builder over the scenario's corpus.

    :param SemanticScholarClient client: Fake-backed provider client.
    :param Scenario scenario: Provider world and corpus.
    :param str mode: ``--semantic-source`` value.
    :param int options: Integer builder overrides such as ``max_papers``.
    :return HybridGraphBuilder: Builder with no corpus binding in candidates mode.
    """
    if mode == "candidates":
        return HybridGraphBuilder(semantic_source=mode, client=client, **options)
    return HybridGraphBuilder(
        semantic_source=mode,
        client=client,
        dataset_source=_DATASET_SOURCE,
        dataset_split=f"train[:{len(scenario.corpus)}]",
        corpus_size=len(scenario.corpus),
        storage_precision="float32",
        **options,
    )


def _members(builder: HybridGraphBuilder, s2_ids: Sequence[str]) -> set[str]:
    """Return the pool members that hold the given provider records.

    :param HybridGraphBuilder builder: Builder after collection.
    :param Sequence[str] s2_ids: Provider record IDs.
    :return set[str]: Member IDs, following corpus takeovers.
    """
    pool = builder.discovery_pool
    assert pool is not None
    members = set()
    for s2_id in s2_ids:
        member = pool.provider_aliases.get(s2_id, s2_id)
        assert member in pool.papers, s2_id
        members.add(member)
    return members


@pytest.mark.parametrize("mode", _MODES)
def test_pool_holds_every_list_of_its_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory, mode: str
) -> None:
    """Each mode pools every reference and citer plus its own semantic lists."""
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    scenario = _scenario()
    with fake_client(scenario.world) as client:
        builder = _builder(client, scenario, mode)
        install_runtime(
            monkeypatch,
            builder.embedding_builder,
            WordHashEncodeModel(),
            scenario.corpus,
        )
        papers = builder.collect_papers(_SEED_ID)

    pool = builder.discovery_pool
    assert pool is not None
    expected = _members(builder, scenario.references) | _members(
        builder, scenario.citers
    )
    if mode != "arxiv-corpus":
        expected |= _members(builder, scenario.recommendations)
    if mode != "candidates":
        # The tiny corpus is within the neighbour limit, so every row joins.
        expected |= {str(row["id"]) for row in scenario.corpus[1:]}
    assert set(pool.papers) == expected
    assert len(builder.discovery_ranking) == len(expected)
    assert len(papers) == 45
    recommended = (
        []
        if mode == "arxiv-corpus"
        else [pool.provider_aliases.get(s2, s2) for s2 in scenario.recommendations]
    )
    assert pool.recommendation_ranks == {
        member: rank for rank, member in enumerate(recommended, start=1)
    }
    assert scenario.world.requests["recommendations"] == int(mode != "arxiv-corpus")
    if mode != "arxiv-corpus":
        assert scenario.world.recommendation_params == [
            {"fields": scenario.world.recommendation_params[0]["fields"]}
            | {"limit": 200, "from": "all-cs"}
        ]


@pytest.mark.parametrize("mode", ["arxiv-corpus", "both"])
def test_indexed_provider_records_use_their_corpus_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory, mode: str
) -> None:
    """Provider records join under their corpus row even when no search finds it.

    Only members without a corpus row are encoded in memory.
    """
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    monkeypatch.setattr(hybrid_module, "HYBRID_CORPUS_NEIGHBOURS", 1)
    scenario = _scenario()
    with fake_client(scenario.world) as client:
        builder = _builder(client, scenario, mode)
        install_runtime(
            monkeypatch,
            builder.embedding_builder,
            WordHashEncodeModel(),
            scenario.corpus,
        )
        encoded: list[str] = []
        embed_documents = builder._embed_documents

        def record_encoding(
            papers: dict[str, Paper],
        ) -> dict[str, NDArray[np.float32]]:
            encoded.extend(papers)
            return embed_documents(papers)

        monkeypatch.setattr(builder, "_embed_documents", record_encoding)
        builder.collect_papers(_SEED_ID)

    pool = builder.discovery_pool
    assert pool is not None
    indexed = {
        s2_id: f"arxiv:2301.{index:05d}"
        for index, s2_id in enumerate(scenario.references)
    }
    if mode == "both":
        indexed |= {
            s2_id: f"arxiv:2302.{index:05d}"
            for index, s2_id in enumerate(scenario.recommendations)
            if index % 2 == 0
        }
    assert {s2_id: pool.provider_aliases.get(s2_id) for s2_id in indexed} == indexed
    assert all(pool.papers[member].is_local_corpus for member in indexed.values())
    assert sorted(encoded) == sorted(
        member for member, paper in pool.papers.items() if not paper.is_local_corpus
    )


def test_candidates_mode_outage_fails_before_the_model_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no corpus to search, a total provider outage stops collection early."""
    client = MagicMock()
    client.get_paper.return_value = Paper(
        paper_id="f" * 40, title="Seed", year=2024, arxiv_id=_SEED_ARXIV
    )
    for method in ("get_paper_references", "get_paper_citations"):
        getattr(client, method).side_effect = SemanticScholarUnavailableError("down")
    client.get_recommended_papers.side_effect = SemanticScholarUnavailableError("down")
    builder = HybridGraphBuilder(client=client)
    load_model = MagicMock()
    monkeypatch.setattr(builder.embedding_builder, "_load_model", load_model)

    with pytest.raises(CandidateAcquisitionError, match="hybrid candidate"):
        builder.collect_papers(_SEED_ID)

    load_model.assert_not_called()
    assert set(builder.candidate_source_status.values()) == {"unavailable"}


def test_citers_are_ranked_without_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every fetched citer is ranked, far beyond the graph's size."""
    scenario = _scenario(citers=1200)
    with fake_client(scenario.world) as client:
        builder = _builder(client, scenario, "candidates", max_papers=10)
        install_runtime(monkeypatch, builder.embedding_builder, WordHashEncodeModel())
        papers = builder.collect_papers(_SEED_ID)

    ranked = {item.paper_id for item in builder.discovery_ranking}
    assert ranked >= set(scenario.citers[:1000])
    assert ranked.isdisjoint(scenario.citers[1000:])
    assert len(papers) == 10
    assert scenario.world.requests["citations"] == 1


def test_publication_version_takes_one_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A citer's journal version folds into its corpus preprint before selection."""
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    scenario = _scenario(citers=0)
    title = "learning rate schedules seed revisited"
    preprint = "arxiv:2402.00001"
    scenario.corpus.append(_corpus_row("2402.00001", title, year=2024))
    publication = scenario.world.add(
        "e" * 40, title, doi="10.1000/revisited", year=2025
    )
    scenario.world.citations["f" * 40] = [publication]
    # Without reference-context competition the pair is the clear top candidate.
    scenario.world.references["f" * 40] = []
    with fake_client(scenario.world) as client:
        builder = _builder(client, scenario, "both", max_papers=4)
        install_runtime(
            monkeypatch,
            builder.embedding_builder,
            WordHashEncodeModel(),
            scenario.corpus,
        )
        papers = builder.collect_papers(_SEED_ID)

    pool = builder.discovery_pool
    assert pool is not None
    assert pool.provider_aliases[publication] == preprint
    assert publication not in pool.papers
    assert pool.sources[preprint] == {"citation", "semantic"}
    assert builder.seed_relations[preprint] == "cites_seed"
    ranked_ids = [item.paper_id for item in builder.discovery_ranking]
    assert ranked_ids[0] == preprint
    assert list(papers)[1:] == ranked_ids[:3]


def _graph_signature(
    graph: nx.Graph,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Return a graph's sorted nodes and weighted edges.

    :param nx.Graph graph: Built graph.
    :return tuple[list[str], list[tuple[str, str, float]]]: Comparable structure.
    """
    edges = sorted(
        (*sorted((str(left), str(right))), round(float(data["weight"]), 9))
        for left, right, data in graph.edges(data=True)
    )
    return sorted(map(str, graph.nodes)), edges


def test_builds_are_deterministic(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Repeated builds rank, select and connect papers identically."""
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    scenario = _scenario()
    model = WordHashEncodeModel()
    outcomes = []
    with fake_client(scenario.world) as client:
        for _ in range(2):
            builder = _builder(client, scenario, "both")
            install_runtime(
                monkeypatch, builder.embedding_builder, model, scenario.corpus
            )
            graph, seed_id = builder.build_graph(_SEED_ID)
            ranking = [
                (item.paper_id, round(item.score, 12))
                for item in builder.discovery_ranking
            ]
            outcomes.append((seed_id, ranking, _graph_signature(graph)))

    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] == _SEED_ID


@pytest.mark.parametrize(
    ("mode", "cold", "warm"),
    [
        (
            "candidates",
            # The seed lookup, its three lists (each carrying its metadata), and
            # the reference lists of the selected citation-sourced papers.
            {
                "citations": 1,
                "paper": 1,
                "recommendations": 1,
                "reference_ids": 36,
                "references": 1,
            },
            {"citations": 1, "recommendations": 1, "references": 1},
        ),
        (
            "arxiv-corpus",
            # The batch enriches selected corpus rows that no provider list named.
            {"batch": 1, "citations": 1, "reference_ids": 44, "references": 1},
            {"citations": 1, "references": 1},
        ),
        (
            "both",
            {
                "batch": 1,
                "citations": 1,
                "recommendations": 1,
                "reference_ids": 37,
                "references": 1,
            },
            {"citations": 1, "recommendations": 1, "references": 1},
        ),
    ],
)
def test_provider_requests_per_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
    mode: str,
    cold: dict[str, int],
    warm: dict[str, int],
) -> None:
    """Pin the provider requests of a cold build and of a warm rebuild.

    Warm rebuilds reuse the paper and reference caches, so only the seed's own
    lists are requested again.
    """
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    scenario = _scenario()
    model = WordHashEncodeModel()
    counts = []
    encoded = []
    with fake_client(scenario.world) as client:
        for _ in range(2):
            scenario.world.requests.clear()
            model.texts.clear()
            builder = _builder(client, scenario, mode)
            install_runtime(
                monkeypatch, builder.embedding_builder, model, scenario.corpus
            )
            builder.build_graph(_SEED_ID)
            counts.append(dict(scenario.world.requests))
            encoded.append(dict(Counter(map(_encoding_role, model.texts))))

    assert counts == [cold, warm]
    assert encoded == _ENCODED_TEXTS[mode]


# Texts encoded per build by role: the seed query, retrieval documents and the
# graph-similarity vectors of the 45 selected papers, which are cached. Candidates
# mode keeps the seed's and its 130 members' documents in the candidate cache, so
# a rebuild encodes only the query. Corpus modes hydrate the 111-row corpus once
# and encode the members without a corpus row (60 citers, plus 20 unindexed
# recommendations in both mode) in memory on every build.
_ENCODED_TEXTS = {
    "candidates": [{"query": 1, "document": 131, "graph": 45}, {"query": 1}],
    "arxiv-corpus": [
        {"query": 1, "document": 171, "graph": 45},
        {"query": 1, "document": 60},
    ],
    "both": [
        {"query": 1, "document": 191, "graph": 45},
        {"query": 1, "document": 80},
    ],
}


def _encoding_role(text: str) -> str:
    """Name the embedding role of one EmbeddingGemma model input.

    :param str text: Formatted model input.
    :return str: ``query``, ``graph`` or ``document``.
    """
    if text.startswith("task: search result"):
        return "query"
    if text.startswith("task: sentence similarity"):
        return "graph"
    return "document"


@pytest.mark.parametrize("mode", ["arxiv-corpus", "both"])
def test_json_export_carries_each_related_papers_discovery_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory, mode: str
) -> None:
    """Related nodes export their fused rank, score, features and list ranks."""
    monkeypatch.setenv("CITEMESH_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    scenario = _scenario()
    with fake_client(scenario.world) as client:
        builder = _builder(client, scenario, mode)
        install_runtime(
            monkeypatch,
            builder.embedding_builder,
            WordHashEncodeModel(),
            scenario.corpus,
        )
        graph, seed_id = builder.build_graph(_SEED_ID)
    payload = GraphExporter(graph, seed_id).graph_payload()
    json.dumps(payload, allow_nan=False)

    # Corpus-only mode fetches no recommendations, so it fuses three lists.
    lists = ["semantic", "reference_context", "direct"]
    if mode == "both":
        lists.append("recommended")
    ranker = {"method": "reciprocal_rank_fusion", "k": 60, "lists": lists}
    assert payload["meta"]["discovery_ranker"] == ranker
    assert payload["dashboard"]["meta"]["discovery_ranker"] == ranker
    ranking = {item.paper_id: item for item in builder.discovery_ranking}
    seed, *related = sorted(payload["nodes"], key=lambda node: not node["is_seed"])
    assert "discovery_rank" not in seed
    assert sorted(node["discovery_rank"] for node in related) == list(range(1, 45))
    for node in related:
        item = ranking[node["id"]]
        assert node["discovery_rank"] == item.rank
        assert node["discovery_score"] == item.score
        assert node["discovery_features"] == {
            "seed_similarity": item.features.cos_dd,
            "bibliography_similarity": item.features.ctx_ref_top5,
            "direct_citation": item.features.direct,
            "recommendation_rank": item.features.recommendation_rank,
        }
        assert node["discovery_list_ranks"] == {
            name: item.list_ranks.get(name) for name in lists
        }
    features = [node["discovery_features"] for node in related]
    assert any(entry["direct_citation"] for entry in features)
    assert (mode == "both") == any(
        entry["recommendation_rank"] is not None for entry in features
    )
