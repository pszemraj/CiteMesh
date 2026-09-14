"""Build-option tables, strategy dispatch, and user-config defaulting.

Owns which ``build`` options each strategy supports, the strategy factory
table, the shared embedding builder keyword assembly, export metadata for
embedding runs, and the merge of user-config defaults into a parsed namespace.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from typing import Any, Protocol

import networkx as nx

from citemesh.data.user_config import UserConfig, format_config_value
from citemesh.services import SemanticScholarClient
from citemesh.strategies.citation import CitationGraphBuilder
from citemesh.strategies.embedding import DEFAULT_DATASET_SOURCE, EmbeddingGraphBuilder
from citemesh.strategies.hybrid import (
    DEFAULT_MAX_SEMANTIC,
    HYBRID_DEFAULT_MAX_CITATIONS,
    HYBRID_DEFAULT_MAX_PAPERS,
    HYBRID_DEFAULT_MAX_REFERENCES,
    HybridGraphBuilder,
)
from citemesh.strategies.recommendation import RecommendationGraphBuilder

from .console import logger


class _StrategyBuilderProtocol(Protocol):
    """Protocol describing strategy builder objects used by CLI dispatch."""

    def build_graph(self, paper_id: str) -> tuple[nx.Graph, str]:
        """Build a graph for a paper identifier.

        :param str paper_id: Raw or normalized seed paper identifier.
        :return tuple[nx.Graph, str]: Built graph and normalized seed paper ID.
        """
        ...


StrategyFactory = Callable[[argparse.Namespace], _StrategyBuilderProtocol]

# Strategy-scoped build options and CLI-token aliases for strict post-parse validation.
_BUILD_STRATEGY_OPTION_SUPPORT: dict[str, set[str]] = {
    "max_citations": {"citation", "hybrid"},
    "max_references": {"citation", "hybrid"},
    "similarity_threshold": {"citation", "recommendation"},
    "no_references": {"citation", "recommendation", "hybrid"},
    "refresh_reference_cache": {"citation", "recommendation", "hybrid"},
    "model": {"embedding", "hybrid"},
    "model_profile": {"embedding", "hybrid"},
    "model_revision": {"embedding", "hybrid"},
    "dataset_source": {"embedding", "hybrid"},
    "dataset_split": {"embedding", "hybrid"},
    "corpus_size": {"embedding", "hybrid"},
    "all_corpus": {"embedding", "hybrid"},
    "top_k": {"embedding"},
    "truncate_dim": {"embedding", "hybrid"},
    "min_semantic_similarity": {"embedding", "hybrid"},
    "streaming": {"embedding", "hybrid"},
    "force_rebuild_cache": {"embedding", "hybrid"},
    "overwrite_cache": {"embedding", "hybrid"},
    "cache_overwrite_reason": {"embedding", "hybrid"},
    "storage_precision": {"embedding", "hybrid"},
    "binary_prefilter": {"embedding", "hybrid"},
    "binary_rescore_multiplier": {"embedding", "hybrid"},
    "calibration_sample_size": {"embedding", "hybrid"},
    "cache_compression": {"embedding", "hybrid"},
    "cache_compression_level": {"embedding", "hybrid"},
    "encode_batch_size": {"embedding", "hybrid"},
    "torch_compile": {"embedding", "hybrid"},
    "device": {"embedding", "hybrid"},
    "semantic_source": {"embedding", "hybrid"},
    "candidate_pool_size": {"embedding", "hybrid"},
    "max_semantic": {"hybrid"},
}
# Flags that only affect arxiv-corpus hydration; providing them implies (or
# requires) --semantic-source arxiv-corpus.
_CORPUS_ONLY_OPTION_DESTS: set[str] = {
    "dataset_source",
    "dataset_split",
    "corpus_size",
    "all_corpus",
    "streaming",
}
# Built-in parser defaults restored when corpus-only config values are ignored
# in candidates mode; kept in sync with the build parser by a contract test.
_CORPUS_ONLY_OPTION_BUILTIN_DEFAULTS: dict[str, object] = {
    "dataset_source": DEFAULT_DATASET_SOURCE,
    "dataset_split": "train",
    "corpus_size": None,
    "all_corpus": False,
    "streaming": False,
}
# Explicit candidate-only options imply candidate sourcing just as explicit
# corpus-only options imply corpus sourcing.
_CANDIDATE_ONLY_OPTION_DESTS: set[str] = {"candidate_pool_size"}
_BUILD_OPTION_PRIMARY_FLAG: dict[str, str] = {
    dest: f"--{dest.replace('_', '-')}" for dest in _BUILD_STRATEGY_OPTION_SUPPORT
}
_BUILD_OPTION_PRIMARY_FLAG["encode_batch_size"] = "--batch-size"


def _build_option_label(args: argparse.Namespace, dest: str) -> str:
    """Name a build option with its effective boolean polarity.

    :param argparse.Namespace args: Parsed option values.
    :param str dest: Parser destination to describe.
    :return str: Positive or negated long option spelling.
    """
    label = _BUILD_OPTION_PRIMARY_FLAG[dest]
    if dest in {"streaming", "binary_prefilter", "torch_compile"} and not getattr(
        args, dest
    ):
        return "--no-" + label[2:]
    return label


_CACHE_COMPRESSION_CHOICES = ("gzip", "lzf")
_HYBRID_BEST_PRACTICE_DEFAULTS: dict[str, int] = {
    "max_papers": HYBRID_DEFAULT_MAX_PAPERS,
    "max_citations": HYBRID_DEFAULT_MAX_CITATIONS,
    "max_references": HYBRID_DEFAULT_MAX_REFERENCES,
}
_HYBRID_EMBEDDING_OPTION_DESTS: set[str] = {
    "model",
    "model_profile",
    "model_revision",
    "dataset_source",
    "dataset_split",
    "corpus_size",
    "all_corpus",
    "truncate_dim",
    "min_semantic_similarity",
    "streaming",
    "force_rebuild_cache",
    "overwrite_cache",
    "cache_overwrite_reason",
    "storage_precision",
    "binary_prefilter",
    "binary_rescore_multiplier",
    "calibration_sample_size",
    "cache_compression",
    "cache_compression_level",
    "encode_batch_size",
    "torch_compile",
    "device",
    "semantic_source",
    "candidate_pool_size",
}


def _shared_embedding_builder_kwargs(cli_args: argparse.Namespace) -> dict[str, object]:
    """Build shared embedding kwargs for embedding-aware strategy builders.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :return Dict[str, object]: Shared kwargs consumed by embedding/hybrid builders.
    """
    return {
        **_configured_client_kwargs(cli_args),
        "model_name": cli_args.model,
        "model_profile": cli_args.model_profile,
        "model_revision": cli_args.model_revision,
        "dataset_source": cli_args.dataset_source,
        "dataset_split": cli_args.dataset_split,
        "corpus_size": None if cli_args.all_corpus else cli_args.corpus_size,
        "truncate_dim": cli_args.truncate_dim,
        "min_semantic_similarity": cli_args.min_semantic_similarity,
        "use_streaming": cli_args.streaming,
        "force_rebuild_cache": cli_args.force_rebuild_cache,
        "force_rebuild_reason": getattr(cli_args, "cache_overwrite_reason", None),
        "storage_precision": cli_args.storage_precision,
        "binary_prefilter": cli_args.binary_prefilter,
        "binary_rescore_multiplier": cli_args.binary_rescore_multiplier,
        "calibration_sample_size": cli_args.calibration_sample_size,
        "cache_compression": cli_args.cache_compression,
        "cache_compression_level": cli_args.cache_compression_level,
        "encode_batch_size": cli_args.encode_batch_size,
        "enable_torch_compile": cli_args.torch_compile,
        "device": cli_args.device,
        "semantic_source": cli_args.semantic_source,
        "candidate_pool_size": cli_args.candidate_pool_size,
    }


def _normalized_cache_reason(raw_reason: str | None) -> str | None:
    """Normalize optional cache-clear rationale into a compact single-line token.

    :param Optional[str] raw_reason: Raw user-provided rationale text.
    :return Optional[str]: Normalized reason, or ``None`` when absent.
    """
    if raw_reason is None:
        return None
    normalized = " ".join(str(raw_reason).split())
    return normalized or None


def _embedding_export_metadata(
    cli_args: argparse.Namespace, runtime_metadata: dict[str, Any] | None = None
) -> dict[str, object]:
    """Build embedding provenance payload persisted in export metadata.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :param Optional[Dict[str, Any]] runtime_metadata: Optional runtime retrieval metadata.
    :return Dict[str, object]: Embedding cache/vector provenance + runtime fields.
    """
    int8_mode = str(cli_args.storage_precision) == "int8"
    binary_prefilter_enabled = bool(cli_args.binary_prefilter and int8_mode)
    binary_prefilter_used_for_query: bool | None
    binary_prefilter_used_for_query = None
    if int8_mode and isinstance(runtime_metadata, dict):
        raw_used = runtime_metadata.get("binary_prefilter_used")
        if isinstance(raw_used, bool):
            binary_prefilter_used_for_query = raw_used
    elif not int8_mode:
        binary_prefilter_used_for_query = False

    effective_device: str | None = None
    effective_compute_dtype: str | None = None
    effective_model = str(cli_args.model)
    effective_model_revision: str | None = None
    model_fingerprint: str | None = None
    effective_truncate_dim = cli_args.truncate_dim
    effective_model_profile = str(cli_args.model_profile)
    retrieval_representation = "retrieval-query/retrieval-document"
    graph_representation = "graph-similarity"
    if isinstance(runtime_metadata, dict):
        raw_device = runtime_metadata.get("device")
        raw_compute_dtype = runtime_metadata.get("compute_dtype")
        raw_active_model = runtime_metadata.get("active_model")
        raw_model_fingerprint = runtime_metadata.get("model_fingerprint")
        raw_resolved_model_revision = runtime_metadata.get("resolved_model_revision")
        raw_truncate_dim = runtime_metadata.get("truncate_dim")
        if isinstance(raw_device, str) and raw_device:
            effective_device = raw_device
        if isinstance(raw_compute_dtype, str) and raw_compute_dtype:
            effective_compute_dtype = raw_compute_dtype
        if isinstance(raw_active_model, str) and raw_active_model:
            effective_model = raw_active_model
        if isinstance(raw_model_fingerprint, str) and raw_model_fingerprint:
            model_fingerprint = raw_model_fingerprint
        if isinstance(raw_resolved_model_revision, str) and raw_resolved_model_revision:
            effective_model_revision = raw_resolved_model_revision
        if isinstance(raw_truncate_dim, int) and not isinstance(raw_truncate_dim, bool):
            effective_truncate_dim = raw_truncate_dim
        raw_model_profile = runtime_metadata.get("model_profile")
        if isinstance(raw_model_profile, str) and raw_model_profile:
            effective_model_profile = raw_model_profile
        raw_retrieval_representation = runtime_metadata.get("retrieval_representation")
        raw_graph_representation = runtime_metadata.get("graph_representation")
        if (
            isinstance(raw_retrieval_representation, str)
            and raw_retrieval_representation
        ):
            retrieval_representation = raw_retrieval_representation
        if isinstance(raw_graph_representation, str) and raw_graph_representation:
            graph_representation = raw_graph_representation

    payload: dict[str, object] = {
        "effective_vector_dtype": "float32",
        "effective_model": effective_model,
        "effective_model_revision": effective_model_revision,
        "model_fingerprint": model_fingerprint,
        "effective_truncate_dim": effective_truncate_dim,
        "effective_device": effective_device,
        "effective_compute_dtype": effective_compute_dtype,
        "model_profile": effective_model_profile,
        "retrieval_representation": retrieval_representation,
        "graph_representation": graph_representation,
        "semantic_source": str(cli_args.semantic_source),
        "candidate_pool_size": int(cli_args.candidate_pool_size),
        "storage_precision": str(cli_args.storage_precision),
        "binary_prefilter_enabled": binary_prefilter_enabled,
        "binary_prefilter_used_for_query": binary_prefilter_used_for_query,
        "binary_rescore_multiplier": (
            int(cli_args.binary_rescore_multiplier) if int8_mode else 1
        ),
        "cache_overwrite_reason": _normalized_cache_reason(
            getattr(cli_args, "cache_overwrite_reason", None)
        ),
    }
    if payload["semantic_source"] == "arxiv-corpus":
        # Corpus builds never consult the candidate pool; recording its default
        # here contradicted the config sidecar, which omits it.
        payload.pop("candidate_pool_size")
    return payload


def _plot_overlay_metadata(export_metadata: dict[str, Any]) -> dict[str, Any]:
    """Return compact metadata suitable for static image overlays.

    :param Dict[str, Any] export_metadata: Full export metadata payload.
    :return Dict[str, Any]: Reduced metadata subset for on-plot annotation.
    """
    overlay_keys = ("paper_id", "strategy", "nodes", "edges", "theme", "timestamp")
    return {
        key: export_metadata[key]
        for key in overlay_keys
        if key in export_metadata and export_metadata[key] is not None
    }


def _resolved_hybrid_max_semantic(cli_args: argparse.Namespace) -> int:
    """Resolve effective hybrid semantic cap from CLI arguments.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :return int: Effective ``max_semantic`` value.
    """
    if cli_args.max_semantic is None:
        return max(0, min(int(DEFAULT_MAX_SEMANTIC), int(cli_args.max_papers) - 1))
    return int(cli_args.max_semantic)


def _apply_hybrid_default_overrides(
    args: argparse.Namespace, provided: set[str]
) -> None:
    """Apply tuned hybrid defaults when budget knobs are omitted.

    :param argparse.Namespace args: Parsed build arguments.
    :param Set[str] provided: Explicit option destinations found in argv.
    :return None: Mutates ``args`` in place for omitted hybrid budget fields.
    """
    if str(args.strategy) != "hybrid":
        return

    for dest, value in _HYBRID_BEST_PRACTICE_DEFAULTS.items():
        if dest in provided:
            continue
        setattr(args, dest, int(value))


def _apply_user_config_defaults(
    args: argparse.Namespace, provided: set[str], user_config: UserConfig
) -> set[str]:
    """Overlay config.toml defaults onto build args the user did not set.

    Precedence: explicit CLI flag > config.toml > built-in default. Values are
    already whitelist-validated at config load time.

    :param argparse.Namespace args: Parsed build arguments.
    :param Set[str] provided: Explicit option destinations found in argv.
    :param UserConfig user_config: Loaded user configuration snapshot.
    :return Set[str]: Destinations that were filled from config.toml.
    """
    applied: set[str] = set()
    for dest in sorted(user_config.defaults):
        if dest in provided or not hasattr(args, dest):
            continue
        value = user_config.defaults[dest]
        setattr(args, dest, list(value) if isinstance(value, list) else value)
        applied.add(dest)
    if applied:
        summary = ", ".join(
            f"{dest}={format_config_value(user_config.defaults[dest])}"
            for dest in sorted(applied)
        )
        logger.debug("Loaded config defaults from %s: %s", user_config.path, summary)
    return applied


def _resolve_user_config_api_key(user_config: UserConfig) -> str | None:
    """Resolve the S2 API key without exposing config secrets to subprocesses.

    Environment presence wins even for an empty value, so ``S2_API_KEY=""``
    still explicitly disables the configured key.

    :param UserConfig user_config: Loaded user configuration snapshot.
    :return str | None: Environment key, configured key, or no configured value.
    """
    if "S2_API_KEY" in os.environ:
        return os.environ["S2_API_KEY"]
    if user_config.s2_api_key:
        logger.debug("Using api.s2_api_key from %s.", user_config.path)
    return user_config.s2_api_key or None


def _configured_client_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Inject a client only when this command has API-specific settings.

    :param argparse.Namespace args: Parsed arguments and resolved API key.
    :return Dict[str, object]: Optional client constructor argument for builders.
    """
    api_key = getattr(args, "_s2_api_key", None)
    refresh = bool(getattr(args, "refresh_paper_cache", False))
    if api_key is None and not refresh:
        return {}
    return {
        "client": SemanticScholarClient(api_key=api_key, refresh_paper_cache=refresh)
    }


def _hybrid_semantic_branch_enabled(cli_args: argparse.Namespace) -> bool:
    """Return whether hybrid semantic branch is effectively enabled.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :return bool: ``True`` when hybrid semantic branch can run.
    """
    return _resolved_hybrid_max_semantic(cli_args) > 0


def _embedding_branch_enabled(cli_args: argparse.Namespace) -> bool:
    """Return whether the active build will execute an embedding-backed workflow.

    :param argparse.Namespace cli_args: Parsed CLI arguments.
    :return bool: ``True`` when embedding runtime, cache, and corpus work may run.
    """
    strategy = str(getattr(cli_args, "strategy", "")).strip().lower()
    if strategy == "embedding":
        return True
    if strategy == "hybrid":
        return _hybrid_semantic_branch_enabled(cli_args)
    return False


def _strategy_score_contract(strategy: str) -> dict[str, object]:
    """Return strategy-specific score semantics metadata for export payloads.

    :param str strategy: Active build strategy.
    :return Dict[str, object]: Score semantics metadata.
    """
    base_contract = {
        "strategy": strategy,
        "comparable_across_strategies": False,
        "range_hint": "[0,1] strategy-specific composite score",
    }
    if strategy == "citation":
        return {
            **base_contract,
            "score_type": "citation_similarity_composite",
        }
    if strategy == "recommendation":
        return {
            **base_contract,
            "score_type": "recommendation_similarity_composite",
        }
    if strategy == "embedding":
        return {
            **base_contract,
            "score_type": "embedding_similarity_composite",
        }
    if strategy == "hybrid":
        return {
            **base_contract,
            "score_type": "hybrid_similarity_composite",
            "adjudication_policy": (
                "seed-relevance-ranked union across citation+semantic candidates "
                "with overlap priority; purely semantic additions are capped by "
                "max_semantic."
            ),
        }
    return {
        **base_contract,
        "score_type": "unknown",
    }


_STRATEGY_DISPATCH: dict[str, StrategyFactory] = {
    "citation": lambda cli_args: CitationGraphBuilder(
        **_configured_client_kwargs(cli_args),
        max_papers=cli_args.max_papers,
        max_citations=cli_args.max_citations,
        max_references=cli_args.max_references,
        similarity_threshold=cli_args.similarity_threshold,
        fetch_references=not cli_args.no_references,
        refresh_reference_cache=cli_args.refresh_reference_cache,
    ),
    "recommendation": lambda cli_args: RecommendationGraphBuilder(
        **_configured_client_kwargs(cli_args),
        max_papers=cli_args.max_papers,
        fetch_references=not cli_args.no_references,
        refresh_reference_cache=cli_args.refresh_reference_cache,
        similarity_threshold=cli_args.similarity_threshold,
    ),
    "embedding": lambda cli_args: EmbeddingGraphBuilder(
        max_papers=cli_args.max_papers,
        top_k=cli_args.top_k,
        **_shared_embedding_builder_kwargs(cli_args),
    ),
    "hybrid": lambda cli_args: HybridGraphBuilder(
        max_papers=cli_args.max_papers,
        max_citations=cli_args.max_citations,
        max_references=cli_args.max_references,
        fetch_references=not cli_args.no_references,
        refresh_reference_cache=cli_args.refresh_reference_cache,
        max_semantic=cli_args.max_semantic,
        **_shared_embedding_builder_kwargs(cli_args),
    ),
}
