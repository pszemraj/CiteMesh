"""Persistent embedding semantics and source provenance, without model imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

INT8_QUANTIZATION = "affine-int8-v1"


class InstalledCorpusError(RuntimeError):
    """Refuse an installed corpus that requires operator action before use."""


class EmbeddingContractData(TypedDict):
    """Serialized fields that identify stored embedding vectors."""

    model_id: str
    model_revision: str
    model_artifact_identity: str
    profile_key: str
    profile_schema: str
    representation: str
    formatter_identity: str
    normalization: str
    dimension: int
    stored_dtype: Literal["int8", "float32"]
    quantization: str | None
    calibration_sample_size: int | None


class CorpusSourceData(TypedDict):
    """Serialized fields that identify a selected source corpus."""

    repo_id: str
    config_name: str | None
    split: str
    revision: str
    selection: Literal["all", "newest"]
    limit: int | None


class NamespaceSelectors(TypedDict, total=False):
    """Typed inputs used to reproduce a native embedding namespace."""

    model_identity: str
    requested_revision: str
    artifact_identity: str
    profile_schema: str
    representation: str
    normalization: str
    dimension: int | None
    stored_dtype: str
    calibration_sample_size: int | None
    formatter_identity: str
    semantic_source: str


@dataclass(frozen=True)
class EmbeddingContract:
    """Describe the model and vector representation persisted in a corpus cache."""

    model_id: str
    model_revision: str
    model_artifact_identity: str
    profile_key: str
    profile_schema: str
    representation: str
    formatter_identity: str
    normalization: str
    dimension: int
    stored_dtype: Literal["int8", "float32"]
    quantization: str | None
    calibration_sample_size: int | None

    def to_dict(self) -> EmbeddingContractData:
        """Return the contract fields in their portable mapping form."""
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_artifact_identity": self.model_artifact_identity,
            "profile_key": self.profile_key,
            "profile_schema": self.profile_schema,
            "representation": self.representation,
            "formatter_identity": self.formatter_identity,
            "normalization": self.normalization,
            "dimension": self.dimension,
            "stored_dtype": self.stored_dtype,
            "quantization": self.quantization,
            "calibration_sample_size": self.calibration_sample_size,
        }


@dataclass(frozen=True)
class CorpusSource:
    """Describe the source selection used to populate a corpus cache."""

    repo_id: str
    config_name: str | None
    split: str
    revision: str
    selection: Literal["all", "newest"]
    limit: int | None

    def to_dict(self) -> CorpusSourceData:
        """Return the source fields in their portable mapping form."""
        return {
            "repo_id": self.repo_id,
            "config_name": self.config_name,
            "split": self.split,
            "revision": self.revision,
            "selection": self.selection,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class ExistingCacheSelection:
    """Resolved contract and source information for an installed cache."""

    namespace_id: str
    namespace_serialization: str
    contract: EmbeddingContract
    source_repo_id: str
    source_config_name: str | None
    source_split: str


def serialize_embedding_namespace(
    *,
    model_identity: str,
    requested_revision: str,
    artifact_identity: str = "unresolved",
    profile_schema: str,
    representation: str,
    normalization: str = "l2-v1",
    dimension: int | None = None,
    stored_dtype: str = "int8",
    calibration_sample_size: int | None = 2000,
    formatter_identity: str = "default",
    semantic_source: str = "arxiv-corpus",
) -> str:
    """Serialize the native cache address from model and representation inputs.

    :param str model_identity: Model identifier used for the cache key.
    :param str requested_revision: Requested model revision token.
    :param str artifact_identity: Resolved checkpoint identity.
    :param str profile_schema: Embedding profile schema token.
    :param str representation: Semantic role and schema of stored vectors.
    :param str normalization: Vector normalization contract.
    :param int | None dimension: Optional truncation dimension.
    :param str stored_dtype: Stored vector dtype.
    :param int calibration_sample_size: Calibration sample count for int8.
    :param str formatter_identity: Document formatter identity.
    :param str semantic_source: Semantic data source used for the namespace.
    :return str: Stable native namespace serialization.
    """
    parts = [
        f"model={model_identity}",
        f"revision={requested_revision}",
        f"artifact={artifact_identity}",
        f"profile={profile_schema}",
        f"representation={representation}",
        f"normalization={normalization}",
    ]
    if dimension is not None:
        parts.append(f"truncate_dim={dimension}")
    parts.extend(
        [
            f"storage_precision={stored_dtype}",
            f"binary_prefilter={int(stored_dtype == 'int8')}",
        ]
    )
    if stored_dtype == "int8":
        parts.append(f"calibration_sample_size={calibration_sample_size}")
    parts.append(f"formatter={formatter_identity}")
    if representation == "retrieval-document-v1" and semantic_source != "arxiv-corpus":
        parts.append(f"mode={semantic_source}")
    return "::".join(parts)


def require_namespace_contract(
    contract: EmbeddingContract,
    selectors: NamespaceSelectors,
    namespace_serialization: str,
) -> None:
    """Check that namespace selectors describe the fixed vector contract.

    :param EmbeddingContract contract: Persisted vector representation.
    :param NamespaceSelectors selectors: Inputs used to create the namespace.
    :param str namespace_serialization: Stored native namespace address.
    :raises ValueError: If the address or selectors conflict with the contract.
    """
    try:
        namespace = serialize_embedding_namespace(**selectors)
    except TypeError as exc:
        raise ValueError("Invalid embedding namespace selectors") from exc
    if namespace != namespace_serialization:
        raise ValueError("Namespace selectors disagree with the native address")
    expected = {
        "model_identity": contract.model_id,
        "artifact_identity": contract.model_artifact_identity,
        "profile_schema": contract.profile_schema,
        "representation": contract.representation,
        "normalization": contract.normalization,
        "stored_dtype": contract.stored_dtype,
        "formatter_identity": contract.formatter_identity,
    }
    defaults = {
        "artifact_identity": "unresolved",
        "normalization": "l2-v1",
        "stored_dtype": "int8",
        "formatter_identity": "default",
    }
    for name, value in expected.items():
        if selectors.get(name, defaults.get(name)) != value:
            raise ValueError(f"Namespace {name} differs from the embedding contract")
    dimension = selectors.get("dimension")
    if dimension is not None and dimension != contract.dimension:
        raise ValueError("Namespace dimension differs from the embedding contract")
    if selectors.get("semantic_source", "arxiv-corpus") != "arxiv-corpus":
        raise ValueError("A corpus namespace cannot contain candidate vectors")
    if contract.stored_dtype == "int8":
        if contract.quantization != INT8_QUANTIZATION:
            raise ValueError("Unsupported corpus int8 quantization")
        if (
            selectors.get("calibration_sample_size", 2000)
            != contract.calibration_sample_size
        ):
            raise ValueError(
                "Namespace calibration size differs from the embedding contract"
            )
    elif contract.quantization is not None:
        raise ValueError("Float32 vectors cannot declare int8 quantization")


def newest_corpus_order_sql(chronology_column: str, ordinal_column: str) -> str:
    """Order dated source records newest first, with stable source-order ties.

    :param str chronology_column: Chronology column used by the query.
    :param str ordinal_column: Source ordinal column used by the query.
    :return str: SQL ordering expression, without the ``ORDER BY`` keyword.
    """
    return (
        f"({chronology_column} IS NULL), {chronology_column} DESC, "
        f"CASE WHEN {chronology_column} IS NOT NULL THEN {ordinal_column} END DESC, "
        f"{ordinal_column} ASC"
    )
