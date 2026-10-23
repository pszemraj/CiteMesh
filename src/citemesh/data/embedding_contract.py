"""Persistent embedding semantics and source provenance, without model imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

INT8_QUANTIZATION = "affine-int8-v1"


@dataclass(frozen=True)
class EmbeddingContract:
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusSource:
    repo_id: str
    config_name: str | None
    split: str
    revision: str
    selection: Literal["all", "newest"]
    limit: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExistingCacheSelection:
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
    """Serialize the existing native address without adding source provenance."""
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
    selectors: dict[str, Any],
    namespace_serialization: str,
) -> None:
    """Check that historical addressing describes the fixed vector contract."""
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
