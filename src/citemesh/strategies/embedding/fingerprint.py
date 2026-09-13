"""Cache-identity fingerprints for the active embedding model.

Owns everything that answers "are the cached vectors still valid for this
model?": the formatter fingerprint, the resolved model fingerprint (including
local HuggingFace snapshot resolution), the inference-artifact digest walk, and
the cache fingerprint check that forces a rebuild when any of them changes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from citemesh.data import format_bytes

from . import deps
from .runtime import (
    _INFERENCE_ARTIFACT_FILENAMES,
    _TORCH_WEIGHT_LAYOUTS,
)
from .text import (
    _FORMATTER_FINGERPRINT_PROBES,
    _GRAPH_SIMILARITY_REPRESENTATION,
    _RETRIEVAL_DOCUMENT_REPRESENTATION,
)

if TYPE_CHECKING:
    from citemesh.data import EmbeddingCache

logger = logging.getLogger(__name__)


class EmbeddingCacheFingerprintMismatchError(RuntimeError):
    """A corpus-scale namespace outlived the model its vectors were built with.

    Raised instead of deleting the payload, so rehydrating hours of GPU time is
    an explicit operator decision rather than a background heuristic.
    """


class _FingerprintMixin:
    """Model- and formatter-identity fingerprints for embedding cache validity.

    Mixed into :class:`~citemesh.strategies.embedding.builder.EmbeddingGraphBuilder`.

    Requires the host to provide: ``model_name``, ``model_profile``,
    ``_model_revision``, ``_active_model_name``, ``_resolved_model_fingerprint``,
    ``_document_formatter_fingerprint``, ``_similarity_formatter_fingerprint``,
    ``_pending_force_rebuild_reason``, ``truncate_dim``, ``compute_dtype``,
    ``storage_precision``, ``embedding_cache`` and ``_clear_embedding_cache``.
    """

    def _resolve_formatter_fingerprint(
        self,
        *,
        formatter: Callable[..., str],
        probe_renderer: Callable[[dict[str, str]], str],
    ) -> str:
        """Resolve a deterministic cache fingerprint for one formatter role.

        :param Callable[..., str] formatter: Underlying formatter identity.
        :param Callable[[Dict[str, str]], str] probe_renderer: Probe rendering callback.
        :return str: SHA-256 digest of profile formatter probes.
        """
        outputs = [
            probe_renderer(dict(metadata)) for metadata in _FORMATTER_FINGERPRINT_PROBES
        ]
        payload = "||".join(
            (
                str(self.model_profile.name),
                str(getattr(formatter, "__module__", "")),
                str(
                    getattr(
                        formatter,
                        "__qualname__",
                        getattr(formatter, "__name__", "formatter"),
                    )
                ),
                *outputs,
            )
        )
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _resolve_model_fingerprint(self) -> str:
        """Resolve deterministic model fingerprint for cache validity checks.

        :return str: Model fingerprint token.
        :raises RuntimeError: If no strong/weak local fingerprint can be resolved.
        """
        if self._resolved_model_fingerprint is not None:
            return self._resolved_model_fingerprint

        model_identity = self._cache_model_identity()
        resolved_path = Path(model_identity).expanduser()
        if resolved_path.exists():
            artifact_digest = self._resolve_inference_artifact_digest(resolved_path)
            fingerprint = (
                f"local::{resolved_path.resolve()}::artifact={artifact_digest}"
            )
            self._resolved_model_fingerprint = fingerprint
            return fingerprint

        model_id = model_identity
        if "/" not in model_id:
            raise RuntimeError(
                "Could not derive immutable artifact identity for embedding model alias "
                f"{model_id!r}; use a local checkpoint path or a full Hugging Face "
                "repository ID."
            )

        requested_revision = self._requested_hf_revision_token()
        if re.fullmatch(r"[0-9a-f]{40}", requested_revision, flags=re.IGNORECASE):
            fingerprint = f"hf::{model_id}::{requested_revision.lower()}"
            self._resolved_model_fingerprint = fingerprint
            return fingerprint
        resolved_sha = ""
        resolution_error: Exception | None = None

        local_snapshot_sha = self._resolve_local_hf_snapshot_sha(
            model_id=model_id,
            requested_revision=requested_revision,
        )
        if local_snapshot_sha is not None:
            resolved_sha = local_snapshot_sha

        if not resolved_sha:
            try:
                huggingface_hub = deps._import_huggingface_hub_module()
                model_info = huggingface_hub.HfApi().model_info(
                    repo_id=model_id,
                    revision=requested_revision,
                )
                resolved_sha = str(getattr(model_info, "sha", "") or "").strip()
            except Exception as exc:
                resolution_error = exc

        if not resolved_sha:
            local_artifact_fingerprint = self._resolve_local_hf_artifact_fingerprint(
                model_id=model_id,
                requested_revision=requested_revision,
            )
            if local_artifact_fingerprint is not None:
                logger.debug(
                    "Could not resolve Hugging Face commit SHA for %s (revision=%s). "
                    "Using a content fingerprint of the complete local inference "
                    "artifact manifest.",
                    model_id,
                    requested_revision,
                )
                self._resolved_model_fingerprint = local_artifact_fingerprint
                return local_artifact_fingerprint

        if not resolved_sha:
            raise RuntimeError(
                "Could not resolve Hugging Face commit SHA for "
                f"{model_id!r} (revision={requested_revision!r}); "
                "refusing to use embedding cache without model fingerprint."
            ) from resolution_error

        fingerprint = f"hf::{model_id}::{resolved_sha}"
        self._resolved_model_fingerprint = fingerprint
        return fingerprint

    def _requested_hf_revision_token(self) -> str:
        """Return normalized Hugging Face revision token for current builder config.

        :return str: Normalized requested revision token.
        """
        return (self.model_revision or "main").strip() or "main"

    @staticmethod
    def _resolve_local_hf_snapshot_path(
        model_id: str, requested_revision: str
    ) -> Path | None:
        """Resolve an existing Hugging Face snapshot without network access.

        :param str model_id: Hugging Face repository ID.
        :param str requested_revision: Requested model revision token.
        :return Optional[Path]: Resolved local snapshot directory, if available.
        """
        try:
            snapshot_download = deps._import_huggingface_hub_module().snapshot_download
            return Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=requested_revision,
                    local_files_only=True,
                )
            ).resolve()
        except Exception:
            return None

    def _resolve_local_hf_snapshot_sha(
        self, model_id: str, requested_revision: str
    ) -> str | None:
        """Best-effort local SHA resolution from existing HF snapshot cache.

        :param str model_id: Hugging Face repository ID.
        :param str requested_revision: Requested model revision token.
        :return Optional[str]: Locally resolved snapshot SHA, if available.
        """
        snapshot_path = self._resolve_local_hf_snapshot_path(
            model_id, requested_revision
        )
        if snapshot_path is None:
            return None

        parts = snapshot_path.parts
        for idx, part in enumerate(parts):
            if part != "snapshots":
                continue
            if idx + 1 >= len(parts):
                continue
            candidate = str(parts[idx + 1]).strip()
            if re.fullmatch(r"[0-9a-f]{40}", candidate, flags=re.IGNORECASE):
                logger.debug(
                    "Resolved Hugging Face snapshot SHA for %s (revision=%s) from local cache.",
                    model_id,
                    requested_revision,
                )
                return candidate.lower()
        return None

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Return hex SHA-256 digest for a file path.

        :param Path path: File path to hash.
        :return str: Lowercase SHA-256 hex digest.
        """
        digest = sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _safe_artifact_reference(root: Path, source: Path, raw_path: object) -> Path:
        """Resolve one manifest reference without allowing lexical path escape.

        Hugging Face snapshot files may be symlinks into the blob store, so this
        validates the logical path before following the file rather than rejecting
        legitimate cache symlink targets.

        :param Path root: Artifact root directory.
        :param Path source: Manifest file containing the reference.
        :param object raw_path: Referenced relative path.
        :return Path: Validated logical path below ``root``.
        :raises RuntimeError: If the reference is absolute or escapes the root.
        """
        reference = str(raw_path or "").strip()
        if not reference:
            raise RuntimeError(f"Empty artifact reference in {source}.")
        candidate = source.parent / reference
        root_absolute = os.path.abspath(root)
        candidate_absolute = os.path.abspath(candidate)
        if os.path.commonpath([root_absolute, candidate_absolute]) != root_absolute:
            raise RuntimeError(
                f"Artifact reference {reference!r} in {source} escapes {root}."
            )
        return Path(candidate_absolute)

    @staticmethod
    def _referenced_python_artifact_paths(
        root: Path,
        raw_reference: object,
    ) -> set[Path]:
        """Resolve locally referenced custom Python model code.

        SentenceTransformers loads repository-local module classes through the
        Transformers dynamic-module loader, which recursively copies relative
        imports. This follows that same bounded dependency graph.

        :param Path root: Complete local model root.
        :param object raw_reference: Dotted module class reference.
        :return Set[Path]: Existing referenced Python artifacts below ``root``.
        """
        if not isinstance(raw_reference, str) or "--" in raw_reference:
            return set()

        module_name, separator, _ = raw_reference.strip().rpartition(".")
        module_parts = module_name.split(".")
        if not separator or any(
            not part or re.fullmatch(r"[A-Za-z_]\w*", part) is None
            for part in module_parts
        ):
            return set()

        module_path = root.joinpath(*module_parts).with_suffix(".py")
        if not module_path.is_file():
            return set()

        paths = {module_path}
        pending = [module_path]
        dependency_root = module_path.parent
        while pending:
            current_path = pending.pop()
            source = current_path.read_text(encoding="utf-8")
            relative_imports = re.findall(
                r"^\s*import\s+\.(\S+)\s*$", source, flags=re.MULTILINE
            )
            relative_imports.extend(
                re.findall(r"^\s*from\s+\.(\S+)\s+import", source, flags=re.MULTILINE)
            )
            for relative_import in relative_imports:
                dependency_path = dependency_root / f"{relative_import}.py"
                if dependency_path.is_file() and dependency_path not in paths:
                    paths.add(dependency_path)
                    pending.append(dependency_path)
        return paths

    @classmethod
    def _inference_artifact_paths(cls, artifact_root: Path) -> list[Path]:
        """Enumerate and validate inference-relevant checkpoint artifacts.

        :param Path artifact_root: Local model file or directory.
        :return List[Path]: Stable sorted artifact paths.
        :raises RuntimeError: If the layout is empty, malformed, or incomplete.
        """
        root = artifact_root.expanduser().resolve()
        if root.is_file():
            return [root]
        if not root.is_dir():
            raise RuntimeError(f"Local embedding model path is not readable: {root}")

        artifacts, artifact_roots, python_references = cls._declared_module_artifacts(
            root
        )
        for current_root in artifact_roots:
            artifacts.update(
                path
                for file_name in _INFERENCE_ARTIFACT_FILENAMES
                if (path := current_root / file_name).is_file()
            )
            artifacts.update(cls._tokenizer_referenced_artifacts(root, current_root))
            artifacts.update(cls._torch_weight_artifacts(root, current_root))

        for reference in python_references:
            artifacts.update(cls._referenced_python_artifact_paths(root, reference))

        if not artifacts:
            raise RuntimeError(
                f"No inference-relevant artifacts found under local model path {root}."
            )
        return sorted(artifacts, key=lambda path: path.relative_to(root).as_posix())

    @classmethod
    def _declared_module_artifacts(
        cls, root: Path
    ) -> tuple[set[Path], set[Path], list[object]]:
        """Read ``modules.json`` and expand the module layout it declares.

        Each declared module contributes its own artifact root (so config and
        tokenizer files under a submodule directory are scanned too) and its
        ``type``, which may name an importable Python module whose source is
        part of the checkpoint's inference behavior.

        :param Path root: Resolved local model directory.
        :return Tuple[Set[Path], Set[Path], List[object]]: Artifacts found so far,
            the roots still to scan, and the declared module type references.
        :raises RuntimeError: If ``modules.json`` is malformed or points at a
            missing path.
        """
        artifacts: set[Path] = set()
        artifact_roots: set[Path] = {root}
        python_references: list[object] = []
        modules_path = root / "modules.json"
        if not modules_path.is_file():
            return artifacts, artifact_roots, python_references

        artifacts.add(modules_path)
        try:
            modules_payload = json.loads(modules_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Malformed SentenceTransformers modules.json: {exc}"
            ) from exc
        if not isinstance(modules_payload, list):
            raise RuntimeError("SentenceTransformers modules.json must be a list.")
        for module in modules_payload:
            if not isinstance(module, dict):
                raise RuntimeError(
                    "SentenceTransformers modules.json entries must be objects."
                )
            python_references.append(module.get("type"))
            module_path = str(module.get("path", "") or "").strip()
            if not module_path:
                continue
            resolved_module = cls._safe_artifact_reference(
                root, modules_path, module_path
            )
            if not resolved_module.exists():
                raise RuntimeError(
                    f"SentenceTransformers module path is missing: {module_path}"
                )
            if resolved_module.is_dir():
                artifact_roots.add(resolved_module)
                artifacts.update(
                    path for path in resolved_module.rglob("*.py") if path.is_file()
                )
            elif resolved_module.is_file():
                artifacts.add(resolved_module)
        return artifacts, artifact_roots, python_references

    @classmethod
    def _tokenizer_referenced_artifacts(
        cls, root: Path, current_root: Path
    ) -> set[Path]:
        """Collect files a ``tokenizer_config.json`` points at by ``*_file(s)`` keys.

        Tokenizer configs reference vocabularies and merge tables by relative
        path. A missing or unreadable config is not fatal here: the config file
        itself is already digested through the declared-filename scan, and an
        unresolvable reference simply contributes nothing.

        :param Path root: Resolved local model directory used as the escape boundary.
        :param Path current_root: Artifact root whose tokenizer config is read.
        :return Set[Path]: Existing files referenced by the tokenizer config.
        """
        tokenizer_config_path = current_root / "tokenizer_config.json"
        if not tokenizer_config_path.is_file():
            return set()
        try:
            tokenizer_config = json.loads(
                tokenizer_config_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            tokenizer_config = {}
        if not isinstance(tokenizer_config, dict):
            return set()

        artifacts: set[Path] = set()
        for key, raw_reference in tokenizer_config.items():
            if key.endswith("_file") and isinstance(raw_reference, str):
                references = [raw_reference]
            elif key.endswith("_files") and isinstance(raw_reference, list):
                references = raw_reference
            else:
                continue
            for reference in references:
                if not isinstance(reference, str):
                    continue
                try:
                    referenced_path = cls._safe_artifact_reference(
                        root, tokenizer_config_path, reference
                    )
                except RuntimeError:
                    continue
                if referenced_path.is_file():
                    artifacts.add(referenced_path)
        return artifacts

    @classmethod
    def _torch_weight_artifacts(cls, root: Path, current_root: Path) -> set[Path]:
        """Select one weight layout and expand a sharded index into its shards.

        Only the first matching layout in :data:`_TORCH_WEIGHT_LAYOUTS` counts, so
        a checkpoint shipping both safetensors and a pickle mirror digests once.
        Unlike the tokenizer references, a declared-but-missing shard is fatal:
        the digest would otherwise claim to cover weights that cannot load.

        :param Path root: Resolved local model directory used as the escape boundary.
        :param Path current_root: Artifact root whose weight layout is selected.
        :return Set[Path]: The selected weight file plus any shards it indexes.
        :raises RuntimeError: If a weight index is malformed or names a missing shard.
        """
        selected_weight = next(
            (
                current_root / file_name
                for file_name in _TORCH_WEIGHT_LAYOUTS
                if (current_root / file_name).is_file()
            ),
            None,
        )
        if selected_weight is None:
            return set()
        artifacts: set[Path] = {selected_weight}
        if not selected_weight.name.endswith(".index.json"):
            return artifacts

        try:
            index_payload = json.loads(selected_weight.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Malformed weight index {selected_weight}: {exc}"
            ) from exc
        weight_map = (
            index_payload.get("weight_map") if isinstance(index_payload, dict) else None
        )
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"Weight index has no weight_map: {selected_weight}")
        for shard_name in sorted({str(value) for value in weight_map.values()}):
            shard_path = cls._safe_artifact_reference(root, selected_weight, shard_name)
            if not shard_path.is_file():
                raise RuntimeError(
                    f"Weight index references a missing shard: {shard_name}"
                )
            artifacts.add(shard_path)
        return artifacts

    @classmethod
    def _resolve_inference_artifact_digest(cls, artifact_root: Path) -> str:
        """Hash a canonical manifest of inference-relevant model contents.

        :param Path artifact_root: Local model file or directory.
        :return str: SHA-256 manifest digest.
        """
        root = artifact_root.expanduser().resolve()
        paths = cls._inference_artifact_paths(root)
        digest = sha256()
        for path in paths:
            relative = (
                path.name if root.is_file() else path.relative_to(root).as_posix()
            )
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(cls._sha256_file(path).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _resolve_local_hf_artifact_fingerprint(
        self, model_id: str, requested_revision: str
    ) -> str | None:
        """Best-effort local artifact fingerprint using config + weights files.

        :param str model_id: Hugging Face repository ID.
        :param str requested_revision: Requested model revision token.
        :return Optional[str]: Deterministic local artifact fingerprint if available.
        """
        snapshot_path = self._resolve_local_hf_snapshot_path(
            model_id, requested_revision
        )
        if snapshot_path is None:
            return None

        try:
            artifact_digest = self._resolve_inference_artifact_digest(snapshot_path)
        except (OSError, RuntimeError):
            return None

        return (
            f"hf::{model_id}::revision={requested_revision}::artifact={artifact_digest}"
        )

    def _ensure_cache_model_fingerprint(
        self,
        *,
        representation: str = _RETRIEVAL_DOCUMENT_REPRESENTATION,
    ) -> None:
        """Verify one task cache is bound to the active model fingerprint.

        :param str representation: Retrieval-document or graph-similarity role.
        :return None: Validates and records model identity on the selected cache.
        :raises EmbeddingCacheFingerprintMismatchError: If a corpus-scale payload
            was built with a different model and no rebuild was authorized.
        """
        try:
            model_fingerprint = self._resolve_model_fingerprint()
        except Exception as exc:
            raise RuntimeError(
                "Could not verify the runtime-active embedding model identity; "
                "refusing persistent cache access instead of adopting an "
                f"unverified payload for {self._cache_model_identity()!r}."
            ) from exc

        self._resolved_model_fingerprint = model_fingerprint
        self._bind_embedding_cache_to_active_model()
        if representation == _RETRIEVAL_DOCUMENT_REPRESENTATION:
            cache = self.embedding_cache
        elif representation == _GRAPH_SIMILARITY_REPRESENTATION:
            cache = self.graph_embedding_cache
        else:
            raise ValueError(
                f"Unsupported embedding cache representation: {representation}"
            )
        operation_lock = (
            cache.hydration_operation_lock()
            if representation == _RETRIEVAL_DOCUMENT_REPRESENTATION
            and self.semantic_source == "arxiv-corpus"
            else nullcontext()
        )
        with operation_lock:
            has_cached_payload = cache.has_cached_payload()
            cached_fingerprint = cache.get_model_fingerprint()
            if has_cached_payload and cached_fingerprint != model_fingerprint:
                self._refuse_expensive_fingerprint_rebuild(
                    cache=cache,
                    cached_fingerprint=cached_fingerprint,
                    model_fingerprint=model_fingerprint,
                )
                logger.warning(
                    "REBUILDING EMBEDDING CACHE — please hang tight. "
                    "Embeddings will be regenerated automatically; this may take a while. "
                    "No action is needed. "
                    "Reason: embedding cache model fingerprint mismatch (cached=%s, active=%s).",
                    cached_fingerprint or "missing",
                    model_fingerprint,
                )
                cache.clear(
                    reason=(
                        "model fingerprint mismatch "
                        f"(cached={cached_fingerprint or 'missing'}, "
                        f"active={model_fingerprint})"
                    )
                )
            if not has_cached_payload or cached_fingerprint != model_fingerprint:
                cache.set_model_fingerprint(model_fingerprint)

    @staticmethod
    def _refuse_expensive_fingerprint_rebuild(
        *,
        cache: EmbeddingCache,
        cached_fingerprint: str | None,
        model_fingerprint: str,
    ) -> None:
        """Abort before a fingerprint mismatch deletes a hydrated corpus.

        A candidate-pool namespace is bounded by ``--candidate-pool-size`` and
        re-encodes in seconds, so it keeps the silent-rebuild path. A corpus
        namespace costs GPU-hours, so its deletion must be an operator decision.
        Any recorded hydration source marks the namespace as corpus-scale --
        including an interrupted hydration, which is just as costly to redo --
        and only the corpus paths ever write that metadata.

        :param EmbeddingCache cache: Namespace whose payload the mismatch targets.
        :param Optional[str] cached_fingerprint: Fingerprint recorded on the payload.
        :param str model_fingerprint: Fingerprint of the runtime-active model.
        :return None: Returns when the namespace is cheap enough to auto-rebuild.
        :raises EmbeddingCacheFingerprintMismatchError: If the namespace carries
            corpus hydration metadata.
        """
        stats = cache.payload_stats()
        if not stats.hydration_dataset_source:
            return

        cached_rows = max(stats.sqlite_rows, stats.embedding_rows)
        raise EmbeddingCacheFingerprintMismatchError(
            "Cached embeddings were built with a different embedding model "
            f"(cached={cached_fingerprint or 'missing'}, active={model_fingerprint}). "
            f"This cache holds {cached_rows:,} corpus paper(s) "
            f"({format_bytes(stats.size_bytes)}) hydrated from "
            f"{stats.hydration_dataset_source}, so CiteMesh refused to delete it "
            "automatically. Re-run with --force-rebuild-cache to discard it and "
            "rebuild (add --overwrite-cache to skip the confirmation prompt in "
            "scripts), or run 'citemesh cache clear' to remove the cache."
        )
