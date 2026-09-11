"""Model load, precision and compile runtime for the embedding builder.

Owns the model contract binding (attention implementation, source dtype, bf16
policy), the SentenceTransformer load with its fallback chain, the autocast /
TF32 / precision contexts wrapped around encoding, and the ``torch.compile``
attempt plus its eager-restore recovery path.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager, nullcontext
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

import numpy as np

from citemesh.core import EMBEDDING_CONFIG
from citemesh.data import (
    DEFAULT_EMBEDDING_MODEL_FALLBACKS,
    resolve_embedding_model_profile,
)
from citemesh.text_batching import encode_texts

from . import deps
from .precision import _model_floating_dtype_names, _PrecisionEncodeProxy
from .runtime import (
    _COMPILE_ELIGIBLE_DEVICES,
    _FA2_FORWARD_DTYPE_WARNING,
    _MPS_MIN_TORCH_VERSION,
    _TF32_COMPILE_BRIDGE_TORCH_VERSIONS,
    EmbeddingBackendCompatibilityError,
    EmbeddingPrecisionCompatibilityError,
    _cpu_native_bf16_supported,
    _cuda_native_bf16_supported,
    _parse_torch_major_minor,
    _require_transformers_compatibility,
    _suppress_expected_fa2_load_dtype_warning,
    _suppress_transformers_progress_for_non_tty,
)

logger = logging.getLogger(__name__)


class _ModelRuntimeMixin:
    """Model loading, precision contexts and compile handling.

    Mixed into :class:`~citemesh.strategies.embedding.builder.EmbeddingGraphBuilder`.

    Requires the host to provide: ``model``, ``model_name``, ``model_profile``,
    ``device``, ``compute_dtype``, ``truncate_dim``, ``embedding_dim``,
    ``encode_batch_size``, ``prefetch_batches``, ``storage_precision``,
    ``_model_revision``, ``_active_model_name``, ``_source_dtype_hint``,
    ``_attention_implementation_hint``, ``_autocast_enabled``,
    ``_autocast_device_type``, ``_tf32_enabled``, ``_inner_model_compiled``,
    ``_compile_status_reason``, ``_bind_model_contract``,
    ``_resolve_truncate_dim``, ``_ensure_cache_model_fingerprint``,
    ``_bind_embedding_cache_to_active_model`` and ``_cached_payload_row_count``.
    """

    def _bind_model_contract(self, model_name_or_path: str) -> None:
        """Bind profile-derived formatting and runtime policy to one load candidate.

        :param str model_name_or_path: Hub identifier or local checkpoint path.
        :return None: Recomputes profile-dependent state when the contract changes.
        """
        resolved_profile = resolve_embedding_model_profile(
            model_name_or_path,
            self.requested_model_profile,
        )
        if resolved_profile == self.model_profile:
            return

        self.model_profile = resolved_profile
        self._document_formatter_fingerprint = self._resolve_formatter_fingerprint(
            formatter=self.model_profile.document_formatter,
            probe_renderer=self._format_retrieval_document_metadata,
        )
        self._similarity_formatter_fingerprint = self._resolve_formatter_fingerprint(
            formatter=self.model_profile.similarity_formatter,
            probe_renderer=self._format_graph_similarity_metadata,
        )
        self.truncate_dim = self._resolve_truncate_dim(self._requested_truncate_dim)
        self._source_dtype_hint = self._resolve_source_dtype_hint()
        self._attention_implementation_hint = (
            self._resolve_attention_implementation_hint()
        )
        self._embedding_cache = None
        self._graph_embedding_cache = None
        self._resolved_model_fingerprint = None
        self._autocast_dtype = None
        self._autocast_device_type = None
        self._autocast_enabled = False
        self._encode_model = None
        self._inner_model_compiled = False
        self._eager_inner_transformer = None
        self._compile_status_reason = None
        self._profile_logged = False
        self._dim_logged = False
        self._runtime_summary_logged = False

    def _resolve_attention_implementation_hint(self) -> Optional[str]:
        """Resolve preferred attention implementation for the resolved device.

        :return Optional[str]: Attention implementation token or ``None``.
        """
        if self.device == "cpu":
            return None

        preferred_attention = str(
            self.model_profile.preferred_attention_implementation or ""
        ).strip()
        if not preferred_attention:
            return None
        if preferred_attention != "flash_attention_2":
            return preferred_attention
        if (
            self.device == "cuda"
            and self._source_dtype_hint == "bfloat16"
            and deps._module_available("flash_attn")
        ):
            return preferred_attention
        # flash_attn ships CUDA-only kernels; the profile's portable fallback is SDPA.
        return "sdpa"

    def _resolve_source_dtype_hint(self) -> str:
        """Resolve effective compute dtype used for cache provenance metadata.

        :return str: Effective compute dtype token.
        """
        try:
            torch = deps._import_torch()
        except ImportError:
            return "float32"

        preferred_dtype = (self.model_profile.preferred_compute_dtype or "").lower()
        if preferred_dtype == "bfloat16" and self._bf16_autocast_allowed(torch):
            return "bfloat16"
        return "float32"

    def _bf16_autocast_allowed(self, torch: Any) -> bool:
        """Return whether the active device can execute bf16 through autocast.

        Model weights use the checkpoint/library's automatic dtype resolution.
        Reduced compute precision is enabled only through a verified autocast
        context; any missing or rejected runtime capability falls back to float32.

        :param Any torch: Imported torch module.
        :return bool: ``True`` when bf16 autocast is usable for the active device.
        """
        if self.device not in self.model_profile.autocast_devices:
            return False

        bf16_dtype = getattr(torch, "bfloat16", None)
        autocast = getattr(torch, "autocast", None)
        if bf16_dtype is None or not callable(autocast):
            logger.warning(
                "%s prefers bfloat16 autocast on %s, but this torch runtime does "
                "not expose the required APIs; falling back to float32.",
                self.model_name,
                self.device,
            )
            return False

        if self.device == "cpu":
            if not _cpu_native_bf16_supported(torch):
                return False
        elif self.device == "cuda":
            if not _cuda_native_bf16_supported(torch):
                return False
        elif self.device == "mps" and not self._mps_bf16_allowed(torch):
            return False

        try:
            with autocast(device_type=self.device, dtype=bf16_dtype):
                pass
        except Exception as exc:
            logger.warning(
                "%s prefers bfloat16 autocast on %s, but the runtime rejected "
                "that context (%s: %s); falling back to float32.",
                self.model_name,
                self.device,
                type(exc).__name__,
                exc,
            )
            return False
        return True

    def _mps_bf16_allowed(self, torch: Any) -> bool:
        """Return whether bf16 autocast is allowed on MPS for this torch build.

        :param Any torch: Imported ``torch`` module object.
        :return bool: ``True`` when torch meets the MPS bf16 policy floor.
        """
        torch_version = _parse_torch_major_minor(getattr(torch, "__version__", ""))
        if torch_version >= _MPS_MIN_TORCH_VERSION:
            return True
        logger.warning(
            "%s prefers bfloat16 on MPS, but torch %s predates the verified "
            "MPS floor %s; falling back to float32.",
            self.model_name,
            getattr(torch, "__version__", "unknown"),
            ".".join(str(part) for part in _MPS_MIN_TORCH_VERSION),
        )
        return False

    def _log_dimension_policy(self) -> None:
        """Log dimensionality and warn once about uncalibrated semantic gates.

        :return None: Reports the active profile's calibration limits.
        """
        if self._dim_logged:
            return
        self._dim_logged = True
        if (
            self.model_profile.name != "google/embeddinggemma"
            or self.truncate_dim != 512
        ):
            logger.warning(
                "Semantic edge threshold %.3f is uncalibrated for profile=%s, "
                "dimension=%s. The default %.3f was calibrated for EmbeddingGemma "
                "at 512 dimensions. Evaluate related/unrelated pairs and set "
                "--min-semantic-similarity or defaults.min_semantic_similarity.",
                self.min_semantic_similarity,
                self.model_profile.name,
                self.truncate_dim or "native",
                EMBEDDING_CONFIG.min_semantic_similarity,
            )

        available_dims = self.model_profile.available_truncate_dims
        if available_dims:
            selected_dim = (
                self.truncate_dim
                if self.truncate_dim is not None
                else available_dims[0]
            )
            available_text = ", ".join(f"{dim}d" for dim in available_dims)
            recommended_dim = self.model_profile.recommended_truncate_dim
            if recommended_dim is not None:
                logger.debug(
                    "%s embedding dimension: using %sd (recommended: %sd; available: %s).",
                    self.model_name,
                    selected_dim,
                    recommended_dim,
                    available_text,
                )
            else:
                logger.debug(
                    "%s embedding dimension: using %sd (available: %s).",
                    self.model_name,
                    selected_dim,
                    available_text,
                )
            return

        if self.truncate_dim is not None:
            logger.debug(
                "%s embedding dimension: using truncate_dim=%sd.",
                self.model_name,
                self.truncate_dim,
            )

    def _resolve_model_kwargs(self) -> Dict[str, Any]:
        """Compute SentenceTransformer kwargs and configure autocast policy.

        :return Dict[str, Any]: ``SentenceTransformer`` constructor kwargs.
        """
        self._autocast_dtype = None
        self._autocast_device_type = None
        self._autocast_enabled = False
        self._encode_model = None
        model_kwargs: Dict[str, Any] = {"dtype": "auto"}
        if self._attention_implementation_hint is not None:
            model_kwargs["attn_implementation"] = self._attention_implementation_hint

        try:
            torch = deps._import_torch()
        except ImportError:
            self._source_dtype_hint = "float32"
            return model_kwargs

        if self._source_dtype_hint != "bfloat16":
            return model_kwargs

        self._autocast_dtype = torch.bfloat16
        self._autocast_device_type = self.device
        self._autocast_enabled = True
        logger.debug(
            "%s will load weights with automatic dtype selection and run with "
            "bfloat16 autocast on %s.",
            self.model_name,
            self.device,
        )

        return model_kwargs

    def _validate_loaded_model_precision(
        self,
        model: Any,
        model_name_or_path: str,
    ) -> None:
        """Reject automatic checkpoint dtypes outside the verified compute policy.

        CiteMesh deliberately keeps Transformers automatic checkpoint loading so
        compatible weights are not needlessly coerced. Live parameters and buffers
        are therefore the authoritative dtype check: float16 is forbidden
        everywhere, and bfloat16 tensors are accepted only when this runtime has
        already selected the verified bfloat16 autocast path.

        :param Any model: Newly loaded SentenceTransformer-compatible model.
        :param str model_name_or_path: Candidate checkpoint that produced ``model``.
        :return None: The model's automatic dtype is compatible.
        :raises EmbeddingPrecisionCompatibilityError: If live tensors resolve to an
            unsupported precision.
        """
        tensor_dtypes = _model_floating_dtype_names(model)
        if not tensor_dtypes:
            raise EmbeddingPrecisionCompatibilityError(
                f"Could not inspect live parameter and buffer dtypes for embedding "
                f"checkpoint {model_name_or_path!r}; automatic precision cannot be verified."
            )
        if "float16" in tensor_dtypes:
            raise EmbeddingPrecisionCompatibilityError(
                f"Embedding checkpoint {model_name_or_path!r} resolved float16 tensors "
                "under automatic dtype loading. CiteMesh forbids float16; choose a "
                "current checkpoint whose saved weights are float32 or bfloat16."
            )
        unsupported_dtypes = tensor_dtypes - {"float32", "bfloat16"}
        if unsupported_dtypes:
            raise EmbeddingPrecisionCompatibilityError(
                f"Embedding checkpoint {model_name_or_path!r} resolved unsupported "
                "automatic tensor dtype(s): "
                f"{', '.join(sorted(unsupported_dtypes))}. CiteMesh supports only "
                "float32 weights or bfloat16 weights on a verified bfloat16 runtime."
            )
        if "bfloat16" in tensor_dtypes and self._source_dtype_hint != "bfloat16":
            raise EmbeddingPrecisionCompatibilityError(
                f"Embedding checkpoint {model_name_or_path!r} resolved bfloat16 tensors "
                f"on device={self.device}, but this runtime has not verified bfloat16 "
                "compute for the active model profile. Choose a float32 checkpoint "
                "instead of mixing bfloat16 execution into a float32 cache namespace."
            )
        logger.debug(
            "%s automatic parameter/buffer dtype(s): %s.",
            model_name_or_path,
            ", ".join(sorted(tensor_dtypes)),
        )

    def _validate_loaded_model_contract(
        self,
        model: Any,
        model_name_or_path: str,
    ) -> None:
        """Require live transformer settings declared by the active model profile.

        :param Any model: Newly loaded SentenceTransformer-compatible model.
        :param str model_name_or_path: Candidate checkpoint that produced ``model``.
        :return None: The runtime-active transformer satisfies its profile.
        :raises EmbeddingBackendCompatibilityError: If a required live setting
            is absent, inaccessible, or disabled.
        """
        if not self.model_profile.requires_bidirectional_attention:
            return

        try:
            active_config = model[0].auto_model.config
            bidirectional_attention = active_config.use_bidirectional_attention
        except Exception as exc:
            raise EmbeddingBackendCompatibilityError(
                f"Embedding checkpoint {model_name_or_path!r} requires bidirectional "
                "attention, but the runtime-active setting at "
                "model[0].auto_model.config.use_bidirectional_attention could not "
                "be verified."
            ) from exc

        if bidirectional_attention is not True:
            raise EmbeddingBackendCompatibilityError(
                f"Embedding checkpoint {model_name_or_path!r} requires bidirectional "
                "attention, but the runtime-active transformer reports "
                f"use_bidirectional_attention={bidirectional_attention!r}."
            )

    def _autocast_context(self) -> Any:
        """Return autocast context for model encoding.

        :return Any: Active autocast context manager or no-op context.
        """
        if (
            not self._autocast_enabled
            or self._autocast_dtype is None
            or self._autocast_device_type is None
        ):
            return nullcontext()

        try:
            torch = deps._import_torch()
        except ImportError:
            return nullcontext()

        return torch.autocast(
            device_type=self._autocast_device_type,
            dtype=self._autocast_dtype,
        )

    @contextmanager
    def _tf32_context(self) -> Iterator[None]:
        """Apply CUDA-only TF32 controls for one encode call and restore them.

        :return Iterator[None]: Context that scopes process-global torch settings.
        """
        if self._tf32_mode not in {"tf32", "tf32-matmul-high"}:
            yield
            return

        try:
            torch = deps._import_torch()
        except ImportError:
            yield
            return

        if self._tf32_mode == "tf32-matmul-high":
            get_precision = getattr(torch, "get_float32_matmul_precision", None)
            set_precision = getattr(torch, "set_float32_matmul_precision", None)
            if not callable(get_precision) or not callable(set_precision):
                yield
                return
            original = get_precision()
            try:
                set_precision("high")
            except Exception:
                logger.warning("Could not enable scoped CUDA TF32 matmul precision.")
                yield
                return
            try:
                yield
            finally:
                try:
                    set_precision(original)
                except Exception:
                    logger.warning(
                        "Could not restore the prior torch float32 matmul precision."
                    )
            return

        backends = getattr(torch, "backends", None)
        cuda_backend = getattr(backends, "cuda", None)
        cudnn_backend = getattr(backends, "cudnn", None)
        targets = (
            getattr(cuda_backend, "matmul", None),
            getattr(cudnn_backend, "conv", None),
        )
        if any(
            target is None or not hasattr(target, "fp32_precision")
            for target in targets
        ):
            yield
            return

        originals = [(target, target.fp32_precision) for target in targets]
        changed: list[tuple[Any, Any]] = []
        try:
            for target, original in originals:
                target.fp32_precision = "tf32"
                changed.append((target, original))
        except Exception:
            for target, original in reversed(changed):
                try:
                    target.fp32_precision = original
                except Exception:
                    pass
            logger.warning("Could not enable scoped CUDA TF32 backend precision.")
            yield
            return

        try:
            yield
        finally:
            for target, original in reversed(originals):
                try:
                    target.fp32_precision = original
                except Exception:
                    logger.warning(
                        "Could not restore a prior CUDA TF32 backend setting."
                    )

    @contextmanager
    def _precision_context(self) -> Iterator[None]:
        """Combine scoped TF32 and autocast controls around model encoding.

        :return Iterator[None]: Active encode-time precision context.
        """
        with ExitStack() as stack:
            stack.enter_context(self._tf32_context())
            stack.enter_context(self._autocast_context())
            if (
                self.device == "cuda"
                and self._autocast_enabled
                and self._autocast_device_type == "cuda"
                and self._attention_implementation_hint == "flash_attention_2"
            ):

                def keep_relevant_warning(record: logging.LogRecord) -> bool:
                    """Reject the expected FP32-to-bf16 attention warning.

                    :param logging.LogRecord record: Candidate Transformers log record.
                    :return bool: Whether the log record should be emitted.
                    """
                    return record.getMessage() != _FA2_FORWARD_DTYPE_WARNING

                flash_logger = logging.getLogger(
                    "transformers.modeling_flash_attention_utils"
                )
                flash_logger.addFilter(keep_relevant_warning)
                stack.callback(flash_logger.removeFilter, keep_relevant_warning)
            if self.device == "cpu" and self._inner_model_compiled:
                # Dynamo must see freezing while capturing weights; backend-only
                # compile options arrive too late. Keep eager fallback weights.
                stack.enter_context(
                    deps._import_torch()._inductor.config.patch(
                        freezing=True, freezing_discard_parameters=False
                    )
                )
            if (
                self._inner_model_compiled
                and self._attention_implementation_hint == "flash_attention_2"
            ):
                from torch._dynamo import config as dynamo_config
                from transformers.modeling_flash_attention_utils import (
                    logger as flash_logger,
                )

                # Transformers logs an autocast conversion inside attention.
                # Tracing that log breaks the decoder loop into eager fragments.
                logging_option = (
                    "ignore_logging_functions"
                    if hasattr(dynamo_config, "ignore_logging_functions")
                    else "reorderable_logging_functions"
                )
                stack.enter_context(
                    dynamo_config.patch(
                        **{
                            logging_option: getattr(dynamo_config, logging_option)
                            | {flash_logger.warning_once}
                        }
                    )
                )
            yield

    def _get_model_for_encoding(self) -> Any:
        """Return model object used for embedding encode calls.

        :return Any: Base model or encode-time precision proxy.
        """
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded.")

        if (
            not self._inner_model_compiled
            and not self._autocast_enabled
            and self._tf32_mode not in {"tf32", "tf32-matmul-high"}
            and self.device != "cuda"
        ):
            return self.model

        if self._encode_model is None:
            self._encode_model = _PrecisionEncodeProxy(
                self.model,
                self._precision_context,
                self._restore_eager_model_after_compile_failure,
                prefetch_batches=self.device == "cuda",
            )

        return self._encode_model

    def _encode_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode text inputs and return normalized float32 embeddings.

        :param List[str] texts: Text payload(s) to encode.
        :param Optional[int] batch_size: Optional batch size override.
        :param bool show_progress_bar: Whether to display encoding progress.
        :return np.ndarray: Embeddings with shape ``(len(texts), dim)``.
        """
        encode_model = self._get_model_for_encoding()
        effective_batch_size = len(texts) if batch_size is None else int(batch_size)
        if effective_batch_size < 1:
            raise ValueError("batch_size must be at least 1 when provided")

        return encode_texts(
            encode_model,
            texts,
            batch_size=min(effective_batch_size, len(texts)),
            show_progress_bar=show_progress_bar,
        )

    def _model_load_candidates(self) -> Tuple[str, ...]:
        """Return ordered candidate model IDs used for lazy model loading.

        Fallbacks are intentionally scoped to default-revision checkpoints so
        explicit revision pins remain deterministic.

        :return Tuple[str, ...]: Ordered model IDs to try.
        """
        requested = str(self.model_name).strip()
        candidates: List[str] = [requested]
        if self.model_revision is not None:
            return tuple(candidates)
        for fallback_model in DEFAULT_EMBEDDING_MODEL_FALLBACKS.get(requested, ()):
            if fallback_model not in candidates:
                candidates.append(fallback_model)
        return tuple(candidates)

    def _cache_hydrated_for_active_spec(self) -> bool:
        """Return whether cache is hydrated for active split/corpus selection.

        :return bool: ``True`` when active cache namespace has a matching hydrated payload.
        """
        return self.embedding_cache.is_hydrated(
            self.dataset_split,
            self.corpus_size,
            dataset_source=self.dataset_source,
        )

    def _should_defer_compile_for_cache_hydration(self) -> bool:
        """Return whether compile should be deferred until cache is hydrated.

        :return bool: ``True`` when runtime should skip compile for current cold-cache run.
        """
        if not self.enable_torch_compile:
            return False
        if not self.model_profile.compile_inner_transformer:
            return False
        if self.device in {"cuda", "cpu"}:
            return False
        if self.semantic_source != "arxiv-corpus":
            return False
        try:
            return not self._cache_hydrated_for_active_spec()
        except Exception:
            # Conservative fallback: avoid compile when cache state cannot be
            # validated before hydration.
            return True

    def _load_model(self) -> None:
        """Lazy load sentence transformer model.

        :return None: Model is initialized in-place on first access.
        """
        if self.model is None:
            sentence_transformer_cls = deps._import_sentence_transformer_class()

            logger.info(f"Loading embedding model: {self.model_name}")
            load_candidates = self._model_load_candidates()
            model_errors: List[Tuple[str, Exception]] = []
            for idx, candidate_model in enumerate(load_candidates):
                try:
                    self._bind_model_contract(candidate_model)
                    _require_transformers_compatibility(self.model_profile)
                    model_kwargs = self._resolve_model_kwargs()
                    st_kwargs: Dict[str, Any] = {"device": self.device}
                    if model_kwargs:
                        st_kwargs["model_kwargs"] = model_kwargs
                    if self.truncate_dim is not None:
                        st_kwargs["truncate_dim"] = self.truncate_dim
                    if self.model_revision is not None:
                        st_kwargs["revision"] = self.model_revision
                    try:
                        with _suppress_transformers_progress_for_non_tty():
                            with _suppress_expected_fa2_load_dtype_warning(
                                enabled=(
                                    self._attention_implementation_hint
                                    == "flash_attention_2"
                                    and self._autocast_enabled
                                    and self._autocast_device_type == "cuda"
                                )
                            ):
                                loaded_model = sentence_transformer_cls(
                                    candidate_model, **st_kwargs
                                )
                    except (ImportError, ValueError) as exc:
                        fa2_error_markers = (
                            "flashattention2",
                            "flash attention 2",
                            "flash_attn",
                            "flash_attention_2",
                        )
                        if (
                            self._attention_implementation_hint != "flash_attention_2"
                            or not any(
                                marker in str(exc).casefold()
                                for marker in fa2_error_markers
                            )
                        ):
                            raise
                        logger.warning(
                            "FlashAttention 2 could not load for %s (%s); retrying with SDPA.",
                            candidate_model,
                            exc,
                        )
                        self._attention_implementation_hint = "sdpa"
                        model_kwargs["attn_implementation"] = "sdpa"
                        with _suppress_transformers_progress_for_non_tty():
                            loaded_model = sentence_transformer_cls(
                                candidate_model, **st_kwargs
                            )
                    self._validate_loaded_model_precision(loaded_model, candidate_model)
                    self._validate_loaded_model_contract(loaded_model, candidate_model)
                    self.model = loaded_model
                except (
                    EmbeddingBackendCompatibilityError,
                    EmbeddingPrecisionCompatibilityError,
                ):
                    raise
                except Exception as exc:
                    model_errors.append((candidate_model, exc))
                    has_more_candidates = idx + 1 < len(load_candidates)
                    if has_more_candidates:
                        logger.warning(
                            "Failed to load embedding model %s (%s: %s). "
                            "Trying fallback checkpoint...",
                            candidate_model,
                            type(exc).__name__,
                            exc,
                        )
                        continue
                    summary = "; ".join(
                        f"{model_id}: {type(error).__name__}: {error}"
                        for model_id, error in model_errors
                    )
                    raise RuntimeError(
                        "Could not load embedding model from candidate chain "
                        f"{load_candidates}: {summary}"
                    ) from exc

                if candidate_model != self.model_name:
                    logger.info(
                        "Using fallback embedding checkpoint: requested=%s active=%s.",
                        self.model_name,
                        candidate_model,
                    )
                if self._active_model_name != candidate_model:
                    self._resolved_model_fingerprint = None
                self._active_model_name = candidate_model
                break

            self._bind_embedding_cache_to_active_model()
            self._configure_tf32_runtime()
            if (
                self.enable_torch_compile
                and self.semantic_source == "arxiv-corpus"
                and self.device == "mps"
            ):
                # Warm-cache compile policy needs the exact artifact namespace.
                self._ensure_cache_model_fingerprint()
            if self._should_defer_compile_for_cache_hydration():
                self._compile_status_reason = (
                    "deferred while hydrating cache; compile resumes on warm-cache runs"
                )
                logger.debug(
                    "Deferring torch.compile for %s until cache hydration completes.",
                    self.model_name,
                )
            else:
                self._maybe_compile_inner_transformer()

            self._log_dimension_policy()
            if self.model_profile.notes and not self._profile_logged:
                logger.debug(self.model_profile.notes)
                self._profile_logged = True
            self._log_runtime_summary()

    def _configure_tf32_runtime(self) -> None:
        """Best-effort TF32 enablement for Ampere+ CUDA devices.

        :return None: Updates runtime backend configuration in-place when available.
        """
        if self._tf32_runtime_configured:
            return
        self._tf32_runtime_configured = True
        self._tf32_mode = "off"

        if self.device != "cuda":
            logger.debug(
                "Skipping TF32 config for %s: device=%s is not CUDA.",
                self.model_name,
                self.device,
            )
            return

        try:
            torch = deps._import_torch()
        except ImportError:
            logger.debug(
                "Skipping TF32 config for %s: torch unavailable.", self.model_name
            )
            return

        cuda_module = getattr(torch, "cuda", None)
        get_capability = getattr(cuda_module, "get_device_capability", None)
        capability = get_capability(0) if callable(get_capability) else None
        if not isinstance(capability, tuple) or len(capability) < 2:
            self._tf32_mode = "unknown"
            logger.debug(
                "Skipping TF32 config for %s: CUDA capability unavailable.",
                self.model_name,
            )
            return
        major, minor = int(capability[0]), int(capability[1])
        if major < 8:
            logger.debug(
                "Skipping TF32 config for %s: GPU capability %s.%s is pre-Ampere.",
                self.model_name,
                major,
                minor,
            )
            return

        torch_version = _parse_torch_major_minor(getattr(torch, "__version__", ""))
        compile_fn = getattr(torch, "compile", None)
        should_use_compile_bridge = (
            self.enable_torch_compile
            and self.model_profile.compile_inner_transformer
            and callable(compile_fn)
            and torch_version in _TF32_COMPILE_BRIDGE_TORCH_VERSIONS
        )
        if should_use_compile_bridge:
            get_matmul_precision = getattr(torch, "get_float32_matmul_precision", None)
            set_matmul_precision = getattr(torch, "set_float32_matmul_precision", None)
            if not callable(get_matmul_precision) or not callable(set_matmul_precision):
                self._tf32_mode = "unsupported"
                logger.debug(
                    "Skipping TF32 config for %s: restorable compile-safe matmul "
                    "precision APIs unavailable.",
                    self.model_name,
                )
                return
            self._tf32_mode = "tf32-matmul-high"
            logger.debug(
                "Will scope compile-safe TF32 matmul precision to encode calls for "
                "%s on torch %s.",
                self.model_name,
                getattr(torch, "__version__", "unknown"),
            )
            return

        backends = getattr(torch, "backends", None)
        cuda_backend = getattr(backends, "cuda", None) if backends is not None else None
        cudnn_backend = (
            getattr(backends, "cudnn", None) if backends is not None else None
        )
        matmul_backend = getattr(cuda_backend, "matmul", None)
        conv_backend = getattr(cudnn_backend, "conv", None)
        if any(
            owner is None or not hasattr(owner, "fp32_precision")
            for owner in (matmul_backend, conv_backend)
        ):
            self._tf32_mode = "unsupported"
            logger.debug(
                "Skipping TF32 config for %s: CUDA-scoped fp32_precision APIs "
                "unavailable.",
                self.model_name,
            )
            return

        self._tf32_mode = "tf32"
        logger.debug("Will scope TF32 to CUDA matmul/conv encode calls (Ampere+ GPU).")

    def _log_runtime_summary(self) -> None:
        """Emit concise one-time runtime summary at info level."""
        if self._runtime_summary_logged:
            return

        selected_dim = self.truncate_dim
        if selected_dim is None and self.model_profile.available_truncate_dims:
            selected_dim = self.model_profile.available_truncate_dims[0]
        dim_label = "full" if selected_dim is None else f"{selected_dim}d"
        compute_dtype_label = self._source_dtype_hint
        if self._autocast_enabled:
            compute_dtype_label = f"{compute_dtype_label}+autocast"
        attention_label = self._attention_implementation_hint or "auto"
        logger.info(
            "%s runtime: device=%s, dim=%s, compute=%s, attn=%s, output=float32, cache=%s, compile=%s, tf32=%s.",
            self.model_name,
            self.device,
            dim_label,
            compute_dtype_label,
            attention_label,
            self.storage_precision,
            "on" if self._inner_model_compiled else "off",
            self._tf32_mode,
        )
        if not self._inner_model_compiled and self._compile_status_reason:
            logger.debug(
                "%s compile status: %s", self.model_name, self._compile_status_reason
            )
        self._runtime_summary_logged = True

    def _maybe_compile_inner_transformer(self) -> None:
        """Best-effort compile of the wrapped HF model for selected profiles.

        SentenceTransformer itself is not compiled due wrapper incompatibilities.
        For supported profiles (currently EmbeddingGemma), we compile only
        the active inner transformer and keep the outer SentenceTransformer intact.

        :return None: Mutates ``self.model`` in place when compilation succeeds.
        """
        if self.model is None or self._inner_model_compiled:
            return

        if not self.enable_torch_compile:
            self._compile_status_reason = "disabled by configuration"
            logger.debug(
                "Skipping torch.compile for %s: disabled by configuration.",
                self.model_name,
            )
            return

        if self.device not in _COMPILE_ELIGIBLE_DEVICES:
            self._compile_status_reason = f"compile disabled for device={self.device}"
            logger.debug(
                "Skipping torch.compile for %s: device=%s is not compile-eligible.",
                self.model_name,
                self.device,
            )
            return

        if not self.model_profile.compile_inner_transformer:
            self._compile_status_reason = "profile does not support inner-model compile"
            return

        try:
            torch = deps._import_torch()
        except ImportError:
            self._compile_status_reason = "torch unavailable"
            logger.debug(
                "%s profile supports inner-model torch.compile, but torch is unavailable.",
                self.model_name,
            )
            return

        compile_fn = getattr(torch, "compile", None)
        if not callable(compile_fn):
            self._compile_status_reason = "torch.compile unavailable"
            logger.debug(
                "%s profile supports inner-model torch.compile, but torch.compile is unavailable.",
                self.model_name,
            )
            return

        try:
            transformer_block = self.model[0]
        except Exception as exc:  # pragma: no cover - defensive for upstream API drift
            self._compile_status_reason = f"model[0] unavailable ({type(exc).__name__})"
            logger.warning(
                "Skipping torch.compile for %s: could not access model[0] (%s).",
                self.model_name,
                exc,
            )
            return

        # SentenceTransformers 6 forwards through .model; assigning its legacy
        # .auto_model alias can register an unused module instead.
        model_attribute = (
            "model" if hasattr(transformer_block, "model") else "auto_model"
        )
        auto_model = getattr(transformer_block, model_attribute, None)
        if auto_model is None:
            self._compile_status_reason = "inner auto_model unavailable"
            logger.warning(
                "Skipping torch.compile for %s: model[0].auto_model is unavailable.",
                self.model_name,
            )
            return

        if auto_model.__class__.__name__ == "OptimizedModule":
            self._inner_model_compiled = True
            self._compile_status_reason = None
            return

        try:
            compile_kwargs: Dict[str, Any] = (
                {"dynamic": True} if self.device in {"cuda", "cpu"} else {}
            )
            if self.device == "cpu":
                compile_kwargs["options"] = {"max_autotune": True}
            compiled_model = compile_fn(auto_model, **compile_kwargs)
            setattr(transformer_block, model_attribute, compiled_model)
        except Exception as exc:
            try:
                setattr(transformer_block, model_attribute, auto_model)
            except Exception:
                pass
            self._compile_status_reason = f"compile failed ({type(exc).__name__})"
            logger.warning(
                "torch.compile failed for %s inner transformer; continuing without compile: %s",
                self.model_name,
                exc,
            )
            return

        self._eager_inner_transformer = auto_model
        self._inner_model_compiled = True
        self._compile_status_reason = None
        if self.device == "mps":
            logger.info(
                "torch.compile on MPS (Inductor/Metal) is experimental; the first "
                "encode will retry from the eager inner model if compilation fails."
            )
        logger.debug(
            "Enabled torch.compile for %s inner transformer (model[0].%s).",
            self.model_name,
            model_attribute,
        )

    def _restore_eager_model_after_compile_failure(self, error: Exception) -> bool:
        """Restore the original inner model after a lazy compiled-call failure.

        ``torch.compile`` normally defers backend compilation until the first
        invocation, so wrapping the module successfully is not proof that it can
        execute. Retrying the failed encode batch is safe: cache writes happen
        only after all batches finish encoding successfully.

        :param Exception error: Failure raised while the compiled model executed.
        :return bool: Whether an eager model was restored for one retry.
        """
        if (
            not self._inner_model_compiled
            or self._eager_inner_transformer is None
            or self.model is None
        ):
            return False
        try:
            transformer_block = self.model[0]
            model_attribute = (
                "model" if hasattr(transformer_block, "model") else "auto_model"
            )
            setattr(transformer_block, model_attribute, self._eager_inner_transformer)
        except Exception:
            logger.warning(
                "Compiled embedding execution failed and the eager inner model "
                "could not be restored.",
                exc_info=True,
            )
            return False

        self._inner_model_compiled = False
        self._eager_inner_transformer = None
        self._encode_model = None
        self._compile_status_reason = (
            f"compiled execution failed ({type(error).__name__}); restored eager model"
        )
        logger.warning(
            "Compiled embedding execution failed for %s (%s: %s); retrying the "
            "affected encode batch with the eager inner model.",
            self.model_name,
            type(error).__name__,
            error,
        )
        return True
