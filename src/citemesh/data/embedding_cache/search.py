"""Similarity scoring kernel mixed into :class:`EmbeddingCache`.

Owns the read-side of a query: calibration-range loading, cached-vector
hydration, Hamming prefiltering over the packed binary index, chunked int8 and
float32 scoring with a bounded global top-k, finite-score enforcement, and the
SQLite metadata join that turns winning row indices back into paper payloads.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from typing import Any

import h5py
import numpy as np

from citemesh.text_batching import l2_normalize_embeddings

from . import constants
from .constants import CALIBRATION_RANGES_DATASET_NAME
from .models import _EmbeddingCacheLayoutError
from .quantization import (
    _POPCOUNT_LUT,
    _dequantize_int8,
    _quantize_ubinary_embeddings,
    _sanitize_ranges,
)
from .sql import _decode_paper_row


class _SearchMixin:
    """Query-time scoring and metadata resolution for :class:`EmbeddingCache`."""

    def _require_calibration_ranges(
        self,
        h5_file: h5py.File,
        embedding_dim: int | None = None,
    ) -> np.ndarray:
        """Load persisted int8 calibration ranges or fail closed.

        Calibration is intentionally explicit at the strategy layer. Bootstrapping
        ranges from whichever request batch happens to arrive first makes the
        namespace path-dependent and can silently skew later quantization quality.

        :param h5py.File h5_file: Open HDF5 cache handle.
        :param Optional[int] embedding_dim: Expected embedding dimension, when known.
        :return np.ndarray: Calibration ranges with shape ``(2, dim)``.
        """
        if self.storage_precision != "int8":
            raise RuntimeError("Calibration ranges are only valid for int8 storage")

        existing = h5_file.get(CALIBRATION_RANGES_DATASET_NAME)
        if existing is None:
            raise RuntimeError(
                "Missing persisted int8 calibration ranges. "
                "Hydrate through EmbeddingGraphBuilder or call "
                "EmbeddingCache.set_calibration_ranges(...) before int8 writes."
            )

        if existing.ndim != 2 or existing.shape[0] != 2:
            raise _EmbeddingCacheLayoutError(
                f"Calibration ranges dataset must have shape (2, dim), got {existing.shape}."
            )
        if embedding_dim is not None and int(existing.shape[1]) != int(embedding_dim):
            raise _EmbeddingCacheLayoutError(
                "Calibration range dimension mismatch in cache: "
                f"{int(existing.shape[1])} != {int(embedding_dim)}"
            )

        ranges = np.asarray(existing, dtype=np.float32)
        try:
            return _sanitize_ranges(ranges)
        except ValueError as exc:
            raise _EmbeddingCacheLayoutError(str(exc)) from exc

    def _load_cached_embeddings(
        self,
        h5_file: h5py.File,
        dataset: h5py.Dataset,
        cached_rows: Sequence[tuple[str, int]],
    ) -> dict[str, np.ndarray]:
        """Load cached embeddings from matrix dataset in row-index order.

        :param h5py.File h5_file: Open HDF5 cache handle.
        :param h5py.Dataset dataset: Matrix dataset containing all embeddings.
        :param Sequence[Tuple[str, int]] cached_rows: Pairs of paper ID and row index.
        :return Dict[str, np.ndarray]: Mapping of paper IDs to float32 embeddings.
        """
        if not cached_rows:
            return {}

        sorted_rows = sorted(cached_rows, key=lambda item: item[1])
        indices = np.asarray([row_idx for _, row_idx in sorted_rows], dtype=np.int64)
        matrix = np.asarray(dataset[indices])

        if self.storage_precision == "int8":
            matrix_f32 = self._dequantize_int8(
                h5_file,
                matrix.astype(np.int8, copy=False),
            )
            matrix_f32 = l2_normalize_embeddings(matrix_f32)
        else:
            matrix_f32 = np.asarray(matrix, dtype=np.float32)

        return {
            paper_id: np.asarray(matrix_f32[idx], dtype=np.float32)
            for idx, (paper_id, _) in enumerate(sorted_rows)
        }

    def _binary_prefilter_rows(
        self,
        binary_dataset: h5py.Dataset,
        query_embedding: np.ndarray,
        candidate_count: int,
    ) -> np.ndarray:
        """Return top candidate row indices via Hamming prefiltering.

        :param h5py.Dataset binary_dataset: Packed binary corpus embeddings.
        :param np.ndarray query_embedding: Float32 query embedding.
        :param int candidate_count: Number of candidate rows to keep.
        :return np.ndarray: Candidate row indices.
        """
        row_count = int(binary_dataset.shape[0])
        if row_count == 0:
            return np.asarray([], dtype=np.int64)

        query_binary = _quantize_ubinary_embeddings(query_embedding)[0]

        keep_k = min(max(int(candidate_count), 0), row_count)
        if keep_k == 0:
            return np.asarray([], dtype=np.int64)
        all_rows: list[np.ndarray] = []
        all_dists: list[np.ndarray] = []
        chunk_size = constants.EMBEDDING_SEARCH_CHUNK_ROWS

        for start in range(0, row_count, chunk_size):
            end = min(start + chunk_size, row_count)
            chunk = np.asarray(binary_dataset[start:end], dtype=np.uint8)
            xor = np.bitwise_xor(chunk, query_binary[None, :])
            dists = _POPCOUNT_LUT[xor].sum(axis=1, dtype=np.int32)

            local_k = min(keep_k, dists.shape[0])
            if local_k == dists.shape[0]:
                local_idx = np.arange(dists.shape[0], dtype=np.int64)
            else:
                distance_cutoff = np.partition(dists, local_k - 1)[local_k - 1]
                closer_idx = np.flatnonzero(dists < distance_cutoff)
                remaining = local_k - int(closer_idx.size)
                tied_idx = np.flatnonzero(dists == distance_cutoff)
                local_idx = np.concatenate(
                    (closer_idx, tied_idx[:remaining]),
                    axis=0,
                )

            all_rows.append((start + local_idx).astype(np.int64, copy=False))
            all_dists.append(dists[local_idx])

        candidate_rows = np.concatenate(all_rows, axis=0)
        candidate_dists = np.concatenate(all_dists, axis=0)

        if candidate_rows.shape[0] > keep_k:
            # Choose a deterministic top-k candidate set by (distance, row_idx).
            ranked_idx = np.lexsort((candidate_rows, candidate_dists))
            candidate_rows = candidate_rows[ranked_idx[:keep_k]]

        # HDF5 fancy indexing requires monotonically increasing integer indices.
        return np.sort(candidate_rows.astype(np.int64, copy=False))

    @staticmethod
    def _matrix_chunk_loader(
        embeddings_dataset: h5py.Dataset,
        dequantize: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> Callable[[int, int], np.ndarray]:
        """Build the chunk loader :meth:`_score_chunked_rows` reads rows through.

        Float32 namespaces read their stored rows directly; int8 namespaces read
        the raw codes and hand them to ``dequantize`` before scoring.

        :param h5py.Dataset embeddings_dataset: Matrix dataset whose rows are scored.
        :param Optional[Callable[[np.ndarray], np.ndarray]] dequantize: Storage-to-float32
            conversion applied to each raw chunk; ``None`` reads float32 rows directly.
        :return Callable[[int, int], np.ndarray]: Loader yielding float32 matrix chunks.
        """
        storage_dtype = np.float32 if dequantize is None else np.int8

        def load_chunk(start: int, end: int) -> np.ndarray:
            """Load one cache matrix chunk as float32.

            :param int start: Inclusive row offset.
            :param int end: Exclusive row offset.
            :return np.ndarray: Float32 matrix chunk.
            """
            chunk = np.asarray(embeddings_dataset[start:end], dtype=storage_dtype)
            return chunk if dequantize is None else dequantize(chunk)

        return load_chunk

    def _score_int8_rows(
        self,
        embeddings_dataset: h5py.Dataset,
        h5_file: h5py.File,
        query_embedding: np.ndarray,
        top_k: int,
        row_indices: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score int8 embeddings against a float32 query.

        :param h5py.Dataset embeddings_dataset: Int8 matrix dataset.
        :param h5py.File h5_file: Open HDF5 file handle.
        :param np.ndarray query_embedding: Float32 query embedding.
        :param int top_k: Top results to keep.
        :param Optional[np.ndarray] row_indices: Optional candidate subset.
        :return Tuple[np.ndarray, np.ndarray, np.ndarray]: Rows, scores, and embeddings.
        """
        query = query_embedding
        if row_indices is not None:
            rows = np.unique(np.asarray(row_indices, dtype=np.int64))
            if rows.size == 0:
                return (
                    np.asarray([], dtype=np.int64),
                    np.asarray([], dtype=np.float32),
                    np.empty((0, int(query_embedding.shape[0])), dtype=np.float32),
                )

            int8_matrix = np.asarray(embeddings_dataset[rows], dtype=np.int8)
            matrix = l2_normalize_embeddings(
                self._dequantize_int8(h5_file, int8_matrix)
            )
            scores = matrix @ query
            self._require_finite_scores(scores)
            return self._select_top_k(rows, scores, matrix, top_k)

        def dequantize_chunk(chunk: np.ndarray) -> np.ndarray:
            """Dequantize and normalize one stored int8 chunk.

            :param np.ndarray chunk: Raw int8 rows read from the matrix dataset.
            :return np.ndarray: Dequantized, normalized float32 matrix.
            """
            return l2_normalize_embeddings(self._dequantize_int8(h5_file, chunk))

        return self._score_chunked_rows(
            embeddings_dataset,
            query,
            top_k,
            self._matrix_chunk_loader(embeddings_dataset, dequantize_chunk),
        )

    def _score_float_rows(
        self,
        embeddings_dataset: h5py.Dataset,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score float32 embeddings against a float32 query.

        :param h5py.Dataset embeddings_dataset: Float matrix dataset.
        :param np.ndarray query_embedding: Float32 query embedding.
        :param int top_k: Top results to keep.
        :return Tuple[np.ndarray, np.ndarray, np.ndarray]: Rows, scores, and embeddings.
        """

        return self._score_chunked_rows(
            embeddings_dataset,
            query_embedding,
            top_k,
            self._matrix_chunk_loader(embeddings_dataset),
        )

    def _score_chunked_rows(
        self,
        embeddings_dataset: h5py.Dataset,
        query_embedding: np.ndarray,
        top_k: int,
        load_chunk: Callable[[int, int], np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score a matrix through a loader while retaining a bounded global top-k.

        :param h5py.Dataset embeddings_dataset: Matrix dataset whose rows are scored.
        :param np.ndarray query_embedding: Float32 query vector.
        :param int top_k: Number of results to retain.
        :param Callable[[int, int], np.ndarray] load_chunk: Matrix chunk loader.
        :return Tuple[np.ndarray, np.ndarray, np.ndarray]: Rows, scores, and vectors.
        """
        row_count = int(embeddings_dataset.shape[0])
        best_rows = np.asarray([], dtype=np.int64)
        best_scores = np.asarray([], dtype=np.float32)
        best_embeddings = np.empty((0, int(query_embedding.shape[0])), dtype=np.float32)

        for start in range(0, row_count, constants.EMBEDDING_SEARCH_CHUNK_ROWS):
            end = min(start + constants.EMBEDDING_SEARCH_CHUNK_ROWS, row_count)
            chunk_matrix = load_chunk(start, end)
            chunk_scores = chunk_matrix @ query_embedding
            self._require_finite_scores(chunk_scores)
            chunk_rows = np.arange(start, end, dtype=np.int64)
            rows, scores, embeddings = self._select_top_k(
                chunk_rows,
                chunk_scores,
                chunk_matrix,
                top_k,
            )
            if rows.size == 0:
                continue

            best_rows, best_scores, best_embeddings = self._select_top_k(
                np.concatenate((best_rows, rows), axis=0),
                np.concatenate((best_scores, scores), axis=0),
                np.concatenate((best_embeddings, embeddings), axis=0),
                top_k,
            )

        return best_rows, best_scores, best_embeddings

    @staticmethod
    def _require_finite_scores(scores: np.ndarray) -> None:
        """Reject non-finite scores from corrupt cached embedding data.

        Query vectors are validated at the public boundary, so non-finite scores
        here indicate a malformed persisted embedding or a numerical failure.

        :param np.ndarray scores: Similarity scores produced from cached vectors.
        :return None: Raises when a score is not finite.
        :raises RuntimeError: If cached vectors produce a non-finite score.
        """
        if not np.all(np.isfinite(scores)):
            raise RuntimeError(
                "Embedding cache integrity error: non-finite scores encountered "
                "while scoring cached embeddings. Rebuild this cache namespace "
                "to restore valid vectors."
            )

    @staticmethod
    def _select_top_k(
        rows: np.ndarray,
        scores: np.ndarray,
        embeddings: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select top-k rows from scored embedding arrays.

        :param np.ndarray rows: Row-index vector.
        :param np.ndarray scores: Score vector.
        :param np.ndarray embeddings: Embedding matrix.
        :param int top_k: Number of rows to keep.
        :return Tuple[np.ndarray, np.ndarray, np.ndarray]: Top rows, scores, embeddings.
        """
        if rows.size == 0:
            return (
                np.asarray([], dtype=np.int64),
                np.asarray([], dtype=np.float32),
                np.empty((0, embeddings.shape[1]), dtype=np.float32),
            )

        keep_k = min(int(top_k), int(rows.size))
        if keep_k == rows.size:
            selected_idx = np.arange(rows.size, dtype=np.int64)
        else:
            score_cutoff = np.partition(scores, -keep_k)[-keep_k]
            higher_score_idx = np.flatnonzero(scores > score_cutoff)
            remaining = keep_k - int(higher_score_idx.size)
            tied_idx = np.flatnonzero(scores == score_cutoff)
            tied_order = np.argsort(rows[tied_idx], kind="stable")
            selected_idx = np.concatenate(
                (higher_score_idx, tied_idx[tied_order[:remaining]]),
                axis=0,
            )

        selected_rows = rows[selected_idx]
        selected_scores = scores[selected_idx]
        selected_embeddings = embeddings[selected_idx]

        order = np.lexsort((selected_rows, -selected_scores))
        return (
            selected_rows[order].astype(np.int64, copy=False),
            selected_scores[order].astype(np.float32, copy=False),
            np.asarray(selected_embeddings[order], dtype=np.float32),
        )

    def _load_metadata_by_rows(
        self,
        conn: sqlite3.Connection,
        row_indices: Sequence[int],
    ) -> dict[int, dict[str, Any]]:
        """Load metadata rows keyed by embedding matrix row index.

        :param sqlite3.Connection conn: Open SQLite connection.
        :param Sequence[int] row_indices: Matrix row indices.
        :return Dict[int, Dict[str, Any]]: Metadata payloads keyed by row index.
        """
        output: dict[int, dict[str, Any]] = {}
        for row in self._query_paper_rows(
            conn,
            [int(idx) for idx in row_indices],
            lookup_column="row_idx",
        ):
            decoded = _decode_paper_row(row, parse_json_lists=True)
            row_idx = decoded.pop("row_idx")
            decoded.pop("text_hash")
            assert row_idx is not None
            output[row_idx] = decoded

        return output

    def _dequantize_int8(
        self, h5_file: h5py.File, int8_embeddings: np.ndarray
    ) -> np.ndarray:
        """Dequantize int8 embeddings to float32 using persisted ranges.

        :param h5py.File h5_file: Open HDF5 file handle.
        :param np.ndarray int8_embeddings: Int8 embeddings.
        :return np.ndarray: Dequantized float32 embeddings.
        """
        return _dequantize_int8(h5_file, int8_embeddings)
