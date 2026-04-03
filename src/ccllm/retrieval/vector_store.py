from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.ccllm.retrieval.embedder import TfidfEmbedder


@dataclass(frozen=True)
class VectorRecord:
    record_id: str
    vector: np.ndarray
    metadata: dict[str, Any]
    text: str


@dataclass(frozen=True)
class VectorSearchResult:
    record_id: str
    score: float
    metadata: dict[str, Any]
    text: str


class InMemoryVectorStore:
    """
    Simple in-memory vector store for retrieval experiments.

    Responsibilities:
    - store vectors with IDs and metadata
    - perform cosine-similarity top-k search
    - support optional exact-match metadata filtering
    """

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []
        self._dimension: int | None = None

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def add(
        self,
        record_id: str,
        vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
        text: str = "",
    ) -> None:
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")
        if not isinstance(vector, np.ndarray):
            raise TypeError("vector must be a numpy.ndarray")
        if vector.ndim != 1:
            raise ValueError("vector must be 1-dimensional")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary or None")
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if self._dimension is None:
            self._dimension = int(vector.shape[0])
        elif vector.shape[0] != self._dimension:
            raise ValueError("vector dimension does not match existing store dimension")

        if any(record.record_id == record_id for record in self._records):
            raise ValueError(f"record_id already exists: {record_id}")

        self._records.append(
            VectorRecord(
                record_id=record_id,
                vector=vector.astype(np.float32, copy=False),
                metadata=metadata or {},
                text=text,
            )
        )

    def add_many(
        self,
        record_ids: Sequence[str],
        vectors: np.ndarray,
        metadatas: Sequence[dict[str, Any]] | None = None,
        texts: Sequence[str] | None = None,
    ) -> None:
        if not isinstance(record_ids, list | tuple):
            raise TypeError("record_ids must be a list or tuple of strings")
        if not all(isinstance(record_id, str) for record_id in record_ids):
            raise TypeError("all record_ids must be strings")
        if not isinstance(vectors, np.ndarray):
            raise TypeError("vectors must be a numpy.ndarray")
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2-dimensional")

        row_count = vectors.shape[0]
        if len(record_ids) != row_count:
            raise ValueError("record_ids length must match vectors row count")

        if metadatas is None:
            metadatas = [{} for _ in range(row_count)]
        if texts is None:
            texts = ["" for _ in range(row_count)]

        if not isinstance(metadatas, list | tuple):
            raise TypeError("metadatas must be a list or tuple of dictionaries")
        if not isinstance(texts, list | tuple):
            raise TypeError("texts must be a list or tuple of strings")
        if len(metadatas) != row_count:
            raise ValueError("metadatas length must match vectors row count")
        if len(texts) != row_count:
            raise ValueError("texts length must match vectors row count")
        if not all(isinstance(metadata, dict) for metadata in metadatas):
            raise TypeError("all metadatas must be dictionaries")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("all texts must be strings")

        for record_id, vector, metadata, text in zip(
            record_ids, vectors, metadatas, texts, strict=True
        ):
            self.add(
                record_id=record_id,
                vector=vector,
                metadata=metadata,
                text=text,
            )

    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        if not isinstance(query_vector, np.ndarray):
            raise TypeError("query_vector must be a numpy.ndarray")
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be 1-dimensional")
        if not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if metadata_filter is not None and not isinstance(metadata_filter, dict):
            raise TypeError("metadata_filter must be a dictionary or None")

        if not self._records:
            return []

        if self._dimension is None:
            return []

        if query_vector.shape[0] != self._dimension:
            raise ValueError("query_vector dimension does not match store dimension")

        filtered_records = self._apply_metadata_filter(
            records=self._records,
            metadata_filter=metadata_filter,
        )
        if not filtered_records:
            return []

        document_matrix = np.vstack(
            [record.vector for record in filtered_records]
        ).astype(
            np.float32,
            copy=False,
        )
        scores = TfidfEmbedder.cosine_similarity(query_vector, document_matrix)

        results = [
            VectorSearchResult(
                record_id=record.record_id,
                score=float(score),
                metadata=record.metadata,
                text=record.text,
            )
            for record, score in zip(filtered_records, scores, strict=True)
            if float(score) > 0.0
        ]

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def get(self, record_id: str) -> VectorRecord | None:
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")

        for record in self._records:
            if record.record_id == record_id:
                return record
        return None

    def delete(self, record_id: str) -> bool:
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")

        for index, record in enumerate(self._records):
            if record.record_id == record_id:
                del self._records[index]
                if not self._records:
                    self._dimension = None
                return True
        return False

    def clear(self) -> None:
        self._records.clear()
        self._dimension = None

    @staticmethod
    def _apply_metadata_filter(
        records: list[VectorRecord],
        metadata_filter: dict[str, Any] | None,
    ) -> list[VectorRecord]:
        if metadata_filter is None:
            return records

        filtered: list[VectorRecord] = []
        for record in records:
            if all(
                record.metadata.get(key) == value
                for key, value in metadata_filter.items()
            ):
                filtered.append(record)
        return filtered
