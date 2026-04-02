from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.memory import MemoryManager

_WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


@dataclass(frozen=True)
class RetrievalResult:
    record_id: str
    score: float
    method: str
    created_at: str
    metadata: dict[str, Any]
    text: str


class MemoryRetriever:
    """
    Lightweight lexical retriever over restored memory texts.

    Current behavior:
    - loads recent memories from MemoryManager
    - tokenizes query and document text
    - scores with a simple TF-IDF-like cosine similarity
    - supports optional exact-match metadata filtering
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        if not isinstance(memory_manager, MemoryManager):
            raise TypeError("memory_manager must be a MemoryManager")

        self.memory_manager = memory_manager

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        search_limit: int = 100,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant stored texts for a query.

        Args:
            query: Search query text.
            limit: Number of results to return.
            search_limit: Number of recent memories to inspect.
            metadata_filter: Optional exact-match metadata filters.

        Returns:
            Ranked RetrievalResult list.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if not isinstance(search_limit, int):
            raise TypeError("search_limit must be an integer")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if search_limit < 1:
            raise ValueError("search_limit must be >= 1")
        if metadata_filter is not None and not isinstance(metadata_filter, dict):
            raise TypeError("metadata_filter must be a dictionary or None")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        restored_items = self.memory_manager.restore_all_texts(limit=search_limit)
        bundles = []
        for item in restored_items:
            bundle = self.memory_manager.export_record_bundle(item["record_id"])
            if bundle is None:
                continue
            bundles.append(bundle)

        filtered_bundles = self._apply_metadata_filter(
            bundles=bundles,
            metadata_filter=metadata_filter,
        )
        if not filtered_bundles:
            return []

        document_tokens = [
            self._tokenize(bundle["record"].get("package_json", ""))
            for bundle in filtered_bundles
        ]
        # Use restored text, not package JSON, for scoring
        document_tokens = [
            (
                self._tokenize(bundle["package_text"])
                if "package_text" in bundle
                else (
                    self._tokenize(bundle["record_text"])
                    if "record_text" in bundle
                    else (
                        self._tokenize(bundle["restored_text"])
                        if "restored_text" in bundle
                        else self._tokenize(self._extract_text(bundle))
                    )
                )
            )
            for bundle in filtered_bundles
        ]

        idf = self._compute_idf(query_tokens=query_tokens, documents=document_tokens)

        scored_results: list[RetrievalResult] = []
        for bundle, tokens in zip(filtered_bundles, document_tokens, strict=True):
            score = self._cosine_similarity(
                query_tokens=query_tokens,
                document_tokens=tokens,
                idf=idf,
            )
            if score <= 0.0:
                continue

            record = bundle["record"]
            metadata = bundle["metadata"]
            text = self._extract_text(bundle)

            scored_results.append(
                RetrievalResult(
                    record_id=record["record_id"],
                    score=score,
                    method=record["method"],
                    created_at=record["created_at"],
                    metadata=metadata,
                    text=text,
                )
            )

        scored_results.sort(
            key=lambda item: (item.score, item.created_at),
            reverse=True,
        )
        return scored_results[:limit]

    def retrieve_texts(
        self,
        query: str,
        limit: int = 5,
        search_limit: int = 100,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Convenience method: return only text payloads from results.
        """
        results = self.retrieve(
            query=query,
            limit=limit,
            search_limit=search_limit,
            metadata_filter=metadata_filter,
        )
        return [result.text for result in results]

    def _apply_metadata_filter(
        self,
        bundles: list[dict[str, Any]],
        metadata_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if metadata_filter is None:
            return bundles

        filtered: list[dict[str, Any]] = []
        for bundle in bundles:
            metadata = bundle["metadata"]
            if all(
                metadata.get(key) == value for key, value in metadata_filter.items()
            ):
                filtered.append(bundle)
        return filtered

    def _compute_idf(
        self,
        query_tokens: list[str],
        documents: list[list[str]],
    ) -> dict[str, float]:
        """
        Compute IDF only for query terms.
        """
        total_docs = len(documents)
        doc_freq: dict[str, int] = {token: 0 for token in set(query_tokens)}

        for doc_tokens in documents:
            unique_tokens = set(doc_tokens)
            for token in doc_freq:
                if token in unique_tokens:
                    doc_freq[token] += 1

        idf: dict[str, float] = {}
        for token, freq in doc_freq.items():
            idf[token] = math.log((1 + total_docs) / (1 + freq)) + 1.0

        return idf

    def _cosine_similarity(
        self,
        query_tokens: list[str],
        document_tokens: list[str],
        idf: dict[str, float],
    ) -> float:
        if not document_tokens:
            return 0.0

        query_counts = Counter(query_tokens)
        doc_counts = Counter(document_tokens)

        query_vector: dict[str, float] = {}
        doc_vector: dict[str, float] = {}

        for token in idf:
            query_tf = query_counts[token]
            doc_tf = doc_counts[token]

            if query_tf > 0:
                query_vector[token] = float(query_tf) * idf[token]
            if doc_tf > 0:
                doc_vector[token] = float(doc_tf) * idf[token]

        if not query_vector or not doc_vector:
            return 0.0

        dot_product = sum(
            query_vector.get(token, 0.0) * doc_vector.get(token, 0.0) for token in idf
        )
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        doc_norm = math.sqrt(sum(value * value for value in doc_vector.values()))

        if query_norm == 0.0 or doc_norm == 0.0:
            return 0.0

        return dot_product / (query_norm * doc_norm)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in _WORD_PATTERN.findall(text)]

    @staticmethod
    def _extract_text(bundle: dict[str, Any]) -> str:
        """
        Extract restored text from a bundle.

        Since export_record_bundle currently returns:
        - record
        - metadata
        - package

        and not the restored text directly, we reconstruct text from restore_all_texts
        at retrieval time via fallback fields if present.
        """
        if "text" in bundle:
            return str(bundle["text"])
        if "restored_text" in bundle:
            return str(bundle["restored_text"])
        if "record_text" in bundle:
            return str(bundle["record_text"])
        if "package_text" in bundle:
            return str(bundle["package_text"])
        raise ValueError("Bundle does not contain restored text")
