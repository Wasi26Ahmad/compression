from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from src.ccllm.memory import MemoryManager
from src.ccllm.retrieval.embedder import TfidfEmbedder
from src.ccllm.retrieval.vector_store import InMemoryVectorStore

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
    Hybrid retriever:
    - lexical (TF-IDF cosine)
    - vector-based (embedder + vector store)

    Modes:
    - "lexical"
    - "vector"
    - "hybrid" (default)

    Hybrid = weighted combination of both scores
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        mode: str = "hybrid",
        alpha: float = 0.5,
    ) -> None:
        if not isinstance(memory_manager, MemoryManager):
            raise TypeError("memory_manager must be a MemoryManager")
        if mode not in {"lexical", "vector", "hybrid"}:
            raise ValueError("mode must be one of: lexical, vector, hybrid")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1")

        self.memory_manager = memory_manager
        self.mode = mode
        self.alpha = alpha

        # Vector components
        self.embedder = TfidfEmbedder()
        self.vector_store = InMemoryVectorStore()
        self._vector_index_ready = False

    # ===============================
    # Public API
    # ===============================

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        search_limit: int = 100,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
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

        bundles = self._load_bundles(search_limit, metadata_filter)
        if not bundles:
            return []

        texts = [bundle["text"] for bundle in bundles]

        results: list[RetrievalResult]

        if self.mode == "lexical":
            scores = self._lexical_scores(query, texts)

        elif self.mode == "vector":
            scores = self._vector_scores(query, texts, bundles)

        else:  # hybrid
            lexical = self._lexical_scores(query, texts)
            vector = self._vector_scores(query, texts, bundles)

            scores = [
                self.alpha * i + (1 - self.alpha) * j
                for i, j in zip(lexical, vector, strict=True)
            ]

        results = [
            RetrievalResult(
                record_id=bundle["record"]["record_id"],
                score=float(score),
                method=bundle["record"]["method"],
                created_at=bundle["record"]["created_at"],
                metadata=bundle["metadata"],
                text=bundle["text"],
            )
            for bundle, score in zip(bundles, scores, strict=True)
            if score > 0.0
        ]

        results.sort(key=lambda r: (r.score, r.created_at), reverse=True)
        return results[:limit]

    def retrieve_texts(self, query: str, limit: int = 5) -> list[str]:
        return [r.text for r in self.retrieve(query, limit)]

    # ===============================
    # Bundle Loading
    # ===============================

    def _load_bundles(
        self,
        limit: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        items = self.memory_manager.restore_all_texts(limit=limit)

        bundles = []
        for item in items:
            bundle = self.memory_manager.export_record_bundle(item["record_id"])
            if bundle is None:
                continue
            if metadata_filter:
                if not all(
                    bundle["metadata"].get(k) == v for k, v in metadata_filter.items()
                ):
                    continue
            bundles.append(bundle)

        return bundles

    # ===============================
    # Lexical Scoring
    # ===============================

    def _lexical_scores(self, query: str, texts: list[str]) -> list[float]:
        query_tokens = self._tokenize(query)
        doc_tokens = [self._tokenize(text) for text in texts]

        idf = self._compute_idf(query_tokens, doc_tokens)

        return [
            self._cosine_similarity(query_tokens, tokens, idf) for tokens in doc_tokens
        ]

    def _tokenize(self, text: str) -> list[str]:
        return [t.lower() for t in _WORD_PATTERN.findall(text)]

    def _compute_idf(
        self,
        query_tokens: list[str],
        docs: list[list[str]],
    ) -> dict[str, float]:
        total_docs = len(docs)
        df = {t: 0 for t in set(query_tokens)}

        for doc in docs:
            unique = set(doc)
            for t in df:
                if t in unique:
                    df[t] += 1

        return {t: math.log((1 + total_docs) / (1 + f)) + 1 for t, f in df.items()}

    def _cosine_similarity(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
        idf: dict[str, float],
    ) -> float:
        q = Counter(query_tokens)
        d = Counter(doc_tokens)

        dot = sum(q[t] * d[t] * idf.get(t, 0.0) for t in idf)
        qn = math.sqrt(sum((q[t] * idf.get(t, 0.0)) ** 2 for t in idf))
        dn = math.sqrt(sum((d[t] * idf.get(t, 0.0)) ** 2 for t in idf))

        if qn == 0 or dn == 0:
            return 0.0
        return dot / (qn * dn)

    # ===============================
    # Vector Scoring
    # ===============================

    def _vector_scores(
        self,
        query: str,
        texts: list[str],
        bundles: list[dict[str, Any]],
    ) -> list[float]:
        # Build index once per call (simple baseline)
        embed_result = self.embedder.fit_transform(texts)

        self.vector_store.clear()
        for bundle, vec in zip(bundles, embed_result.vectors, strict=True):
            self.vector_store.add(
                record_id=bundle["record"]["record_id"],
                vector=vec,
                metadata=bundle["metadata"],
                text=bundle["text"],
            )

        query_vec = self.embedder.transform([query])[0]
        results = self.vector_store.search(query_vec, limit=len(texts))

        score_map = {r.record_id: r.score for r in results}

        return [score_map.get(bundle["record"]["record_id"], 0.0) for bundle in bundles]
