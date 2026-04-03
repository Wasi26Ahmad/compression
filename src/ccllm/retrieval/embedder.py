from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray
    vocabulary_size: int
    document_count: int


class TfidfEmbedder:
    """
    Lightweight TF-IDF embedder for text retrieval.

    Design goals:
    - deterministic output
    - easy testing
    - no external model downloads
    - good baseline for vector retrieval

    Notes:
    - fit_transform() should be used on a document corpus
    - transform() should be used for query texts after fitting
    """

    def __init__(
        self,
        lowercase: bool = True,
        stop_words: str | None = None,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int | None = None,
    ) -> None:
        if not isinstance(lowercase, bool):
            raise TypeError("lowercase must be a boolean")
        if stop_words is not None and not isinstance(stop_words, str):
            raise TypeError("stop_words must be a string or None")
        if (
            not isinstance(ngram_range, tuple)
            or len(ngram_range) != 2
            or not all(isinstance(value, int) for value in ngram_range)
        ):
            raise TypeError("ngram_range must be a tuple of two integers")
        if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
            raise ValueError("ngram_range must satisfy 1 <= min_n <= max_n")
        if max_features is not None:
            if not isinstance(max_features, int):
                raise TypeError("max_features must be an integer or None")
            if max_features < 1:
                raise ValueError("max_features must be >= 1")

        self.vectorizer = TfidfVectorizer(
            lowercase=lowercase,
            stop_words=stop_words,
            ngram_range=ngram_range,
            max_features=max_features,
        )
        self._is_fitted = False

    def fit_transform(self, texts: Sequence[str]) -> EmbeddingResult:
        """
        Fit the TF-IDF vectorizer on the input corpus and return dense vectors.
        """
        self._validate_texts(texts)

        if len(texts) == 0:
            raise ValueError("texts must not be empty")

        matrix = self.vectorizer.fit_transform(texts)
        vectors = matrix.toarray().astype(np.float32, copy=False)
        self._is_fitted = True

        return EmbeddingResult(
            vectors=vectors,
            vocabulary_size=len(self.vectorizer.vocabulary_),
            document_count=len(texts),
        )

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        """
        Transform new texts into the already-fitted TF-IDF vector space.
        """
        self._validate_texts(texts)

        if not self._is_fitted:
            raise ValueError("embedder must be fitted before calling transform")

        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().astype(np.float32, copy=False)

    def fit(self, texts: Sequence[str]) -> None:
        """
        Fit the vectorizer without returning vectors.
        """
        self.fit_transform(texts)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def vocabulary_size(self) -> int:
        if not self._is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)

    @staticmethod
    def cosine_similarity(
        query_vector: np.ndarray, document_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between one query vector and multiple document
        vectors.
        """
        if not isinstance(query_vector, np.ndarray):
            raise TypeError("query_vector must be a numpy.ndarray")
        if not isinstance(document_matrix, np.ndarray):
            raise TypeError("document_matrix must be a numpy.ndarray")

        if query_vector.ndim != 1:
            raise ValueError("query_vector must be 1-dimensional")
        if document_matrix.ndim != 2:
            raise ValueError("document_matrix must be 2-dimensional")

        if document_matrix.shape[1] != query_vector.shape[0]:
            raise ValueError("query_vector and document_matrix dimensions do not match")

        query_norm = np.linalg.norm(query_vector)
        doc_norms = np.linalg.norm(document_matrix, axis=1)

        if query_norm == 0.0:
            return np.zeros(document_matrix.shape[0], dtype=np.float32)

        safe_doc_norms = np.where(doc_norms == 0.0, 1.0, doc_norms)
        similarities = document_matrix @ query_vector / (safe_doc_norms * query_norm)
        similarities = np.where(doc_norms == 0.0, 0.0, similarities)

        return similarities.astype(np.float32, copy=False)

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> None:
        if not isinstance(texts, list | tuple):
            raise TypeError("texts must be a list or tuple of strings")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("all texts must be strings")
