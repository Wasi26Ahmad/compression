from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.embedder import EmbeddingResult, TfidfEmbedder


def test_embedder_fit_transform_returns_embedding_result() -> None:
    embedder = TfidfEmbedder()
    texts = [
        "cattle weight estimation from images",
        "road anomaly detection from sensors",
    ]

    result = embedder.fit_transform(texts)

    assert isinstance(result, EmbeddingResult)
    assert isinstance(result.vectors, np.ndarray)
    assert result.document_count == 2
    assert result.vocabulary_size > 0
    assert result.vectors.shape[0] == 2


def test_embedder_sets_fitted_state_after_fit_transform() -> None:
    embedder = TfidfEmbedder()

    embedder.fit_transform(["text one", "text two"])

    assert embedder.is_fitted is True
    assert embedder.vocabulary_size > 0


def test_embedder_fit_sets_fitted_state() -> None:
    embedder = TfidfEmbedder()

    embedder.fit(["alpha beta", "gamma delta"])

    assert embedder.is_fitted is True
    assert embedder.vocabulary_size > 0


def test_transform_returns_vectors_after_fit() -> None:
    embedder = TfidfEmbedder()
    embedder.fit(["machine learning", "deep learning"])

    vectors = embedder.transform(["machine learning"])

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == 1
    assert vectors.shape[1] == embedder.vocabulary_size


def test_transform_rejects_call_before_fit() -> None:
    embedder = TfidfEmbedder()

    with pytest.raises(
        ValueError, match="embedder must be fitted before calling transform"
    ):
        embedder.transform(["test query"])


def test_fit_transform_rejects_empty_texts() -> None:
    embedder = TfidfEmbedder()

    with pytest.raises(ValueError, match="texts must not be empty"):
        embedder.fit_transform([])


def test_fit_transform_rejects_invalid_texts_container() -> None:
    embedder = TfidfEmbedder()

    with pytest.raises(TypeError, match="texts must be a list or tuple of strings"):
        embedder.fit_transform("not-a-list")  # type: ignore[arg-type]


def test_fit_transform_rejects_non_string_items() -> None:
    embedder = TfidfEmbedder()

    with pytest.raises(TypeError, match="all texts must be strings"):
        embedder.fit_transform(["valid", 123])  # type: ignore[list-item]


def test_transform_rejects_invalid_texts_container() -> None:
    embedder = TfidfEmbedder()
    embedder.fit(["alpha beta"])

    with pytest.raises(TypeError, match="texts must be a list or tuple of strings"):
        embedder.transform("not-a-list")  # type: ignore[arg-type]


def test_transform_rejects_non_string_items() -> None:
    embedder = TfidfEmbedder()
    embedder.fit(["alpha beta"])

    with pytest.raises(TypeError, match="all texts must be strings"):
        embedder.transform(["valid", 123])  # type: ignore[list-item]


def test_cosine_similarity_returns_scores_for_documents() -> None:
    embedder = TfidfEmbedder()
    corpus = [
        "cattle weight estimation from images",
        "road anomaly detection from sensors",
    ]
    fit_result = embedder.fit_transform(corpus)
    query_vector = embedder.transform(["cattle image estimation"])[0]

    scores = embedder.cosine_similarity(query_vector, fit_result.vectors)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)
    assert scores[0] >= scores[1]


def test_cosine_similarity_returns_zero_for_zero_query_vector() -> None:
    query_vector = np.zeros(4, dtype=np.float32)
    document_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    scores = TfidfEmbedder.cosine_similarity(query_vector, document_matrix)

    assert np.allclose(scores, np.array([0.0, 0.0], dtype=np.float32))


def test_cosine_similarity_rejects_invalid_query_vector_type() -> None:
    document_matrix = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="query_vector must be a numpy.ndarray"):
        TfidfEmbedder.cosine_similarity([1.0, 0.0], document_matrix)  # type: ignore[arg-type]


def test_cosine_similarity_rejects_invalid_document_matrix_type() -> None:
    query_vector = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(TypeError, match="document_matrix must be a numpy.ndarray"):
        TfidfEmbedder.cosine_similarity(query_vector, [[1.0, 0.0]])  # type: ignore[arg-type]


def test_cosine_similarity_rejects_invalid_query_vector_shape() -> None:
    query_vector = np.array([[1.0, 0.0]], dtype=np.float32)
    document_matrix = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="query_vector must be 1-dimensional"):
        TfidfEmbedder.cosine_similarity(query_vector, document_matrix)


def test_cosine_similarity_rejects_invalid_document_matrix_shape() -> None:
    query_vector = np.array([1.0, 0.0], dtype=np.float32)
    document_matrix = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="document_matrix must be 2-dimensional"):
        TfidfEmbedder.cosine_similarity(query_vector, document_matrix)


def test_cosine_similarity_rejects_dimension_mismatch() -> None:
    query_vector = np.array([1.0, 0.0, 0.5], dtype=np.float32)
    document_matrix = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(
        ValueError, match="query_vector and document_matrix dimensions do not match"
    ):
        TfidfEmbedder.cosine_similarity(query_vector, document_matrix)


def test_embedder_rejects_invalid_lowercase_type() -> None:
    with pytest.raises(TypeError, match="lowercase must be a boolean"):
        TfidfEmbedder(lowercase="yes")  # type: ignore[arg-type]


def test_embedder_rejects_invalid_stop_words_type() -> None:
    with pytest.raises(TypeError, match="stop_words must be a string or None"):
        TfidfEmbedder(stop_words=["english"])  # type: ignore[arg-type]


def test_embedder_rejects_invalid_ngram_range_type() -> None:
    with pytest.raises(TypeError, match="ngram_range must be a tuple of two integers"):
        TfidfEmbedder(ngram_range=[1, 2])  # type: ignore[arg-type]


def test_embedder_rejects_invalid_ngram_range_value() -> None:
    with pytest.raises(
        ValueError, match="ngram_range must satisfy 1 <= min_n <= max_n"
    ):
        TfidfEmbedder(ngram_range=(2, 1))


def test_embedder_rejects_invalid_max_features_type() -> None:
    with pytest.raises(TypeError, match="max_features must be an integer or None"):
        TfidfEmbedder(max_features="100")  # type: ignore[arg-type]


def test_embedder_rejects_invalid_max_features_value() -> None:
    with pytest.raises(ValueError, match="max_features must be >= 1"):
        TfidfEmbedder(max_features=0)
