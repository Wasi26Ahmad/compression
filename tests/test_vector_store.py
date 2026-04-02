from __future__ import annotations

import numpy as np
import pytest

from src.retrieval import InMemoryVectorStore, VectorRecord, VectorSearchResult


def test_add_single_vector_record() -> None:
    store = InMemoryVectorStore()
    vector = np.array([1.0, 0.0, 0.5], dtype=np.float32)

    store.add(
        record_id="r1",
        vector=vector,
        metadata={"topic": "cattle"},
        text="cattle weight estimation",
    )

    assert store.size == 1
    assert store.dimension == 3


def test_get_returns_added_record() -> None:
    store = InMemoryVectorStore()
    vector = np.array([1.0, 0.0], dtype=np.float32)
    store.add("r1", vector, {"topic": "road"}, "road anomaly")

    record = store.get("r1")

    assert record is not None
    assert isinstance(record, VectorRecord)
    assert record.record_id == "r1"
    assert record.metadata["topic"] == "road"


def test_get_returns_none_for_missing_record() -> None:
    store = InMemoryVectorStore()

    assert store.get("missing") is None


def test_add_many_inserts_multiple_records() -> None:
    store = InMemoryVectorStore()
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    store.add_many(
        record_ids=["r1", "r2"],
        vectors=vectors,
        metadatas=[{"topic": "a"}, {"topic": "b"}],
        texts=["text a", "text b"],
    )

    assert store.size == 2
    assert store.dimension == 2


def test_search_returns_ranked_results() -> None:
    store = InMemoryVectorStore()

    store.add(
        "r1",
        np.array([1.0, 0.0], dtype=np.float32),
        {"topic": "cattle"},
        "cattle text",
    )
    store.add(
        "r2",
        np.array([0.0, 1.0], dtype=np.float32),
        {"topic": "road"},
        "road text",
    )

    query_vector = np.array([1.0, 0.0], dtype=np.float32)
    results = store.search(query_vector, limit=2)

    assert len(results) >= 1
    assert isinstance(results[0], VectorSearchResult)
    assert results[0].record_id == "r1"
    assert results[0].score >= 0.0


def test_search_respects_limit() -> None:
    store = InMemoryVectorStore()
    store.add("r1", np.array([1.0, 0.0], dtype=np.float32), {}, "a")
    store.add("r2", np.array([0.9, 0.1], dtype=np.float32), {}, "b")
    store.add("r3", np.array([0.8, 0.2], dtype=np.float32), {}, "c")

    query_vector = np.array([1.0, 0.0], dtype=np.float32)
    results = store.search(query_vector, limit=2)

    assert len(results) <= 2


def test_search_can_filter_by_metadata() -> None:
    store = InMemoryVectorStore()
    store.add(
        "r1",
        np.array([1.0, 0.0], dtype=np.float32),
        {"topic": "cattle", "type": "vision"},
        "cattle vision",
    )
    store.add(
        "r2",
        np.array([1.0, 0.0], dtype=np.float32),
        {"topic": "cattle", "type": "health"},
        "cattle health",
    )

    query_vector = np.array([1.0, 0.0], dtype=np.float32)
    results = store.search(
        query_vector,
        limit=5,
        metadata_filter={"type": "vision"},
    )

    assert len(results) == 1
    assert results[0].metadata["type"] == "vision"


def test_search_returns_empty_when_store_is_empty() -> None:
    store = InMemoryVectorStore()
    query_vector = np.array([1.0, 0.0], dtype=np.float32)

    results = store.search(query_vector)

    assert results == []


def test_search_returns_empty_when_metadata_filter_matches_nothing() -> None:
    store = InMemoryVectorStore()
    store.add("r1", np.array([1.0, 0.0], dtype=np.float32), {"topic": "cattle"}, "x")

    query_vector = np.array([1.0, 0.0], dtype=np.float32)
    results = store.search(query_vector, metadata_filter={"topic": "road"})

    assert results == []


def test_delete_removes_record() -> None:
    store = InMemoryVectorStore()
    store.add("r1", np.array([1.0, 0.0], dtype=np.float32), {}, "text")

    deleted = store.delete("r1")

    assert deleted is True
    assert store.get("r1") is None
    assert store.size == 0
    assert store.dimension is None


def test_delete_returns_false_for_missing_record() -> None:
    store = InMemoryVectorStore()

    assert store.delete("missing") is False


def test_clear_removes_all_records() -> None:
    store = InMemoryVectorStore()
    store.add("r1", np.array([1.0, 0.0], dtype=np.float32), {}, "a")
    store.add("r2", np.array([0.0, 1.0], dtype=np.float32), {}, "b")

    store.clear()

    assert store.size == 0
    assert store.dimension is None


def test_add_rejects_invalid_record_id_type() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(TypeError, match="record_id must be a string"):
        store.add(1, np.array([1.0], dtype=np.float32))  # type: ignore[arg-type]


def test_add_rejects_invalid_vector_type() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(TypeError, match="vector must be a numpy.ndarray"):
        store.add("r1", [1.0, 2.0])  # type: ignore[arg-type]


def test_add_rejects_non_1d_vector() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="vector must be 1-dimensional"):
        store.add("r1", np.array([[1.0, 2.0]], dtype=np.float32))


def test_add_rejects_invalid_metadata_type() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(TypeError, match="metadata must be a dictionary or None"):
        store.add(
            "r1",
            np.array([1.0], dtype=np.float32),
            metadata="bad",  # type: ignore[arg-type]
        )


def test_add_rejects_invalid_text_type() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(TypeError, match="text must be a string"):
        store.add(
            "r1",
            np.array([1.0], dtype=np.float32),
            text=123,  # type: ignore[arg-type]
        )


def test_add_rejects_duplicate_record_id() -> None:
    store = InMemoryVectorStore()
    vector = np.array([1.0, 0.0], dtype=np.float32)
    store.add("r1", vector)

    with pytest.raises(ValueError, match="record_id already exists: r1"):
        store.add("r1", vector)


def test_add_rejects_dimension_mismatch() -> None:
    store = InMemoryVectorStore()
    store.add("r1", np.array([1.0, 0.0], dtype=np.float32))

    with pytest.raises(
        ValueError,
        match="vector dimension does not match existing store dimension",
    ):
        store.add("r2", np.array([1.0, 0.0, 0.5], dtype=np.float32))


def test_add_many_rejects_invalid_record_ids_type() -> None:
    store = InMemoryVectorStore()
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(
        TypeError, match="record_ids must be a list or tuple of strings"
    ):
        store.add_many("bad", vectors)  # type: ignore[arg-type]


def test_add_many_rejects_non_string_record_ids() -> None:
    store = InMemoryVectorStore()
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="all record_ids must be strings"):
        store.add_many([1], vectors)  # type: ignore[list-item]


def test_add_many_rejects_invalid_vectors_type() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(TypeError, match="vectors must be a numpy.ndarray"):
        store.add_many(["r1"], [[1.0, 0.0]])  # type: ignore[arg-type]


def test_add_many_rejects_non_2d_vectors() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="vectors must be 2-dimensional"):
        store.add_many(["r1"], np.array([1.0, 0.0], dtype=np.float32))


def test_add_many_rejects_length_mismatch_with_record_ids() -> None:
    store = InMemoryVectorStore()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with pytest.raises(
        ValueError, match="record_ids length must match vectors row count"
    ):
        store.add_many(["r1"], vectors)


def test_add_many_rejects_invalid_metadatas_type() -> None:
    store = InMemoryVectorStore()
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(
        TypeError, match="metadatas must be a list or tuple of dictionaries"
    ):
        store.add_many(["r1"], vectors, metadatas="bad")  # type: ignore[arg-type]


def test_add_many_rejects_invalid_texts_type() -> None:
    store = InMemoryVectorStore()
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="texts must be a list or tuple of strings"):
        store.add_many(["r1"], vectors, texts="bad")  # type: ignore[arg-type]


def test_add_many_rejects_metadata_length_mismatch() -> None:
    store = InMemoryVectorStore()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with pytest.raises(
        ValueError, match="metadatas length must match vectors row count"
    ):
        store.add_many(["r1", "r2"], vectors, metadatas=[{}])


def test_add_many_rejects_texts_length_mismatch() -> None:
    store = InMemoryVectorStore()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="texts length must match vectors row count"):
        store.add_many(["r1", "r2"], vectors, texts=["a"])


def test_add_many_rejects_non_dict_metadatas() -> None:
    store = InMemoryVectorStore()
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="all metadatas must be dictionaries"):
        store.add_many(["r1"], vectors, metadatas=["bad"])  # type: ignore[list-item]


def test_add_many_rejects_non_string_texts() -> None:
    store = InMemoryVectorStore()
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="all texts must be strings"):
        store.add_many(["r1"], vectors, texts=[1])  # type: ignore[list-item]


def test_search_rejects_invalid_query_vector_type() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(TypeError, match="query_vector must be a numpy.ndarray"):
        store.search([1.0, 0.0])  # type: ignore[arg-type]


def test_search_rejects_non_1d_query_vector() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="query_vector must be 1-dimensional"):
        store.search(np.array([[1.0, 0.0]], dtype=np.float32))


def test_search_rejects_invalid_limit_type() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(TypeError, match="limit must be an integer"):
        store.search(np.array([1.0], dtype=np.float32), limit="5")  # type: ignore[arg-type]


def test_search_rejects_invalid_limit_value() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="limit must be >= 1"):
        store.search(np.array([1.0], dtype=np.float32), limit=0)


def test_search_rejects_invalid_metadata_filter_type() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(TypeError, match="metadata_filter must be a dictionary or None"):
        store.search(
            np.array([1.0], dtype=np.float32),
            metadata_filter="bad",  # type: ignore[arg-type]
        )


def test_search_rejects_dimension_mismatch() -> None:
    store = InMemoryVectorStore()
    store.add("r1", np.array([1.0, 0.0], dtype=np.float32))

    with pytest.raises(
        ValueError,
        match="query_vector dimension does not match store dimension",
    ):
        store.search(np.array([1.0, 0.0, 0.5], dtype=np.float32))


def test_get_rejects_invalid_record_id_type() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(TypeError, match="record_id must be a string"):
        store.get(123)  # type: ignore[arg-type]


def test_delete_rejects_invalid_record_id_type() -> None:
    store = InMemoryVectorStore()

    with pytest.raises(TypeError, match="record_id must be a string"):
        store.delete(123)  # type: ignore[arg-type]
