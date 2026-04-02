from __future__ import annotations

import pytest

from src.memory import MemoryManager
from src.retrieval import MemoryRetriever, RetrievalResult
from src.storage import CompressionStorage


def test_retriever_rejects_invalid_memory_manager() -> None:
    with pytest.raises(TypeError, match="memory_manager must be a MemoryManager"):
        MemoryRetriever(memory_manager="not-a-manager")  # type: ignore[arg-type]


def test_retrieve_returns_relevant_results(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage, default_method="dictionary")
    retriever = MemoryRetriever(manager)

    manager.save_text(
        "Cattle weight estimation uses side-view and rear-view images.",
        metadata={"topic": "cattle"},
    )
    manager.save_text(
        "Road anomaly detection uses accelerometer and gyroscope signals.",
        metadata={"topic": "road"},
    )

    results = retriever.retrieve("cattle image estimation", limit=5)

    assert len(results) >= 1
    assert isinstance(results[0], RetrievalResult)
    assert "cattle" in results[0].text.lower()


def test_retrieve_respects_limit(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    for idx in range(5):
        manager.save_text(f"Machine learning text sample {idx}")

    results = retriever.retrieve("machine learning", limit=2)

    assert len(results) <= 2


def test_retrieve_returns_empty_for_empty_query_tokens(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    manager.save_text("Some stored text")

    results = retriever.retrieve("   !!!   ", limit=5)

    assert results == []


def test_retrieve_returns_empty_when_no_memories_exist(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    results = retriever.retrieve("anything", limit=5)

    assert results == []


def test_retrieve_can_filter_by_metadata(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    manager.save_text(
        "Cattle body measurements help estimate weight.",
        metadata={"topic": "cattle", "type": "vision"},
    )
    manager.save_text(
        "Cattle nutrition affects growth rate.",
        metadata={"topic": "cattle", "type": "health"},
    )
    manager.save_text(
        "Road anomalies can be detected by vehicle motion sensors.",
        metadata={"topic": "road", "type": "sensor"},
    )

    results = retriever.retrieve(
        "cattle",
        limit=10,
        metadata_filter={"type": "vision"},
    )

    assert len(results) >= 1
    assert all(result.metadata["type"] == "vision" for result in results)


def test_retrieve_returns_empty_when_metadata_filter_matches_nothing(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    manager.save_text(
        "Some text about cattle.",
        metadata={"topic": "cattle"},
    )

    results = retriever.retrieve(
        "cattle",
        limit=5,
        metadata_filter={"topic": "road"},
    )

    assert results == []


def test_retrieve_texts_returns_only_text_payloads(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    manager.save_text("Transformer models are useful for sequence tasks.")
    manager.save_text("Convolutional models are useful for image tasks.")

    texts = retriever.retrieve_texts("image tasks", limit=2)

    assert len(texts) >= 1
    assert all(isinstance(text, str) for text in texts)


def test_retrieve_results_are_sorted_by_score_descending(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    manager.save_text("cattle cattle cattle weight estimation from images")
    manager.save_text("cattle estimation")
    manager.save_text("road anomaly detection")

    results = retriever.retrieve("cattle estimation", limit=3)

    assert len(results) >= 2
    assert results[0].score >= results[1].score


def test_retrieve_rejects_invalid_query_type(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    with pytest.raises(TypeError, match="query must be a string"):
        retriever.retrieve(123, limit=5)  # type: ignore[arg-type]


def test_retrieve_rejects_invalid_limit_type(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    with pytest.raises(TypeError, match="limit must be an integer"):
        retriever.retrieve("test", limit="5")  # type: ignore[arg-type]


def test_retrieve_rejects_invalid_search_limit_type(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    with pytest.raises(TypeError, match="search_limit must be an integer"):
        retriever.retrieve("test", search_limit="10")  # type: ignore[arg-type]


def test_retrieve_rejects_invalid_limit_value(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    with pytest.raises(ValueError, match="limit must be >= 1"):
        retriever.retrieve("test", limit=0)


def test_retrieve_rejects_invalid_search_limit_value(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    with pytest.raises(ValueError, match="search_limit must be >= 1"):
        retriever.retrieve("test", search_limit=0)


def test_retrieve_rejects_invalid_metadata_filter_type(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    with pytest.raises(TypeError, match="metadata_filter must be a dictionary or None"):
        retriever.retrieve("test", metadata_filter="bad-filter")  # type: ignore[arg-type]


def test_retrieve_uses_search_limit_to_reduce_scope(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    manager.save_text("older cattle record", metadata={"order": 1})
    manager.save_text("middle road record", metadata={"order": 2})
    manager.save_text("newest transformer record", metadata={"order": 3})

    results = retriever.retrieve("cattle", limit=5, search_limit=1)

    # Only the most recent record is searched, so cattle text should not appear.
    assert results == []


def test_retrieve_dictionary_compressed_texts(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage, default_method="dictionary")
    retriever = MemoryRetriever(manager)

    manager.save_text(
        "Alpha beta alpha beta alpha beta gamma",
        compressor_kwargs={
            "min_phrase_len": 2,
            "max_phrase_len": 5,
            "min_frequency": 2,
        },
    )

    results = retriever.retrieve("alpha beta", limit=5)

    assert len(results) >= 1
    assert "alpha beta" in results[0].text.lower()


def test_retrieve_returns_result_fields(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    manager.save_text(
        "Biomedical signal processing supports diagnostic systems.",
        metadata={"domain": "biomedical"},
    )

    results = retriever.retrieve("diagnostic systems", limit=1)

    assert len(results) == 1
    result = results[0]
    assert isinstance(result.record_id, str)
    assert isinstance(result.score, float)
    assert isinstance(result.method, str)
    assert isinstance(result.created_at, str)
    assert isinstance(result.metadata, dict)
    assert isinstance(result.text, str)
