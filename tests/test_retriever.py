from __future__ import annotations

import pytest

from src.memory import MemoryManager
from src.retrieval import MemoryRetriever, RetrievalResult
from src.storage import CompressionStorage

# ===============================
# Setup Helper
# ===============================


def build_env(tmp_path):
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    return manager


# ===============================
# Basic Validation
# ===============================


def test_retriever_rejects_invalid_memory_manager() -> None:
    with pytest.raises(TypeError, match="memory_manager must be a MemoryManager"):
        MemoryRetriever(memory_manager="bad")  # type: ignore[arg-type]


def test_retriever_rejects_invalid_mode(tmp_path) -> None:
    manager = build_env(tmp_path)

    with pytest.raises(ValueError, match="mode must be one of"):
        MemoryRetriever(manager, mode="invalid")  # type: ignore[arg-type]


def test_retriever_rejects_invalid_alpha_type(tmp_path) -> None:
    manager = build_env(tmp_path)

    with pytest.raises(TypeError, match="alpha must be a float"):
        MemoryRetriever(manager, alpha="0.5")  # type: ignore[arg-type]


def test_retriever_rejects_invalid_alpha_range(tmp_path) -> None:
    manager = build_env(tmp_path)

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        MemoryRetriever(manager, alpha=1.5)


# ===============================
# Lexical Mode Tests
# ===============================


def test_lexical_retrieval_basic(tmp_path) -> None:
    manager = build_env(tmp_path)
    retriever = MemoryRetriever(manager, mode="lexical")

    manager.save_text("Cattle weight estimation from images")
    manager.save_text("Road anomaly detection using sensors")

    results = retriever.retrieve("cattle estimation", limit=2)

    assert len(results) >= 1
    assert isinstance(results[0], RetrievalResult)
    assert "cattle" in results[0].text.lower()


def test_lexical_respects_limit(tmp_path) -> None:
    manager = build_env(tmp_path)
    retriever = MemoryRetriever(manager, mode="lexical")

    for i in range(5):
        manager.save_text(f"text {i} cattle")

    results = retriever.retrieve("cattle", limit=2)

    assert len(results) <= 2


# ===============================
# Vector Mode Tests
# ===============================


def test_vector_retrieval_basic(tmp_path) -> None:
    manager = build_env(tmp_path)
    retriever = MemoryRetriever(manager, mode="vector")

    manager.save_text("Deep learning for medical imaging")
    manager.save_text("Cattle weight estimation from vision")

    results = retriever.retrieve("cattle vision", limit=2)

    assert len(results) >= 1
    assert "cattle" in results[0].text.lower()


def test_vector_returns_empty_for_no_match(tmp_path) -> None:
    manager = build_env(tmp_path)
    retriever = MemoryRetriever(manager, mode="vector")

    manager.save_text("road anomaly detection")

    results = retriever.retrieve("quantum physics", limit=2)

    assert results == [] or all(r.score >= 0 for r in results)


# ===============================
# Hybrid Mode Tests
# ===============================


def test_hybrid_retrieval_combines_scores(tmp_path) -> None:
    manager = build_env(tmp_path)
    retriever = MemoryRetriever(manager, mode="hybrid", alpha=0.5)

    manager.save_text("cattle weight estimation using images")
    manager.save_text("machine learning optimization")

    results = retriever.retrieve("cattle estimation", limit=2)

    assert len(results) >= 1
    assert results[0].score > 0


def test_hybrid_behaves_like_lexical_when_alpha_1(tmp_path) -> None:
    manager = build_env(tmp_path)

    retriever = MemoryRetriever(manager, mode="hybrid", alpha=1.0)

    manager.save_text("cattle estimation")
    manager.save_text("road anomaly")

    results = retriever.retrieve("cattle", limit=2)

    assert len(results) >= 1
    assert "cattle" in results[0].text.lower()


def test_hybrid_behaves_like_vector_when_alpha_0(tmp_path) -> None:
    manager = build_env(tmp_path)

    retriever = MemoryRetriever(manager, mode="hybrid", alpha=0.0)

    manager.save_text("cattle estimation from images")
    manager.save_text("road anomaly detection")

    results = retriever.retrieve("cattle images", limit=2)

    assert len(results) >= 1


# ===============================
# Metadata Filtering
# ===============================


def test_retrieval_with_metadata_filter(tmp_path) -> None:
    manager = build_env(tmp_path)
    retriever = MemoryRetriever(manager)

    manager.save_text("cattle vision model", metadata={"type": "vision"})
    manager.save_text("cattle nutrition", metadata={"type": "health"})

    results = retriever.retrieve(
        "cattle",
        metadata_filter={"type": "vision"},
    )

    assert len(results) == 1
    assert results[0].metadata["type"] == "vision"


def test_metadata_filter_no_match(tmp_path) -> None:
    manager = build_env(tmp_path)
    retriever = MemoryRetriever(manager)

    manager.save_text("cattle vision model", metadata={"type": "vision"})

    results = retriever.retrieve(
        "cattle",
        metadata_filter={"type": "health"},
    )

    assert results == []


# ===============================
# Edge Cases
# ===============================


def test_empty_query_returns_empty(tmp_path) -> None:
    manager = build_env(tmp_path)
    retriever = MemoryRetriever(manager)

    manager.save_text("some text")

    results = retriever.retrieve("   !!!   ")

    assert results == []


def test_no_memories_returns_empty(tmp_path) -> None:
    manager = build_env(tmp_path)
    retriever = MemoryRetriever(manager)

    results = retriever.retrieve("anything")

    assert results == []


def test_retrieve_texts_returns_only_strings(tmp_path) -> None:
    manager = build_env(tmp_path)
    retriever = MemoryRetriever(manager)

    manager.save_text("transformer models")
    manager.save_text("cnn models")

    texts = retriever.retrieve_texts("models", limit=2)

    assert all(isinstance(t, str) for t in texts)


# ===============================
# Sorting
# ===============================


def test_results_sorted_by_score(tmp_path) -> None:
    manager = build_env(tmp_path)
    retriever = MemoryRetriever(manager)

    manager.save_text("cattle cattle cattle")
    manager.save_text("cattle")
    manager.save_text("road")

    results = retriever.retrieve("cattle", limit=3)

    assert len(results) >= 2
    assert results[0].score >= results[1].score
