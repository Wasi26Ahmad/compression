from __future__ import annotations

import pytest

from src.memory import MemoryManager
from src.retrieval import MemoryRetriever
from src.storage import CompressionStorage

# ===============================
# Full Pipeline Test (All Modes)
# ===============================


@pytest.mark.parametrize("mode", ["lexical", "vector", "hybrid"])
def test_full_pipeline_retrieval_modes(tmp_path, mode: str) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager, mode=mode)

    # Store diverse texts
    manager.save_text(
        "Cattle weight estimation using side-view and rear-view images",
        metadata={"domain": "cattle"},
    )
    manager.save_text(
        "Road anomaly detection using accelerometer and gyroscope",
        metadata={"domain": "road"},
    )
    manager.save_text(
        "Deep learning models for medical image analysis",
        metadata={"domain": "medical"},
    )

    results = retriever.retrieve("cattle image estimation", limit=2)

    assert len(results) >= 1
    assert "cattle" in results[0].text.lower()


# ===============================
# Metadata + Retrieval Integration
# ===============================


def test_retrieval_with_metadata_filter_integration(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager, mode="hybrid")

    manager.save_text(
        "Cattle vision model using convolutional neural networks",
        metadata={"type": "vision"},
    )
    manager.save_text(
        "Cattle nutrition and feeding strategies",
        metadata={"type": "health"},
    )

    results = retriever.retrieve(
        "cattle",
        metadata_filter={"type": "vision"},
    )

    assert len(results) == 1
    assert results[0].metadata["type"] == "vision"


# ===============================
# Compression + Retrieval Compatibility
# ===============================


@pytest.mark.parametrize("method", ["zlib", "lzma", "dictionary"])
def test_retrieval_works_with_all_compression_methods(tmp_path, method: str) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager, mode="hybrid")

    manager.save_text(
        "Cattle weight estimation from images",
        method=method,
    )
    manager.save_text(
        "Road anomaly detection",
        method=method,
    )

    results = retriever.retrieve("cattle estimation", limit=2)

    assert len(results) >= 1
    assert "cattle" in results[0].text.lower()


# ===============================
# Large Batch Integration
# ===============================


def test_retrieval_with_multiple_records(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    for i in range(20):
        manager.save_text(f"Cattle data sample {i}")

    results = retriever.retrieve("cattle data", limit=5)

    assert len(results) <= 5
    assert len(results) > 0


# ===============================
# Deletion Impact on Retrieval
# ===============================


def test_deleted_records_are_not_retrieved(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    record = manager.save_text("Cattle important record")
    manager.save_text("Another cattle record")

    # Delete one
    manager.delete_memory(record.record_id)

    results = retriever.retrieve("cattle", limit=5)

    # Ensure deleted record not present
    assert all(r.record_id != record.record_id for r in results)


# ===============================
# Stability Across Multiple Calls
# ===============================


def test_retrieval_consistency_across_multiple_calls(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)
    retriever = MemoryRetriever(manager)

    manager.save_text("Cattle weight estimation system")

    results1 = retriever.retrieve("cattle estimation")
    results2 = retriever.retrieve("cattle estimation")

    assert results1[0].text == results2[0].text
    assert abs(results1[0].score - results2[0].score) < 1e-6


# ===============================
# Hybrid Scoring Behavior
# ===============================


def test_hybrid_score_differs_from_pure_modes(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    manager.save_text("cattle cattle cattle estimation")
    manager.save_text("cattle estimation")

    lexical = MemoryRetriever(manager, mode="lexical")
    vector = MemoryRetriever(manager, mode="vector")
    hybrid = MemoryRetriever(manager, mode="hybrid", alpha=0.5)

    l_res = lexical.retrieve("cattle estimation", limit=1)
    v_res = vector.retrieve("cattle estimation", limit=1)
    h_res = hybrid.retrieve("cattle estimation", limit=1)

    assert len(l_res) >= 1
    assert len(v_res) >= 1
    assert len(h_res) >= 1

    # Hybrid score should be between lexical and vector
    assert (
        min(l_res[0].score, v_res[0].score)
        <= h_res[0].score
        <= max(l_res[0].score, v_res[0].score)
    )
