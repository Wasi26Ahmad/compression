from __future__ import annotations

import pytest

from src.ccllm.compression import CompressionPackage
from src.ccllm.memory import MemoryManager
from src.ccllm.storage import CompressionStorage, StoredCompressionRecord


def test_memory_manager_rejects_invalid_storage() -> None:
    with pytest.raises(TypeError, match="storage must be a CompressionStorage"):
        MemoryManager(storage="not-storage")  # type: ignore[arg-type]


def test_save_text_returns_stored_record(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage, default_method="zlib")

    record = manager.save_text("Hello memory manager")

    assert isinstance(record, StoredCompressionRecord)
    assert isinstance(record.record_id, str)
    assert record.method == "zlib"


def test_save_text_with_explicit_method(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage, default_method="zlib")

    record = manager.save_text(
        text="alpha beta alpha beta",
        method="dictionary",
    )

    assert record.method == "dictionary"


def test_save_text_rejects_invalid_text(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    with pytest.raises(TypeError, match="text must be a string"):
        manager.save_text(123)  # type: ignore[arg-type]


def test_get_record_returns_saved_record(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    saved = manager.save_text("retrieve record")

    loaded = manager.get_record(saved.record_id)

    assert loaded is not None
    assert loaded.record_id == saved.record_id


def test_get_record_returns_none_for_missing_id(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    assert manager.get_record("missing-id") is None


def test_get_package_returns_reconstructed_package(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage, default_method="dictionary")

    saved = manager.save_text("go stop go stop go stop")
    package = manager.get_package(saved.record_id)

    assert package is not None
    assert isinstance(package, CompressionPackage)
    assert package.method == "dictionary"


def test_get_package_returns_none_for_missing_id(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    assert manager.get_package("missing-id") is None


def test_get_text_restores_original_text(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage, default_method="zlib")

    text = "You are a helpful assistant."
    saved = manager.save_text(text)

    restored = manager.get_text(saved.record_id)

    assert restored == text


def test_get_text_restores_original_text_for_dictionary_method(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage, default_method="dictionary")

    text = "alpha beta alpha beta alpha beta"
    saved = manager.save_text(
        text,
        compressor_kwargs={
            "min_phrase_len": 2,
            "max_phrase_len": 5,
            "min_frequency": 2,
        },
    )

    restored = manager.get_text(saved.record_id)

    assert restored == text


def test_get_text_returns_none_for_missing_id(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    assert manager.get_text("missing-id") is None


def test_list_memories_returns_saved_records(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    manager.save_text("memory 1")
    manager.save_text("memory 2")

    records = manager.list_memories()

    assert len(records) == 2
    assert all(isinstance(record, StoredCompressionRecord) for record in records)


def test_list_memories_respects_limit(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    for idx in range(5):
        manager.save_text(f"memory {idx}")

    records = manager.list_memories(limit=3)

    assert len(records) == 3


def test_delete_memory_removes_record(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    saved = manager.save_text("delete this memory")

    deleted = manager.delete_memory(saved.record_id)

    assert deleted is True
    assert manager.get_record(saved.record_id) is None
    assert manager.memory_exists(saved.record_id) is False


def test_delete_memory_returns_false_for_missing_id(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    assert manager.delete_memory("missing-id") is False


def test_memory_exists_returns_true_for_saved_record(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    saved = manager.save_text("exists")

    assert manager.memory_exists(saved.record_id) is True


def test_memory_exists_returns_false_for_missing_record(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    assert manager.memory_exists("missing-id") is False


def test_count_memories_reflects_saved_records(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    assert manager.count_memories() == 0

    saved = manager.save_text("first")
    manager.save_text("second")

    assert manager.count_memories() == 2

    manager.delete_memory(saved.record_id)

    assert manager.count_memories() == 1


def test_export_record_bundle_returns_complete_data(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage, default_method="dictionary")

    text = "repeat repeat repeat repeat"
    saved = manager.save_text(
        text,
        metadata={"source": "unit_test", "kind": "memory"},
    )

    bundle = manager.export_record_bundle(saved.record_id)

    assert bundle is not None
    assert "record" in bundle
    assert "metadata" in bundle
    assert "package" in bundle
    assert bundle["record"]["record_id"] == saved.record_id
    assert bundle["metadata"]["source"] == "unit_test"
    assert bundle["metadata"]["kind"] == "memory"
    assert bundle["package"]["method"] == "dictionary"


def test_export_record_bundle_returns_none_for_missing_id(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    assert manager.export_record_bundle("missing-id") is None


def test_restore_all_texts_returns_restored_items(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    manager.save_text("text one")
    manager.save_text("text two")

    restored_items = manager.restore_all_texts()

    assert len(restored_items) == 2
    assert all("record_id" in item for item in restored_items)
    assert all("method" in item for item in restored_items)
    assert all("created_at" in item for item in restored_items)
    assert all("text" in item for item in restored_items)


def test_restore_all_texts_respects_limit(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    for idx in range(5):
        manager.save_text(f"text {idx}")

    restored_items = manager.restore_all_texts(limit=2)

    assert len(restored_items) == 2


def test_save_text_preserves_metadata_in_export_bundle(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    saved = manager.save_text(
        text="metadata test",
        metadata={"source": "test_case", "category": "prompt"},
    )

    bundle = manager.export_record_bundle(saved.record_id)

    assert bundle is not None
    assert bundle["metadata"]["source"] == "test_case"
    assert bundle["metadata"]["category"] == "prompt"


def test_save_text_with_custom_record_id(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    saved = manager.save_text(
        text="custom id memory",
        record_id="custom-memory-id",
    )

    assert saved.record_id == "custom-memory-id"
    assert manager.memory_exists("custom-memory-id") is True


def test_save_text_with_compressor_kwargs_uses_dictionary_options(tmp_path) -> None:
    storage = CompressionStorage(tmp_path / "compression.db")
    manager = MemoryManager(storage=storage)

    text = "A B A B A B"
    saved = manager.save_text(
        text=text,
        method="dictionary",
        compressor_kwargs={
            "min_phrase_len": 2,
            "max_phrase_len": 4,
            "min_frequency": 2,
        },
    )

    package = manager.get_package(saved.record_id)

    assert package is not None
    assert package.method == "dictionary"
    assert package.dictionary is not None
