from __future__ import annotations

import json

import pytest

from src.compression import TextCompressor, TextDecompressor
from src.storage import CompressionStorage


@pytest.mark.parametrize("method", ["zlib", "dictionary"])
def test_end_to_end_storage_round_trip(tmp_path, method: str) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method=method)  # type: ignore[arg-type]
    decompressor = TextDecompressor()

    text = (
        "You are a helpful assistant.\n"
        "Summarize this text exactly.\n"
        "Do not omit any important details."
    )

    package = compressor.compress(text)
    record = storage.save_package(package, metadata={"source": "integration_test"})

    loaded_package = storage.get_package(record.record_id)
    assert loaded_package is not None

    restored = decompressor.decompress(loaded_package)

    assert restored == text
    assert record.method == method
    assert storage.count_records() == 1


def test_storage_round_trip_preserves_dictionary_package(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(
        method="dictionary",
        min_phrase_len=2,
        max_phrase_len=5,
        min_frequency=2,
    )
    decompressor = TextDecompressor()

    text = "alpha beta alpha beta alpha beta gamma"
    package = compressor.compress(text)
    record = storage.save_package(package)

    loaded_package = storage.get_package(record.record_id)
    assert loaded_package is not None

    restored = decompressor.decompress(loaded_package)

    assert loaded_package.method == "dictionary"
    assert loaded_package.dictionary is not None
    assert len(loaded_package.dictionary) > 0
    assert restored == text


def test_storage_round_trip_preserves_metadata_json(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method="zlib")

    text = "Metadata integration test text"
    package = compressor.compress(text)
    record = storage.save_package(
        package,
        metadata={
            "source": "integration_test",
            "category": "prompt",
            "version": 1,
        },
    )

    loaded_record = storage.get_record(record.record_id)
    assert loaded_record is not None

    metadata = json.loads(loaded_record.metadata_json)

    assert metadata["source"] == "integration_test"
    assert metadata["category"] == "prompt"
    assert metadata["version"] == 1


def test_storage_round_trip_with_json_package_reconstruction(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method="dictionary")
    decompressor = TextDecompressor()

    text = "repeat this repeat this repeat this"
    package = compressor.compress(text)
    record = storage.save_package(package)

    loaded_record = storage.get_record(record.record_id)
    assert loaded_record is not None

    restored_package_json = loaded_record.package_json
    restored_text = decompressor.decompress_from_json(restored_package_json)

    assert restored_text == text


def test_storage_multiple_records_can_be_saved_and_restored(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    decompressor = TextDecompressor()

    texts = [
        ("zlib", "First integration sample text."),
        ("dictionary", "go stop go stop go stop"),
        ("none", "Plain storage round trip."),
    ]

    saved_ids: list[str] = []

    for method, text in texts:
        compressor = TextCompressor(method=method)  # type: ignore[arg-type]
        package = compressor.compress(text)
        record = storage.save_package(package, metadata={"method": method})
        saved_ids.append(record.record_id)

    assert storage.count_records() == 3

    for (method, original_text), record_id in zip(texts, saved_ids, strict=True):
        loaded_package = storage.get_package(record_id)
        assert loaded_package is not None

        restored = decompressor.decompress(loaded_package)

        assert loaded_package.method == method
        assert restored == original_text


def test_storage_delete_breaks_restore_path(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method="zlib")

    text = "Delete path integration test"
    package = compressor.compress(text)
    record = storage.save_package(package)

    assert storage.get_package(record.record_id) is not None

    deleted = storage.delete_record(record.record_id)

    assert deleted is True
    assert storage.get_record(record.record_id) is None
    assert storage.get_package(record.record_id) is None
    assert storage.count_records() == 0
