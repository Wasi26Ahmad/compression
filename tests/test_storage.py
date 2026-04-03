from __future__ import annotations

import json

import pytest

from src.ccllm.compression import CompressionPackage, TextCompressor
from src.ccllm.storage import CompressionStorage, StoredCompressionRecord


def test_storage_initializes_database_file(tmp_path) -> None:
    db_path = tmp_path / "compression.db"

    storage = CompressionStorage(db_path)

    assert db_path.exists()
    assert storage.count_records() == 0


def test_save_package_returns_record(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method="zlib")

    package = compressor.compress("Hello compression")
    record = storage.save_package(package)

    assert isinstance(record, StoredCompressionRecord)
    assert isinstance(record.record_id, str)
    assert record.method == "zlib"
    assert record.original_length == len("Hello compression")
    assert record.token_count == package.token_count


def test_save_package_preserves_metadata_json(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method="dictionary")

    package = compressor.compress("alpha beta alpha beta")
    record = storage.save_package(
        package,
        metadata={"source": "unit_test", "kind": "prompt"},
    )

    parsed_metadata = json.loads(record.metadata_json)
    assert parsed_metadata["source"] == "unit_test"
    assert parsed_metadata["kind"] == "prompt"


def test_save_package_with_custom_record_id(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method="none")

    package = compressor.compress("custom id text")
    record = storage.save_package(package, record_id="my-record-id")

    assert record.record_id == "my-record-id"
    assert storage.record_exists("my-record-id") is True


def test_get_record_returns_saved_record(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method="lzma")

    package = compressor.compress("retrieve me")
    saved = storage.save_package(package)

    fetched = storage.get_record(saved.record_id)

    assert fetched is not None
    assert fetched.record_id == saved.record_id
    assert fetched.original_sha256 == saved.original_sha256
    assert fetched.package_json == saved.package_json


def test_get_record_returns_none_for_missing_id(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)

    fetched = storage.get_record("missing-id")

    assert fetched is None


def test_get_package_reconstructs_package(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method="dictionary")

    original_package = compressor.compress("go stop go stop go stop")
    saved = storage.save_package(original_package)

    loaded_package = storage.get_package(saved.record_id)

    assert loaded_package is not None
    assert isinstance(loaded_package, CompressionPackage)
    assert loaded_package == original_package


def test_get_package_returns_none_for_missing_id(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)

    loaded_package = storage.get_package("missing-id")

    assert loaded_package is None


def test_list_records_returns_saved_records(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method="zlib")

    package_a = compressor.compress("record A")
    package_b = compressor.compress("record B")

    storage.save_package(package_a, record_id="a")
    storage.save_package(package_b, record_id="b")

    records = storage.list_records()

    assert len(records) == 2
    assert all(isinstance(record, StoredCompressionRecord) for record in records)


def test_list_records_respects_limit(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method="zlib")

    for idx in range(5):
        package = compressor.compress(f"record {idx}")
        storage.save_package(package, record_id=f"id-{idx}")

    records = storage.list_records(limit=3)

    assert len(records) == 3


def test_delete_record_removes_saved_record(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method="zlib")

    package = compressor.compress("delete me")
    saved = storage.save_package(package)

    deleted = storage.delete_record(saved.record_id)

    assert deleted is True
    assert storage.get_record(saved.record_id) is None
    assert storage.record_exists(saved.record_id) is False


def test_delete_record_returns_false_for_missing_id(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)

    deleted = storage.delete_record("missing-id")

    assert deleted is False


def test_record_exists_returns_true_for_saved_record(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method="none")

    package = compressor.compress("exists")
    saved = storage.save_package(package)

    assert storage.record_exists(saved.record_id) is True


def test_record_exists_returns_false_for_missing_record(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)

    assert storage.record_exists("missing-id") is False


def test_count_records_reflects_saved_and_deleted_records(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method="dictionary")

    assert storage.count_records() == 0

    record_1 = storage.save_package(compressor.compress("alpha beta alpha beta"))
    storage.save_package(compressor.compress("gamma delta gamma delta"))

    assert storage.count_records() == 2

    storage.delete_record(record_1.record_id)

    assert storage.count_records() == 1


def test_save_package_rejects_invalid_package_type(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)

    with pytest.raises(TypeError, match="package must be a CompressionPackage"):
        storage.save_package("not-a-package")  # type: ignore[arg-type]


def test_get_record_rejects_invalid_record_id_type(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)

    with pytest.raises(TypeError, match="record_id must be a string"):
        storage.get_record(123)  # type: ignore[arg-type]


def test_get_package_rejects_invalid_record_id_type(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)

    with pytest.raises(TypeError, match="record_id must be a string"):
        storage.get_package(123)  # type: ignore[arg-type]


def test_delete_record_rejects_invalid_record_id_type(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)

    with pytest.raises(TypeError, match="record_id must be a string"):
        storage.delete_record(123)  # type: ignore[arg-type]


def test_record_exists_rejects_invalid_record_id_type(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)

    with pytest.raises(TypeError, match="record_id must be a string"):
        storage.record_exists(123)  # type: ignore[arg-type]


def test_list_records_rejects_invalid_limit_type(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)

    with pytest.raises(TypeError, match="limit must be an integer"):
        storage.list_records(limit="10")  # type: ignore[arg-type]


def test_list_records_rejects_invalid_limit_value(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)

    with pytest.raises(ValueError, match="limit must be >= 1"):
        storage.list_records(limit=0)


def test_saved_package_json_can_be_decoded(tmp_path) -> None:
    db_path = tmp_path / "compression.db"
    storage = CompressionStorage(db_path)
    compressor = TextCompressor(method="dictionary")

    package = compressor.compress("repeat me repeat me repeat me")
    record = storage.save_package(package)

    decoded = CompressionPackage.from_json(record.package_json)

    assert decoded == package
