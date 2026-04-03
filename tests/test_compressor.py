from __future__ import annotations

import json

import pytest

from src.ccllm.compression import (
    CompressionPackage,
    CompressionStats,
    TextCompressor,
)


def test_available_methods() -> None:
    compressor = TextCompressor()

    methods = compressor.available_methods()

    assert methods == ["none", "zlib", "lzma", "dictionary"]


def test_compress_returns_package() -> None:
    compressor = TextCompressor(method="zlib")
    text = "You are a helpful assistant."

    package = compressor.compress(text)

    assert isinstance(package, CompressionPackage)
    assert package.version == "1.1.0"
    assert package.method == "zlib"
    assert package.original_length == len(text)
    assert package.token_count > 0
    assert isinstance(package.stats, CompressionStats)
    assert isinstance(package.compressed_payload_b64, str)
    assert len(package.original_sha256) == 64
    assert package.dictionary is None


def test_compress_preserves_metadata() -> None:
    compressor = TextCompressor(method="zlib")
    text = "Hello world"
    metadata = {"source": "unit_test", "kind": "prompt"}

    package = compressor.compress(text, metadata=metadata)

    assert package.metadata == metadata


@pytest.mark.parametrize("method", ["none", "zlib", "lzma"])
def test_compress_supported_standard_methods(method: str) -> None:
    compressor = TextCompressor(method=method)  # type: ignore[arg-type]
    text = "Repeated text. Repeated text. Repeated text."

    package = compressor.compress(text)

    assert package.method == method
    assert package.stats.original_bytes == len(text.encode("utf-8"))
    assert package.stats.compressed_bytes >= 0
    assert package.stats.compressed_token_count is None
    assert package.dictionary is None


def test_compress_supported_dictionary_method() -> None:
    compressor = TextCompressor(
        method="dictionary",
        min_phrase_len=2,
        max_phrase_len=5,
        min_frequency=2,
    )
    text = "alpha beta alpha beta alpha beta"

    package = compressor.compress(text)

    assert package.method == "dictionary"
    assert package.stats.original_bytes == len(text.encode("utf-8"))
    assert package.stats.compressed_bytes >= 0
    assert package.stats.compressed_token_count is not None
    assert package.dictionary is not None
    assert isinstance(package.dictionary, dict)


def test_none_method_keeps_original_byte_length() -> None:
    compressor = TextCompressor(method="none")
    text = "abc 123"

    package = compressor.compress(text)

    assert package.stats.original_bytes == len(text.encode("utf-8"))
    assert package.stats.compressed_bytes == len(text.encode("utf-8"))
    assert package.stats.compression_ratio == 1.0
    assert package.stats.space_saving_ratio == 0.0
    assert package.stats.compressed_token_count is None


def test_compress_to_json_returns_valid_json() -> None:
    compressor = TextCompressor(method="zlib")
    text = "Hello JSON"

    json_output = compressor.compress_to_json(text)
    parsed = json.loads(json_output)

    assert parsed["version"] == "1.1.0"
    assert parsed["method"] == "zlib"
    assert parsed["original_length"] == len(text)
    assert parsed["dictionary"] is None


def test_compress_to_json_returns_valid_json_for_dictionary_method() -> None:
    compressor = TextCompressor(method="dictionary")
    text = "repeat me repeat me repeat me"

    json_output = compressor.compress_to_json(text)
    parsed = json.loads(json_output)

    assert parsed["version"] == "1.1.0"
    assert parsed["method"] == "dictionary"
    assert parsed["original_length"] == len(text)
    assert parsed["dictionary"] is not None
    assert parsed["stats"]["compressed_token_count"] is not None


def test_package_to_dict_and_from_dict_round_trip() -> None:
    compressor = TextCompressor(method="zlib")
    text = "Round-trip package test"

    package = compressor.compress(text)
    package_dict = package.to_dict()
    restored = CompressionPackage.from_dict(package_dict)

    assert restored == package


def test_package_to_dict_and_from_dict_round_trip_for_dictionary_method() -> None:
    compressor = TextCompressor(method="dictionary")
    text = "go go stop go go stop"

    package = compressor.compress(text)
    package_dict = package.to_dict()
    restored = CompressionPackage.from_dict(package_dict)

    assert restored == package


def test_package_to_json_and_from_json_round_trip() -> None:
    compressor = TextCompressor(method="lzma")
    text = "JSON package reconstruction"

    package = compressor.compress(text)
    package_json = package.to_json()
    restored = CompressionPackage.from_json(package_json)

    assert restored == package


def test_package_to_json_and_from_json_round_trip_for_dictionary_method() -> None:
    compressor = TextCompressor(method="dictionary")
    text = "up down up down up down"

    package = compressor.compress(text)
    package_json = package.to_json()
    restored = CompressionPackage.from_json(package_json)

    assert restored == package


def test_compress_empty_string() -> None:
    compressor = TextCompressor(method="zlib")

    package = compressor.compress("")

    assert package.original_length == 0
    assert package.token_count == 0
    assert package.stats.original_bytes == 0
    assert package.stats.compression_ratio == 1.0
    assert package.stats.space_saving_ratio == 0.0
    assert package.dictionary is None


def test_compress_empty_string_with_dictionary_method() -> None:
    compressor = TextCompressor(method="dictionary")

    package = compressor.compress("")

    assert package.original_length == 0
    assert package.token_count == 0
    assert package.stats.original_bytes == 0
    assert package.stats.compression_ratio == 1.0
    assert package.stats.space_saving_ratio == 0.0
    assert package.stats.compressed_token_count == 0
    assert package.dictionary == {}


def test_rejects_invalid_text_input() -> None:
    compressor = TextCompressor(method="zlib")

    with pytest.raises(TypeError, match="text must be a string"):
        compressor.compress(123)  # type: ignore[arg-type]


def test_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="Unsupported compression method"):
        TextCompressor(method="brotli")  # type: ignore[arg-type]


def test_rejects_invalid_zlib_level() -> None:
    with pytest.raises(ValueError, match="zlib_level must be between 0 and 9"):
        TextCompressor(method="zlib", zlib_level=10)


def test_rejects_invalid_lzma_preset() -> None:
    with pytest.raises(ValueError, match="lzma_preset must be between 0 and 9"):
        TextCompressor(method="lzma", lzma_preset=10)


def test_repeated_text_usually_compresses_with_zlib() -> None:
    compressor = TextCompressor(method="zlib")
    text = ("You are a helpful assistant. " * 200).strip()

    package = compressor.compress(text)

    assert package.stats.compressed_bytes < package.stats.original_bytes
    assert package.stats.compression_ratio < 1.0
    assert package.stats.space_saving_ratio > 0.0


def test_repeated_text_usually_compresses_with_lzma() -> None:
    compressor = TextCompressor(method="lzma")
    text = ("Compression test text. " * 200).strip()

    package = compressor.compress(text)

    assert package.stats.compressed_bytes < package.stats.original_bytes
    assert package.stats.compression_ratio < 1.0
    assert package.stats.space_saving_ratio > 0.0


def test_dictionary_method_populates_dictionary_when_repetition_exists() -> None:
    compressor = TextCompressor(
        method="dictionary",
        min_phrase_len=2,
        max_phrase_len=4,
        min_frequency=2,
    )
    text = "machine learning machine learning machine learning"

    package = compressor.compress(text)

    assert package.method == "dictionary"
    assert package.dictionary is not None
    assert len(package.dictionary) > 0


def test_dictionary_method_tracks_compressed_token_count() -> None:
    compressor = TextCompressor(
        method="dictionary",
        min_phrase_len=2,
        max_phrase_len=4,
        min_frequency=2,
    )
    text = "A B A B A B"

    package = compressor.compress(text)

    assert package.stats.compressed_token_count is not None
    assert package.stats.compressed_token_count <= package.token_count


def test_dictionary_method_can_fall_back_to_plain_token_stream() -> None:
    compressor = TextCompressor(
        method="dictionary",
        min_phrase_len=3,
        max_phrase_len=5,
        min_frequency=2,
    )
    text = "unique text only once"

    package = compressor.compress(text)

    assert package.method == "dictionary"
    assert package.dictionary == {}
    assert package.stats.compressed_token_count == package.token_count


def test_dictionary_method_preserves_metadata() -> None:
    compressor = TextCompressor(method="dictionary")
    text = "alpha alpha beta beta alpha alpha"
    metadata = {"source": "unit_test", "stage": "dictionary"}

    package = compressor.compress(text, metadata=metadata)

    assert package.metadata == metadata
    assert package.dictionary is not None


def test_sha256_is_stable_for_same_input() -> None:
    compressor = TextCompressor(method="zlib")
    text = "Stable hash input"

    package_1 = compressor.compress(text)
    package_2 = compressor.compress(text)

    assert package_1.original_sha256 == package_2.original_sha256


def test_sha256_is_stable_for_same_input_in_dictionary_method() -> None:
    compressor = TextCompressor(method="dictionary")
    text = "same same same text"

    package_1 = compressor.compress(text)
    package_2 = compressor.compress(text)

    assert package_1.original_sha256 == package_2.original_sha256
