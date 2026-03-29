from __future__ import annotations

import base64
import json

import pytest

from src.compression.compressor import (
    CompressionPackage,
    CompressionStats,
    TextCompressor,
)
from src.compression.decompressor import TextDecompressor


def test_decompress_round_trip_with_zlib() -> None:
    compressor = TextCompressor(method="zlib")
    decompressor = TextDecompressor()

    text = "You are a helpful assistant.\nYou are a helpful assistant."
    package = compressor.compress(text)
    restored = decompressor.decompress(package)

    assert restored == text


def test_decompress_round_trip_with_lzma() -> None:
    compressor = TextCompressor(method="lzma")
    decompressor = TextDecompressor()

    text = "Compression test. Compression test. Compression test."
    package = compressor.compress(text)
    restored = decompressor.decompress(package)

    assert restored == text


def test_decompress_round_trip_with_none_method() -> None:
    compressor = TextCompressor(method="none")
    decompressor = TextDecompressor()

    text = "Plain text with no compression."
    package = compressor.compress(text)
    restored = decompressor.decompress(package)

    assert restored == text


def test_decompress_round_trip_with_dictionary_method() -> None:
    compressor = TextCompressor(
        method="dictionary",
        min_phrase_len=2,
        max_phrase_len=5,
        min_frequency=2,
    )
    decompressor = TextDecompressor()

    text = (
        "You are a helpful assistant. "
        "You are a helpful assistant. "
        "You are a helpful assistant."
    )
    package = compressor.compress(text)
    restored = decompressor.decompress(package)

    assert package.method == "dictionary"
    assert package.dictionary is not None
    assert restored == text


def test_decompress_from_json_package() -> None:
    compressor = TextCompressor(method="zlib")
    decompressor = TextDecompressor()

    text = "JSON package restore test."
    package_json = compressor.compress_to_json(text)
    restored = decompressor.decompress_from_json(package_json)

    assert restored == text


def test_decompress_from_json_package_with_dictionary_method() -> None:
    compressor = TextCompressor(method="dictionary")
    decompressor = TextDecompressor()

    text = "Alpha beta alpha beta alpha beta"
    package_json = compressor.compress_to_json(text)
    restored = decompressor.decompress_from_json(package_json)

    assert restored == text


def test_decompress_empty_string() -> None:
    compressor = TextCompressor(method="zlib")
    decompressor = TextDecompressor()

    package = compressor.compress("")
    restored = decompressor.decompress(package)

    assert restored == ""


def test_decompress_preserves_whitespace_and_newlines() -> None:
    compressor = TextCompressor(method="zlib")
    decompressor = TextDecompressor()

    text = "Hello,\tworld!\n\nNext line\r\nEnd"
    package = compressor.compress(text)
    restored = decompressor.decompress(package)

    assert restored == text


def test_decompress_dictionary_preserves_whitespace_and_newlines() -> None:
    compressor = TextCompressor(
        method="dictionary",
        min_phrase_len=2,
        max_phrase_len=4,
        min_frequency=2,
    )
    decompressor = TextDecompressor()

    text = "Hello,\tworld!\nHello,\tworld!\nHello,\tworld!"
    package = compressor.compress(text)
    restored = decompressor.decompress(package)

    assert restored == text


def test_decompress_rejects_invalid_package_type() -> None:
    decompressor = TextDecompressor()

    with pytest.raises(TypeError, match="package must be a CompressionPackage"):
        decompressor.decompress("not-a-package")  # type: ignore[arg-type]


def test_decompress_rejects_invalid_json() -> None:
    decompressor = TextDecompressor()

    with pytest.raises(ValueError, match="Invalid JSON input for CompressionPackage"):
        decompressor.decompress_from_json("{invalid json")


def test_decompress_detects_sha256_mismatch() -> None:
    compressor = TextCompressor(method="zlib")
    decompressor = TextDecompressor()

    text = "Integrity check text."
    package = compressor.compress(text)

    tampered = CompressionPackage(
        version=package.version,
        method=package.method,
        original_length=package.original_length,
        original_sha256="0" * 64,
        token_count=package.token_count,
        compressed_payload_b64=package.compressed_payload_b64,
        stats=package.stats,
        metadata=package.metadata,
        dictionary=package.dictionary,
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        decompressor.decompress(tampered)


def test_decompress_dictionary_detects_sha256_mismatch() -> None:
    compressor = TextCompressor(method="dictionary")
    decompressor = TextDecompressor()

    text = "repeat repeat repeat repeat"
    package = compressor.compress(text)

    tampered = CompressionPackage(
        version=package.version,
        method=package.method,
        original_length=package.original_length,
        original_sha256="0" * 64,
        token_count=package.token_count,
        compressed_payload_b64=package.compressed_payload_b64,
        stats=package.stats,
        metadata=package.metadata,
        dictionary=package.dictionary,
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        decompressor.decompress(tampered)


def test_decompress_detects_length_mismatch() -> None:
    compressor = TextCompressor(method="zlib")
    decompressor = TextDecompressor()

    text = "Length check text."
    package = compressor.compress(text)

    tampered = CompressionPackage(
        version=package.version,
        method=package.method,
        original_length=999,
        original_sha256=package.original_sha256,
        token_count=package.token_count,
        compressed_payload_b64=package.compressed_payload_b64,
        stats=package.stats,
        metadata=package.metadata,
        dictionary=package.dictionary,
    )

    with pytest.raises(ValueError, match="Original length mismatch"):
        decompressor.decompress(tampered)


def test_decompress_dictionary_detects_length_mismatch() -> None:
    compressor = TextCompressor(method="dictionary")
    decompressor = TextDecompressor()

    text = "one two one two one two"
    package = compressor.compress(text)

    tampered = CompressionPackage(
        version=package.version,
        method=package.method,
        original_length=999,
        original_sha256=package.original_sha256,
        token_count=package.token_count,
        compressed_payload_b64=package.compressed_payload_b64,
        stats=package.stats,
        metadata=package.metadata,
        dictionary=package.dictionary,
    )

    with pytest.raises(ValueError, match="Original length mismatch"):
        decompressor.decompress(tampered)


def test_decompress_rejects_invalid_base64_payload() -> None:
    decompressor = TextDecompressor()

    package = CompressionPackage(
        version="1.1.0",
        method="zlib",
        original_length=5,
        original_sha256="0" * 64,
        token_count=1,
        compressed_payload_b64="%%%invalid-base64%%%",
        stats=CompressionStats(
            original_bytes=5,
            compressed_bytes=5,
            compression_ratio=1.0,
            space_saving_ratio=0.0,
            original_token_count=1,
            compressed_token_count=None,
        ),
        metadata={},
        dictionary=None,
    )

    with pytest.raises(ValueError, match="Invalid base64 payload"):
        decompressor.decompress(package)


def test_decompress_rejects_unsupported_method() -> None:
    decompressor = TextDecompressor()

    package = CompressionPackage(
        version="1.1.0",
        method="zlib",
        original_length=4,
        original_sha256="0" * 64,
        token_count=1,
        compressed_payload_b64=base64.b64encode(b"test").decode("ascii"),
        stats=CompressionStats(
            original_bytes=4,
            compressed_bytes=4,
            compression_ratio=1.0,
            space_saving_ratio=0.0,
            original_token_count=1,
            compressed_token_count=None,
        ),
        metadata={},
        dictionary=None,
    )

    hacked = json.loads(package.to_json())
    hacked["method"] = "brotli"
    hacked_package = CompressionPackage.from_dict(hacked)

    with pytest.raises(ValueError, match="Unsupported compression method"):
        decompressor.decompress(hacked_package)


def test_dictionary_package_missing_dictionary_data() -> None:
    compressor = TextCompressor(method="dictionary")
    decompressor = TextDecompressor()

    text = "alpha beta alpha beta"
    package = compressor.compress(text)

    tampered = CompressionPackage(
        version=package.version,
        method="dictionary",
        original_length=package.original_length,
        original_sha256=package.original_sha256,
        token_count=package.token_count,
        compressed_payload_b64=package.compressed_payload_b64,
        stats=package.stats,
        metadata=package.metadata,
        dictionary=None,
    )

    with pytest.raises(
        ValueError,
        match="Dictionary compression package is missing dictionary data",
    ):
        decompressor.decompress(tampered)


def test_dictionary_package_rejects_invalid_payload_json() -> None:
    decompressor = TextDecompressor()

    invalid_json_payload = base64.b64encode(b"{not valid json").decode("ascii")

    package = CompressionPackage(
        version="1.1.0",
        method="dictionary",
        original_length=5,
        original_sha256="0" * 64,
        token_count=1,
        compressed_payload_b64=invalid_json_payload,
        stats=CompressionStats(
            original_bytes=5,
            compressed_bytes=5,
            compression_ratio=1.0,
            space_saving_ratio=0.0,
            original_token_count=1,
            compressed_token_count=1,
        ),
        metadata={},
        dictionary={"@P0": ["hello"]},
    )

    with pytest.raises(ValueError, match="Invalid dictionary payload JSON"):
        decompressor.decompress(package)


def test_dictionary_package_rejects_non_list_payload() -> None:
    decompressor = TextDecompressor()

    invalid_payload = base64.b64encode(json.dumps({"a": 1}).encode("utf-8")).decode(
        "ascii"
    )

    package = CompressionPackage(
        version="1.1.0",
        method="dictionary",
        original_length=5,
        original_sha256="0" * 64,
        token_count=1,
        compressed_payload_b64=invalid_payload,
        stats=CompressionStats(
            original_bytes=5,
            compressed_bytes=5,
            compression_ratio=1.0,
            space_saving_ratio=0.0,
            original_token_count=1,
            compressed_token_count=1,
        ),
        metadata={},
        dictionary={"@P0": ["hello"]},
    )

    with pytest.raises(
        ValueError,
        match="Dictionary payload must decode to a list of tokens",
    ):
        decompressor.decompress(package)


def test_dictionary_package_rejects_non_string_tokens_in_payload() -> None:
    decompressor = TextDecompressor()

    invalid_payload = base64.b64encode(
        json.dumps(["hello", 123]).encode("utf-8")
    ).decode("ascii")

    package = CompressionPackage(
        version="1.1.0",
        method="dictionary",
        original_length=5,
        original_sha256="0" * 64,
        token_count=1,
        compressed_payload_b64=invalid_payload,
        stats=CompressionStats(
            original_bytes=5,
            compressed_bytes=5,
            compression_ratio=1.0,
            space_saving_ratio=0.0,
            original_token_count=1,
            compressed_token_count=2,
        ),
        metadata={},
        dictionary={"@P0": ["hello"]},
    )

    with pytest.raises(
        ValueError,
        match="Dictionary payload tokens must all be strings",
    ):
        decompressor.decompress(package)


def test_dictionary_package_rejects_non_dict_dictionary() -> None:
    decompressor = TextDecompressor()

    payload = base64.b64encode(json.dumps(["@P0"]).encode("utf-8")).decode("ascii")

    package = CompressionPackage(
        version="1.1.0",
        method="dictionary",
        original_length=5,
        original_sha256="0" * 64,
        token_count=1,
        compressed_payload_b64=payload,
        stats=CompressionStats(
            original_bytes=5,
            compressed_bytes=5,
            compression_ratio=1.0,
            space_saving_ratio=0.0,
            original_token_count=1,
            compressed_token_count=1,
        ),
        metadata={},
        dictionary="not-a-dict",  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError, match="dictionary must be a dict\\[str, list\\[str\\]\\]"
    ):
        decompressor.decompress(package)


def test_dictionary_package_rejects_non_string_phrase_id() -> None:
    decompressor = TextDecompressor()

    payload = base64.b64encode(json.dumps(["x"]).encode("utf-8")).decode("ascii")

    package = CompressionPackage(
        version="1.1.0",
        method="dictionary",
        original_length=1,
        original_sha256="0" * 64,
        token_count=1,
        compressed_payload_b64=payload,
        stats=CompressionStats(
            original_bytes=1,
            compressed_bytes=1,
            compression_ratio=1.0,
            space_saving_ratio=0.0,
            original_token_count=1,
            compressed_token_count=1,
        ),
        metadata={},
        dictionary={1: ["x"]},  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="Dictionary phrase IDs must be strings"):
        decompressor.decompress(package)


def test_dictionary_package_rejects_non_list_phrase_value() -> None:
    decompressor = TextDecompressor()

    payload = base64.b64encode(json.dumps(["@P0"]).encode("utf-8")).decode("ascii")

    package = CompressionPackage(
        version="1.1.0",
        method="dictionary",
        original_length=1,
        original_sha256="0" * 64,
        token_count=1,
        compressed_payload_b64=payload,
        stats=CompressionStats(
            original_bytes=1,
            compressed_bytes=1,
            compression_ratio=1.0,
            space_saving_ratio=0.0,
            original_token_count=1,
            compressed_token_count=1,
        ),
        metadata={},
        dictionary={"@P0": "x"},  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match="Dictionary phrase values must be lists of strings",
    ):
        decompressor.decompress(package)


def test_dictionary_package_rejects_non_string_items_in_phrase_value() -> None:
    decompressor = TextDecompressor()

    payload = base64.b64encode(json.dumps(["@P0"]).encode("utf-8")).decode("ascii")

    package = CompressionPackage(
        version="1.1.0",
        method="dictionary",
        original_length=1,
        original_sha256="0" * 64,
        token_count=1,
        compressed_payload_b64=payload,
        stats=CompressionStats(
            original_bytes=1,
            compressed_bytes=1,
            compression_ratio=1.0,
            space_saving_ratio=0.0,
            original_token_count=1,
            compressed_token_count=1,
        ),
        metadata={},
        dictionary={"@P0": ["x", 1]},  # type: ignore[list-item]
    )

    with pytest.raises(
        ValueError,
        match="Dictionary phrase values must contain only strings",
    ):
        decompressor.decompress(package)


def test_dictionary_package_rejects_invalid_dictionary_entry_during_expansion() -> None:
    decompressor = TextDecompressor()

    payload = base64.b64encode(json.dumps(["@P0"]).encode("utf-8")).decode("ascii")

    package = CompressionPackage(
        version="1.1.0",
        method="dictionary",
        original_length=1,
        original_sha256="0" * 64,
        token_count=1,
        compressed_payload_b64=payload,
        stats=CompressionStats(
            original_bytes=1,
            compressed_bytes=1,
            compression_ratio=1.0,
            space_saving_ratio=0.0,
            original_token_count=1,
            compressed_token_count=1,
        ),
        metadata={},
        dictionary={"@P0": ["x"]},
    )

    # valid structure, but hash is intentionally wrong, so expansion happens first
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        decompressor.decompress(package)
