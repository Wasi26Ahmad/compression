from __future__ import annotations

import base64
import json

import pytest

from src.compression.compressor import CompressionPackage, TextCompressor
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


def test_decompress_from_json_package() -> None:
    compressor = TextCompressor(method="zlib")
    decompressor = TextDecompressor()

    text = "JSON package restore test."
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
    )

    with pytest.raises(ValueError, match="Original length mismatch"):
        decompressor.decompress(tampered)


def test_decompress_rejects_invalid_base64_payload() -> None:
    decompressor = TextDecompressor()

    package = CompressionPackage(
        version="1.0.0",
        method="zlib",
        original_length=5,
        original_sha256="0" * 64,
        token_count=1,
        compressed_payload_b64="%%%invalid-base64%%%",
        stats={
            "original_bytes": 5,
            "compressed_bytes": 5,
            "compression_ratio": 1.0,
            "space_saving_ratio": 0.0,
            "original_token_count": 1,
        },  # type: ignore[arg-type]
        metadata={},
    )

    with pytest.raises(ValueError, match="Invalid base64 payload"):
        decompressor.decompress(package)


def test_decompress_rejects_unsupported_method() -> None:
    decompressor = TextDecompressor()

    package = CompressionPackage(
        version="1.0.0",
        method="zlib",  # temporary valid creation
        original_length=4,
        original_sha256="0" * 64,
        token_count=1,
        compressed_payload_b64=base64.b64encode(b"test").decode("ascii"),
        stats={
            "original_bytes": 4,
            "compressed_bytes": 4,
            "compression_ratio": 1.0,
            "space_saving_ratio": 0.0,
            "original_token_count": 1,
        },  # type: ignore[arg-type]
        metadata={},
    )

    hacked = json.loads(package.to_json())
    hacked["method"] = "brotli"
    hacked_package = CompressionPackage.from_dict(hacked)

    with pytest.raises(ValueError, match="Unsupported compression method"):
        decompressor.decompress(hacked_package)