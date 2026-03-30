from __future__ import annotations

import pytest

from src.compression import CompressionPackage, TextCompressor, TextDecompressor


@pytest.mark.parametrize("method", ["none", "zlib", "lzma", "dictionary"])
def test_end_to_end_round_trip_for_all_methods(method: str) -> None:
    compressor = TextCompressor(method=method)  # type: ignore[arg-type]
    decompressor = TextDecompressor()

    text = (
        "You are a helpful assistant.\n"
        "Summarize the following text exactly.\n"
        "Do not omit any important details."
    )

    package = compressor.compress(text)
    restored = decompressor.decompress(package)

    assert restored == text


@pytest.mark.parametrize("method", ["none", "zlib", "lzma", "dictionary"])
def test_end_to_end_json_round_trip_for_all_methods(method: str) -> None:
    compressor = TextCompressor(method=method)  # type: ignore[arg-type]
    decompressor = TextDecompressor()

    text = (
        "Prompt:\n"
        "Convert the following paragraph into bullet points.\n"
        "Keep all numeric values unchanged."
    )

    package_json = compressor.compress_to_json(text)
    restored = decompressor.decompress_from_json(package_json)

    assert restored == text


def test_dictionary_method_end_to_end_with_repeated_prompt_patterns() -> None:
    compressor = TextCompressor(
        method="dictionary",
        min_phrase_len=2,
        max_phrase_len=6,
        min_frequency=2,
    )
    decompressor = TextDecompressor()

    text = (
        "You are a helpful assistant. "
        "You are a helpful assistant. "
        "You are a helpful assistant. "
        "Answer clearly and accurately."
    )

    package = compressor.compress(text)
    restored = decompressor.decompress(package)

    assert package.method == "dictionary"
    assert package.dictionary is not None
    assert restored == text


def test_dictionary_method_json_round_trip_with_dictionary_entries() -> None:
    compressor = TextCompressor(
        method="dictionary",
        min_phrase_len=2,
        max_phrase_len=5,
        min_frequency=2,
    )
    decompressor = TextDecompressor()

    text = "alpha beta alpha beta alpha beta gamma"
    package = compressor.compress(text)

    assert package.dictionary is not None
    assert len(package.dictionary) > 0

    package_json = package.to_json()
    restored_package = CompressionPackage.from_json(package_json)
    restored_text = decompressor.decompress(restored_package)

    assert restored_text == text


def test_end_to_end_preserves_whitespace_tabs_and_newlines() -> None:
    compressor = TextCompressor(method="dictionary")
    decompressor = TextDecompressor()

    text = "Hello,\tworld!\n\nLine two.\r\nLine three.\n    Indented block."

    package = compressor.compress(text)
    restored = decompressor.decompress(package)

    assert restored == text


def test_end_to_end_with_punctuation_heavy_text() -> None:
    compressor = TextCompressor(method="dictionary")
    decompressor = TextDecompressor()

    text = (
        "System: You are an AI assistant.\n"
        "User: Explain 'lossless compression'—briefly, clearly, and exactly.\n"
        "Assistant: Sure; here it is..."
    )

    package = compressor.compress(text)
    restored = decompressor.decompress(package)

    assert restored == text


def test_end_to_end_empty_string_for_all_methods() -> None:
    decompressor = TextDecompressor()

    for method in ["none", "zlib", "lzma", "dictionary"]:
        compressor = TextCompressor(method=method)  # type: ignore[arg-type]
        package = compressor.compress("")
        restored = decompressor.decompress(package)

        assert restored == ""


def test_multiple_compress_decompress_cycles_remain_stable() -> None:
    compressor = TextCompressor(method="dictionary")
    decompressor = TextDecompressor()

    text = "repeat this sentence. repeat this sentence. repeat this sentence."

    package_1 = compressor.compress(text)
    restored_1 = decompressor.decompress(package_1)

    package_2 = compressor.compress(restored_1)
    restored_2 = decompressor.decompress(package_2)

    assert restored_1 == text
    assert restored_2 == text


def test_json_package_round_trip_remains_stable_across_multiple_cycles() -> None:
    compressor = TextCompressor(method="dictionary")
    decompressor = TextDecompressor()

    text = "Prompt template: summarize, classify, summarize, classify."

    json_1 = compressor.compress_to_json(text)
    restored_1 = decompressor.decompress_from_json(json_1)

    json_2 = compressor.compress_to_json(restored_1)
    restored_2 = decompressor.decompress_from_json(json_2)

    assert restored_1 == text
    assert restored_2 == text


def test_dictionary_method_reports_valid_token_statistics_in_end_to_end_flow() -> None:
    compressor = TextCompressor(
        method="dictionary",
        min_phrase_len=2,
        max_phrase_len=4,
        min_frequency=2,
    )
    decompressor = TextDecompressor()

    text = "machine learning machine learning machine learning"
    package = compressor.compress(text)
    restored = decompressor.decompress(package)

    assert restored == text
    assert package.stats.original_token_count == package.token_count
    assert package.stats.compressed_token_count is not None
    assert package.stats.compressed_token_count <= package.token_count


def test_standard_methods_keep_dictionary_field_none_in_end_to_end_flow() -> None:
    decompressor = TextDecompressor()
    text = "Standard compression methods should not use phrase dictionaries."

    for method in ["none", "zlib", "lzma"]:
        compressor = TextCompressor(method=method)  # type: ignore[arg-type]
        package = compressor.compress(text)
        restored = decompressor.decompress(package)

        assert restored == text
        assert package.dictionary is None
        assert package.stats.compressed_token_count is None
