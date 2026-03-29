from __future__ import annotations

import pytest

from src.compression.tokenizer import TextTokenizer, TokenizationResult


def test_tokenize_returns_tokenization_result() -> None:
    tokenizer = TextTokenizer()
    text = "Hello world"

    result = tokenizer.tokenize(text)

    assert isinstance(result, TokenizationResult)
    assert result.tokens == ["Hello", " ", "world"]
    assert result.token_count == 3
    assert result.original_length == len(text)


def test_tokenize_preserves_spaces_tabs_and_newlines() -> None:
    tokenizer = TextTokenizer()
    text = "Hello,\tworld!\n\nNext line\r\nEnd"

    result = tokenizer.tokenize(text)

    assert tokenizer.detokenize(result.tokens) == text
    assert "\t" in result.tokens
    assert "\n" in result.tokens
    assert "\r\n" in result.tokens


def test_tokenize_preserves_multiple_spaces() -> None:
    tokenizer = TextTokenizer()
    text = "Hello,   world!"

    result = tokenizer.tokenize(text)

    assert result.tokens == ["Hello", ",", "   ", "world", "!"]
    assert tokenizer.detokenize(result.tokens) == text


def test_tokenize_empty_string() -> None:
    tokenizer = TextTokenizer()

    result = tokenizer.tokenize("")

    assert result.tokens == []
    assert result.token_count == 0
    assert result.original_length == 0
    assert tokenizer.detokenize(result.tokens) == ""


def test_detokenize_round_trip() -> None:
    tokenizer = TextTokenizer()
    text = "Prompt: summarize this exactly.\nDo not omit anything."

    result = tokenizer.tokenize(text)
    restored = tokenizer.detokenize(result.tokens)

    assert restored == text


def test_validate_round_trip_true() -> None:
    tokenizer = TextTokenizer()
    text = "A  B\tC\nD\r\nE"

    assert tokenizer.validate_round_trip(text) is True


def test_token_lengths() -> None:
    tokenizer = TextTokenizer()
    tokens = ["Hello", " ", "world", "!"]

    lengths = tokenizer.token_lengths(tokens)

    assert lengths == [5, 1, 5, 1]


def test_tokenize_rejects_non_string_input() -> None:
    tokenizer = TextTokenizer()

    with pytest.raises(TypeError, match="text must be a string"):
        tokenizer.tokenize(123)  # type: ignore[arg-type]


def test_detokenize_rejects_invalid_container_type() -> None:
    tokenizer = TextTokenizer()

    with pytest.raises(TypeError, match="tokens must be a list or tuple of strings"):
        tokenizer.detokenize("not-a-token-list")  # type: ignore[arg-type]


def test_detokenize_rejects_non_string_tokens() -> None:
    tokenizer = TextTokenizer()

    with pytest.raises(TypeError, match="all tokens must be strings"):
        tokenizer.detokenize(["Hello", 123])  # type: ignore[list-item]


def test_token_lengths_rejects_non_string_tokens() -> None:
    tokenizer = TextTokenizer()

    with pytest.raises(TypeError, match="all tokens must be strings"):
        tokenizer.token_lengths(["a", 1])  # type: ignore[list-item]