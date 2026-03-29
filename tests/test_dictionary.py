from __future__ import annotations

import pytest

from src.compression.dictionary import DictionaryEntry, PhraseDictionaryBuilder


def test_build_returns_dictionary_entries() -> None:
    builder = PhraseDictionaryBuilder()
    tokens = ["machine", " ", "learning", " ", "machine", " ", "learning"]

    entries = builder.build(tokens)

    assert isinstance(entries, list)
    assert all(isinstance(entry, DictionaryEntry) for entry in entries)


def test_build_finds_repeated_phrase() -> None:
    builder = PhraseDictionaryBuilder(
        min_phrase_len=2,
        max_phrase_len=3,
        min_frequency=2,
    )
    tokens = ["A", " ", "B", " ", "A", " ", "B"]

    entries = builder.build(tokens)
    phrases = [entry.phrase for entry in entries]

    assert ("A", " ") in phrases or (" ", "B") in phrases or ("A", " ", "B") in phrases


def test_build_returns_empty_for_short_input() -> None:
    builder = PhraseDictionaryBuilder(min_phrase_len=3)
    tokens = ["hello", "world"]

    entries = builder.build(tokens)

    assert entries == []


def test_build_respects_max_dictionary_size() -> None:
    builder = PhraseDictionaryBuilder(
        min_phrase_len=2,
        max_phrase_len=2,
        min_frequency=2,
        max_dictionary_size=1,
    )
    tokens = ["a", "b", "a", "b", "c", "d", "c", "d"]

    entries = builder.build(tokens)

    assert len(entries) == 1


def test_build_lookup_returns_phrase_to_id_mapping() -> None:
    builder = PhraseDictionaryBuilder()
    tokens = ["x", "y", "x", "y"]

    entries = builder.build(tokens)
    lookup = builder.build_lookup(entries)

    for entry in entries:
        assert lookup[entry.phrase] == entry.phrase_id


def test_build_reverse_lookup_returns_id_to_phrase_mapping() -> None:
    builder = PhraseDictionaryBuilder()
    tokens = ["x", "y", "x", "y"]

    entries = builder.build(tokens)
    reverse_lookup = builder.build_reverse_lookup(entries)

    for entry in entries:
        assert reverse_lookup[entry.phrase_id] == list(entry.phrase)


def test_skip_all_whitespace_phrases() -> None:
    builder = PhraseDictionaryBuilder(
        min_phrase_len=2,
        max_phrase_len=2,
        min_frequency=2,
        skip_all_whitespace_phrases=True,
    )
    tokens = [" ", " ", " ", " "]

    entries = builder.build(tokens)

    assert entries == []


def test_rejects_invalid_tokens_type() -> None:
    builder = PhraseDictionaryBuilder()

    with pytest.raises(TypeError, match="tokens must be a list or tuple of strings"):
        builder.build("not-a-token-list")  # type: ignore[arg-type]


def test_rejects_non_string_tokens() -> None:
    builder = PhraseDictionaryBuilder()

    with pytest.raises(TypeError, match="all tokens must be strings"):
        builder.build(["hello", 1, "world"])  # type: ignore[list-item]


def test_rejects_invalid_min_phrase_len() -> None:
    with pytest.raises(ValueError, match="min_phrase_len must be >= 2"):
        PhraseDictionaryBuilder(min_phrase_len=1)


def test_rejects_invalid_max_phrase_len() -> None:
    with pytest.raises(ValueError, match="max_phrase_len must be >= min_phrase_len"):
        PhraseDictionaryBuilder(min_phrase_len=4, max_phrase_len=3)


def test_rejects_invalid_min_frequency() -> None:
    with pytest.raises(ValueError, match="min_frequency must be >= 2"):
        PhraseDictionaryBuilder(min_frequency=1)


def test_rejects_invalid_max_dictionary_size() -> None:
    with pytest.raises(ValueError, match="max_dictionary_size must be >= 1"):
        PhraseDictionaryBuilder(max_dictionary_size=0)


def test_rejects_invalid_min_estimated_savings() -> None:
    with pytest.raises(ValueError, match="min_estimated_savings must be >= 1"):
        PhraseDictionaryBuilder(min_estimated_savings=0)
