from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Phrase = tuple[str, ...]


@dataclass(frozen=True)
class PhraseStats:
    phrase: Phrase
    frequency: int
    phrase_length: int
    estimated_savings: int


@dataclass(frozen=True)
class DictionaryEntry:
    phrase_id: str
    phrase: Phrase
    frequency: int
    estimated_savings: int


class PhraseDictionaryBuilder:
    """
    Build a reusable phrase dictionary from a token stream.

    Goal:
    - find repeated multi-token phrases
    - keep only phrases that are likely to improve compression
    - return deterministic, ranked dictionary entries

    Notes:
    - this module only discovers and ranks phrases
    - actual substitution is handled later in compressor.py
    - phrases are token-based, not character-based
    """

    def __init__(
        self,
        min_phrase_len: int = 2,
        max_phrase_len: int = 8,
        min_frequency: int = 2,
        max_dictionary_size: int = 256,
        min_estimated_savings: int = 1,
        skip_all_whitespace_phrases: bool = True,
    ) -> None:
        if min_phrase_len < 2:
            raise ValueError("min_phrase_len must be >= 2")
        if max_phrase_len < min_phrase_len:
            raise ValueError("max_phrase_len must be >= min_phrase_len")
        if min_frequency < 2:
            raise ValueError("min_frequency must be >= 2")
        if max_dictionary_size < 1:
            raise ValueError("max_dictionary_size must be >= 1")
        if min_estimated_savings < 1:
            raise ValueError("min_estimated_savings must be >= 1")

        self.min_phrase_len = min_phrase_len
        self.max_phrase_len = max_phrase_len
        self.min_frequency = min_frequency
        self.max_dictionary_size = max_dictionary_size
        self.min_estimated_savings = min_estimated_savings
        self.skip_all_whitespace_phrases = skip_all_whitespace_phrases

    def build(self, tokens: Sequence[str]) -> list[DictionaryEntry]:
        self._validate_tokens(tokens)

        if len(tokens) < self.min_phrase_len:
            return []

        phrase_counts = self._count_phrases(tokens)
        candidates = self._score_candidates(phrase_counts)
        selected = self._select_entries(candidates)

        return selected

    def build_lookup(self, entries: Sequence[DictionaryEntry]) -> dict[Phrase, str]:
        lookup: dict[Phrase, str] = {}

        for entry in entries:
            lookup[entry.phrase] = entry.phrase_id

        return lookup

    def build_reverse_lookup(
        self, entries: Sequence[DictionaryEntry]
    ) -> dict[str, list[str]]:
        reverse_lookup: dict[str, list[str]] = {}

        for entry in entries:
            reverse_lookup[entry.phrase_id] = list(entry.phrase)

        return reverse_lookup

    def _count_phrases(self, tokens: Sequence[str]) -> dict[Phrase, int]:
        counts: dict[Phrase, int] = {}
        n_tokens = len(tokens)

        for phrase_len in range(self.min_phrase_len, self.max_phrase_len + 1):
            for start_idx in range(0, n_tokens - phrase_len + 1):
                phrase = tuple(tokens[start_idx : start_idx + phrase_len])

                if self.skip_all_whitespace_phrases and self._is_all_whitespace_phrase(
                    phrase
                ):
                    continue

                counts[phrase] = counts.get(phrase, 0) + 1

        return counts

    def _score_candidates(self, phrase_counts: dict[Phrase, int]) -> list[PhraseStats]:
        candidates: list[PhraseStats] = []

        for phrase, frequency in phrase_counts.items():
            if frequency < self.min_frequency:
                continue

            phrase_length = len(phrase)
            estimated_savings = self._estimate_savings(
                phrase_length=phrase_length,
                frequency=frequency,
            )

            if estimated_savings < self.min_estimated_savings:
                continue

            candidates.append(
                PhraseStats(
                    phrase=phrase,
                    frequency=frequency,
                    phrase_length=phrase_length,
                    estimated_savings=estimated_savings,
                )
            )

        candidates.sort(
            key=lambda item: (
                item.estimated_savings,
                item.phrase_length,
                item.frequency,
                item.phrase,
            ),
            reverse=True,
        )

        return candidates

    def _select_entries(
        self, candidates: Sequence[PhraseStats]
    ) -> list[DictionaryEntry]:
        selected: list[DictionaryEntry] = []
        selected_phrases: set[Phrase] = set()

        for candidate in candidates:
            if len(selected) >= self.max_dictionary_size:
                break

            if candidate.phrase in selected_phrases:
                continue

            phrase_id = self._make_phrase_id(len(selected))

            selected.append(
                DictionaryEntry(
                    phrase_id=phrase_id,
                    phrase=candidate.phrase,
                    frequency=candidate.frequency,
                    estimated_savings=candidate.estimated_savings,
                )
            )
            selected_phrases.add(candidate.phrase)

        return selected

    @staticmethod
    def _estimate_savings(phrase_length: int, frequency: int) -> int:
        return (frequency * (phrase_length - 1)) - phrase_length

    @staticmethod
    def _is_all_whitespace_phrase(phrase: Sequence[str]) -> bool:
        return all(token.isspace() for token in phrase)

    @staticmethod
    def _make_phrase_id(index: int) -> str:
        return f"@P{index}"

    @staticmethod
    def _validate_tokens(tokens: Sequence[str]) -> None:
        if not isinstance(tokens, list | tuple):
            raise TypeError("tokens must be a list or tuple of strings")

        if not all(isinstance(token, str) for token in tokens):
            raise TypeError("all tokens must be strings")
