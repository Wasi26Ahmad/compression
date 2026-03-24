from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence


# Lossless tokenization pattern.
# Order matters:
# 1. Windows newline
# 2. Unix newline
# 3. tabs
# 4. runs of spaces
# 5. alphanumeric/underscore words
# 6. any remaining single non-space, non-word character (punctuation/symbols)
_TOKEN_PATTERN = re.compile(
    r"\r\n|\n|\t|[ ]+|[A-Za-z0-9_]+|[^\w\s]",
    re.UNICODE,
)


@dataclass(frozen=True)
class TokenizationResult:
    tokens: List[str]
    token_count: int
    original_length: int


class TextTokenizer:
    def tokenize(self, text: str) -> TokenizationResult:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if text == "":
            return TokenizationResult(
                tokens=[],
                token_count=0,
                original_length=0,
            )

        tokens = _TOKEN_PATTERN.findall(text)
        reconstructed = "".join(tokens)

        if reconstructed != text:
            raise ValueError(
                "Lossless tokenization failed: reconstructed text does not match input."
            )

        return TokenizationResult(
            tokens=tokens,
            token_count=len(tokens),
            original_length=len(text),
        )

    def detokenize(self, tokens: Sequence[str]) -> str:
        if not isinstance(tokens, (list, tuple)):
            raise TypeError("tokens must be a list or tuple of strings")

        if not all(isinstance(token, str) for token in tokens):
            raise TypeError("all tokens must be strings")

        return "".join(tokens)

    def validate_round_trip(self, text: str) -> bool:
        result = self.tokenize(text)
        return self.detokenize(result.tokens) == text

    def token_lengths(self, tokens: Sequence[str]) -> List[int]:
        if not all(isinstance(token, str) for token in tokens):
            raise TypeError("all tokens must be strings")

        return [len(token) for token in tokens]