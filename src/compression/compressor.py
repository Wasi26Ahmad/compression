from __future__ import annotations

import base64
import hashlib
import json
import lzma
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Literal

from src.compression.dictionary import DictionaryEntry, PhraseDictionaryBuilder
from src.compression.tokenizer import TextTokenizer

CompressionMethod = Literal["none", "zlib", "lzma", "dictionary"]


@dataclass(frozen=True)
class CompressionStats:
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    space_saving_ratio: float
    original_token_count: int
    compressed_token_count: int | None = None


@dataclass(frozen=True)
class CompressionPackage:
    version: str
    method: CompressionMethod
    original_length: int
    original_sha256: str
    token_count: int
    compressed_payload_b64: str
    stats: CompressionStats
    metadata: dict[str, Any]
    dictionary: dict[str, list[str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompressionPackage:
        stats = CompressionStats(**data["stats"])
        return cls(
            version=data["version"],
            method=data["method"],
            original_length=data["original_length"],
            original_sha256=data["original_sha256"],
            token_count=data["token_count"],
            compressed_payload_b64=data["compressed_payload_b64"],
            stats=stats,
            metadata=data.get("metadata", {}),
            dictionary=data.get("dictionary"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> CompressionPackage:
        return cls.from_dict(json.loads(json_str))


class TextCompressor:
    """
    Baseline + custom dictionary text compressor.

    Supported methods:
    - none
    - zlib
    - lzma
    - dictionary

    The dictionary method is token-aware:
    - tokenize text losslessly
    - build repeated phrase dictionary
    - replace repeated phrases with phrase IDs
    - serialize encoded token stream as UTF-8 JSON
    """

    SUPPORTED_METHODS: tuple[CompressionMethod, ...] = (
        "none",
        "zlib",
        "lzma",
        "dictionary",
    )

    def __init__(
        self,
        method: CompressionMethod = "zlib",
        zlib_level: int = 9,
        lzma_preset: int = 6,
        min_phrase_len: int = 2,
        max_phrase_len: int = 8,
        min_frequency: int = 2,
        max_dictionary_size: int = 256,
        min_estimated_savings: int = 1,
        skip_all_whitespace_phrases: bool = True,
    ) -> None:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported compression method: {method}. "
                f"Supported: {self.SUPPORTED_METHODS}"
            )

        if not 0 <= zlib_level <= 9:
            raise ValueError("zlib_level must be between 0 and 9")

        if not 0 <= lzma_preset <= 9:
            raise ValueError("lzma_preset must be between 0 and 9")

        self.method = method
        self.zlib_level = zlib_level
        self.lzma_preset = lzma_preset
        self.tokenizer = TextTokenizer()
        self.dictionary_builder = PhraseDictionaryBuilder(
            min_phrase_len=min_phrase_len,
            max_phrase_len=max_phrase_len,
            min_frequency=min_frequency,
            max_dictionary_size=max_dictionary_size,
            min_estimated_savings=min_estimated_savings,
            skip_all_whitespace_phrases=skip_all_whitespace_phrases,
        )

    def compress(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> CompressionPackage:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        metadata = metadata or {}
        tokenization = self.tokenizer.tokenize(text)
        original_bytes = text.encode("utf-8")

        if self.method == "dictionary":
            return self._compress_with_dictionary(
                text=text,
                original_bytes=original_bytes,
                token_count=tokenization.token_count,
                tokens=tokenization.tokens,
                metadata=metadata,
            )

        compressed_bytes = self._compress_bytes(original_bytes)
        payload_b64 = base64.b64encode(compressed_bytes).decode("ascii")

        stats = self._build_stats(
            original_bytes_len=len(original_bytes),
            compressed_bytes_len=len(compressed_bytes),
            original_token_count=tokenization.token_count,
            compressed_token_count=None,
        )

        return CompressionPackage(
            version="1.1.0",
            method=self.method,
            original_length=len(text),
            original_sha256=self._sha256_hex(original_bytes),
            token_count=tokenization.token_count,
            compressed_payload_b64=payload_b64,
            stats=stats,
            metadata=metadata,
            dictionary=None,
        )

    def compress_to_json(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self.compress(text=text, metadata=metadata).to_json()

    def _compress_with_dictionary(
        self,
        text: str,
        original_bytes: bytes,
        token_count: int,
        tokens: list[str],
        metadata: dict[str, Any],
    ) -> CompressionPackage:
        entries = self.dictionary_builder.build(tokens)
        dictionary_map = self.dictionary_builder.build_reverse_lookup(entries)
        encoded_tokens = self._encode_with_dictionary(tokens, entries)

        serialized_stream = json.dumps(
            encoded_tokens,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        payload_b64 = base64.b64encode(serialized_stream).decode("ascii")

        stats = self._build_stats(
            original_bytes_len=len(original_bytes),
            compressed_bytes_len=len(serialized_stream),
            original_token_count=token_count,
            compressed_token_count=len(encoded_tokens),
        )

        return CompressionPackage(
            version="1.1.0",
            method="dictionary",
            original_length=len(text),
            original_sha256=self._sha256_hex(original_bytes),
            token_count=token_count,
            compressed_payload_b64=payload_b64,
            stats=stats,
            metadata=metadata,
            dictionary=dictionary_map,
        )

    def _encode_with_dictionary(
        self,
        tokens: list[str],
        entries: list[DictionaryEntry],
    ) -> list[str]:
        if not entries:
            return list(tokens)

        phrase_to_id = self.dictionary_builder.build_lookup(entries)
        available_lengths = sorted(
            {len(entry.phrase) for entry in entries},
            reverse=True,
        )

        encoded_tokens: list[str] = []
        token_index = 0
        total_tokens = len(tokens)

        while token_index < total_tokens:
            matched_phrase_id: str | None = None
            matched_phrase_len = 0

            for phrase_len in available_lengths:
                if token_index + phrase_len > total_tokens:
                    continue

                candidate = tuple(tokens[token_index : token_index + phrase_len])
                phrase_id = phrase_to_id.get(candidate)

                if phrase_id is not None:
                    matched_phrase_id = phrase_id
                    matched_phrase_len = phrase_len
                    break

            if matched_phrase_id is not None:
                encoded_tokens.append(matched_phrase_id)
                token_index += matched_phrase_len
            else:
                encoded_tokens.append(tokens[token_index])
                token_index += 1

        return encoded_tokens

    def _compress_bytes(self, data: bytes) -> bytes:
        if self.method == "none":
            return data

        if self.method == "zlib":
            return zlib.compress(data, level=self.zlib_level)

        if self.method == "lzma":
            return lzma.compress(data, preset=self.lzma_preset)

        raise ValueError(f"Unhandled compression method: {self.method}")

    @staticmethod
    def _sha256_hex(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _build_stats(
        original_bytes_len: int,
        compressed_bytes_len: int,
        original_token_count: int,
        compressed_token_count: int | None,
    ) -> CompressionStats:
        if original_bytes_len < 0 or compressed_bytes_len < 0:
            raise ValueError("byte lengths must be non-negative")

        if original_bytes_len == 0:
            compression_ratio = 1.0
            space_saving_ratio = 0.0
        else:
            compression_ratio = compressed_bytes_len / original_bytes_len
            space_saving_ratio = 1.0 - compression_ratio

        return CompressionStats(
            original_bytes=original_bytes_len,
            compressed_bytes=compressed_bytes_len,
            compression_ratio=compression_ratio,
            space_saving_ratio=space_saving_ratio,
            original_token_count=original_token_count,
            compressed_token_count=compressed_token_count,
        )

    def available_methods(self) -> list[str]:
        return list(self.SUPPORTED_METHODS)
