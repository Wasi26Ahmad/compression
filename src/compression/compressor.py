from __future__ import annotations

import base64
import hashlib
import json
import lzma
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Literal

from src.compression.tokenizer import TextTokenizer

CompressionMethod = Literal["none", "zlib", "lzma"]


@dataclass(frozen=True)
class CompressionStats:
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    space_saving_ratio: float
    original_token_count: int


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompressionPackage:
        stats_data = data["stats"]
        stats = CompressionStats(**stats_data)
        return cls(
            version=data["version"],
            method=data["method"],
            original_length=data["original_length"],
            original_sha256=data["original_sha256"],
            token_count=data["token_count"],
            compressed_payload_b64=data["compressed_payload_b64"],
            stats=stats,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> CompressionPackage:
        return cls.from_dict(json.loads(json_str))


class TextCompressor:
    SUPPORTED_METHODS: tuple[CompressionMethod, ...] = ("none", "zlib", "lzma")

    def __init__(
        self,
        method: CompressionMethod = "zlib",
        zlib_level: int = 9,
        lzma_preset: int = 6,
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

    def compress(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> CompressionPackage:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        metadata = metadata or {}

        tokenization = self.tokenizer.tokenize(text)
        original_bytes = text.encode("utf-8")
        compressed_bytes = self._compress_bytes(original_bytes)
        payload_b64 = base64.b64encode(compressed_bytes).decode("ascii")

        stats = self._build_stats(
            original_bytes_len=len(original_bytes),
            compressed_bytes_len=len(compressed_bytes),
            original_token_count=tokenization.token_count,
        )

        package = CompressionPackage(
            version="1.0.0",
            method=self.method,
            original_length=len(text),
            original_sha256=self._sha256_hex(original_bytes),
            token_count=tokenization.token_count,
            compressed_payload_b64=payload_b64,
            stats=stats,
            metadata=metadata,
        )
        return package

    def compress_to_json(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> str:
        return self.compress(text=text, metadata=metadata).to_json()

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
        )

    def available_methods(self) -> list[str]:
        return list(self.SUPPORTED_METHODS)
