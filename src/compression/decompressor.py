from __future__ import annotations

import base64
import binascii
import hashlib
import json
import lzma
import zlib

from src.compression.compressor import CompressionPackage


class TextDecompressor:
    """
    Baseline lossless text decompressor.

    Responsibilities:
    - decode base64 payload
    - decompress bytes according to package method
    - restore UTF-8 text
    - validate original text length
    - validate SHA-256 integrity
    """

    SUPPORTED_METHODS = ("none", "zlib", "lzma")

    def decompress(self, package: CompressionPackage) -> str:
        if not isinstance(package, CompressionPackage):
            raise TypeError("package must be a CompressionPackage")

        compressed_bytes = self._decode_base64(package.compressed_payload_b64)
        original_bytes = self._decompress_bytes(compressed_bytes, package.method)

        text = self._decode_utf8(original_bytes)
        self._validate_length(text=text, expected_length=package.original_length)
        self._validate_sha256(data=original_bytes, expected_sha256=package.original_sha256)

        return text

    def decompress_from_json(self, json_str: str) -> str:
        try:
            package = CompressionPackage.from_json(json_str)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid JSON input for CompressionPackage") from exc

        return self.decompress(package)

    def _decode_base64(self, payload_b64: str) -> bytes:
        if not isinstance(payload_b64, str):
            raise TypeError("compressed_payload_b64 must be a string")

        try:
            return base64.b64decode(payload_b64.encode("ascii"), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid base64 payload") from exc

    def _decompress_bytes(self, data: bytes, method: str) -> bytes:
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Unsupported compression method: {method}")

        if method == "none":
            return data

        if method == "zlib":
            try:
                return zlib.decompress(data)
            except zlib.error as exc:
                raise ValueError("Failed to decompress zlib payload") from exc

        if method == "lzma":
            try:
                return lzma.decompress(data)
            except lzma.LZMAError as exc:
                raise ValueError("Failed to decompress lzma payload") from exc

        raise ValueError(f"Unsupported compression method: {method}")

    @staticmethod
    def _decode_utf8(data: bytes) -> str:
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Decompressed payload is not valid UTF-8 text") from exc

    @staticmethod
    def _validate_length(text: str, expected_length: int) -> None:
        if len(text) != expected_length:
            raise ValueError(
                f"Original length mismatch: expected {expected_length}, got {len(text)}"
            )

    @staticmethod
    def _validate_sha256(data: bytes, expected_sha256: str) -> None:
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )