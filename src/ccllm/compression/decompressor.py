from __future__ import annotations

import base64
import binascii
import hashlib
import json
import lzma
import zlib
from typing import Any

from src.ccllm.compression.compressor import CompressionPackage


class TextDecompressor:
    """
    Lossless text decompressor.

    Supported methods:
    - none
    - zlib
    - lzma
    - dictionary

    For dictionary mode:
    - payload stores a JSON-serialized token stream
    - package.dictionary stores phrase_id -> original phrase tokens
    - decompression expands phrase IDs back into the original token sequence
    """

    SUPPORTED_METHODS = ("none", "zlib", "lzma", "dictionary")

    def decompress(self, package: CompressionPackage) -> str:
        """
        Decompress a CompressionPackage back to the original text.
        """
        if not isinstance(package, CompressionPackage):
            raise TypeError("package must be a CompressionPackage")

        if package.method == "dictionary":
            text = self._decompress_dictionary_package(package)
        else:
            compressed_bytes = self._decode_base64(package.compressed_payload_b64)
            original_bytes = self._decompress_bytes(compressed_bytes, package.method)
            text = self._decode_utf8(original_bytes)
            self._validate_sha256(
                data=original_bytes,
                expected_sha256=package.original_sha256,
            )

        self._validate_length(text=text, expected_length=package.original_length)
        return text

    def decompress_from_json(self, json_str: str) -> str:
        """
        Deserialize a JSON package and decompress it.
        """
        try:
            package = CompressionPackage.from_json(json_str)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid JSON input for CompressionPackage") from exc

        return self.decompress(package)

    def _decompress_dictionary_package(self, package: CompressionPackage) -> str:
        if package.dictionary is None:
            raise ValueError(
                "Dictionary compression package is missing dictionary data"
            )

        self._validate_dictionary(package.dictionary)

        serialized_stream = self._decode_base64(package.compressed_payload_b64)
        encoded_tokens = self._decode_dictionary_payload(serialized_stream)
        expanded_tokens = self._expand_dictionary_tokens(
            encoded_tokens=encoded_tokens,
            dictionary=package.dictionary,
        )
        text = "".join(expanded_tokens)

        self._validate_sha256(
            data=text.encode("utf-8"),
            expected_sha256=package.original_sha256,
        )
        return text

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
    def _decode_dictionary_payload(data: bytes) -> list[str]:
        try:
            decoded: Any = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid dictionary payload JSON") from exc

        if not isinstance(decoded, list):
            raise ValueError("Dictionary payload must decode to a list of tokens")

        if not all(isinstance(token, str) for token in decoded):
            raise ValueError("Dictionary payload tokens must all be strings")

        return decoded

    @staticmethod
    def _expand_dictionary_tokens(
        encoded_tokens: list[str],
        dictionary: dict[str, list[str]],
    ) -> list[str]:
        expanded_tokens: list[str] = []

        for token in encoded_tokens:
            if token in dictionary:
                phrase_tokens = dictionary[token]
                if not isinstance(phrase_tokens, list) or not all(
                    isinstance(item, str) for item in phrase_tokens
                ):
                    raise ValueError(f"Invalid dictionary entry for phrase ID: {token}")
                expanded_tokens.extend(phrase_tokens)
            else:
                expanded_tokens.append(token)

        return expanded_tokens

    @staticmethod
    def _validate_dictionary(dictionary: dict[str, list[str]]) -> None:
        if not isinstance(dictionary, dict):
            raise TypeError("dictionary must be a dict[str, list[str]]")

        for phrase_id, phrase_tokens in dictionary.items():
            if not isinstance(phrase_id, str):
                raise ValueError("Dictionary phrase IDs must be strings")
            if not isinstance(phrase_tokens, list):
                raise ValueError("Dictionary phrase values must be lists of strings")
            if not all(isinstance(token, str) for token in phrase_tokens):
                raise ValueError("Dictionary phrase values must contain only strings")

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
