from __future__ import annotations

import json
from typing import Any

from src.compression import CompressionPackage, TextCompressor, TextDecompressor
from src.storage import CompressionStorage, StoredCompressionRecord


class MemoryManager:
    """
    High-level orchestration layer for compressed text memory.

    Responsibilities:
    - compress and save text
    - load stored records and packages
    - restore original text
    - list stored memories
    - delete memories

    This sits between:
    - compression layer
    - storage layer

    """

    def __init__(
        self,
        storage: CompressionStorage,
        default_method: str = "zlib",
    ) -> None:
        if not isinstance(storage, CompressionStorage):
            raise TypeError("storage must be a CompressionStorage")

        self.storage = storage
        self.default_method = default_method
        self.decompressor = TextDecompressor()

    def save_text(
        self,
        text: str,
        method: str | None = None,
        metadata: dict[str, Any] | None = None,
        record_id: str | None = None,
        compressor_kwargs: dict[str, Any] | None = None,
    ) -> StoredCompressionRecord:
        """
        Compress text and save it to storage.

        Args:
            text: Original text to store.
            method: Compression method to use. Falls back to default_method.
            metadata: Optional metadata stored alongside the record.
            record_id: Optional caller-provided record ID.
            compressor_kwargs: Optional extra arguments for TextCompressor.

        Returns:
            StoredCompressionRecord
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        final_method = method or self.default_method
        final_metadata = metadata or {}
        final_compressor_kwargs = compressor_kwargs or {}

        compressor = TextCompressor(
            method=final_method,
            **final_compressor_kwargs,
        )
        package = compressor.compress(text=text, metadata=final_metadata)

        return self.storage.save_package(
            package=package,
            metadata=final_metadata,
            record_id=record_id,
        )

    def get_record(self, record_id: str) -> StoredCompressionRecord | None:
        """
        Return the stored record metadata by record ID.
        """
        return self.storage.get_record(record_id)

    def get_package(self, record_id: str) -> CompressionPackage | None:
        """
        Return the reconstructed CompressionPackage by record ID.
        """
        return self.storage.get_package(record_id)

    def get_text(self, record_id: str) -> str | None:
        """
        Load, decompress, and return the original text by record ID.
        """
        package = self.storage.get_package(record_id)
        if package is None:
            return None
        return self.decompressor.decompress(package)

    def list_memories(self, limit: int = 100) -> list[StoredCompressionRecord]:
        """
        List recent stored memory records.
        """
        return self.storage.list_records(limit=limit)

    def delete_memory(self, record_id: str) -> bool:
        """
        Delete a stored memory by record ID.
        """
        return self.storage.delete_record(record_id)

    def memory_exists(self, record_id: str) -> bool:
        """
        Check whether a memory record exists.
        """
        return self.storage.record_exists(record_id)

    def count_memories(self) -> int:
        """
        Return total number of stored memories.
        """
        return self.storage.count_records()

    def export_record_bundle(self, record_id: str) -> dict[str, Any] | None:
        """
        Export a complete record bundle including:
        - stored record fields
        - decoded metadata JSON
        - reconstructed compression package as dict
        - restored original text

        """
        record = self.storage.get_record(record_id)
        if record is None:
            return None

        package = CompressionPackage.from_json(record.package_json)

        try:
            metadata = json.loads(record.metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Stored metadata_json is invalid JSON") from exc

        text = self.decompressor.decompress(package)

        return {
            "record": record.to_dict(),
            "metadata": metadata,
            "package": package.to_dict(),
            "text": text,
        }

    def restore_all_texts(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Restore original text for recent stored records.

        Returns a list of dictionaries containing:
        - record_id
        - method
        - created_at
        - text
        """
        records = self.storage.list_records(limit=limit)
        restored_items: list[dict[str, Any]] = []

        for record in records:
            package = CompressionPackage.from_json(record.package_json)
            text = self.decompressor.decompress(package)

            restored_items.append(
                {
                    "record_id": record.record_id,
                    "method": record.method,
                    "created_at": record.created_at,
                    "text": text,
                }
            )

        return restored_items
