from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from src.ccllm.compression import CompressionPackage


@dataclass(frozen=True)
class StoredCompressionRecord:
    """
    Database-facing record for a saved compression package.
    """

    record_id: str
    created_at: str
    method: str
    original_sha256: str
    original_length: int
    token_count: int
    compressed_bytes: int
    compression_ratio: float
    package_json: str
    metadata_json: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StorageCreateRequest:
    """
    Input model used when creating a storage record from a compression package.
    """

    record_id: str
    package: CompressionPackage
    package_json: str
    metadata_json: str

    def to_record(self) -> StoredCompressionRecord:
        return StoredCompressionRecord(
            record_id=self.record_id,
            created_at=_utc_now_iso(),
            method=self.package.method,
            original_sha256=self.package.original_sha256,
            original_length=self.package.original_length,
            token_count=self.package.token_count,
            compressed_bytes=self.package.stats.compressed_bytes,
            compression_ratio=self.package.stats.compression_ratio,
            package_json=self.package_json,
            metadata_json=self.metadata_json,
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
