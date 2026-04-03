from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from src.ccllm.compression import CompressionPackage
from src.ccllm.storage.models import StorageCreateRequest, StoredCompressionRecord


class CompressionStorage:
    """
    SQLite-backed storage for compression packages.

    Responsibilities:
    - create and initialize the database
    - save compression packages
    - fetch a record by ID
    - list saved records
    - delete saved records
    """

    def __init__(self, db_path: str | Path = "data/compression.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS compression_records (
                    record_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    method TEXT NOT NULL,
                    original_sha256 TEXT NOT NULL,
                    original_length INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    compressed_bytes INTEGER NOT NULL,
                    compression_ratio REAL NOT NULL,
                    package_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """)
            connection.commit()

    def save_package(
        self,
        package: CompressionPackage,
        metadata: dict[str, Any] | None = None,
        record_id: str | None = None,
    ) -> StoredCompressionRecord:
        """
        Save a CompressionPackage into SQLite.

        Args:
            package: Compression package to persist.
            metadata: Optional extra storage metadata.
            record_id: Optional caller-provided record ID.

        Returns:
            StoredCompressionRecord
        """
        if not isinstance(package, CompressionPackage):
            raise TypeError("package must be a CompressionPackage")

        metadata = metadata or {}
        final_record_id = record_id or self._generate_record_id()
        package_json = package.to_json()
        metadata_json = json.dumps(metadata, ensure_ascii=False)

        create_request = StorageCreateRequest(
            record_id=final_record_id,
            package=package,
            package_json=package_json,
            metadata_json=metadata_json,
        )
        record = create_request.to_record()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO compression_records (
                    record_id,
                    created_at,
                    method,
                    original_sha256,
                    original_length,
                    token_count,
                    compressed_bytes,
                    compression_ratio,
                    package_json,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.created_at,
                    record.method,
                    record.original_sha256,
                    record.original_length,
                    record.token_count,
                    record.compressed_bytes,
                    record.compression_ratio,
                    record.package_json,
                    record.metadata_json,
                ),
            )
            connection.commit()

        return record

    def get_record(self, record_id: str) -> StoredCompressionRecord | None:
        """
        Fetch a stored record by ID.
        """
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    record_id,
                    created_at,
                    method,
                    original_sha256,
                    original_length,
                    token_count,
                    compressed_bytes,
                    compression_ratio,
                    package_json,
                    metadata_json
                FROM compression_records
                WHERE record_id = ?
                """,
                (record_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def get_package(self, record_id: str) -> CompressionPackage | None:
        """
        Fetch and reconstruct a CompressionPackage by record ID.
        """
        record = self.get_record(record_id)
        if record is None:
            return None
        return CompressionPackage.from_json(record.package_json)

    def list_records(self, limit: int = 100) -> list[StoredCompressionRecord]:
        """
        List recent stored records.
        """
        if not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    record_id,
                    created_at,
                    method,
                    original_sha256,
                    original_length,
                    token_count,
                    compressed_bytes,
                    compression_ratio,
                    package_json,
                    metadata_json
                FROM compression_records
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def delete_record(self, record_id: str) -> bool:
        """
        Delete a record by ID.

        Returns:
            True if a row was deleted, False otherwise.
        """
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")

        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM compression_records WHERE record_id = ?",
                (record_id,),
            )
            connection.commit()

        return cursor.rowcount > 0

    def record_exists(self, record_id: str) -> bool:
        """
        Check whether a record exists.
        """
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string")

        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM compression_records WHERE record_id = ? LIMIT 1",
                (record_id,),
            ).fetchone()

        return row is not None

    def count_records(self) -> int:
        """
        Return total number of stored records.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM compression_records"
            ).fetchone()

        if row is None:
            return 0

        return int(row["total"])

    @staticmethod
    def _generate_record_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> StoredCompressionRecord:
        return StoredCompressionRecord(
            record_id=row["record_id"],
            created_at=row["created_at"],
            method=row["method"],
            original_sha256=row["original_sha256"],
            original_length=row["original_length"],
            token_count=row["token_count"],
            compressed_bytes=row["compressed_bytes"],
            compression_ratio=row["compression_ratio"],
            package_json=row["package_json"],
            metadata_json=row["metadata_json"],
        )
