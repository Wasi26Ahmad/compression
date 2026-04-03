from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.memory import MemoryManager
from src.retrieval import MemoryRetriever
from src.storage import CompressionStorage

# -----------------------------
# App setup
# -----------------------------

storage = CompressionStorage("data/compression.db")
memory_manager = MemoryManager(storage=storage, default_method="zlib")

app = FastAPI(
    title="Compression Memory API",
    version="1.0.0",
    description=(
        "API for storing, restoring, listing, deleting, and retrieving "
        "compressed text memories."
    ),
)


# -----------------------------
# Request / Response Models
# -----------------------------

CompressionMethod = Literal["none", "zlib", "lzma", "dictionary"]
RetrievalMode = Literal["lexical", "vector", "hybrid"]


class StoreTextRequest(BaseModel):
    text: str = Field(..., description="Original text to compress and store.")
    method: CompressionMethod | None = Field(
        default=None,
        description="Compression method to use. Falls back to default if omitted.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata stored with the record.",
    )
    record_id: str | None = Field(
        default=None,
        description="Optional custom record ID.",
    )
    compressor_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Optional compressor configuration.",
    )


class StoreTextResponse(BaseModel):
    record_id: str
    created_at: str
    method: str
    original_sha256: str
    original_length: int
    token_count: int
    compressed_bytes: int
    compression_ratio: float
    metadata_json: str


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Search query text.")
    mode: RetrievalMode = Field(
        default="hybrid",
        description="Retrieval mode: lexical, vector, or hybrid.",
    )
    limit: int = Field(default=5, ge=1, description="Maximum number of results.")
    search_limit: int = Field(
        default=100,
        ge=1,
        description="Maximum number of stored memories to inspect.",
    )
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description="Optional exact-match metadata filter.",
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Hybrid weighting. Used only in hybrid mode.",
    )


class RetrievalItem(BaseModel):
    record_id: str
    score: float
    method: str
    created_at: str
    metadata: dict[str, Any]
    text: str


class DeleteResponse(BaseModel):
    deleted: bool
    record_id: str


class HealthResponse(BaseModel):
    status: str
    total_memories: int


# -----------------------------
# Routes
# -----------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        total_memories=memory_manager.count_memories(),
    )


@app.post("/store", response_model=StoreTextResponse)
def store_text(payload: StoreTextRequest) -> StoreTextResponse:
    try:
        record = memory_manager.save_text(
            text=payload.text,
            method=payload.method,
            metadata=payload.metadata,
            record_id=payload.record_id,
            compressor_kwargs=payload.compressor_kwargs,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StoreTextResponse(
        record_id=record.record_id,
        created_at=record.created_at,
        method=record.method,
        original_sha256=record.original_sha256,
        original_length=record.original_length,
        token_count=record.token_count,
        compressed_bytes=record.compressed_bytes,
        compression_ratio=record.compression_ratio,
        metadata_json=record.metadata_json,
    )


@app.get("/memories")
def list_memories(
    limit: int = Query(default=100, ge=1, description="Maximum number of records."),
) -> list[dict[str, Any]]:
    try:
        records = memory_manager.list_memories(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return [record.to_dict() for record in records]


@app.get("/memories/{record_id}")
def get_memory(record_id: str) -> dict[str, Any]:
    bundle = memory_manager.export_record_bundle(record_id)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return bundle


@app.get("/memories/{record_id}/text")
def get_memory_text(record_id: str) -> dict[str, Any]:
    text = memory_manager.get_text(record_id)
    if text is None:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"record_id": record_id, "text": text}


@app.delete("/memories/{record_id}", response_model=DeleteResponse)
def delete_memory(record_id: str) -> DeleteResponse:
    deleted = memory_manager.delete_memory(record_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Record not found")

    return DeleteResponse(
        deleted=True,
        record_id=record_id,
    )


@app.post("/retrieve", response_model=list[RetrievalItem])
def retrieve_texts(payload: RetrieveRequest) -> list[RetrievalItem]:
    try:
        retriever = MemoryRetriever(
            memory_manager=memory_manager,
            mode=payload.mode,
            alpha=payload.alpha,
        )
        results = retriever.retrieve(
            query=payload.query,
            limit=payload.limit,
            search_limit=payload.search_limit,
            metadata_filter=payload.metadata_filter,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return [
        RetrievalItem(
            record_id=result.record_id,
            score=result.score,
            method=result.method,
            created_at=result.created_at,
            metadata=result.metadata,
            text=result.text,
        )
        for result in results
    ]
