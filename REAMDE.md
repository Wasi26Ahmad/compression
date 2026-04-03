# Compression Memory System

A modular, production-ready Python system for **lossless text
compression, storage, memory management, and intelligent retrieval** ---
supporting both **lexical and vector-based search**.

------------------------------------------------------------------------

## What this project does

This project provides a complete pipeline to:

-   Compress text efficiently (multiple algorithms)
-   Store compressed data with metadata
-   Restore original text perfectly (lossless)
-   Manage memory like a database
-   Retrieve relevant information using:
    -   **Lexical search (TF-IDF)**
    -   **Vector search (embeddings)**
    -   **Hybrid retrieval (best of both worlds)**

------------------------------------------------------------------------

### Compression Performance Summary

  -------------------------------------------------------------------------------
  Dataset    Method       Compression   Space      Compress Time  Decompress Time
  Type                    Ratio ↓       Saving ↑   (ms) ↓         (ms) ↓
  ---------- ------------ ------------- ---------- -------------- ---------------
  Long       none         1.00          0%         0.20           0.01
  Context                                                         

             zlib         **0.33**      **67%**    0.36           0.02

             lzma         0.45          55%        7.21           0.15

             dictionary   0.45          55%        2.91           0.39

  Repeated   none         1.00          0%         0.03           0.01
  Text                                                            

             zlib         0.35          65%        0.05           0.03

             lzma         0.59          41%        6.38           0.09

             dictionary   **0.33**      **66%**    1.33           0.17

  Short Text none         1.00          0%         0.02           0.00

             zlib         0.85          15%        0.04           0.01

             lzma         ❌ 1.39       -39%       6.45           0.09

             dictionary   ❌ 1.83       -83%       0.72           0.06
  -------------------------------------------------------------------------------

------------------------------------------------------------------------

##  System Architecture

### Compression Layer

- zlib ->  fast + efficient
- lzma -> high compression
- dictionary -> adaptive phrase compression

### Storage Layer
- SQLite backend
- SHA256 integrity verification
- metadata-aware

### Memory Layer
- CURD operations
- full bundle export
- scalable abstraction

### Retrieval Layer
- Lexical
- Vector
- Hybrid

------------------------------------------------------------------------

#  Installation

## From PyPI

pip install ccllm

# From source

git clone
https://github.com/`Wasi26Ahmad`{=html}/compression.git\
cd compression\
pip install -e .\[dev\]

------------------------------------------------------------------------

#  Quick Start

## Store Text

``` python
from src.memory import MemoryManager
from src.storage import CompressionStorage

storage = CompressionStorage("data.db")
manager = MemoryManager(storage)

record = manager.save_text(
    "Cattle weight estimation using images",
    method="dictionary",
    metadata={"topic": "cattle"},
)
print(record.record_id)
```

## Retrieve Text

``` python
from src.retrieval import MemoryRetriever

retriever = MemoryRetriever(manager, mode="hybrid")
results = retriever.retrieve("cattle image estimation")

for r in results:
    print(r.text, r.score)
```

## Restore Text

``` python
text = manager.get_text(record.record_id)
print(text)
```

------------------------------------------------------------------------

#  Core Features

## Compression Engine

-   none
-   zlib
-   lzma
-   dictionary

## Storage

-   SQLite-based
-   Metadata + hashing

## Retrieval

-   Lexical (TF-IDF)
-   Vector (Embeddings)
-   Hybrid

------------------------------------------------------------------------

#  API

Run:

uvicorn src.api.app:app --reload

Docs: http://127.0.0.1:8000/docs

------------------------------------------------------------------------


#  License

MIT License
