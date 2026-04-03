from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["total_memories"], int)


def test_store_text_endpoint() -> None:
    response = client.post(
        "/store",
        json={
            "text": "Cattle weight estimation uses side-view images.",
            "method": "zlib",
            "metadata": {"topic": "cattle"},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "record_id" in data
    assert data["method"] == "zlib"
    assert data["original_length"] == len(
        "Cattle weight estimation uses side-view images."
    )


def test_store_text_endpoint_with_dictionary_method() -> None:
    response = client.post(
        "/store",
        json={
            "text": "alpha beta alpha beta alpha beta",
            "method": "dictionary",
            "metadata": {"topic": "test"},
            "compressor_kwargs": {
                "min_phrase_len": 2,
                "max_phrase_len": 5,
                "min_frequency": 2,
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "dictionary"


def test_store_text_rejects_bad_payload() -> None:
    response = client.post(
        "/store",
        json={
            "text": 123,
            "method": "zlib",
        },
    )

    assert response.status_code == 422


def test_list_memories_endpoint() -> None:
    client.post("/store", json={"text": "memory one", "method": "zlib"})
    client.post("/store", json={"text": "memory two", "method": "zlib"})

    response = client.get("/memories?limit=10")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2


def test_get_memory_endpoint() -> None:
    store_response = client.post(
        "/store",
        json={
            "text": "Get memory API test",
            "method": "zlib",
            "metadata": {"kind": "api-test"},
        },
    )
    record_id = store_response.json()["record_id"]

    response = client.get(f"/memories/{record_id}")

    assert response.status_code == 200
    data = response.json()
    assert "record" in data
    assert "metadata" in data
    assert "package" in data
    assert "text" in data
    assert data["record"]["record_id"] == record_id
    assert data["text"] == "Get memory API test"


def test_get_memory_returns_404_for_missing_record() -> None:
    response = client.get("/memories/missing-record-id")

    assert response.status_code == 404


def test_get_memory_text_endpoint() -> None:
    store_response = client.post(
        "/store",
        json={
            "text": "Restore original text through API",
            "method": "lzma",
        },
    )
    record_id = store_response.json()["record_id"]

    response = client.get(f"/memories/{record_id}/text")

    assert response.status_code == 200
    data = response.json()
    assert data["record_id"] == record_id
    assert data["text"] == "Restore original text through API"


def test_get_memory_text_returns_404_for_missing_record() -> None:
    response = client.get("/memories/missing-record-id/text")

    assert response.status_code == 404


def test_delete_memory_endpoint() -> None:
    store_response = client.post(
        "/store",
        json={
            "text": "Delete me through API",
            "method": "zlib",
        },
    )
    record_id = store_response.json()["record_id"]

    delete_response = client.delete(f"/memories/{record_id}")

    assert delete_response.status_code == 200
    data = delete_response.json()
    assert data["deleted"] is True
    assert data["record_id"] == record_id

    get_response = client.get(f"/memories/{record_id}")
    assert get_response.status_code == 404


def test_delete_memory_returns_404_for_missing_record() -> None:
    response = client.delete("/memories/missing-record-id")

    assert response.status_code == 404


def test_retrieve_endpoint_lexical_mode() -> None:
    client.post(
        "/store",
        json={
            "text": "Cattle weight estimation from side-view and rear-view images.",
            "method": "dictionary",
            "metadata": {"topic": "cattle"},
        },
    )
    client.post(
        "/store",
        json={
            "text": "Road anomaly detection using smartphone sensors.",
            "method": "zlib",
            "metadata": {"topic": "road"},
        },
    )

    response = client.post(
        "/retrieve",
        json={
            "query": "cattle image estimation",
            "mode": "lexical",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "cattle" in data[0]["text"].lower()


def test_retrieve_endpoint_vector_mode() -> None:
    client.post(
        "/store",
        json={
            "text": "Medical image analysis with deep learning.",
            "method": "zlib",
            "metadata": {"topic": "medical"},
        },
    )

    response = client.post(
        "/retrieve",
        json={
            "query": "medical image",
            "mode": "vector",
            "limit": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_retrieve_endpoint_hybrid_mode_with_metadata_filter() -> None:
    client.post(
        "/store",
        json={
            "text": "Cattle vision model for weight estimation.",
            "method": "dictionary",
            "metadata": {"type": "vision"},
        },
    )
    client.post(
        "/store",
        json={
            "text": "Cattle nutrition and feeding analysis.",
            "method": "zlib",
            "metadata": {"type": "health"},
        },
    )

    response = client.post(
        "/retrieve",
        json={
            "query": "cattle",
            "mode": "hybrid",
            "limit": 5,
            "metadata_filter": {"type": "vision"},
            "alpha": 0.5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert all(item["metadata"]["type"] == "vision" for item in data)


def test_retrieve_endpoint_rejects_bad_payload() -> None:
    response = client.post(
        "/retrieve",
        json={
            "query": 123,
            "mode": "hybrid",
        },
    )

    assert response.status_code == 422
