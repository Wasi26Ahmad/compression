from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_full_api_pipeline_store_retrieve_get_text_delete() -> None:
    store_response = client.post(
        "/store",
        json={
            "text": "Cattle weight estimation uses side-view and rear-view images.",
            "method": "dictionary",
            "metadata": {"topic": "cattle", "source": "integration"},
            "compressor_kwargs": {
                "min_phrase_len": 2,
                "max_phrase_len": 5,
                "min_frequency": 2,
            },
        },
    )
    assert store_response.status_code == 200
    stored = store_response.json()
    record_id = stored["record_id"]

    retrieve_response = client.post(
        "/retrieve",
        json={
            "query": "cattle image estimation",
            "mode": "hybrid",
            "limit": 3,
            "metadata_filter": {"topic": "cattle"},
            "alpha": 0.5,
        },
    )
    assert retrieve_response.status_code == 200
    retrieved = retrieve_response.json()
    assert len(retrieved) >= 1
    assert any(item["record_id"] == record_id for item in retrieved)

    memory_response = client.get(f"/memories/{record_id}")
    assert memory_response.status_code == 200
    memory_bundle = memory_response.json()
    assert memory_bundle["record"]["record_id"] == record_id
    assert memory_bundle["text"] == (
        "Cattle weight estimation uses side-view and rear-view images."
    )

    text_response = client.get(f"/memories/{record_id}/text")
    assert text_response.status_code == 200
    text_data = text_response.json()
    assert text_data["record_id"] == record_id
    assert text_data["text"] == (
        "Cattle weight estimation uses side-view and rear-view images."
    )

    delete_response = client.delete(f"/memories/{record_id}")
    assert delete_response.status_code == 200
    delete_data = delete_response.json()
    assert delete_data["deleted"] is True
    assert delete_data["record_id"] == record_id

    missing_response = client.get(f"/memories/{record_id}")
    assert missing_response.status_code == 404


def test_api_pipeline_across_multiple_compression_methods() -> None:
    payloads = [
        {
            "text": "Plain storage sample for none method.",
            "method": "none",
            "metadata": {"method_group": "baseline"},
        },
        {
            "text": "Repeated repeated repeated text for zlib compression.",
            "method": "zlib",
            "metadata": {"method_group": "baseline"},
        },
        {
            "text": "Another repeated repeated repeated text for lzma compression.",
            "method": "lzma",
            "metadata": {"method_group": "baseline"},
        },
        {
            "text": "alpha beta alpha beta alpha beta",
            "method": "dictionary",
            "metadata": {"method_group": "custom"},
            "compressor_kwargs": {
                "min_phrase_len": 2,
                "max_phrase_len": 5,
                "min_frequency": 2,
            },
        },
    ]

    record_ids: list[str] = []

    for payload in payloads:
        response = client.post("/store", json=payload)
        assert response.status_code == 200
        record_ids.append(response.json()["record_id"])

    list_response = client.get("/memories?limit=20")
    assert list_response.status_code == 200
    memories = list_response.json()
    returned_ids = {item["record_id"] for item in memories}

    for record_id in record_ids:
        assert record_id in returned_ids

    retrieve_response = client.post(
        "/retrieve",
        json={
            "query": "alpha beta",
            "mode": "hybrid",
            "limit": 5,
        },
    )
    assert retrieve_response.status_code == 200
    retrieved = retrieve_response.json()
    assert isinstance(retrieved, list)
    assert len(retrieved) >= 1

    for record_id in record_ids:
        response = client.get(f"/memories/{record_id}/text")
        assert response.status_code == 200
        assert response.json()["record_id"] == record_id


def test_api_health_and_memory_count_work_together() -> None:
    before = client.get("/health")
    assert before.status_code == 200
    before_count = before.json()["total_memories"]

    store_response = client.post(
        "/store",
        json={
            "text": "Health endpoint count integration test.",
            "method": "zlib",
        },
    )
    assert store_response.status_code == 200
    record_id = store_response.json()["record_id"]

    after = client.get("/health")
    assert after.status_code == 200
    after_count = after.json()["total_memories"]

    assert after_count >= before_count + 1

    delete_response = client.delete(f"/memories/{record_id}")
    assert delete_response.status_code == 200
