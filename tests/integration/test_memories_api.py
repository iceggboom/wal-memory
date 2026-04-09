import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_mem0():
    return MagicMock()


@pytest.fixture
def app_client(mock_mem0, mock_env):
    from memory_platform.api.memories import create_router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(create_router(mock_mem0))
    return TestClient(app)


class TestPostMemories:
    def test_add_memory_success(self, app_client, mock_mem0):
        mock_mem0.add.return_value = {
            "results": [{"id": "m1", "memory": "test", "event": "ADD"}]
        }
        resp = app_client.post(
            "/v1/memories",
            headers={"authorization": "Bearer test-key-123"},
            json={
                "user_id": "u1",
                "app_id": "app1",
                "memories": [{"text": "是一名Java工程师"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["added"] == 1

    def test_missing_api_key(self, app_client):
        resp = app_client.post(
            "/v1/memories",
            json={"user_id": "u1", "app_id": "app1", "memories": [{"text": "test"}]},
        )
        assert resp.status_code == 401

    def test_missing_user_id(self, app_client):
        resp = app_client.post(
            "/v1/memories",
            headers={"authorization": "Bearer test-key-123"},
            json={"app_id": "app1", "memories": [{"text": "test"}]},
        )
        assert resp.status_code == 422


class TestPostExtract:
    def test_extract_success(self, app_client, mock_mem0):
        mock_mem0.add.return_value = {
            "results": [{"id": "m1", "memory": "用户是工程师", "event": "ADD"}]
        }
        resp = app_client.post(
            "/v1/memories/extract",
            headers={"authorization": "Bearer test-key-123"},
            json={
                "user_id": "u1",
                "app_id": "app1",
                "messages": [{"role": "user", "content": "我是一名Java工程师"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["added"] == 1


class TestPostSearch:
    def test_search_success(self, app_client, mock_mem0):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        mock_mem0.search.return_value = {
            "results": [{
                "id": "m1", "memory": "是工程师", "score": 0.9,
                "metadata": {"memory_layer": "L1", "scope": "shared", "app_id": "app1"},
                "created_at": now.isoformat(), "updated_at": now.isoformat(),
            }]
        }
        resp = app_client.post(
            "/v1/memories/search",
            headers={"authorization": "Bearer test-key-123"},
            json={"user_id": "u1", "app_id": "app1", "query": "工程师"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) >= 1
        assert "confidence" in data["results"][0]

    def test_search_degradation_on_error(self, app_client, mock_mem0):
        mock_mem0.search.side_effect = Exception("DB down")
        resp = app_client.post(
            "/v1/memories/search",
            headers={"authorization": "Bearer test-key-123"},
            json={"user_id": "u1", "app_id": "app1", "query": "test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []
        assert data["total"] == 0


class TestGetMemories:
    def test_get_all_memories(self, app_client, mock_mem0):
        mock_mem0.get_all.return_value = {"results": []}
        resp = app_client.get(
            "/v1/memories?user_id=u1&app_id=app1",
            headers={"authorization": "Bearer test-key-123"},
        )
        assert resp.status_code == 200


class TestDeleteMemory:
    def test_delete_success(self, app_client, mock_mem0):
        mock_mem0.delete.return_value = {"message": "ok"}
        resp = app_client.delete(
            "/v1/memories/m1?user_id=u1&app_id=app1",
            headers={"authorization": "Bearer test-key-123"},
        )
        assert resp.status_code == 200
