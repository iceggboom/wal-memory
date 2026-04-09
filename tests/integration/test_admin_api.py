"""Integration tests for Admin API."""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_mem0():
    return MagicMock()


@pytest.fixture
def mock_app_registry():
    registry = MagicMock()
    registry.list_apps.return_value = []
    registry.count.return_value = 0
    # Valid key matches the env-based key for backward compatibility
    registry.get_by_api_key = MagicMock(
        side_effect=lambda k: {"app_id": "test-app", "name": "Test", "api_key": k, "status": "active"}
        if k == "test-key-123" else None
    )
    return registry


@pytest.fixture
def app_client(mock_mem0, mock_app_registry, mock_env):
    from memory_platform.api.admin import create_router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(create_router(mock_mem0, app_registry=mock_app_registry))
    return TestClient(app)


class TestAdminApps:
    def test_list_apps_empty(self, app_client, mock_env):
        resp = app_client.get("/v1/admin/apps", headers={"authorization": "Bearer test-key-123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["apps"] == []

    def test_register_app(self, app_client, mock_app_registry, mock_env):
        mock_app_registry.register.return_value = {
            "app_id": "new-app",
            "name": "New App",
            "api_key": "mpk-generated-key",
            "status": "active",
        }

        resp = app_client.post(
            "/v1/admin/apps",
            headers={"authorization": "Bearer test-key-123"},
            json={"app_id": "new-app", "name": "New App"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["app_id"] == "new-app"
        assert data["name"] == "New App"
        assert data["status"] == "active"
        assert "api_key" in data

    def test_register_app_returns_generated_key(self, app_client, mock_app_registry, mock_env):
        mock_app_registry.register.return_value = {
            "app_id": "test-app",
            "name": "Test",
            "api_key": "mpk-abc123",
            "status": "active",
        }

        resp = app_client.post(
            "/v1/admin/apps",
            headers={"authorization": "Bearer test-key-123"},
            json={"app_id": "test-app", "name": "Test"},
        )
        data = resp.json()
        assert data["api_key"].startswith("mpk-")


class TestAdminUsers:
    def test_get_user_memories(self, app_client, mock_mem0, mock_env):
        mock_mem0.get_all.return_value = {"results": []}
        resp = app_client.get(
            "/v1/admin/users/u1/memories?app_id=app1",
            headers={"authorization": "Bearer test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []
        assert data["total"] == 0

    def test_delete_user_memories(self, app_client, mock_mem0, mock_env):
        mock_mem0.delete_all.return_value = {"message": "ok"}
        resp = app_client.delete(
            "/v1/admin/users/u1/memories?app_id=app1",
            headers={"authorization": "Bearer test-key-123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["message"] == "User memories deleted successfully"


class TestAdminStats:
    def test_get_stats(self, app_client, mock_app_registry, mock_env):
        mock_app_registry.count.return_value = 3

        resp = app_client.get("/v1/admin/stats", headers={"authorization": "Bearer test-key-123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_apps"] == 3


class TestAdminAuth:
    def test_unauthorized_access(self, app_client, mock_env):
        resp = app_client.get("/v1/admin/apps")
        assert resp.status_code == 401

    def test_invalid_api_key(self, app_client, mock_env):
        resp = app_client.get("/v1/admin/apps", headers={"authorization": "Bearer wrong-key"})
        assert resp.status_code == 401
