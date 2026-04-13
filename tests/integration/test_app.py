import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(mock_env):
    from unittest.mock import MagicMock

    mock_mem0 = MagicMock()
    from memory_platform.main import create_app
    app = create_app(mem0=mock_mem0)
    yield TestClient(app), mock_mem0


class TestHealthEndpoint:
    def test_health(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestOpenAPIDocs:
    def test_docs_available(self, client):
        c, _ = client
        resp = c.get("/docs")
        assert resp.status_code == 200

    def test_openapi_json(self, client):
        c, _ = client
        resp = c.get("/openapi.json")
        assert resp.status_code == 200
