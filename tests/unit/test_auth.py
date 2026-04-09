import pytest
from unittest.mock import MagicMock

from memory_platform.middleware.auth import get_api_key, require_auth


class TestGetApiKey:
    def test_extracts_bearer_token(self):
        request = MagicMock()
        request.headers = {"authorization": "Bearer test-key"}
        assert get_api_key(request) == "test-key"

    def test_returns_none_if_no_header(self):
        request = MagicMock()
        request.headers = {}
        assert get_api_key(request) is None

    def test_returns_none_for_non_bearer(self):
        request = MagicMock()
        request.headers = {"authorization": "Basic abc123"}
        assert get_api_key(request) is None


class TestRequireAuth:
    def test_valid_key_passes(self, mock_env):
        result = require_auth("test-key-123")
        assert result is True

    def test_invalid_key_raises(self, mock_env):
        with pytest.raises(ValueError, match="Invalid API key"):
            require_auth("wrong-key")

    def test_none_key_raises(self):
        with pytest.raises(ValueError, match="Missing API key"):
            require_auth(None)

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="Missing API key"):
            require_auth("")
