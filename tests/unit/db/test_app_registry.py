"""Tests for AppRegistry — apps table CRUD."""

from datetime import datetime
from unittest.mock import MagicMock

from memory_platform.db.app_registry import AppRegistry


class TestAppRegistry:
    def _make_registry(self):
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.get_connection.return_value = mock_conn
        # Make __enter__ return the cursor mock itself so with-statement works
        cursor_mock = mock_conn.cursor.return_value
        cursor_mock.__enter__ = MagicMock(return_value=cursor_mock)
        cursor_mock.__exit__ = MagicMock(return_value=False)
        return AppRegistry(db=mock_pool), mock_conn

    def test_register_inserts_app(self):
        """register should INSERT into apps table."""
        registry, mock_conn = self._make_registry()
        cursor = mock_conn.cursor.return_value

        registry.register(
            app_id="hr-assistant",
            name="HR 助手",
            api_key="key-hr-001",
        )

        insert_calls = [
            c for c in cursor.execute.call_args_list
            if "INSERT" in str(c)
        ]
        assert len(insert_calls) >= 1

    def test_list_apps_returns_list(self):
        """list_apps should return list of app dicts."""
        registry, mock_conn = self._make_registry()
        cursor = mock_conn.cursor.return_value
        cursor.fetchall.return_value = [
            {
                "app_id": "hr-assistant",
                "name": "HR 助手",
                "api_key": "key-hr-001",
                "status": "active",
                "created_at": datetime(2026, 4, 8),
                "updated_at": datetime(2026, 4, 8),
            }
        ]

        result = registry.list_apps()

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["app_id"] == "hr-assistant"

    def test_get_by_api_key_returns_app(self):
        """get_by_api_key should return app dict or None."""
        registry, mock_conn = self._make_registry()
        cursor = mock_conn.cursor.return_value
        cursor.fetchone.return_value = {
            "app_id": "hr-assistant",
            "name": "HR 助手",
            "api_key": "key-hr-001",
            "status": "active",
            "created_at": datetime(2026, 4, 8),
            "updated_at": datetime(2026, 4, 8),
        }

        result = registry.get_by_api_key("key-hr-001")

        assert result is not None
        assert result["app_id"] == "hr-assistant"

    def test_get_by_api_key_returns_none_for_unknown(self):
        """get_by_api_key should return None for unknown key."""
        registry, mock_conn = self._make_registry()
        cursor = mock_conn.cursor.return_value
        cursor.fetchone.return_value = None

        result = registry.get_by_api_key("unknown-key")

        assert result is None

    def test_count_returns_number(self):
        """count should return number of active apps."""
        registry, mock_conn = self._make_registry()
        cursor = mock_conn.cursor.return_value
        cursor.fetchone.return_value = {"cnt": 5}

        result = registry.count()

        assert result == 5

    def test_returns_connection_to_pool(self):
        """Each method should return connection to pool after use."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.get_connection.return_value = mock_conn
        registry = AppRegistry(db=mock_pool)
        cursor = mock_conn.cursor.return_value
        cursor.fetchall.return_value = []

        registry.list_apps()

        mock_pool.return_connection.assert_called_with(mock_conn)
