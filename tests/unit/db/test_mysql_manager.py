"""Tests for MySQLManager — mem0 history storage backed by MySQL."""

from unittest.mock import MagicMock

from memory_platform.db.mysql_manager import MySQLManager


class TestMySQLManager:
    def _make_manager(self):
        """Create a MySQLManager with a mocked connection pool."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.get_connection.return_value = mock_conn
        # Make __enter__ return the cursor mock itself so with-statement works
        cursor_mock = mock_conn.cursor.return_value
        cursor_mock.__enter__ = MagicMock(return_value=cursor_mock)
        cursor_mock.__exit__ = MagicMock(return_value=False)
        return MySQLManager(db=mock_pool), mock_conn

    def test_create_history_table_on_init(self):
        """MySQLManager should create history table on init."""
        manager, mock_conn = self._make_manager()
        cursor = mock_conn.cursor.return_value

        create_calls = [
            c for c in cursor.execute.call_args_list
            if "CREATE TABLE" in str(c)
        ]
        assert len(create_calls) >= 1

    def test_add_history_inserts_record(self):
        """add_history should INSERT a record."""
        manager, mock_conn = self._make_manager()
        cursor = mock_conn.cursor.return_value

        manager.add_history(
            memory_id="mem-1",
            old_memory=None,
            new_memory="test memory",
            event="ADD",
            created_at="2026-04-08T00:00:00",
            updated_at="2026-04-08T00:00:00",
        )

        insert_calls = [
            c for c in cursor.execute.call_args_list
            if "INSERT" in str(c)
        ]
        assert len(insert_calls) >= 1

    def test_get_history_returns_list(self):
        """get_history should return list of dicts."""
        manager, mock_conn = self._make_manager()
        cursor = mock_conn.cursor.return_value
        cursor.fetchall.return_value = [
            {
                "id": "h1",
                "memory_id": "mem-1",
                "old_memory": None,
                "new_memory": "test",
                "event": "ADD",
                "created_at": "2026-04-08T00:00:00",
                "updated_at": "2026-04-08T00:00:00",
                "is_deleted": 0,
                "actor_id": None,
                "role": None,
            }
        ]

        result = manager.get_history(memory_id="mem-1")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["memory_id"] == "mem-1"
        assert result[0]["event"] == "ADD"

    def test_reset_drops_and_recreates(self):
        """reset should DROP and re-create the history table."""
        manager, mock_conn = self._make_manager()
        cursor = mock_conn.cursor.return_value

        manager.reset()

        drop_calls = [
            c for c in cursor.execute.call_args_list
            if "DROP TABLE" in str(c)
        ]
        assert len(drop_calls) >= 1

    def test_returns_connection_to_pool(self):
        """Each method should return connection to pool after use."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.get_connection.return_value = mock_conn

        manager = MySQLManager(db=mock_pool)
        mock_conn.cursor.return_value.fetchall.return_value = []

        manager.get_history(memory_id="mem-1")

        mock_pool.return_connection.assert_called_with(mock_conn)
