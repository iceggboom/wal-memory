"""Tests for MySQL connection pool."""

from unittest.mock import MagicMock, patch

from memory_platform.db.connection import MySQLConnectionPool


class TestMySQLConnectionPool:
    def test_init_creates_pool(self):
        """Pool should create connections on init."""
        with patch("memory_platform.db.connection.pymysql") as mock_pymysql:
            mock_conn = MagicMock()
            mock_pymysql.connect.return_value = mock_conn

            MySQLConnectionPool(
                host="localhost",
                port=3306,
                database="test_db",
                username="root",
                password="secret",
                pool_size=3,
            )

            assert mock_pymysql.connect.call_count == 3

    def test_get_connection_returns_connection(self):
        """get_connection should return a connection from the pool."""
        with patch("memory_platform.db.connection.pymysql") as mock_pymysql:
            mock_conn = MagicMock()
            mock_pymysql.connect.return_value = mock_conn

            pool = MySQLConnectionPool(
                host="localhost", port=3306, database="test",
                username="root", password="", pool_size=2,
            )

            conn = pool.get_connection()
            assert conn is not None

    def test_return_connection_puts_back(self):
        """return_connection should put connection back to pool."""
        with patch("memory_platform.db.connection.pymysql") as mock_pymysql:
            mock_conn = MagicMock()
            mock_pymysql.connect.return_value = mock_conn

            pool = MySQLConnectionPool(
                host="localhost", port=3306, database="test",
                username="root", password="", pool_size=1,
            )

            conn = pool.get_connection()
            pool.return_connection(conn)

            conn2 = pool.get_connection()
            assert conn2 is not None

    def test_close_all(self):
        """close_all should close all connections."""
        with patch("memory_platform.db.connection.pymysql") as mock_pymysql:
            mock_conn = MagicMock()
            mock_pymysql.connect.return_value = mock_conn

            pool = MySQLConnectionPool(
                host="localhost", port=3306, database="test",
                username="root", password="", pool_size=2,
            )

            pool.close_all()

            assert mock_conn.close.call_count == 2
