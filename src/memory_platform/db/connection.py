"""MySQL connection pool management."""

import logging
import queue

import pymysql
import pymysql.cursors

logger = logging.getLogger(__name__)


class MySQLConnectionPool:
    """Simple MySQL connection pool using queue.Queue."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3306,
        database: str = "memory_platform",
        username: str = "root",
        password: str = "",
        pool_size: int = 5,
    ) -> None:
        self._pool: queue.Queue = queue.Queue(maxsize=pool_size)
        self._all_connections: list = []

        for _ in range(pool_size):
            conn = pymysql.connect(
                host=host,
                port=port,
                user=username,
                password=password,
                database=database,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True,
            )
            self._pool.put(conn)
            self._all_connections.append(conn)

        logger.info(
            "MySQL connection pool created: %s:%d/%s pool_size=%d",
            host, port, database, pool_size,
        )

    def get_connection(self):
        """Get a connection from the pool (blocking)."""
        return self._pool.get()

    def return_connection(self, conn) -> None:
        """Return a connection back to the pool."""
        self._pool.put(conn)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        for conn in self._all_connections:
            try:
                conn.close()
            except Exception:
                pass
        self._all_connections.clear()
        logger.info("MySQL connection pool closed")
