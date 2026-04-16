"""MySQL-backed storage manager — replaces mem0's SQLiteManager."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from memory_platform.db.connection import MySQLConnectionPool

logger = logging.getLogger(__name__)


class MySQLManager:
    """Drop-in replacement for mem0's SQLiteManager using MySQL."""

    def __init__(self, db: MySQLConnectionPool) -> None:
        self.db = db
        self._create_history_table()

    def _create_history_table(self) -> None:
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS history (
                        id           VARCHAR(36) PRIMARY KEY,
                        memory_id    VARCHAR(36),
                        old_memory   TEXT,
                        new_memory   TEXT,
                        event        VARCHAR(16),
                        created_at   DATETIME,
                        updated_at   DATETIME,
                        is_deleted   TINYINT DEFAULT 0,
                        actor_id     VARCHAR(255),
                        role         VARCHAR(64),
                        user_id      VARCHAR(255)
                    )
                    """
                )
                # 兼容旧表：如果缺少 user_id 列则自动添加
                cursor.execute(
                    """
                    SELECT COUNT(*) AS cnt FROM information_schema.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = 'history'
                      AND COLUMN_NAME = 'user_id'
                    """
                )
                row = cursor.fetchone()
                if row and row["cnt"] == 0:
                    cursor.execute(
                        "ALTER TABLE history ADD COLUMN user_id VARCHAR(255) AFTER role"
                    )
        finally:
            self.db.return_connection(conn)

    def add_history(
        self,
        memory_id: str,
        old_memory: str | None,
        new_memory: str | None,
        event: str,
        *,
        created_at: str | None = None,
        updated_at: str | None = None,
        is_deleted: int = 0,
        actor_id: str | None = None,
        role: str | None = None,
        user_id: str | None = None,
    ) -> None:
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO history (
                        id, memory_id, old_memory, new_memory, event,
                        created_at, updated_at, is_deleted, actor_id, role, user_id
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()),
                        memory_id,
                        old_memory,
                        new_memory,
                        event,
                        created_at,
                        updated_at,
                        is_deleted,
                        actor_id,
                        role,
                        user_id,
                    ),
                )
        except Exception as e:
            logger.error("Failed to add history record: %s", e)
            raise
        finally:
            self.db.return_connection(conn)

    def get_history(self, memory_id: str) -> list[dict[str, Any]]:
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, memory_id, old_memory, new_memory, event,
                           created_at, updated_at, is_deleted, actor_id, role, user_id
                    FROM history
                    WHERE memory_id = %s
                    ORDER BY created_at ASC, updated_at ASC
                    """,
                    (memory_id,),
                )
                rows = cursor.fetchall()
        finally:
            self.db.return_connection(conn)

        return [
            {
                "id": r["id"],
                "memory_id": r["memory_id"],
                "old_memory": r["old_memory"],
                "new_memory": r["new_memory"],
                "event": r["event"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "is_deleted": bool(r["is_deleted"]),
                "actor_id": r["actor_id"],
                "role": r["role"],
                "user_id": r.get("user_id"),
            }
            for r in rows
        ]

    def reset(self) -> None:
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS history")
            self._create_history_table()
        except Exception as e:
            logger.error("Failed to reset history table: %s", e)
            raise
        finally:
            self.db.return_connection(conn)

    def close(self) -> None:
        """No-op — connection pool manages lifecycle."""
        pass
