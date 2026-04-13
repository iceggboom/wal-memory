"""AppRegistry — apps table CRUD for application management."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from memory_platform.db.connection import MySQLConnectionPool

logger = logging.getLogger(__name__)


class AppRegistry:
    """Manages the apps table in MySQL."""

    def __init__(self, db: MySQLConnectionPool) -> None:
        self.db = db
        self._create_table()

    def _create_table(self) -> None:
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS apps (
                        app_id     VARCHAR(64) PRIMARY KEY,
                        name       VARCHAR(255) NOT NULL,
                        api_key    VARCHAR(255) NOT NULL UNIQUE,
                        status     VARCHAR(16) NOT NULL DEFAULT 'active',
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    )
                    """
                )
        finally:
            self.db.return_connection(conn)

    def register(self, app_id: str, name: str, api_key: str) -> dict[str, Any]:
        """Register a new application."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO apps (app_id, name, api_key, status, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (app_id, name, api_key, "active", now, now),
                )
        finally:
            self.db.return_connection(conn)

        return {
            "app_id": app_id,
            "name": name,
            "api_key": api_key,
            "status": "active",
            "created_at": now,
            "updated_at": now,
        }

    def get(self, app_id: str) -> dict[str, Any] | None:
        """Get an app by app_id."""
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT app_id, name, api_key, status, created_at, updated_at FROM apps WHERE app_id = %s",
                    (app_id,),
                )
                return cursor.fetchone()
        finally:
            self.db.return_connection(conn)

    def get_by_api_key(self, api_key: str) -> dict[str, Any] | None:
        """Get an app by its API key."""
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT app_id, name, api_key, status, created_at, updated_at FROM apps WHERE api_key = %s AND status = 'active'",
                    (api_key,),
                )
                return cursor.fetchone()
        finally:
            self.db.return_connection(conn)

    def list_apps(self) -> list[dict[str, Any]]:
        """List all registered applications."""
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT app_id, name, api_key, status, created_at, updated_at FROM apps ORDER BY created_at"
                )
                return cursor.fetchall()
        finally:
            self.db.return_connection(conn)

    def update_status(self, app_id: str, status: str) -> None:
        """Update app status (active/inactive)."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "UPDATE apps SET status = %s, updated_at = %s WHERE app_id = %s",
                    (status, now, app_id),
                )
        finally:
            self.db.return_connection(conn)

    def count(self) -> int:
        """Count active apps."""
        conn = self.db.get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as cnt FROM apps WHERE status = 'active'")
                row = cursor.fetchone()
                return row["cnt"] if row else 0
        finally:
            self.db.return_connection(conn)
