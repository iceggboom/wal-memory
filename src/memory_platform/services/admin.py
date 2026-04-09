"""管理服务 — 用户记忆的管理操作

核心职责：
1. 获取指定用户的全部记忆
2. 删除指定用户的全部记忆
3. 应用注册管理（可选 MySQL 持久化）

使用场景：
- 用户注销时清理数据
- 管理员查看用户记忆
- 数据导出/迁移

使用示例：
    >>> svc = AdminService(mem0)
    >>> # 获取用户记忆
    >>> memories = svc.get_user_memories("user-1", "app-1")
    >>> # 删除用户记忆
    >>> svc.delete_user_memories("user-1", "app-1")
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mem0 import Memory

if TYPE_CHECKING:
    from memory_platform.db.app_registry import AppRegistry


class AdminService:
    """管理服务

    提供用户级别的记忆管理操作和应用注册管理。
    """

    def __init__(self, mem0: Memory, app_registry: AppRegistry | None = None):
        self.mem0 = mem0
        self.app_registry = app_registry

    def register_app(self, app_id: str, name: str, api_key: str) -> dict[str, Any]:
        """Register a new application."""
        if self.app_registry is None:
            raise RuntimeError("App registry not configured (MySQL not enabled)")
        return self.app_registry.register(app_id=app_id, name=name, api_key=api_key)

    def list_apps(self) -> list[dict[str, Any]]:
        """List all registered applications."""
        if self.app_registry is None:
            return []
        return self.app_registry.list_apps()

    def stats(self) -> dict[str, Any]:
        """Get platform statistics."""
        if self.app_registry is None:
            return {"total_memories": 0, "total_users": 0, "total_apps": 0}
        return {
            "total_apps": self.app_registry.count(),
            "total_memories": 0,
            "total_users": 0,
        }

    def get_user_memories(
        self, user_id: str, agent_id: str, limit: int = 100
    ) -> list[dict]:
        raw = self.mem0.get_all(
            user_id=user_id, agent_id=agent_id, limit=limit
        )
        return raw.get("results", [])

    def delete_user_memories(self, user_id: str, agent_id: str) -> None:
        self.mem0.delete_all(user_id=user_id, agent_id=agent_id)
