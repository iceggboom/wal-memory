"""Scope 过滤模块 — 多应用隔离的记忆可见性控制

核心职责：
1. 定义记忆的可见性范围（Scope）
2. 按 Scope 过滤记忆列表
3. 去重记忆（基于 hash）

Scope 类型：
- shared: 共享记忆，跨应用可见（如用户职业、偏好）
- private: 私有记忆，仅当前应用可见（如会话上下文）
- all: 包含 shared 和 private

使用场景：
- 多应用共享同一用户的记忆库
- 某些记忆需要应用间隔离

使用示例：
    >>> from memory_platform.ext.scope import Scope, apply_scope_filter
    >>> memories = [{"metadata": {"scope": "shared"}}, {"metadata": {"scope": "private"}}]
    >>> filtered = apply_scope_filter(memories, Scope.SHARED)
    >>> len(filtered)
    1
"""
from enum import Enum


class Scope(str, Enum):
    """记忆可见性范围枚举

    Attributes:
        ALL: 包含所有记忆（shared + private）
        SHARED: 仅共享记忆，跨应用可见
        PRIVATE: 仅私有记忆，应用内可见
    """

    ALL = "all"
    SHARED = "shared"
    PRIVATE = "private"

    @property
    def include_shared(self) -> bool:
        """是否包含 shared 记忆"""
        return self in (Scope.ALL, Scope.SHARED)

    @property
    def include_private(self) -> bool:
        """是否包含 private 记忆"""
        return self in (Scope.ALL, Scope.PRIVATE)


def apply_scope_filter(memories: list[dict], scope: Scope) -> list[dict]:
    """按 Scope 过滤记忆列表

    过滤规则：
    - scope=ALL: 返回所有记忆
    - scope=SHARED: 仅返回 metadata.scope="shared" 的记忆
    - scope=PRIVATE: 仅返回 metadata.scope="private" 的记忆
    - 未设置 scope 的记忆默认视为 shared

    Args:
        memories: 记忆列表
        scope: 过滤范围

    Returns:
        过滤后的记忆列表

    Example:
        >>> memories = [{"metadata": {"scope": "shared"}}]
        >>> apply_scope_filter(memories, Scope.PRIVATE)
        []
    """
    filtered = []
    for mem in memories:
        # 提取记忆的 scope，默认为 shared
        mem_scope = (mem.get("metadata") or {}).get("scope", "shared")

        # 根据过滤条件判断是否保留
        if mem_scope == "shared" and scope.include_shared:
            filtered.append(mem)
        elif mem_scope == "private" and scope.include_private:
            filtered.append(mem)

    return filtered


def deduplicate_memories(memories: list[dict]) -> list[dict]:
    """去重记忆列表

    基于 memory.hash 字段去重，保留首次出现的记忆。

    Args:
        memories: 记忆列表

    Returns:
        去重后的记忆列表

    Example:
        >>> memories = [{"hash": "a1"}, {"hash": "a1"}, {"hash": "b2"}]
        >>> deduplicate_memories(memories)
        [{"hash": "a1"}, {"hash": "b2"}]
    """
    seen: set[str] = set()
    unique = []

    for mem in memories:
        h = mem.get("hash")
        # 已存在则跳过
        if h and h in seen:
            continue
        # 记录 hash
        if h:
            seen.add(h)
        unique.append(mem)

    return unique
