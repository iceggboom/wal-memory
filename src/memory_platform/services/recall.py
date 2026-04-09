"""记忆检索服务 — 负责记忆的搜索、查询、删除

核心职责：
1. 向量相似度搜索
2. 置信度衰减计算和过滤
3. 按 layer/scope 过滤
4. 获取全部记忆
5. 删除记忆

检索流程：
    search（搜索）:
    1. 调用 mem0.search 获取候选记忆
    2. 按 scope 过滤（shared/private/all）
    3. 按 layer 过滤（L1/L2/L3/L4）
    4. 计算置信度（similarity × 时间衰减）
    5. 按置信度排序，返回 top N

    get_all（获取全部）:
    1. 调用 mem0.get_all 获取记忆
    2. 按 scope/layer 过滤
    3. 格式化返回

使用示例：
    >>> svc = RecallService(mem0)
    >>> # 搜索记忆
    >>> results = svc.search("Python工程师", user_id="u1", agent_id="a1")
    >>> # 获取全部记忆
    >>> all_memories = svc.get_all(user_id="u1", agent_id="a1")
"""
from datetime import datetime
from typing import TYPE_CHECKING

from mem0 import Memory

from memory_platform.ext.confidence import filter_by_confidence
from memory_platform.ext.layer import parse_layer_filter
from memory_platform.ext.scope import Scope, apply_scope_filter

if TYPE_CHECKING:
    from memory_platform.services.cross_collection import CrossCollectionSearcher


class RecallService:
    """记忆检索服务

    提供三种操作：
    1. search: 向量相似度搜索 + 置信度过滤
    2. get_all: 获取用户全部记忆
    3. delete: 删除指定记忆
    """

    def __init__(
        self,
        mem0: Memory,
        cross_collection_searcher: "CrossCollectionSearcher | None" = None,
    ):
        """初始化服务

        Args:
            mem0: mem0 Memory 实例
            cross_collection_searcher: 可选的跨 Collection 搜索器
        """
        self.mem0 = mem0
        self.cross_collection_searcher = cross_collection_searcher

    def search(
        self,
        query: str,
        user_id: str,
        agent_id: str,
        limit: int = 10,
        min_confidence: float = 0.5,
        memory_layer: str | None = None,
        scope: str = "all",
        now: datetime | None = None,
        all_app_ids: list[str] | None = None,
    ) -> list[dict]:
        """搜索记忆

        核心流程：
        1. 判断是否需要跨 Collection 搜索
        2. 向量相似度搜索（单 Collection 或跨 Collection）
        3. Scope 过滤（shared/private/all）
        4. Layer 过滤（L1/L2/L3/L4）
        5. 置信度衰减计算
        6. 过滤低置信度记忆
        7. 按置信度降序排序

        Args:
            query: 搜索查询文本
            user_id: 用户唯一标识
            agent_id: 应用唯一标识
            limit: 最大返回数量
            min_confidence: 最小置信度阈值 (0-1)
            memory_layer: 层级过滤，如 "L1,L2"，None 表示全部
            scope: 可见性过滤，"all"/"shared"/"private"
            now: 当前时间（测试时可注入）
            all_app_ids: 所有已注册应用 ID 列表，用于跨 Collection 搜索

        Returns:
            记忆列表，每项包含：
            - id: 记忆 ID
            - text: 记忆文本
            - memory_layer: 记忆层级
            - confidence: 置信度 (0-1)
            - similarity: 原始相似度
            - scope: 可见性
            - created_at/updated_at: 时间戳

        Example:
            >>> results = svc.search("Python", "u1", "a1", limit=5)
            >>> len(results)
            3
        """
        # Step 1: 判断搜索策略
        if (
            scope in ("shared", "all")
            and self.cross_collection_searcher is not None
            and all_app_ids is not None
            and len(all_app_ids) > 1
        ):
            # 跨 Collection 搜索 shared 记忆
            memories = self._search_cross_collection(
                query=query,
                user_id=user_id,
                all_app_ids=all_app_ids,
                limit=limit * 3,
            )
        else:
            # 单 Collection 搜索
            raw = self.mem0.search(
                query,
                user_id=user_id,
                agent_id=agent_id,
                limit=limit * 3,
            )
            memories = raw.get("results", [])

        # Step 2: Scope 过滤
        scope_enum = Scope(scope)
        memories = apply_scope_filter(memories, scope_enum)

        # Step 3: Layer 过滤
        allowed_layers = parse_layer_filter(memory_layer)
        allowed_values = [layer.value for layer in allowed_layers]
        memories = [
            m
            for m in memories
            if (m.get("metadata") or {}).get("memory_layer", "L1") in allowed_values
        ]

        # Step 4-6: 置信度计算 + 过滤 + 排序
        scored = filter_by_confidence(
            memories, min_confidence=min_confidence, limit=limit, now=now
        )

        # 格式化返回结果
        results = []
        for mem, confidence in scored:
            metadata = mem.get("metadata") or {}
            results.append(
                {
                    "id": mem["id"],
                    "text": mem["memory"],
                    "memory_layer": metadata.get("memory_layer", "L1"),
                    "confidence": round(confidence, 4),
                    "similarity": mem.get("score", 0.0),
                    "scope": metadata.get("scope", "shared"),
                    "created_at": mem.get("created_at", ""),
                    "updated_at": mem.get("updated_at", ""),
                    "source_app_id": mem.get("_source_app_id", agent_id),
                }
            )

        return results

    def get_all(
        self,
        user_id: str,
        agent_id: str,
        memory_layer: str | None = None,
        scope: str = "all",
        limit: int = 100,
    ) -> list[dict]:
        """获取用户全部记忆

        核心流程：
        1. 调用 mem0.get_all
        2. 按 scope/layer 过滤
        3. 格式化返回

        Args:
            user_id: 用户唯一标识
            agent_id: 应用唯一标识
            memory_layer: 层级过滤
            scope: 可见性过滤
            limit: 最大返回数量

        Returns:
            记忆列表

        Example:
            >>> memories = svc.get_all("u1", "a1")
            >>> len(memories)
            5
        """
        raw = self.mem0.get_all(
            user_id=user_id,
            agent_id=agent_id,
            limit=limit,
        )
        memories = raw.get("results", [])

        # Scope 过滤
        scope_enum = Scope(scope)
        memories = apply_scope_filter(memories, scope_enum)

        # Layer 过滤
        allowed_layers = parse_layer_filter(memory_layer)
        allowed_values = [layer.value for layer in allowed_layers]
        memories = [
            m
            for m in memories
            if (m.get("metadata") or {}).get("memory_layer", "L1") in allowed_values
        ]

        # 格式化返回
        results = []
        for mem in memories:
            metadata = mem.get("metadata") or {}
            results.append(
                {
                    "id": mem["id"],
                    "text": mem["memory"],
                    "memory_layer": metadata.get("memory_layer", "L1"),
                    "scope": metadata.get("scope", "shared"),
                    "created_at": mem.get("created_at", ""),
                    "updated_at": mem.get("updated_at", ""),
                }
            )

        return results

    def _search_cross_collection(
        self,
        query: str,
        user_id: str,
        all_app_ids: list[str],
        limit: int,
    ) -> list[dict]:
        """跨 Collection 搜索实现

        Args:
            query: 搜索查询
            user_id: 用户 ID
            all_app_ids: 所有应用 ID
            limit: 结果数量限制

        Returns:
            合并后的记忆列表
        """
        if self.cross_collection_searcher is None:
            return []

        return self.cross_collection_searcher.search(
            query=query,
            user_id=user_id,
            app_ids=all_app_ids,
            limit=limit,
            scope="shared",
        )

    def delete(self, memory_id: str, user_id: str, agent_id: str) -> None:
        """删除指定记忆

        Args:
            memory_id: 记忆唯一标识
            user_id: 用户唯一标识（用于权限校验）
            agent_id: 应用唯一标识（用于权限校验）

        Note:
            当前实现直接调用 mem0.delete，未校验 user_id/agent_id
            TODO: 添加权限校验
        """
        self.mem0.delete(memory_id)
