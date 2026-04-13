"""跨 Collection 搜索服务 — 多应用共享记忆检索

核心职责：
1. 遍历多个应用的 Collection 进行搜索
2. 合并、去重、排序结果
3. 过滤 private 记忆（跨 Collection 时只返回 shared）

使用场景：
- scope=shared 时，需要搜索用户在所有应用中的共享记忆
- 多应用协作场景，如 HR 助手查询培训助手的用户信息

使用示例：
    >>> searcher = CrossCollectionSearcher(get_memory_for_app=memory_factory)
    >>> results = searcher.search("Python", "u1", ["app-1", "app-2"], limit=10)
"""
import logging
from typing import Callable

from mem0 import Memory

logger = logging.getLogger(__name__)


class CrossCollectionSearcher:
    """跨 Collection 搜索器

    核心流程：
    1. 遍历 app_ids，获取每个应用的 Memory 实例
    2. 并行（或串行）调用 search
    3. 合并结果，过滤 private
    4. 基于 hash 去重
    5. 按 score 降序排序
    6. 返回前 limit 条
    """

    def __init__(
        self,
        get_memory_for_app: Callable[[str], Memory | None],
    ):
        """初始化搜索器

        Args:
            get_memory_for_app: 根据 app_id 获取 Memory 实例的工厂函数
        """
        self.get_memory_for_app = get_memory_for_app

    def search(
        self,
        query: str,
        user_id: str,
        app_ids: list[str],
        limit: int = 10,
        scope: str = "shared",
        filters: dict | None = None,
    ) -> list[dict]:
        """跨 Collection 搜索

        Args:
            query: 搜索查询文本
            user_id: 用户唯一标识
            app_ids: 要搜索的应用 ID 列表
            limit: 最大返回数量
            scope: 可见性过滤，跨 Collection 时通常为 "shared"
            filters: 额外的过滤条件

        Returns:
            合并、去重、排序后的记忆列表

        Example:
            >>> results = searcher.search("Python", "u1", ["app-1", "app-2"])
            >>> len(results)
            5
        """
        all_results: list[dict] = []
        seen_hashes: set[str] = set()

        for app_id in app_ids:
            # 获取 Memory 实例
            memory = self.get_memory_for_app(app_id)
            if memory is None:
                logger.warning("Memory instance not found for app: %s", app_id)
                continue

            try:
                # 搜索该 Collection
                raw = memory.search(
                    query,
                    user_id=user_id,
                    agent_id=app_id,
                    limit=limit * 2,  # 获取更多，为去重留余量
                    filters=filters,
                )

                for mem in raw.get("results", []):
                    # 跨 Collection 时只返回 shared 记忆
                    mem_scope = (mem.get("metadata") or {}).get("scope", "shared")
                    if scope == "shared" and mem_scope != "shared":
                        continue

                    # 基于 hash 去重
                    mem_hash = mem.get("hash")
                    if mem_hash and mem_hash in seen_hashes:
                        continue
                    if mem_hash:
                        seen_hashes.add(mem_hash)

                    # 添加来源应用标识
                    mem["_source_app_id"] = app_id
                    all_results.append(mem)

            except Exception as e:
                logger.error("Search failed for app %s: %s", app_id, e)
                continue

        # 按 score 降序排序
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

        # 返回前 limit 条
        return all_results[:limit]
