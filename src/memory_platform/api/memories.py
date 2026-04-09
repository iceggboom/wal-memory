"""记忆 API 路由 — 提供记忆的 CRUD 和搜索接口

核心职责：
1. POST /v1/memories — 批量添加记忆
2. POST /v1/memories/extract — 从对话中提取记忆
3. POST /v1/memories/search — 搜索记忆
4. GET /v1/memories — 获取全部记忆
5. DELETE /v1/memories/{id} — 删除记忆

认证方式：
- 所有接口需要 Bearer Token 认证
- Header: Authorization: Bearer <api_key>

API 示例：
    # 添加记忆
    POST /v1/memories
    {
        "user_id": "u1",
        "app_id": "a1",
        "memories": [{"text": "我是Python工程师"}]
    }

    # 搜索记忆
    POST /v1/memories/search
    {
        "user_id": "u1",
        "app_id": "a1",
        "query": "Python"
    }
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query, Request
from mem0 import Memory
from pydantic import BaseModel

from memory_platform.middleware.auth import get_api_key, require_auth
from memory_platform.middleware.degradation import memory_degradation_handler
from memory_platform.services.write import WriteService, AddMemoryItem
from memory_platform.services.recall import RecallService

if TYPE_CHECKING:
    from anthropic import Anthropic
    from memory_platform.services.cross_collection import CrossCollectionSearcher

logger = logging.getLogger(__name__)


# ============================================================================
# Request Models
# ============================================================================


class MemoriesRequest(BaseModel):
    """批量添加记忆请求

    Attributes:
        user_id: 用户唯一标识
        app_id: 应用唯一标识
        memories: 记忆项列表
    """

    user_id: str
    app_id: str
    memories: list[AddMemoryItem]


class ExtractRequest(BaseModel):
    """从对话提取记忆请求

    Attributes:
        user_id: 用户唯一标识
        app_id: 应用唯一标识
        messages: 对话消息列表 [{"role": "user", "content": "..."}]
        memory_layer: 显式指定的记忆层级（可选）
    """

    user_id: str
    app_id: str
    messages: list[dict]
    memory_layer: str | None = None


class SearchRequest(BaseModel):
    """搜索记忆请求

    Attributes:
        user_id: 用户唯一标识
        app_id: 应用唯一标识
        query: 搜索查询文本
        limit: 最大返回数量，默认 10
        min_confidence: 最小置信度阈值，默认 0.5
        memory_layer: 层级过滤（如 "L1,L2"），可选
        scope: 可见性过滤，默认 "all"
    """

    user_id: str
    app_id: str
    query: str
    limit: int = 10
    min_confidence: float = 0.5
    memory_layer: str | None = None
    scope: str = "all"


# ============================================================================
# Router Factory
# ============================================================================


def create_router(
    mem0: Memory,
    llm_client: Anthropic | None = None,
    cross_collection_searcher: CrossCollectionSearcher | None = None,
    all_app_ids: list[str] | None = None,
    app_registry: Any = None,
) -> APIRouter:
    """创建记忆 API 路由"""
    router = APIRouter(prefix="/v1")
    write_svc = WriteService(mem0=mem0, llm_client=llm_client)
    recall_svc = RecallService(mem0=mem0, cross_collection_searcher=cross_collection_searcher)

    def _auth(request: Request) -> None:
        key = get_api_key(request)
        try:
            require_auth(key, app_registry=app_registry)
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e))

    # ========================================================================
    # POST /v1/memories — 批量添加记忆
    # ========================================================================
    @router.post("/memories")
    def add_memories(req: MemoriesRequest, request: Request):
        """批量添加记忆

        直接写入记忆到向量存储，不调用 LLM 推理。

        Request Body:
            user_id: 用户 ID
            app_id: 应用 ID
            memories: 记忆项列表 [{"text": "...", "memory_layer": "L1", "scope": "shared"}]

        Returns:
            {"added": n, "updated": n, "unchanged": n}
        """
        _auth(request)
        try:
            return write_svc.add_memory(
                user_id=req.user_id,
                agent_id=req.app_id,
                items=req.memories,
            )
        except Exception as e:
            logger.error("add_memories failed: %s", e)
            raise

    # ========================================================================
    # POST /v1/memories/extract — 从对话提取记忆
    # ========================================================================
    @router.post("/memories/extract")
    def extract_memories(req: ExtractRequest, request: Request):
        """从对话中提取记忆

        调用 LLM 从对话中提取结构化记忆，自动去重和更新。

        Request Body:
            user_id: 用户 ID
            app_id: 应用 ID
            messages: 对话消息列表
            memory_layer: 显式指定的层级（可选）

        Returns:
            {"added": n, "updated": n, "deleted": n, "memories": [...]}
        """
        _auth(request)
        try:
            return write_svc.extract(
                user_id=req.user_id,
                agent_id=req.app_id,
                messages=req.messages,
                memory_layer=req.memory_layer,
            )
        except Exception as e:
            logger.error("extract_memories failed: %s", e)
            raise

    # ========================================================================
    # POST /v1/memories/search — 搜索记忆
    # ========================================================================
    @router.post("/memories/search")
    def search_memories(req: SearchRequest, request: Request):
        """搜索记忆

        基于向量相似度搜索，应用置信度衰减和过滤。

        Request Body:
            user_id: 用户 ID
            app_id: 应用 ID
            query: 搜索查询
            limit: 最大返回数量
            min_confidence: 最小置信度
            memory_layer: 层级过滤
            scope: 可见性过滤

        Returns:
            {"results": [...], "total": n}

        Note:
            搜索失败时触发降级，返回空结果而非报错
        """
        _auth(request)
        try:
            results = recall_svc.search(
                query=req.query,
                user_id=req.user_id,
                agent_id=req.app_id,
                limit=req.limit,
                min_confidence=req.min_confidence,
                memory_layer=req.memory_layer,
                scope=req.scope,
                all_app_ids=all_app_ids,
            )
            return {"results": results, "total": len(results)}
        except Exception as e:
            # 降级处理：返回空结果，不阻塞主流程
            return memory_degradation_handler(e)

    # ========================================================================
    # GET /v1/memories — 获取全部记忆
    # ========================================================================
    @router.get("/memories")
    def get_memories(
        request: Request,
        user_id: str = Query(...),
        app_id: str = Query(...),
        memory_layer: str | None = None,
        scope: str = "all",
    ):
        """获取用户全部记忆

        Query Params:
            user_id: 用户 ID（必填）
            app_id: 应用 ID（必填）
            memory_layer: 层级过滤（可选）
            scope: 可见性过滤（默认 all）

        Returns:
            {"results": [...], "total": n}

        Note:
            获取失败时触发降级，返回空结果
        """
        _auth(request)
        try:
            results = recall_svc.get_all(
                user_id=user_id,
                agent_id=app_id,
                memory_layer=memory_layer,
                scope=scope,
            )
            return {"results": results, "total": len(results)}
        except Exception as e:
            return memory_degradation_handler(e)

    # ========================================================================
    # DELETE /v1/memories/{id} — 删除记忆
    # ========================================================================
    @router.delete("/memories/{memory_id}")
    def delete_memory(
        memory_id: str,
        request: Request,
        user_id: str = Query(...),
        app_id: str = Query(...),
    ):
        """删除指定记忆

        Path Params:
            memory_id: 记忆唯一标识

        Query Params:
            user_id: 用户 ID（用于权限校验）
            app_id: 应用 ID（用于权限校验）

        Returns:
            {"message": "Memory deleted successfully"}
        """
        _auth(request)
        recall_svc.delete(memory_id=memory_id, user_id=user_id, agent_id=app_id)
        return {"message": "Memory deleted successfully"}

    return router
