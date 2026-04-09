"""管理 API 路由 — 提供应用和用户级别的管理接口"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from mem0 import Memory
from pydantic import BaseModel

from memory_platform.middleware.auth import get_api_key, require_auth
from memory_platform.middleware.degradation import memory_degradation_handler
from memory_platform.services.admin import AdminService

logger = logging.getLogger(__name__)


# ============================================================================
# Request Models
# ============================================================================


class RegisterAppRequest(BaseModel):
    """注册应用请求

    Attributes:
        app_id: 应用唯一标识
        name: 应用显示名称
    """

    app_id: str
    name: str


# ============================================================================
# Router Factory
# ============================================================================


def create_router(mem0: Memory, app_registry: Any = None) -> APIRouter:
    """创建管理 API 路由"""
    router = APIRouter(prefix="/v1/admin")
    admin_svc = AdminService(mem0=mem0, app_registry=app_registry)

    def _auth(request: Request) -> None:
        key = get_api_key(request)
        try:
            require_auth(key, app_registry=app_registry)
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e))

    # ========================================================================
    # GET /v1/admin/apps — 获取应用列表
    # ========================================================================
    @router.get("/apps")
    def list_apps(request: Request):
        _auth(request)
        apps = admin_svc.list_apps()
        return {"apps": apps}

    # ========================================================================
    # POST /v1/admin/apps — 注册新应用
    # ========================================================================
    @router.post("/apps")
    def register_app(req: RegisterAppRequest, request: Request):
        _auth(request)
        import secrets
        api_key = f"mpk-{secrets.token_hex(16)}"
        try:
            result = admin_svc.register_app(app_id=req.app_id, name=req.name, api_key=api_key)
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ========================================================================
    # GET /v1/admin/users/{user_id}/memories — 获取用户记忆
    # ========================================================================
    @router.get("/users/{user_id}/memories")
    def get_user_memories(
        user_id: str,
        request: Request,
        app_id: str = Query(...),
    ):
        """获取指定用户的全部记忆

        Path Params:
            user_id: 用户唯一标识

        Query Params:
            app_id: 应用唯一标识（必填）

        Returns:
            {"results": [...], "total": n}

        Note:
            失败时触发降级，返回空结果
        """
        _auth(request)
        try:
            memories = admin_svc.get_user_memories(user_id=user_id, agent_id=app_id)
            return {"results": memories, "total": len(memories)}
        except Exception as e:
            return memory_degradation_handler(e)

    # ========================================================================
    # DELETE /v1/admin/users/{user_id}/memories — 删除用户记忆
    # ========================================================================
    @router.delete("/users/{user_id}/memories")
    def delete_user_memories(
        user_id: str,
        request: Request,
        app_id: str = Query(...),
    ):
        """删除指定用户的全部记忆

        警告：此操作不可逆！

        Path Params:
            user_id: 用户唯一标识

        Query Params:
            app_id: 应用唯一标识（必填）

        Returns:
            {"message": "User memories deleted successfully"}

        Note:
            失败时触发降级，返回空结果
        """
        _auth(request)
        try:
            admin_svc.delete_user_memories(user_id=user_id, agent_id=app_id)
            return {"message": "User memories deleted successfully"}
        except Exception as e:
            return memory_degradation_handler(e)

    # ========================================================================
    # GET /v1/admin/stats — 获取统计信息
    # ========================================================================
    @router.get("/stats")
    def get_stats(request: Request):
        _auth(request)
        return admin_svc.stats()

    return router
