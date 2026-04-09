"""认证中间件 — API Key 验证"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from memory_platform.db.app_registry import AppRegistry


def get_api_key(request: Request) -> str | None:
    """从请求头中提取 Bearer Token。"""
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def require_auth(api_key: str | None, app_registry: AppRegistry | None = None) -> bool:
    """验证 API Key。优先使用 AppRegistry，回退到环境变量配置。"""
    if not api_key:
        raise ValueError("Missing API key")

    # MySQL mode: check apps table
    if app_registry is not None:
        app = app_registry.get_by_api_key(api_key)
        if app is None:
            raise ValueError("Invalid API key")
        return True

    # Fallback: env-based validation
    from memory_platform.config import get_settings

    settings = get_settings()
    if not settings.validate_api_key(api_key):
        raise ValueError("Invalid API key")
    return True
