"""FastAPI 应用入口 — AI Memory Platform

核心职责：
1. 创建和配置 FastAPI 应用实例
2. 初始化 mem0 Memory 实例
3. 注册 API 路由（memories + admin）

启动方式：
    # 开发环境
    uv run python -m uvicorn memory_platform.main:app --reload

    # 生产环境
    uv run gunicorn memory_platform.main:app -w 4 -k uvicorn.workers.UvicornWorker

API 端点：
    GET  /health              — 健康检查
    POST /v1/memories         — 批量添加记忆
    POST /v1/memories/extract — 从对话提取记忆
    POST /v1/memories/search  — 搜索记忆
    GET  /v1/memories         — 获取全部记忆
    DELETE /v1/memories/{id}  — 删除记忆
    GET  /v1/admin/stats      — 获取统计信息
"""
import logging

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from mem0 import Memory

from memory_platform.config import build_mem0_config, get_settings
from memory_platform.api.memories import create_router as create_memories_router
from memory_platform.api.admin import create_router as create_admin_router

# 配置日志格式为 JSON，便于日志平台采集
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)


def create_app(mem0: Memory | None = None) -> FastAPI:
    app = FastAPI(title="AI Memory Platform", version="0.1.0", docs_url="/docs")
    settings = get_settings()

    # MySQL 初始化
    db_pool = None
    app_registry = None

    if settings.mysql_enabled:
        from memory_platform.db.connection import MySQLConnectionPool
        from memory_platform.db.app_registry import AppRegistry

        try:
            db_pool = MySQLConnectionPool(
                host=settings.mysql_host,
                port=settings.mysql_port,
                database=settings.mysql_database,
                username=settings.mysql_username,
                password=settings.mysql_password,
                pool_size=settings.mysql_pool_size,
            )
            app_registry = AppRegistry(db=db_pool)
        except Exception as e:
            logging.getLogger(__name__).warning("MySQL connection failed, running without DB: %s", e)
            db_pool = None
            app_registry = None

    # mem0 初始化
    if mem0 is None:
        config = build_mem0_config()

        # 注入 MySQLManager 替换 SQLiteManager
        if db_pool is not None:
            from memory_platform.db.mysql_manager import MySQLManager
            config._storage_manager = MySQLManager(db=db_pool)

        try:
            logging.getLogger(__name__).info(
                "初始化 mem0: llm_provider=%s, vector_store=%s",
                config.llm.provider, config.vector_store.provider,
            )
            mem0 = Memory(config=config)
            logging.getLogger(__name__).info("mem0 初始化成功")
        except Exception as e:
            logging.getLogger(__name__).error("mem0 初始化失败: %s", e, exc_info=True)
            raise

    # LLM 实例（用于层级分类，通过工厂统一创建）
    llm_instance = None
    if settings.llm_provider == "wal":
        from mem0.utils.factory import LlmFactory
        llm_instance = LlmFactory.create(
            "wal",
            config={
                "model": settings.llm_model,
                "wal_base_url": settings.wal_base_url,
                "aloha_app_name": settings.aloha_app_name,
                "access_token": settings.access_token,
            },
        )

    cross_collection_searcher = None

    app.include_router(create_memories_router(
        mem0=mem0,
        llm=llm_instance,
        cross_collection_searcher=cross_collection_searcher,
        app_registry=app_registry,
    ))
    app.include_router(create_admin_router(mem0, app_registry=app_registry))

    @app.get("/health")
    def health():
        return {"status": "ok", "mysql": "enabled" if settings.mysql_enabled else "disabled"}

    # 挂载静态文件（前端页面）
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


def _get_app() -> FastAPI:
    """延迟创建应用实例，避免模块导入时初始化外部连接"""
    return create_app()


# uvicorn 通过字符串 "memory_platform.main:app" 引用时，
# 会触发 __getattr__ 延迟创建，避免测试并行导入时的 Qdrant 锁冲突
def __getattr__(name: str):
    if name == "app":
        return _get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    import uvicorn

    # 直接运行时启动开发服务器
    uvicorn.run("memory_platform.main:app", host="0.0.0.0", port=8000)
