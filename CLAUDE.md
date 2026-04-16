# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

AI Memory Platform — 基于 mem0 fork 的智能记忆平台，为多应用提供统一的用户记忆存储、检索和管理能力。支持记忆层级分类（L1-L4）、置信度衰减、多应用隔离（Scope）和跨 Collection 搜索。

## 交互要求

- Thinking 思考过程全程使用**中文**
- 最终输出的所有回答内容**必须使用中文**（代码语法英文关键词除外）

## 注释要求

遵循中文注释规范，所有模块/类/函数使用中文文档字符串，包含参数说明、返回值和使用示例。参考现有代码中的注释风格。

## 常用命令

```bash
# 安装依赖（使用 uv）
uv sync

# 运行全部单元测试
uv run python -m pytest tests/unit/ -v

# 运行单个测试文件
uv run python -m pytest tests/unit/test_config.py -v

# 运行单个测试函数
uv run python -m pytest tests/unit/test_layer.py::TestClassifyLayer::test_profile_keywords -v

# 代码检查
uv run ruff check src/ tests/

# 格式化
uv run ruff format src/ tests/

# 启动开发服务器
uv run python -m uvicorn memory_platform.main:app --reload
```

## 项目结构

```
src/
├── memory_platform/          # 主应用
│   ├── config.py             # 配置中心（从 config.yaml 读取所有配置）
│   ├── main.py               # FastAPI 入口，应用组装
│   ├── api/                  # API 路由层
│   │   ├── memories.py       # 记忆 CRUD + 搜索接口
│   │   └── admin.py          # 管理接口（应用注册、用户管理、Collection 重置）
│   ├── services/             # 业务逻辑层
│   │   ├── write.py          # 记忆写入（直接添加 + LLM 提取）
│   │   ├── recall.py         # 记忆检索（搜索 + 置信度过滤）
│   │   ├── cross_collection.py # 跨 Collection 搜索
│   │   └── admin.py          # 管理服务
│   ├── ext/                  # 核心扩展模块
│   │   ├── layer.py          # 4层记忆模型（L1-L4）+ 关键词/LLM 分类
│   │   ├── scope.py          # 记忆可见性（shared/private/all）
│   │   └── confidence.py     # 置信度衰减算法
│   ├── middleware/            # 中间件
│   │   ├── auth.py           # API Key 认证
│   │   └── degradation.py    # 降级处理（记忆异常时返回空结果）
│   ├── db/                   # 数据库层
│   │   ├── connection.py     # MySQL 连接池
│   │   ├── mysql_manager.py  # 替代 mem0 的 SQLiteManager
│   │   └── app_registry.py   # 应用注册表（apps 表 CRUD）
│   ├── adapters/
│   │   └── tencent_vector.py # 腾讯云向量 DB 适配器（mem0 VectorStoreBase 实现）
│   └── embeddings/
│       └── mock.py           # Mock Embedder（开发测试用）
├── mem0/                     # mem0 fork（记忆引擎核心）
│   ├── configs/              # mem0 配置类（LLM、向量存储、Embedder 等）
│   │   └── llms/wal.py       # Wal LLM 配置类
│   ├── llms/
│   │   └── wal.py            # Wal LLM Provider（Walmart 内部网关，httpx HTTP 调用）
│   ├── embeddings/           # Embedding 提供者实现
│   ├── vector_stores/        # 向量存储实现
│   ├── memory/telemetry.py   # 遥测开关（默认关闭）
│   └── utils/factory.py      # 工厂模式注册（LlmFactory, EmbedderFactory 等）
tests/
├── unit/                     # 单元测试（mock 外部依赖）
├── integration/              # 集成测试（需要 MySQL/向量 DB）
└── e2e/                      # 端到端测试（需要 LLM API）
```

## 架构要点

### 配置系统

所有配置集中在 `config.yaml`，不再依赖 `.env` 环境变量。`Settings` 类通过 `_load_yaml()` 解析 YAML 并映射到平铺字段名。配置优先级：代码显式传入 > config.yaml > 环境变量 > 默认值。

### LLM 调用链

LLM 统一通过 Wal Provider 调用 Walmart 内部 LLM 网关：
- `LlmFactory.create("wal", config={...})` 创建 `WalLLM` 实例
- `WalLLM` 使用 httpx 同步 HTTP POST，自定义 Header 认证（alohaAppName, accessToken）
- OpenAI 兼容响应格式，`verify=False` 跳过内网 SSL 验证
- 两处使用：(1) mem0 记忆提取/去重 (2) `ext/layer.py` 层级分类
- 工厂注册在 `mem0/utils/factory.py`，白名单在 `mem0/llms/configs.py`

### 请求处理流程

```
Request → auth 中间件 → API 路由 → Service 层 → mem0 SDK → 向量存储/MySQL
```

### 向量存储适配器（TencentVectorStore）

实现 mem0 的 `VectorStoreBase` 接口，对接腾讯云 VectorDB：
- 支持真实模式（tcvectordb RPC）和 Mock 模式（内存）
- 使用 VectorDB 内置 Embedding（bge-base-zh），文本由服务端自动向量化
- Collection 包含 FilterIndex：user_id, agent_id, run_id, hash, memory_layer, scope, app_id
- 过滤器使用 `Filter.In()` 而非 `Filter.Include()`（腾讯云要求 `in` 操作符）
- `search_by_text` 返回 `List[List[Dict]]`，非 dict，search/list/get 方法需兼容 dict 访问
- 删除 Collection 使用 `db.drop_collection(name=...)`，非 `collection.drop()`
- Collection 重建脚本：`rebuild_collection.py`（需内网环境运行）

### 4层记忆模型

- **L1 Profile**: 长期属性（职业、身份），衰减最慢（λ=0.001）
- **L2 Preference**: 偏好习惯，衰减较慢（λ=0.005）
- **L3 Episodic**: 具体经历，衰减较快（λ=0.02）
- **L4 Relational**: 社交关系，衰减中等（λ=0.01）

分类策略：显式指定 > 关键词匹配 > LLM 辅助。

### 记忆可见性（Scope）

- **shared**: 跨应用可见（如用户职业、偏好）
- **private**: 仅当前应用可见（如会话上下文）
- 跨 Collection 搜索时只返回 shared 记忆

### 降级策略

记忆服务是增强型功能，异常时返回 `{"results": [], "total": 0}` 不阻塞主流程。

### mem0 扩展机制

- `LlmFactory.provider_to_class["wal"]` 注册 Wal LLM Provider
- `EmbedderFactory.provider_to_class["mock"]` 注册自定义 Embedder
- `MySQLManager` 替代 mem0 的 `SQLiteManager`（通过 `config._storage_manager` 注入）
- `TencentVectorStore` 实现 `VectorStoreBase` 接口对接腾讯云向量 DB

### 应用启动

`main.py` 使用 `__getattr__` 延迟创建 FastAPI 实例，避免模块导入时初始化外部连接。初始化顺序：MySQL → mem0 Config → Memory 实例 → WalLLM → 路由注册。

## 技术栈

- Python 3.12+, FastAPI, uvicorn
- mem0 (fork), Pydantic v2, pydantic-settings
- SQLAlchemy, PyMySQL（MySQL）
- tcvectordb（腾讯云向量存储）
- httpx（Wal LLM HTTP 调用）
- 测试: pytest, pytest-asyncio | 代码质量: ruff
