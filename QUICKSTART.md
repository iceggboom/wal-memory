# AI Memory Platform 快速开始

## 环境要求

- Python 3.12+
- MySQL 8.0+（可选，不配置时降级运行）
- [uv](https://docs.astral.sh/uv/) 包管理器

## 1. 安装依赖

```bash
git clone <repo-url> && cd wal-memory
uv sync
```

## 2. 配置

复制配置模板并填入实际值：

```bash
cp config.example.yaml config.yaml
```

主要配置项（`config.yaml`）：

```yaml
# API 认证（app_id: api_key 映射）
api_keys:
  test-app: "your-api-key"

# LLM（Walmart 内部 LLM 网关，用于记忆提取/去重/层级分类）
llm:
  provider: "wal"
  model: "DeepSeekV3.1"
  wal_base_url: "https://your-llm-gateway-url/chat/completions"
  aloha_app_name: "your-app-name"
  access_token: "your-access-token"

# 向量存储（腾讯云 VectorDB）
vector_store:
  provider: "tencent_vector"
  config:
    url: "http://your-tcvdb-endpoint:80"
    username: "root"
    key: "your-tcvdb-api-key"
    collection_name: "memory"
    embedding_model_dims: 768
    database_name: "ai_platform"
    embedding_model: "bge-base-zh"
    mock: false  # 开发测试设为 true 可跳过真实向量数据库

# MySQL（可选）
mysql:
  host: "127.0.0.1"
  port: 3306
  database: "memory_platform"
  username: "root"
  password: ""
```

> 开发阶段 `vector_store.config.mock: true` 可跳过真实向量数据库，使用内存模拟存储。

## 3. 启动服务

```bash
uv run python -m uvicorn memory_platform.main:app --reload
```

服务启动后：

| 地址 | 说明 |
|------|------|
| http://localhost:8000/ | 前端可视化页面 |
| http://localhost:8000/docs | Swagger API 文档 |
| http://localhost:8000/health | 健康检查 |

## 4. 使用 API

所有接口需要在请求头中携带 API Key：

```
Authorization: Bearer your-api-key
```

### 4.1 注册应用

首次使用前先注册应用，获取专用 API Key：

```bash
curl -X POST http://localhost:8000/v1/admin/apps \
  -H "Authorization: Bearer your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"app_id": "my-app", "name": "我的应用"}'
```

返回示例：

```json
{"app_id": "my-app", "name": "我的应用", "api_key": "mpk-xxxx..."}
```

### 4.2 批量添加记忆

直接添加结构化记忆，支持指定层级和可见性：

```bash
curl -X POST http://localhost:8000/v1/memories \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-1",
    "app_id": "my-app",
    "memories": [
      {"text": "我是Python工程师", "memory_layer": "L1", "scope": "shared"},
      {"text": "我喜欢用VS Code开发", "memory_layer": "L2", "scope": "private"}
    ]
  }'
```

### 4.3 从对话提取记忆

传入对话内容，LLM 自动提取结构化记忆并去重存储：

```bash
curl -X POST http://localhost:8000/v1/memories/extract \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-1",
    "app_id": "my-app",
    "messages": [
      {"role": "user", "content": "我刚入职了一家新的互联网公司，担任后端工程师"},
      {"role": "assistant", "content": "恭喜！新工作感觉怎么样？"},
      {"role": "user", "content": "还不错，团队用Go语言开发微服务"}
    ]
  }'
```

### 4.4 搜索记忆

基于向量语义相似度搜索：

```bash
curl -X POST http://localhost:8000/v1/memories/search \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-1",
    "app_id": "my-app",
    "query": "工作相关"
  }'
```

### 4.5 获取全部记忆

```bash
curl "http://localhost:8000/v1/memories?user_id=user-1&app_id=my-app" \
  -H "Authorization: Bearer your-api-key"
```

### 4.6 删除记忆

```bash
curl -X DELETE "http://localhost:8000/v1/memories/{memory_id}?user_id=user-1&app_id=my-app" \
  -H "Authorization: Bearer your-api-key"
```

### 4.7 管理接口

```bash
# 获取应用列表
curl http://localhost:8000/v1/admin/apps \
  -H "Authorization: Bearer your-admin-key"

# 获取统计信息
curl http://localhost:8000/v1/admin/stats \
  -H "Authorization: Bearer your-admin-key"

# 获取用户全部记忆（管理视角）
curl "http://localhost:8000/v1/admin/users/user-1/memories?app_id=my-app" \
  -H "Authorization: Bearer your-admin-key"

# 重置向量 Collection（危险！清空所有记忆数据）
curl -X POST http://localhost:8000/v1/admin/reset-collection \
  -H "Authorization: Bearer your-admin-key"
```

## 5. 核心概念

### 记忆层级（L1-L4）

| 层级 | 类型 | 说明 | 衰减系数 |
|------|------|------|----------|
| L1 | Profile | 长期属性（职业、身份） | 0.001 |
| L2 | Preference | 偏好习惯 | 0.005 |
| L3 | Episodic | 具体经历 | 0.02 |
| L4 | Relational | 社交关系 | 0.01 |

### 记忆可见性（Scope）

- `shared` — 跨应用可见（如用户职业、偏好）
- `private` — 仅当前应用可见（如会话上下文）

### 降级策略

记忆服务是增强型功能，异常时返回空结果（`{"results": [], "total": 0}`），不阻塞主流程。

### 向量搜索

搜索基于向量语义相似度（bge-base-zh Embedding + 余弦相似度），不是关键词匹配。数据量较少时可能出现低相关度的召回结果，随着记忆增多会自然改善。

## 6. 运行测试

```bash
# 全部单元测试
uv run python -m pytest tests/unit/ -v

# 单个测试文件
uv run python -m pytest tests/unit/test_config.py -v

# 集成测试（需要 MySQL）
uv run python -m pytest tests/integration/ -v

# 代码检查
uv run ruff check src/ tests/

# 格式化
uv run ruff format src/ tests/
```

## 7. 项目结构

```
src/
├── memory_platform/          # 主应用
│   ├── main.py               # FastAPI 入口
│   ├── config.py             # 配置中心
│   ├── api/                  # API 路由（memories, admin）
│   ├── services/             # 业务逻辑（write, recall, cross_collection）
│   ├── ext/                  # 核心扩展（layer, scope, confidence）
│   ├── middleware/            # 中间件（auth, degradation）
│   ├── db/                   # MySQL 连接池、管理器、应用注册表
│   ├── adapters/             # 腾讯云向量数据库适配器
│   └── static/               # 前端页面
├── mem0/                     # mem0 fork（记忆引擎核心）
│   ├── llms/wal.py           # Wal LLM Provider（Walmart 内部网关）
│   ├── configs/llms/wal.py   # Wal LLM 配置类
│   └── utils/factory.py      # 工厂模式注册（LLM/Embedder/VectorStore）
config.yaml                   # 运行时配置
config.example.yaml           # 配置模板
```

## 8. 常见问题

### Q: 向量 DB Collection 报 Field Not Found 错误

Collection 需要包含 mem0 所需的过滤字段索引。运行重建脚本：

```bash
# 确保在内网环境
uv run python rebuild_collection.py
```

或通过 API 重置：

```bash
curl -X POST http://localhost:8000/v1/admin/reset-collection \
  -H "Authorization: Bearer your-api-key"
```

### Q: LLM 调用 SSL 证书验证失败

Wal LLM Provider 已内置 `verify=False` 跳过内网自签名证书验证。

### Q: 搜索结果不相关

向量语义搜索在数据量较少时可能召回低相关度结果，这是正常现象。可通过 `min_confidence` 参数提高过滤阈值。
