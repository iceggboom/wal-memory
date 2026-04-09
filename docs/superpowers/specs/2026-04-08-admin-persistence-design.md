# Admin 应用管理持久化 + mem0 SQLite→MySQL 设计文档

## Context

当前 Admin API 的应用注册（`POST /v1/admin/apps`）、应用列表（`GET /v1/admin/apps`）、统计（`GET /v1/admin/stats`）均为 TODO 状态，返回硬编码数据。认证依赖环境变量 `API_KEYS`，无动态管理能力。mem0 内置的 `SQLiteManager` 使用 SQLite 存储记忆操作历史，需替换为 MySQL。

**目标：**
1. 应用注册/查询/统计持久化到 MySQL
2. mem0 的 `SQLiteManager` 替换为 `MySQLManager`
3. 认证中间件从数据库查询 API Key
4. 所有配置统一到根目录 `config.yaml`

## 配置

根目录 `config.yaml` 新增 MySQL 配置段：

```yaml
mysql:
  host: "127.0.0.1"
  port: 3306
  database: "memory_platform"
  username: "root"
  password: "secret"
  pool_size: 5
```

`Settings` 类（`src/memory_platform/config.py`）新增 MySQL 相关字段，从 `config.yaml` 加载，与 `.env` 并存。

## 数据模型

共用同一个 MySQL 数据库 `memory_platform`。

### apps 表 — 应用注册

| 字段 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `app_id` | VARCHAR(64) | PK | 应用标识 |
| `name` | VARCHAR(255) | NOT NULL | 应用名称 |
| `api_key` | VARCHAR(255) | UNIQUE NOT NULL | 应用 API Key |
| `status` | VARCHAR(16) | DEFAULT 'active' | 状态：active / inactive |
| `created_at` | DATETIME | NOT NULL | 注册时间 |
| `updated_at` | DATETIME | NOT NULL | 最后更新时间 |

### history 表 — 替换 mem0 SQLite

与 mem0 现有 `SQLiteManager.history` 表结构完全一致：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | VARCHAR(36) PK | 记录 UUID |
| `memory_id` | VARCHAR(36) | 被操作的记忆 ID |
| `old_memory` | TEXT | 操作前内容 |
| `new_memory` | TEXT | 操作后内容 |
| `event` | VARCHAR(16) | 操作类型（ADD/UPDATE/DELETE） |
| `created_at` | DATETIME | 创建时间 |
| `updated_at` | DATETIME | 更新时间 |
| `is_deleted` | TINYINT | 是否已删除 |
| `actor_id` | VARCHAR(255) | 操作者 ID |
| `role` | VARCHAR(64) | 角色 |

## 模块设计

### 新增文件

```
src/memory_platform/db/
├── __init__.py           # 导出 get_db_connection, AppRegistry
├── connection.py         # MySQL 连接池管理
├── app_registry.py       # AppRegistry 类 — apps 表 CRUD
└── mysql_manager.py      # MySQLManager 类 — 替换 SQLiteManager
```

### connection.py — 连接池

```python
class MySQLConnectionPool:
    """管理 pymysql 连接池。"""

    def __init__(self, host, port, database, username, password, pool_size=5): ...
    def get_connection(self) -> Connection: ...
    def close_all(self) -> None: ...
```

使用 `queue.Queue` 实现简单连接池。启动时创建 `pool_size` 个连接，按需取用/归还。

### app_registry.py — 应用注册表

```python
class AppRegistry:
    """apps 表 CRUD 操作。"""

    def __init__(self, db: MySQLConnectionPool): ...
    def register(self, app_id: str, name: str, api_key: str) -> dict: ...
    def get(self, app_id: str) -> dict | None: ...
    def get_by_api_key(self, api_key: str) -> dict | None: ...
    def list_apps(self) -> list[dict]: ...
    def update_status(self, app_id: str, status: str) -> None: ...
    def count(self) -> int: ...
```

### mysql_manager.py — 替换 SQLiteManager

```python
class MySQLManager:
    """替换 mem0 的 SQLiteManager，接口完全一致。"""

    def __init__(self, db: MySQLConnectionPool): ...
    def _create_history_table(self) -> None: ...
    def add_history(self, memory_id, old_memory, new_memory, event, **kwargs) -> None: ...
    def get_history(self, memory_id: str) -> list[dict]: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...
```

接口与 `mem0.memory.storage.SQLiteManager` 完全一致，mem0 的 `Memory.__init__` 中注入即可。

### 修改文件

#### `src/memory_platform/config.py`

- 新增 `MySQLConfig` 内嵌 model
- `Settings` 新增 `mysql: MySQLConfig` 字段
- 新增 `load_yaml_config()` 函数读取 `config.yaml`
- `get_settings()` 改为从 YAML 加载后与 env 合并

#### `src/memory_platform/main.py`

- `create_app()` 初始化 `MySQLConnectionPool`
- 创建 `MySQLManager` 注入 mem0（替换 SQLiteManager）
- 创建 `AppRegistry` 注入 `AdminService` 和 `Auth` 中间件

#### `src/memory_platform/services/admin.py`

- 构造函数新增 `app_registry: AppRegistry` 参数
- `register_app()` — 调用 `app_registry.register()`
- `list_apps()` — 调用 `app_registry.list_apps()`
- `stats()` — 查询 `app_registry.count()` + `MySQLManager` 统计

#### `src/memory_platform/api/admin.py`

- `create_router()` 新增 `app_registry` 参数
- `POST /v1/admin/apps` — 调用 `AdminService.register_app()`
- `GET /v1/admin/apps` — 调用 `AdminService.list_apps()`
- `GET /v1/admin/stats` — 调用 `AdminService.stats()`

#### `src/memory_platform/middleware/auth.py`

- `require_auth()` 改为从 `AppRegistry.get_by_api_key()` 查询验证
- 不再依赖 `Settings.api_keys`

## 数据流

```
启动:
  config.yaml → Settings → MySQLConnectionPool
                            ├── MySQLManager → 注入 Memory(config)
                            └── AppRegistry → 注入 AdminService + Auth

注册应用:
  POST /v1/admin/apps → Auth 验证 → AdminService.register_app()
    → AppRegistry.register() → INSERT INTO apps

查询应用:
  GET /v1/admin/apps → Auth 验证 → AdminService.list_apps()
    → AppRegistry.list_apps() → SELECT FROM apps

统计:
  GET /v1/admin/stats → Auth 验证 → AdminService.stats()
    → AppRegistry.count() + history 表 COUNT

认证:
  请求 → require_auth(api_key)
    → AppRegistry.get_by_api_key() → SELECT FROM apps WHERE api_key=?

记忆操作历史:
  mem0.add/update/delete → MySQLManager.add_history() → INSERT INTO history
```

## API Key 兼容策略

迁移期间保持向后兼容：
- 如果 `config.yaml` 中无 MySQL 配置，回退到环境变量 `API_KEYS` + 内存存储
- `POST /v1/admin/apps` 仅在 MySQL 模式下可用，否则返回 501

## 测试策略

- **单元测试**：`AppRegistry` 和 `MySQLManager` 用 pymysql 的 mock 或 SQLite 内存库测试
- **集成测试**：更新 `test_admin_api.py`，验证持久化后的 CRUD 和认证
- **验证命令**：提供本地 MySQL 验证脚本
