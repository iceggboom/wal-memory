# AI 中台 · 记忆模块 设计文档

> 基于 PRD 设计，以 mem0 为核心组件，为约 10 个 AI 应用提供共享记忆基础设施。

## 1. 系统整体架构

```
                         ┌──────────────┐
                         │  AI 应用 (×~10)  │
                         │  HR助手 / 培训... │
                         └──────┬───────┘
                                │ REST API
                         ┌──────▼───────┐
                         │ API Gateway  │
                         │  (FastAPI)   │
                         └──────┬───────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
     ┌────────▼──────┐ ┌──────▼────────┐ ┌───────▼────────┐
     │ Write Service │ │ Recall Service│ │ Admin Service  │
     │               │ │              │ │                │
     │ · 显式写入     │ │ · 向量检索    │ │ · 记忆管理     │
     │ · 对话提取     │ │ · 置信度衰减  │ │ · 用户管理     │
     │ · 层级标记     │ │ · 层级过滤    │ │ · 应用管理     │
     │ · scope 控制   │ │ · scope 控制  │ │ · 监控统计     │
     └────────┬──────┘ └──────┬────────┘ └───────┬────────┘
              │                │                  │
              └────────────────┼──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Memory Extension   │
                    │      Layer          │
                    │                     │
                    │ · 4层记忆映射       │
                    │ · 置信度衰减计算     │
                    │ · 多应用scope路由   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │     mem0 SDK        │
                    │                     │
                    │ · 提取 (LLM驱动)    │
                    │ · 去重/冲突解决     │
                    │ · 向量CRUD          │
                    │ · Reranker          │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                 │
     ┌────────▼──────┐ ┌──────▼────────┐ ┌───────▼────────┐
     │ 腾讯云向量DB   │ │  GLM 5.0      │ │  SQLite       │
     │ (自定义适配器) │ │ (via LiteLLM) │ │ (mem0内置历史) │
     └───────────────┘ └───────────────┘ └───────────────┘
```

**核心原则：**
- mem0 负责"怎么存、怎么查、怎么去重" — 记忆引擎层
- Extension Layer 负责"存什么层级、怎么衰减、谁可见" — 业务策略层
- REST API 负责"外部怎么接入" — 接入层
- 三层职责清晰，互不侵入

## 2. 核心模块职责

### 2.1 Write Service — 记忆写入

两条写入路径：

| 路径 | 触发方式 | 流程 |
|------|---------|------|
| **显式写入** | 应用直接调用 `POST /v1/memories` | API → Extension Layer 标记层级 → mem0.add() → 返回 |
| **提取写入** | 应用提交对话 `POST /v1/memories/extract` | API → mem0.add(messages) → mem0 自动提取 facts → LLM 去重/冲突 → Extension Layer 标记层级 → 返回 |

**层级标记逻辑：**
- 调用方可通过 `memory_layer` 字段显式指定层级
- 未指定时，Extension Layer 根据提取出的 fact 内容自动归类（基于关键词规则 + 可选 LLM 辅助判断）

### 2.2 Recall Service — 记忆检索

```
请求(query + filters + min_confidence + scope)
    → mem0.search(query, filters)  # 原始向量检索
    → Extension Layer 后处理：
        ├─ 置信度衰减：confidence = similarity × e^(-λ × Δt)
        ├─ min_confidence 过滤
        ├─ 按 memory_layer 过滤
        └─ scope 过滤（共享/私有）
    → 排序返回
```

### 2.3 Admin Service — 管理后台

- 记忆 CRUD（查看、修改、删除指定用户的记忆）
- 应用管理（注册/注销应用，配置 scope 默认策略）
- 统计监控（记忆量、提取准确率、召回相关率）

### 2.4 Memory Extension Layer — 核心扩展

封装为 Python 包 `memory_ext`，不修改 mem0 源码，所有扩展通过 mem0 的 metadata 机制和结果后处理实现：

```
memory_ext/
├── __init__.py
├── layer.py          # 4层记忆映射与自动归类
├── confidence.py     # 置信度衰减计算
├── scope.py          # 多应用 scope 路由
└── config.py         # 衰减参数 λ (按层级差异化)
```

### 2.5 自定义适配器

```
adapters/
├── tencent_vector.py   # VectorStoreBase 实现，接入腾讯云向量DB
└── glm_llm.py          # (备选) 自定义 LLM provider，若 LiteLLM 不兼容时使用
```

## 3. 4 层记忆模型

### 3.1 层级定义

| PRD 层级 | 含义 | mem0 MemoryType | metadata.memory_layer | 典型内容 | λ 衰减系数 |
|----------|------|-----------------|---------------------|---------|-----------|
| **L1 Profile** | 角色·职业·兴趣 | semantic | `L1` | "是一名Java工程师"、"喜欢摄影" | 0.001（极慢衰减，长期稳定） |
| **L2 Preference** | 风格·偏好 | semantic | `L2` | "喜欢简洁的沟通风格"、"偏好下午开会" | 0.005（慢衰减） |
| **L3 Episodic** | 具体经历 | episodic | `L3` | "上周参加了Java培训"、"上次请假因为发烧" | 0.02（中等衰减） |
| **L4 Relational** | 团队·社交 | semantic | `L4` | "和Alice同在项目组"、"直属领导是Bob" | 0.01（较慢衰减） |

### 3.2 层级自动归类

调用方可显式指定 `memory_layer`，不指定时由 Extension Layer 自动判断：

```
事实内容 → 关键词规则匹配（第一轮，快速）
    ├─ 匹配成功 → 标记层级
    └─ 未匹配 → LLM 辅助分类（第二轮，仅对规则未覆盖的内容）
```

**规则示例：**
- 包含"是/担任/职位/工程师/经理" → L1
- 包含"喜欢/偏好/习惯/风格" → L2
- 包含"上周/昨天/参加了/去了/那次" → L3
- 包含"同事/领导/团队/同组/一起" → L4

规则匹配覆盖约 80% 场景，剩余 20% 交给 LLM 分类。

### 3.3 检索层级控制

- 默认检索所有层级
- 调用方可通过 `memory_layer` 参数指定层级（支持多选：`L1,L2`）
- 检索结果中包含每条记忆的层级标签

## 4. 置信度衰减机制

### 4.1 计算公式

```
confidence = similarity × e^(-λ × Δt)
```

- **similarity**：mem0 向量搜索返回的原始相似度分数（0~1）
- **λ**：衰减系数，按层级差异化（L1: 0.001, L2: 0.005, L3: 0.02, L4: 0.01）
- **Δt**：距上次更新的时间差（小时为单位）

### 4.2 衰减曲线（初始 similarity = 0.9）

| 时间跨度 | L1 (λ=0.001) | L2 (λ=0.005) | L3 (λ=0.02) | L4 (λ=0.01) |
|---------|-------------|-------------|-------------|-------------|
| 1天 | 0.978 | 0.897 | 0.637 | 0.786 |
| 7天 | 0.851 | 0.536 | 0.069 | 0.221 |
| 30天 | 0.497 | 0.055 | ~0 | ~0 |
| 90天 | 0.060 | ~0 | ~0 | ~0 |

每次记忆被 update（如冲突解决时的 UPDATE 操作），`updated_at` 刷新，Δt 归零。

### 4.3 min_confidence 过滤

- 调用方通过 `min_confidence` 参数设定阈值（默认 0.5）
- 置信度低于阈值的结果直接过滤掉
- 历史久远但未被刷新的记忆自然淘汰，无需定期清理任务

### 4.4 实现位置

查询后处理，不侵入 mem0 内部：

```python
def recall(query, min_confidence=0.5, ...):
    results = mem0.search(query, filters=filters, limit=limit)
    now = datetime.utcnow()
    scored = []
    for mem in results:
        layer = mem.metadata.get("memory_layer", "L1")
        lam = DECAY_LAMBDA[layer]
        delta_t = (now - mem.updated_at).total_seconds() / 3600
        confidence = mem.score * math.exp(-lam * delta_t)
        if confidence >= min_confidence:
            scored.append((mem, confidence))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:limit]
```

## 5. 数据模型与存储

### 5.1 存储架构

| 组件 | 用途 | 选型 |
|------|------|------|
| **腾讯云向量DB** | 记忆向量 + 原始数据存储 | mem0 自定义适配器 |
| **MySQL** | 记忆历史记录（add/update/delete 操作日志）+ 应用注册表 | MySQLManager 替代 SQLiteManager |

### 5.2 向量 DB Collection 设计

每个应用（app_id）对应一个 Collection：

```
Collection: app_{app_id}
├─ id: string (UUID)
├─ text: string (记忆原文)
├─ embedding: float[] (向量，腾讯云自带 Embedding 模型)
├─ metadata: {
│     user_id: string,
│     memory_layer: "L1" | "L2" | "L3" | "L4",
│     hash: string,
│     created_at: datetime,
│     updated_at: datetime,
│     app_id: string,
│     scope: "shared" | "private"
│  }
└─ score: float (向量相似度，查询时返回)
```

### 5.3 跨应用共享记忆

- `scope = "shared"`：记忆存入源应用的 Collection，检索时跨 Collection 搜索
- `scope = "private"`：仅在本应用的 Collection 内检索
- MVP 阶段串行检索（~10 个应用），后续可改为并行

### 5.4 数据流

```
写入:  API请求 → Extension Layer(标记层级+scope) → mem0.add() → 腾讯云向量DB
提取:  对话 → mem0.add(messages) → LLM提取facts → 去重/冲突 → 写入向量DB
检索:  API请求 → mem0.search() → Extension Layer(衰减+过滤+排序) → 返回
```

## 6. 矛盾检测与冲突解决机制

记忆系统需要处理三类语义冲突：**重复**（同一信息多次输入）、**补充**（部分重叠但有新信息）、**矛盾**（新旧信息互斥）。系统采用三层机制覆盖不同粒度的冲突。

### 6.1 第一层：LLM 驱动的向量记忆冲突（核心）

这是最主要的矛盾处理机制，发生在**写入时**（`mem0.add(infer=True)`），覆盖 `POST /v1/memories/extract` 路径。

#### 端到端流程

```
新对话消息
    │
    ▼
[Phase 1: LLM 事实提取]
    │  src/mem0/configs/prompts.py
    │  根据 agent_id 是否存在，选择 USER_MEMORY_EXTRACTION_PROMPT
    │  或 AGENT_MEMORY_EXTRACTION_PROMPT 提取结构化事实
    │  normalize_facts() 处理 LLM 输出不一致（dict vs str）
    ▼
[Phase 2: 向量相似度搜索]
    │  src/mem0/memory/main.py:542-569
    │  对每条新事实，搜索 Top-5 语义最相似的旧记忆
    │  搜索范围限定：user_id + agent_id + run_id
    │  去重合并检索到的旧记忆（按 ID 去重）
    ▼
[Phase 3: LLM 矛盾裁决]
    │  src/mem0/configs/prompts.py:175-323 (DEFAULT_UPDATE_MEMORY_PROMPT)
    │  将旧记忆 + 新事实一起交给 LLM，输出四种事件：
    │  ├── ADD    → 全新信息，无任何重叠
    │  ├── UPDATE → 有重叠但信息更丰富（合并）
    │  ├── DELETE → 矛盾信息（互斥）
    │  └── NONE   → 信息已存在，无变化
    ▼
[Phase 4: 执行 LLM 决策]
    │  src/mem0/memory/main.py:607-693
    │  遍历每个事件，执行 create / update / delete
    │  每条记忆分配/更新 MD5 hash（用于后续去重）
    │  操作记录写入 history 表（SQLite / MySQL）
    ▼
[Phase 5: 知识图谱处理]（如果启用 graph）
    │  与向量处理并行执行，见 6.2 节
    ▼
写入完成
```

#### 四种冲突事件的判定规则

| 事件 | 判定条件 | 处理方式 | 示例 |
|------|---------|---------|------|
| **ADD** | 新事实与任何旧记忆无语义重叠 | 创建新记忆，分配新 ID | 新增"喜欢摄影" |
| **UPDATE** | 新事实与旧记忆有重叠，但包含**额外信息** | 原地更新旧记忆，保持同一 ID | "喜欢芝士披萨" + "也爱鸡肉披萨" → "喜欢芝士和鸡肉披萨" |
| **DELETE** | 新事实与旧记忆**互相矛盾** | 删除旧记忆（硬删除） | "喜欢芝士披萨" + "讨厌芝士披萨" → 删除旧的 |
| **NONE** | 新事实的信息已存在于旧记忆中 | 不操作 | "是Java工程师" + "是Java工程师" → 无变化 |

#### UUID 映射防幻觉

LLM 无法准确返回 UUID，系统在 Phase 3 前将旧记忆的 ID 替换为整数索引（0, 1, 2...），LLM 返回后再映射回真实 UUID：

```python
# src/mem0/memory/main.py:571-575
temp_uuid_mapping = {}
for idx, item in enumerate(retrieved_old_memory):
    temp_uuid_mapping[str(idx)] = item["id"]
    retrieved_old_memory[idx]["id"] = str(idx)
```

### 6.2 第二层：知识图谱矛盾（Graph Layer）

当启用 mem0 的 Graph Memory 功能时，额外的图谱层冲突检测并行执行。

#### 关系矛盾检测

`src/mem0/graphs/utils.py` 中的 `DELETE_RELATIONS_SYSTEM_PROMPT` 指导 LLM 识别应删除的关系：

- **过时/不准确**：新信息比旧信息更近期或更准确 → 删除旧关系
- **互相矛盾**：新信息否定旧信息 → 删除旧关系
- **同类型不同目标**：不删除（如 "属于A组" 和 "属于B组" 可共存）

#### 软删除机制

与向量记忆的硬删除不同，图谱关系采用**软删除**（`valid = false`），保留时间推理能力：

```cypher
MATCH (n)-[r:RELATIONSHIP]->(m)
WHERE r.valid IS NULL OR r.valid = true
SET r.valid = false, r.invalidated_at = datetime()
```

当同一关系被重新建立时，恢复为有效：
```cypher
ON MATCH SET r.valid = true, r.invalidated_at = null
```

#### 实体节点合并

通过 embedding 余弦相似度（阈值 0.9）检测重复实体节点，合并而非重复创建：

```cypher
// 搜索相似节点
WITH source_candidate,
    round(2 * vector.similarity.cosine(source_candidate.embedding, $embedding) - 1, 4) AS similarity
WHERE similarity >= 0.9
```

### 6.3 第三层：跨集合哈希去重（读取时）

跨应用搜索（`CrossCollectionSearcher`）时，通过 MD5 hash 去重，确保同一条记忆出现在多个 app 中只返回一次：

```python
# src/memory_platform/services/cross_collection.py:74-105
seen_hashes: set[str] = set()
for app_id in app_ids:
    for mem in search_results:
        mem_hash = mem.get("hash")
        if mem_hash and mem_hash in seen_hashes:
            continue  # 跳过重复
        if mem_hash:
            seen_hashes.add(mem_hash)
        all_results.append(mem)
```

### 6.4 关键设计决策

| 决策 | 说明 |
|------|------|
| 矛盾 → 删除，非合并 | 新事实与旧记忆矛盾时，**硬删除**旧记忆。新事实需由 LLM 另外产生 ADD 事件 |
| 仅写入时检测 | 冲突检测只发生在写入阶段，读取时只做置信度衰减过滤 |
| LLM 判断，非规则引擎 | 依赖 LLM 语义理解区分"相同"(NONE) vs "补充"(UPDATE) vs "矛盾"(DELETE) |
| 向量 + 图谱双通道 | 向量存储和知识图谱各自独立检测矛盾，互不依赖 |
| 图谱软删除保留历史 | 关系矛盾不物理删除，通过 `valid` 字段标记，支持时间线回溯 |
| 无预写 hash 去重 | 写入前不检查 hash 去重。去重仅通过 LLM 裁决（NONE 事件）和读取时 hash 去重 |
| `add_memory(infer=False)` 跳过冲突检测 | 显式写入路径不经过 LLM 裁决，直接写入向量存储 |

### 6.5 与数据流的整合

```
写入（显式）:  API → WriteService.add_memory() → mem0.add(infer=False)
               → 直接写入向量DB → 跳过冲突检测

写入（提取）:  API → WriteService.extract() → mem0.add(infer=True)
               → [Phase 1-5 完整冲突检测] → 写入向量DB + 图谱

检索:          API → RecallService.search() → mem0.search()
               → Extension Layer（置信度衰减 + 层级过滤）
               → CrossCollectionSearcher（跨集合 hash 去重）
               → 返回

管理删除:      API → AdminService.delete_user_memories() → mem0.delete_all()
               → 硬删除所有匹配记忆
```

## 7. REST API 设计

### 7.1 认证

应用级 API Key，请求头 `Authorization: Bearer {api_key}`。

### 7.2 核心 API

#### POST /v1/memories — 显式写入

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_id | string | 是 | 用户标识 |
| app_id | string | 是 | 应用标识 |
| memories | array | 是 | `[{text, memory_layer?, scope?}]` |

响应：`{added: 3, updated: 1, unchanged: 0}`

#### POST /v1/memories/extract — 对话提取

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_id | string | 是 | 用户标识 |
| app_id | string | 是 | 应用标识 |
| messages | array | 是 | `[{role: "user/assistant", content}]` |
| memory_layer? | string | 否 | 强制指定层级 |

响应：`{added: 5, updated: 2, deleted: 1, memories: [...]}`

#### POST /v1/memories/search — 记忆检索

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_id | string | 是 | 用户标识 |
| app_id | string | 是 | 应用标识 |
| query | string | 是 | 查询文本 |
| limit? | int | 否 | 最大返回数，默认 10 |
| min_confidence? | float | 否 | 最低置信度，默认 0.5 |
| memory_layer? | string | 否 | 层级过滤，如 `L1,L2` |
| scope? | string | 否 | `"shared"` / `"private"` / `"all"`，默认 `"all"` |

响应：
```json
{
  "results": [
    {
      "id": "uuid",
      "text": "是一名Java工程师",
      "memory_layer": "L1",
      "confidence": 0.92,
      "similarity": 0.95,
      "scope": "shared",
      "created_at": "...",
      "updated_at": "..."
    }
  ],
  "total": 3
}
```

#### GET /v1/memories — 获取用户全部记忆

查询参数：`user_id`, `app_id`, `memory_layer?`, `scope?`

#### DELETE /v1/memories/{memory_id} — 删除记忆

查询参数：`user_id`, `app_id`

### 7.3 管理 API

```
GET    /v1/admin/apps                      # 应用列表
POST   /v1/admin/apps                      # 注册应用
GET    /v1/admin/users/{user_id}/memories   # 查看用户所有记忆
DELETE /v1/admin/users/{user_id}/memories   # 清空用户记忆
GET    /v1/admin/stats                      # 平台统计
```

### 7.4 错误码

| 状态码 | 含义 |
|--------|------|
| 200 | 成功 |
| 400 | 参数错误 |
| 401 | API Key 无效 |
| 403 | 无权访问该资源 |
| 404 | 记忆/用户不存在 |
| 500 | 服务内部错误 |

## 8. 部署架构

### 8.1 TKE 部署

```
┌─────────────────────────────────────────────────────┐
│                     TKE 集群                          │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │            Namespace: memory-platform         │   │
│  │                                              │   │
│  │  ┌─────────────┐    2-3 Pod（HPA 自动伸缩）   │   │
│  │  │ memory-api   │◄── CLB (负载均衡)           │   │
│  │  │ (FastAPI)    │    资源: 1C2G / Pod         │   │
│  │  └──────┬──────┘                              │   │
│  │         │                                      │   │
│  │  ┌──────▼──────┐                              │   │
│  │  │ SQLite PVC   │    持久化存储（记忆历史）     │   │
│  │  └─────────────┘                              │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
└─────────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│ 腾讯云向量DB      │    │  GLM 5.0 API    │
│ (全托管服务)      │    │  (腾讯云部署)    │
└─────────────────┘    └─────────────────┘
```

- **单服务部署**：API + Extension Layer + mem0 SDK 打包为一个容器镜像
- **HPA 弹性伸缩**：基于 CPU 利用率（70%）自动扩缩容，2 Pod 起，最大 10 Pod
- **SQLite 持久化**：通过腾讯云 CBS PVC 挂载，后续可迁移至 TDSQL

### 8.2 降级策略

任何 mem0/向量DB/LLM 调用失败，一律返回空结果（HTTP 200），不阻塞主链：

```
正常:  API → mem0.search() → 返回记忆
降级:  API catch 异常 → 返回 {"results": [], "total": 0} + 日志 + 告警
```

### 8.3 可观测性

- **日志**：stdout JSON 格式，腾讯云 CLS 收集
- **指标**：Prometheus `/metrics`，关键指标：请求量、延迟 P99、记忆操作数、降级次数
- **链路**：请求中带 `trace_id`，贯穿写入→提取→检索全链路

## 9. 项目结构与测试

### 9.1 项目结构

```
wal-memory/
├── pyproject.toml
├── Dockerfile
├── CLAUDE.md
├── src/
│   └── memory_platform/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── memories.py
│       │   └── admin.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── write.py
│       │   ├── recall.py
│       │   └── admin.py
│       ├── ext/
│       │   ├── __init__.py
│       │   ├── layer.py
│       │   ├── confidence.py
│       │   └── scope.py
│       ├── adapters/
│       │   ├── __init__.py
│       │   └── tencent_vector.py
│       └── middleware/
│           ├── __init__.py
│           ├── auth.py
│           └── degradation.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_layer.py
│   │   ├── test_confidence.py
│   │   ├── test_scope.py
│   │   └── test_write.py
│   ├── integration/
│   │   ├── test_memories_api.py
│   │   ├── test_recall.py
│   │   └── test_admin_api.py
│   └── e2e/
│       └── test_full_flow.py
└── docs/
    └── superpowers/specs/
```

### 9.2 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | 异步原生，自动 OpenAPI 文档 |
| 包管理 | uv | 快速，锁定依赖 |
| 测试框架 | pytest + pytest-asyncio | 异步测试支持 |
| 代码质量 | ruff (lint + format) | 替代 flake8 + black |

### 9.3 测试策略

- **单元测试**（覆盖率目标 ≥80%）：`ext/` 下的纯函数 + `services/` 的业务逻辑（mock mem0）
- **集成测试**：API 端点测试，mock mem0 SDK，使用 `httpx.AsyncClient` + FastAPI `TestClient`
- **端到端测试**（CI 按需触发）：连接真实的腾讯云向量 DB 和 GLM 5.0

### 9.4 PRD 验收指标

| PRD 指标 | 验证方式 |
|---------|---------|
| 召回相关率 ≥70% | E2E 测试 + 人工评估集 |
| 提取准确率 ≥85% | 单元测试 + 人工抽检 |
| 重复沟通率下降 ≥40% | 上线后 A/B 对照统计 |
| 平均活跃记忆置信分 ≥0.75 | Admin 统计 API 数据 |
| 接入成本 ≤1天 | API 文档 + 示例代码 |
