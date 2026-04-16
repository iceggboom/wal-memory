# Wal LLM Provider 设计文档

## 概述

新增 `wal` LLM provider，通过 HTTP POST 调用 Walmart 内部 LLM 网关（`{base_url}/chat/completions`），替代当前基于 Anthropic SDK 的两套调用路径，统一所有 LLM 调用。

## 背景

当前项目存在两套独立的 LLM 调用路径：
1. **mem0 工厂模式**：`LlmFactory.create("anthropic")` → `AnthropicLLM`，用于事实提取、记忆更新、程序性记忆
2. **直接 SDK 调用**：`Anthropic(api_key, base_url)`，用于 `layer.py` 中的记忆层级分类

两套路径使用相同的 Anthropic SDK 代理（z.ai），维护成本高且不一致。需要统一为通过 HTTP 直接调用 Walmart LLM 网关的方式。

参考实现：`com.walmart.reimbursement.service.impl.GenAiServiceImpl`

## 设计

### 新增文件

#### 1. `src/mem0/configs/llms/wal.py` — WalConfig 配置类

继承 `BaseLlmConfig`，新增字段：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `wal_base_url` | `str` | 必填 | LLM 网关地址 |
| `aloha_app_name` | `str` | 必填 | 应用标识 |
| `access_token` | `str` | 必填 | 访问令牌 |
| `supplier_type` | `str` | `"2"` | 供应商类型 |

继承自 `BaseLlmConfig` 的已有字段：`model`、`temperature`（默认 0.1）、`max_tokens`（默认 2000）、`top_p`、`top_k`。

#### 2. `src/mem0/llms/wal.py` — WalLLM 实现类

继承 `LLMBase`，实现核心方法 `generate_response()`。

**请求构造：**
- HTTP 方法：POST
- URL：`{wal_base_url}/chat/completions`
- 请求头：
  - `alohaAppName`: 配置中的应用标识
  - `accessToken`: 配置中的访问令牌
  - `model`: 配置中的模型名称
  - `supplierType`: `"2"`
  - `Content-Type`: `"application/json"`
- 请求体：
  ```json
  {
    "messages": [{"role": "user", "content": "..."}],
    "temperature": 0.1,
    "model": "DeepSeekV3.2"
  }
  ```

**响应解析：**
- 从 `response.json()["choices"][0]["message"]["content"]` 提取内容
- 返回 mem0 期望的格式：`{"content": "提取的内容"}`

**错误处理（fail-safe）：**
- HTTP 错误（4xx/5xx）：记录警告日志，返回 `{"content": ""}`
- 网络异常：记录错误日志，返回 `{"content": ""}`
- 响应解析失败：记录警告日志，返回 `{"content": ""}`
- temperature 校验：仅当 `0 < temperature < 1` 时发送，否则使用默认值

**HTTP 客户端：** 使用 `httpx`（同步模式），与项目现有依赖一致。

### 需修改文件

#### 3. `src/mem0/utils/factory.py`

在 `LlmFactory` 的 provider 映射中注册 `wal`：
```python
"wal": ("mem0.llms.wal", "WalLLM"),
```

同时在 `import` 或映射中关联 `WalConfig`。

#### 4. `config.yaml`

```yaml
llm:
  provider: "wal"
  model: "DeepSeekV3.2"
  wal_base_url: "https://xxx/api"    # LLM 网关地址
  aloha_app_name: "your-app-name"     # 应用标识
  access_token: "your-token"          # 访问令牌
```

移除原有的 `api_key` 和 `base_url` 字段（不再需要 Anthropic SDK 的配置）。

#### 5. `src/memory_platform/config.py`

**Settings 类变更：**
- 新增字段：`wal_base_url: str`、`aloha_app_name: str`、`access_token: str`
- `_load_yaml()` 中解析 `llm.wal_base_url`、`llm.aloha_app_name`、`llm.access_token`
- 移除或保留 `llm_api_key`、`llm_base_url`（兼容性考虑，可保留但不使用）

**`build_mem0_config()` 变更：**
- 当 `provider == "wal"` 时，传递 `wal_base_url`、`aloha_app_name`、`access_token` 到 config dict

#### 6. `src/memory_platform/main.py`

- 移除直接创建 `Anthropic` 客户端的代码（第 84-86 行区域）
- 改为通过 `LlmFactory.create(settings.llm_provider, config_dict)` 获取 LLM 实例
- 将该实例传递给需要 LLM 的服务（如 `WriteService`）

#### 7. `src/memory_platform/ext/layer.py`

- `classify_layer_with_llm()` 函数签名调整：接收 LLM 实例（`LLMBase`）而非 Anthropic 客户端
- 内部调用从 `llm_client.messages.create(model=..., max_tokens=100, messages=[...])` 改为 `llm.generate_response(messages=[...])`
- 提取响应内容的方式从 `response.content[0].text` 改为 `response["content"]`

### 调用链路

**改造前：**
```
config.yaml → Settings → build_mem0_config()
  ├→ LlmFactory.create("anthropic") → AnthropicLLM → Anthropic SDK → z.ai 代理
  └→ Anthropic(api_key, base_url) → 直接 SDK 调用 → z.ai 代理 (layer.py)
```

**改造后：**
```
config.yaml → Settings → build_mem0_config()
  └→ LlmFactory.create("wal") → WalLLM → httpx POST → LLM 网关
      ├→ mem0 内部调用（事实提取/记忆更新/程序性记忆）
      └→ layer.py 层级分类调用
```

### 测试

新增 `tests/unit/test_wal_llm.py`：

1. **请求构造验证**：mock httpx 响应，验证请求头包含正确的 `alohaAppName`、`accessToken`、`model`、`supplierType`
2. **响应解析验证**：验证从 `choices[0].message.content` 正确提取内容
3. **fail-safe 验证**：HTTP 4xx/5xx、网络超时、响应格式异常时均返回 `{"content": ""}`
4. **temperature 校验**：验证仅 `0 < temp < 1` 时传递，超出范围使用默认值
5. **配置映射验证**：`build_mem0_config()` 正确传递 wal 专属配置字段

### 不在范围内

- 图像识别（`imageToWord`）端点暂不实现，仅实现 `/chat/completions`
- 流式响应暂不支持
- `httpx` 之外的 HTTP 客户端不考虑
