#!/bin/bash
# AI Memory Platform API 验证脚本
# 用法：./test_api.sh [BASE_URL] [API_KEY]

set -e

BASE_URL="${1:-http://localhost:8000}"
API_KEY="${2:-test-key-123}"

echo "=== AI Memory Platform 可用性验证 ==="
echo "服务地址: $BASE_URL"
echo "API Key: ${API_KEY:0:8}..."
echo ""

# 1. 健康检查
echo "1. 健康检查..."
HEALTH=$(curl -sf "$BASE_URL/health")
STATUS=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "FAIL")
MYSQL=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin)['mysql'])" 2>/dev/null || echo "unknown")
if [ "$STATUS" = "ok" ]; then
    echo "   ✓ 服务正常 (MySQL: $MYSQL)"
else
    echo "   ✗ 服务异常"
    exit 1
fi

# 2. 应用列表
echo "e 2. 查看应用列表..."
curl -sf "$BASE_URL/v1/admin/apps" \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

# 3. 统计信息
echo -e "\n3. 平台统计..."
curl -sf "$BASE_URL/v1/admin/stats" \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

# 4. 添加记忆
echo -e "\n4. 批量添加记忆..."
ADD_RESULT=$(curl -sf -X POST "$BASE_URL/v1/memories" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "user_id": "test-user-001",
    "app_id": "test-app",
    "memories": [
      {"text": "我是Python后端工程师", "scope": "shared"},
      {"text": "我喜欢用FastAPI开发API", "scope": "shared"},
      {"text": "上周参加了架构评审会议", "scope": "private"}
    ]
  }')
echo "$ADD_RESULT" | python3 -m json.tool
ADDED=$(echo "$ADD_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['added'])" 2>/dev/null || echo "0")
if [ "$ADDED" -gt 0 ]; then
    echo "   ✓ 成功添加 $ADDED 条记忆"
else
    echo "   ✗ 添加记忆失败"
fi

# 5. 搜索记忆
echo -e "\n5. 搜索记忆..."
SEARCH_RESULT=$(curl -sf -X POST "$BASE_URL/v1/memories/search" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"user_id": "test-user-001", "app_id": "test-app", "query": "Python工程师"}')
echo "$SEARCH_RESULT" | python3 -m json.tool
TOTAL=$(echo "$SEARCH_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['total'])" 2>/dev/null || echo "0")
if [ "$TOTAL" -gt 0 ]; then
    echo "   ✓ 搜索到 $TOTAL 条记忆"
else
    echo "   ✗ 搜索无结果"
fi

# 6. 获取全部记忆
echo -e "\n6. 获取全部记忆..."
curl -sf "$BASE_URL/v1/memories?user_id=test-user-001&app_id=test-app" \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

echo -e "\n=== 验证完成 ==="
