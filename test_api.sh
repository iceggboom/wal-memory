#!/bin/bash
# AI Memory Platform API 测试脚本

BASE_URL="${1:-http://localhost:8000}"
API_KEY="${2:-test-key-123}"

echo "=== Testing AI Memory Platform ==="
echo "Base URL: $BASE_URL"
echo "API Key: $API_KEY"
echo ""

# 1. 健康检查
echo "1. Health Check..."
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""

# 2. 提取记忆
echo "2. Extract Memory..."
curl -s -X POST "$BASE_URL/v1/memories/extract" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "user_id": "test-user-001",
    "app_id": "test-app",
    "messages": [{"role": "user", "content": "我是Python后端工程师，喜欢用FastAPI开发API，喜欢吃火锅"}]
  }' | python3 -m json.tool
echo ""

# 3. 搜索记忆
echo "3. Search Memory..."
curl -s -X POST "$BASE_URL/v1/memories/search" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "user_id": "test-user-001",
    "app_id": "test-app",
    "query": "Python工程师",
    "limit": 5
  }' | python3 -m json.tool
echo ""

# 4. 获取全部记忆
echo "4. Get All Memories..."
curl -s "$BASE_URL/v1/memories?user_id=test-user-001&app_id=test-app" \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
echo ""

# 5. Admin 统计
echo "5. Admin Stats..."
curl -s "$BASE_URL/v1/admin/stats" \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool
echo ""

echo "=== Test Complete ==="
