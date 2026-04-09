#!/bin/bash
# AI Memory Platform 启动脚本

set -e

# 设置必要的环境变量
export GLM_API_KEY="${GLM_API_KEY:-1d160cd4628a4d27ac904e469506e8ef.covWKalV0W6HzP50}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$GLM_API_KEY}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.z.ai/api/anthropic}"

echo "Starting AI Memory Platform..."
echo "GLM_API_KEY: ${GLM_API_KEY:0:8}..."
echo "ANTHROPIC_BASE_URL: $ANTHROPIC_BASE_URL"

# 停止之前的进程
pkill -f "uvicorn memory_platform" 2>/dev/null || true
sleep 1

# 清理 Qdrant 锁（避免并发访问错误）
rm -rf /tmp/qdrant 2>/dev/null || true

cd "$(dirname "$0")"
uv run python -m uvicorn memory_platform.main:app --host 0.0.0.0 --port 8000
