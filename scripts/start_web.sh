#!/bin/bash
# Web 服务启动脚本
# 启动基于 FastAPI + Uvicorn 的 HTTPS 服务

set -e

# 加载 .env 文件（如果存在）
ENV_FILE="/workspace/docker/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

SSL_DIR="/workspace/python/alchemist/web/ssl"
CERT="$SSL_DIR/cert.pem"
KEY="$SSL_DIR/key.pem"

# 默认配置
HOST="${WEB_HOST:-0.0.0.0}"
PORT="${WEB_PORT:-8443}"
RELOAD="${WEB_RELOAD:-true}"

# 检查 SSL 证书
if [ ! -f "$CERT" ] || [ ! -f "$KEY" ]; then
    echo "SSL 证书不存在，正在生成..."
    /workspace/scripts/generate_ssl.sh
fi

echo "=========================================="
echo "  Alchemist2026 Web 服务"
echo "=========================================="
echo "  主机: $HOST"
echo "  端口: $PORT"
echo "  HTTPS: 已启用"
echo "  热重载: $RELOAD"
echo "=========================================="
echo ""
echo "访问地址:"
echo "  - 首页:     https://localhost:$PORT/"
echo "  - API 文档: https://localhost:$PORT/docs"
echo "  - 健康检查: https://localhost:$PORT/health"
echo ""

# 切换到 Python 源码目录
cd /workspace/python/alchemist

# 启动 Uvicorn (HTTPS)
if [ "$RELOAD" = "true" ]; then
    exec uvicorn web.app:app \
        --host "$HOST" \
        --port "$PORT" \
        --ssl-keyfile="$KEY" \
        --ssl-certfile="$CERT" \
        --reload \
        --reload-dir="/workspace/python/alchemist"
else
    exec uvicorn web.app:app \
        --host "$HOST" \
        --port "$PORT" \
        --ssl-keyfile="$KEY" \
        --ssl-certfile="$CERT"
fi
