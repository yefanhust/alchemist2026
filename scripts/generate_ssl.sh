#!/bin/bash
# 自签名 SSL 证书生成脚本
# 用于 HTTPS Web 服务

set -e

SSL_DIR="/workspace/python/alchemist/web/ssl"

# 创建 SSL 目录
mkdir -p "$SSL_DIR"

# 检查证书是否已存在
if [ -f "$SSL_DIR/cert.pem" ] && [ -f "$SSL_DIR/key.pem" ]; then
    echo "SSL 证书已存在:"
    echo "  证书: $SSL_DIR/cert.pem"
    echo "  私钥: $SSL_DIR/key.pem"
    echo ""
    read -p "是否重新生成? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消"
        exit 0
    fi
fi

# 生成私钥和自签名证书 (有效期 365 天)
echo "正在生成 SSL 证书..."
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$SSL_DIR/key.pem" \
    -out "$SSL_DIR/cert.pem" \
    -subj "/C=CN/ST=Shanghai/L=Shanghai/O=Alchemist2026/OU=QuantDev/CN=localhost"

# 设置权限
chmod 600 "$SSL_DIR/key.pem"
chmod 644 "$SSL_DIR/cert.pem"

echo ""
echo "SSL 证书已生成:"
echo "  证书: $SSL_DIR/cert.pem"
echo "  私钥: $SSL_DIR/key.pem"
echo "  有效期: 365 天"
echo ""
echo "注意: 这是自签名证书，浏览器会显示安全警告"
