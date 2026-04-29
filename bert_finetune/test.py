# debug_proxy.py
import os
import socket
import requests

PROXY_HOST = "host.docker.internal"  # 或改成宿主机局域网 IP
PROXY_PORT = "7890"
TEST_URL = "https://huggingface.co"

print(f"🔍 测试代理: {PROXY_HOST}:{PROXY_PORT}\n")

# 1. DNS 解析测试
try:
    ip = socket.gethostbyname(PROXY_HOST)
    print(f"✅ DNS 解析: {PROXY_HOST} → {ip}")
except Exception as e:
    print(f"❌ DNS 解析失败: {e}")
    PROXY_HOST = input("👉 请输入宿主机局域网 IP (如 192.168.1.100): ")

# 2. 端口连通性测试
try:
    sock = socket.create_connection((PROXY_HOST, int(PROXY_PORT)), timeout=3)
    sock.close()
    print(f"✅ 端口连通: {PROXY_HOST}:{PROXY_PORT}")
except Exception as e:
    print(f"❌ 端口连接失败: {e}")
    print("💡 检查: 代理是否运行？防火墙？Docker 网络模式？")
    exit(1)

# 3. 代理功能测试
proxies = {
    "http": f"http://{PROXY_HOST}:{PROXY_PORT}",
    "https": f"http://{PROXY_HOST}:{PROXY_PORT}",
}
try:
    resp = requests.get(TEST_URL, proxies=proxies, timeout=10)
    print(f"✅ 代理工作正常! 状态码: {resp.status_code}")
except requests.exceptions.ProxyError as e:
    print(f"❌ 代理协议错误: {e}")
    print("💡 检查: 代理是 HTTP 还是 SOCKS5？协议头写对了吗？")
except Exception as e:
    print(f"❌ 请求失败: {e}")