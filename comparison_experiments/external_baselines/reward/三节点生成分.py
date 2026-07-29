import random
import json


def generate_academic_gas_test_data(num_groups):
    budget_per_group_scaled = 100000
    paired_data = []

    for _ in range(num_groups):
        reward_high = random.randint(65000, 85000)
        reward_moderate = budget_per_group_scaled - reward_high
        reward_low = 0
        group_rewards = [reward_high, reward_moderate, reward_low]

        for reward in group_rewards:
            # 生成标准的 40 位 16 进制以太坊/FISCO地址格式
            addr = "0x" + "".join(random.choices("0123456789abcdef", k=40))
            paired_data.append((addr, reward))

    # 【神级防御适配】：严格按地址数值从小到大排序，绕过链上昂贵的 mapping 查重
    paired_data.sort(key=lambda x: int(x[0], 16))

    final_addresses = [item[0] for item in paired_data]
    final_rewards = [item[1] for item in paired_data]

    print(f"=== 实验三 V4.1 测试数据：{num_groups} 组 (共 {num_groups * 3} 个节点) ===")
    print("\n【workers 数组 (请直接复制下方中括号及内容)】:")
    print(json.dumps(final_addresses).replace(" ", ""))

    print("\n【rewards 数组 (请直接复制下方中括号及内容)】:")
    print(json.dumps(final_rewards).replace(" ", ""))


if __name__ == "__main__":
    generate_academic_gas_test_data(100)
