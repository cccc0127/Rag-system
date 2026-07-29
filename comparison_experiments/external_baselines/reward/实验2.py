import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置学术风格
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.grid": True,
    "grid.linestyle": "--"
})


def simulate_academic_rewards(rounds=200):
    total_budget = 100  # 每轮发放的总积分
    # 设定距离阈值：超过 1.0 的更新被视为无效
    distance_threshold = 1.0

    # 存储累计数据
    history = {"High": [], "Moderate": [], "Low": []}
    totals = {"High": 0, "Moderate": 0, "Low": 0}

    for r in range(rounds):
        # 1. 模拟三类节点的实测距离 D^2
        # 高质量：始终保持在 0.2 左右
        d_high = 0.2 + np.random.normal(0, 0.02)

        # 中等质量：距离随轮次增加，模拟边际贡献下降 (从 0.4 线性增加到 0.9)
        d_moderate = 0.4 + (r / rounds) * 0.5 + np.random.normal(0, 0.03)

        # 低质量：始终处于阈值边缘或之上 (如 1.2)
        d_low = 1.2 + np.random.normal(0, 0.05)

        # 2. 计算原始贡献得分 (q_i = 1 / D_i^2)
        # 引入门限判断：若 D^2 > 阈值，则得分归零
        s_high = 1.0 / (d_high)
        s_moderate = 1.0 / (d_moderate) if d_moderate < distance_threshold else 0
        s_low = 1.0 / (d_low) if d_low < distance_threshold else 0

        # 3. 模拟 Moderate 节点的“斜率趋 0”效应 (信誉饱和/衰减)
        # 随着轮次增加，系统对重复性中等贡献的激励权重降低
        s_moderate *= (1 - (r / (rounds * 1.1)))

        # 4. 奖励分配
        total_s = s_high + s_moderate + s_low

        # 计算当轮收益
        reward_high = (s_high / total_s) * total_budget
        reward_moderate = (s_moderate / total_s) * total_budget
        reward_low = (s_low / total_s) * total_budget if s_low > 0 else 0

        # 累加并记录
        totals["High"] += reward_high
        totals["Moderate"] += reward_moderate
        totals["Low"] += reward_low

        history["High"].append(totals["High"])
        history["Moderate"].append(totals["Moderate"])
        history["Low"].append(totals["Low"])

    return pd.DataFrame(history)


# 执行并绘图
df = simulate_academic_rewards(200)

plt.figure(figsize=(10, 6))
plt.plot(df["High"], label="High-quality Worker", color="#2ca02c", linewidth=2.5)
plt.plot(df["Moderate"], label="Moderate Worker", color="#ff7f0e", linewidth=2.5)
plt.plot(df["Low"], label="Low-quality Worker", color="#d62728", linewidth=2.5)

plt.title("Reward Evolution: Impact of Contribution Quality & Threshold", pad=15)
plt.xlabel("Training Rounds (Batch Number)")
plt.ylabel("Cumulative Reward (Reputation Score)")
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)  # 强调 0 基准线
plt.legend()
plt.tight_layout()

plt.savefig("experiment_2_nonlinear_rewards.png", dpi=300)
print("实验二(非线性版)图表已生成。")