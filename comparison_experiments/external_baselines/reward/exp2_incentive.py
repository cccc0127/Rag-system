import os

# 解决 Windows 下的底层绘图库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import copy

# ==========================================
# 1. 环境准备与数据加载
# ==========================================
print("正在加载本地 MNIST 数据集...")
data_root = r"E:\区块链\reward机制\data"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root=data_root, train=True, download=False, transform=transform)

num_nodes = 5
data_per_node = 400
loaders = []
for i in range(num_nodes):
    subset = torch.utils.data.Subset(train_dataset, range(i * data_per_node, (i + 1) * data_per_node))
    loaders.append(torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True))


# ==========================================
# 2. 定义模型结构与工具函数
# ==========================================
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)


def get_flat_weights(model):
    return torch.cat([param.view(-1) for param in model.parameters()]).detach().numpy()


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)


# ==========================================
# 3. 节点行为 (诚实、懒惰、投毒)
# ==========================================
def train_honest(model, dataloader):
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for data, target in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
    return model


def train_lazy(model):
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * 0.05
            param.add_(noise)
    return model


def train_malicious(model, dataloader):
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for data, target in dataloader:
        poisoned_target = 9 - target  # 标签翻转攻击
        optimizer.zero_grad()
        loss = criterion(model(data), poisoned_target)
        loss.backward()
        optimizer.step()
    return model


# ==========================================
# 4. 动态门限联邦学习与结算
# ==========================================
ROUNDS = 10
TOTAL_BOUNTY = 100
node_labels = ['Honest-1', 'Honest-2', 'Honest-3', 'Lazy (Noise)', 'Malicious (Poison)']
colors = ['#2ecc71', '#27ae60', '#1abc9c', '#f1c40f', '#e74c3c']

cumulative_tokens = np.zeros(num_nodes)
token_history = {i: [0] for i in range(num_nodes)}
sim_history = {i: [] for i in range(num_nodes)}
threshold_history = []

global_model = SimpleModel()
print("\n🚀 开始模拟：动态门限区块链 Token 结算过程...\n")

for round_idx in range(ROUNDS):
    print(f"--- Round {round_idx + 1} / {ROUNDS} ---")
    local_models = []

    # A. 节点本地训练
    for i in range(num_nodes):
        local_model = copy.deepcopy(global_model)
        if i < 3:
            local_model = train_honest(local_model, loaders[i])
        elif i == 3:
            local_model = train_lazy(local_model)
        else:
            local_model = train_malicious(local_model, loaders[i])
        local_models.append(local_model)

    # B. 中心聚合
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([m.state_dict()[key] for m in local_models], 0).mean(0)
    global_model.load_state_dict(global_dict)

    # C. 计算原始余弦相似度
    flat_global = get_flat_weights(global_model)
    raw_sims = np.zeros(num_nodes)
    for i in range(num_nodes):
        flat_local = get_flat_weights(local_models[i])
        raw_sims[i] = cosine_similarity(flat_local, flat_global)
        sim_history[i].append(raw_sims[i])

    # 【核心创新】：计算动态及格线 (中位数 - 容忍度)
    tolerance = 0.015  # 容忍度极小，偏离大部队1.5%直接淘汰
    dynamic_threshold = np.median(raw_sims) - tolerance
    threshold_history.append(dynamic_threshold)

    # D. 动态过滤与打分
    scores = np.zeros(num_nodes)
    for i in range(num_nodes):
        if raw_sims[i] < dynamic_threshold:
            scores[i] = 0  # 彻底打死，收益归零！
            print(f" ⚠️ 节点 {i} ({node_labels[i]}) 跌破动态门限，收益归 0！")
        else:
            # 采用相对得分，进一步放大诚实节点间的奖励差异
            scores[i] = raw_sims[i] - dynamic_threshold

    # E. 归一化分发 Token
    sum_score = np.sum(scores)
    if sum_score > 0:
        rewards = (scores / sum_score) * TOTAL_BOUNTY
    else:
        rewards = np.zeros(num_nodes)

    for i in range(num_nodes):
        cumulative_tokens[i] += rewards[i]
        token_history[i].append(cumulative_tokens[i])

    print(f"💰 本轮 Token 分配: {np.round(rewards, 2)}")
    print(f"🏆 累计 Token 余额: {np.round(cumulative_tokens, 2)}\n")

# ==========================================
# 5. 生成论文专属：双排版精美可视化
# ==========================================
plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：原始余弦相似度与动态门限的博弈
for i in range(num_nodes):
    ax1.plot(range(1, ROUNDS + 1), sim_history[i], marker='s', markersize=6,
             linewidth=2, color=colors[i], label=node_labels[i], alpha=0.8)

# 绘制动态及格线（生死线）
ax1.plot(range(1, ROUNDS + 1), threshold_history, color='black', linestyle='--',
         linewidth=2.5, label='Dynamic Threshold (Death Line)')

ax1.set_title('Cosine Similarity & Dynamic Threshold', fontsize=14, fontweight='bold')
ax1.set_xlabel('Federated Learning Rounds', fontsize=12)
ax1.set_ylabel('Cosine Similarity Score', fontsize=12)
ax1.set_xticks(range(1, ROUNDS + 1))
ax1.legend(loc='lower right')
ax1.grid(True, linestyle=':', alpha=0.6)

# 右图：Token 累计趋势 (彻底封杀坏节点)
for i in range(num_nodes):
    ax2.plot(range(ROUNDS + 1), token_history[i], marker='o', markersize=6,
             linewidth=2.5, color=colors[i], label=node_labels[i])

ax2.set_title('Token Accumulation (With Strict Penalty)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Federated Learning Rounds', fontsize=12)
ax2.set_ylabel('Cumulative Tokens (Rewards)', fontsize=12)
ax2.set_xticks(range(ROUNDS + 1))
ax2.legend(loc='upper left')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
filename = "experiment2_dynamic_penalty.png"
full_path = os.path.abspath(filename)
plt.savefig(full_path, dpi=300)

print(f"✅ 实验 2 优化版完成！图表已成功生成。")
print(f"👉 完整保存路径为: {full_path}")
plt.show()