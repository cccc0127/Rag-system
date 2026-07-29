import os
# 【防崩溃神器】解决 Windows 下 PyTorch 与 Matplotlib 的底层库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import tenseal as ts
import gc

# ==========================================
# 1. 从本地绝对路径读取数据集 (MNIST)
# ==========================================
print("正在从本地读取 MNIST 数据集...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 【关键修改 1】：使用绝对路径，加 r 防止转义
# 【关键修改 2】：路径只写到 data 这一层，PyTorch 会自动去找里面的 MNIST/raw
data_root = r"E:\区块链\reward机制\data"

# 【终极修改】：把 False 改回 True
train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
train_subset = torch.utils.data.Subset(train_dataset, range(2000))
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
print(f"✅ 本地数据集读取成功！已截取 {len(train_subset)} 个样本用于快速训练。")

# ==========================================
# 2. 定义并训练简单的模型
# ==========================================
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)

def train_model(model, epochs=1, seed=42):
    torch.manual_seed(seed)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model

print("\n正在训练全局模型 (Global Model) 和本地模型 (Local Model)...")
global_model = train_model(SimpleModel(), epochs=2, seed=10)
local_model = train_model(SimpleModel(), epochs=1, seed=42)

# ==========================================
# 3. 提取权重、展平并归一化
# ==========================================
def extract_and_normalize_weights(model):
    weights = torch.cat([param.view(-1) for param in model.parameters()]).detach().numpy()
    weights = weights[:1000] # 截取前 1000 个参数进行加密测试
    norm = np.linalg.norm(weights)
    return weights / norm

w_global = extract_and_normalize_weights(global_model)
w_local = extract_and_normalize_weights(local_model)

plaintext_cosine_sim = np.dot(w_global, w_local)
print(f"\n[明文计算] 本地模型与全局模型的余弦相似度: {plaintext_cosine_sim:.6f}")

# ==========================================
# 4. 同态加密设置
# ==========================================
print("\n正在初始化 TenSEAL 同态加密上下文...")
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40

# ==========================================
# 5. 密文计算余弦相似度
# ==========================================
print("正在加密权重向量...")
enc_w_local = ts.ckks_vector(context, w_local)
enc_w_global = ts.ckks_vector(context, w_global)

print("正在密文状态下执行点积运算 (Cosine Similarity)...")
enc_cosine_sim = enc_w_local.dot(enc_w_global)

decrypted_cosine_sim = enc_cosine_sim.decrypt()[0]
print(f"[密文计算并解密] 得到的余弦相似度: {decrypted_cosine_sim:.6f}")

error = abs(plaintext_cosine_sim - decrypted_cosine_sim)
print(f"\n[结果对照] 绝对误差 (Absolute Error): {error:.8e}")

del enc_w_local, enc_w_global, enc_cosine_sim
gc.collect()

# ==========================================
# 6. 生成论文用可视化图表
# ==========================================
print("\n正在生成可视化图表，请稍候...")
plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(w_global, bins=50, alpha=0.5, label='Global Model Weights')
ax1.hist(w_local, bins=50, alpha=0.5, label='Local Model Weights')
ax1.set_title('Normalized Weight Distribution (Subset)')
ax1.set_xlabel('Weight Value')
ax1.set_ylabel('Frequency')
ax1.legend()

labels = ['Plaintext', 'Ciphertext (HE)']
values = [plaintext_cosine_sim, decrypted_cosine_sim]
bars = ax2.bar(labels, values, color=['#3498db', '#2ecc71'], width=0.5)
ax2.set_title('Cosine Similarity: Plaintext vs. HE')
ax2.set_ylabel('Similarity Score')
ax2.set_ylim(0, 1.1)

for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 5), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

filename = "experiment1_evaluation.png"
full_path = os.path.abspath(filename)
plt.savefig(full_path, dpi=300)

print(f"\n✅ 图表已成功生成！")
print(f"👉 完整保存路径为: {full_path}")
plt.show()