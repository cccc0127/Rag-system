from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dimension_reduction import JLProjector, johnson_lindenstrauss_min_dim

# 测试JL最小维度计算
print("🔹 JL最小维度估算：")
print(f"  n=100, eps=0.3: {johnson_lindenstrauss_min_dim(100, 0.3)}")
print(f"  n=1000, eps=0.3: {johnson_lindenstrauss_min_dim(1000, 0.3)}")
print(f"  n=1000, eps=0.1: {johnson_lindenstrauss_min_dim(1000, 0.1)}")

# 测试投影矩阵生成
print("\n🔹 投影矩阵测试：")
projector = JLProjector(target_dim=128, eps=0.3, random_state=42)
test_vectors = np.random.randn(10, 768).astype(np.float32)
reduced = projector.fit_transform(test_vectors)
print(f"  原始维度: {test_vectors.shape}")
print(f"  降维后维度: {reduced.shape}")
print(f"  投影矩阵形状: {projector.projection_matrix.shape}")

# 测试保存和加载
print("\n🔹 保存/加载测试：")
projection_path = Path(__file__).resolve().parent / "test_projection.npz"
projector.save(projection_path)
loaded_projector = JLProjector.load(projection_path)
reloaded = loaded_projector.transform(test_vectors)
print(f"  加载后转换结果一致: {np.allclose(reduced, reloaded)}")
