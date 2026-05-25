# 重建FAISS索引，将文档内容向量化存储。
# main.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
from config import config
from chunk import chunk_documents
from dimension_reduction import fit_reduce_embeddings, l2_normalize
from gaussian_noise import AnalyticGaussianCalibrator
from privacy_judge import PrivacyScorer
import logging

# ========================
# 模块初始化配置
# 配置日志输出
# ========================
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 加载预训练的嵌入模型（Embedding Model）
embedding_model_path = os.path.abspath(config.EMBEDDING_MODEL)
embedding_model = SentenceTransformer(embedding_model_path, device=config.EMBEDDING_DEVICE)

def rebuild_index():
    """重新加载所有文档，并重建 FAISS 索引"""
    print("🔄 开始重建 FAISS 索引...")

    # 使用配置中的知识库目录
    knowledge_dir = config.REFERENCE_FOLDER  # 这里改为 `REFERENCE_FOLDER`
    privacy_scorer = PrivacyScorer()
    noise_calibrator = AnalyticGaussianCalibrator(
        delta=getattr(config, "DP_DELTA", 1e-5),
        utility_scale=getattr(config, "DP_UTILITY_SCALE", 0.01),
        random_state=getattr(config, "DP_RANDOM_SEED", None),
    )

    # chunk_documents 会批量读取 knowledge-base，并在每个 chunk 生成后立即做隐私评估。
    chunk_records = chunk_documents(
        knowledge_dir,
        chunk_size=getattr(config, "CHUNK_SIZE", 1000),
        overlap=getattr(config, "OVERLAP", 200),
        scorer=privacy_scorer,
    )
    if not chunk_records:
        print("⚠️ 没有可用分块，索引未更新。")
        return "⚠️ 没有可用分块，索引未更新。"

    print(f"🔐 已完成 {len(chunk_records)} 个分块的隐私评估")

    texts = [record["content"] for record in chunk_records]
    filenames = [
        f"{record['filename']}#chunk-{record['chunk_id']}"
        for record in chunk_records
    ]

    # 1) 将分块文本转化为原始 Embedding 向量。
    raw_embeddings = np.array(embedding_model.encode(texts), dtype=np.float32)  # 确保是 NumPy 数组

    # 2) 先进行 JL 降维。后续 DP 噪声在检索空间中施加，避免在 1024 维
    #    空间中累积过大的高维 Gaussian 能量。
    original_dim = raw_embeddings.shape[1]
    projection_path = config.FAISS_CACHE / "jl_projection.npz"
    reduced_embeddings = fit_reduce_embeddings(
        raw_embeddings,
        save_path=projection_path,
        target_dim=getattr(config, "JL_TARGET_DIM", 256),
        eps=getattr(config, "JL_EPSILON", 0.3),
        random_state=getattr(config, "JL_RANDOM_SEED", 42),
    )
    print(f"✅ JL 降维完成：{original_dim} -> {reduced_embeddings.shape[1]}")

    # 3) 在 256 维检索空间中施加 DP 高斯机制。
    #    apply_noise 内部会在加噪前做 L2 clipping，并用 utility_scale/sqrt(dim)
    #    控制逐维噪声，最后再归一化进入 cosine/L2 检索空间。
    private_embeddings = np.vstack(
        [
            noise_calibrator.apply_noise(
                embedding_vec,
                raw_score=record["raw_sensitivity_score"],
            )
            for embedding_vec, record in zip(reduced_embeddings, chunk_records)
        ]
    ).astype(np.float32)
    private_embeddings = l2_normalize(private_embeddings)
    print("✅ DP 高斯噪声注入并归一化完成")

    # 使用 FAISS 建立索引
    dim = private_embeddings.shape[1]  # 向量维度
    index = faiss.IndexFlatL2(dim)
    index.add(private_embeddings)

    # 保存 FAISS 索引和文档文件名列表
    faiss.write_index(index, str(config.FAISS_CACHE / "docs.index"))
    
    with open(str(config.FAISS_CACHE / "filenames.pkl"), "wb") as f:
        pickle.dump(filenames, f)

    with open(str(config.FAISS_CACHE / "chunk_metadata.pkl"), "wb") as f:
        pickle.dump(chunk_records, f)

    print("✅ FAISS 索引已成功重建！")
    return "✅ FAISS 索引已成功重建！"

# 如果 `main.py` 直接运行，则自动创建索引
if __name__ == "__main__":
    rebuild_index()
