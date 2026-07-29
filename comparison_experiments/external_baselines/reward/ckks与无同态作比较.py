import tenseal as ts
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle

def run_fl_contribution_experiment():
    print("=== 开始 SciSec 顶会级同态加密联邦学习对比实验 ===")

    # 模拟真实的联邦学习参数维度，凸显 SIMD 密文打包优势
    vector_sizes = [128, 512, 1024, 4096]
    num_runs = 20  # 独立实验次数，用于计算误差棒

    # 统一密码学参数 (确保 BFV 和 CKKS 在相同安全级别和模数位宽下竞争)
    POLY_MOD_DEGREE = 8192
    COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]

    # 数据存储结构
    results = {size: {"Baseline": {}, "CKKS": {}, "BFV": {}} for size in vector_sizes}

    for size in vector_sizes:
        print(f"\n--- 正在测试模型维度: {size} ---")

        # 临时存储 20 次运行的数据
        metrics = {"Baseline": {"err": [], "time": [], "size": []},
                   "CKKS": {"err": [], "time": [], "size": []},
                   "BFV": {"err": [], "time": [], "size": []}}

        for run in range(num_runs):
            # 1. 模拟浮点数梯度/权重参数
            np.random.seed(42 + run)
            global_model = np.random.uniform(-0.1, 0.1, size)
            local_model = np.random.uniform(-0.1, 0.1, size)

            # ==========================================
            # Baseline: 明文欧氏距离平方
            # ==========================================
            t0 = time.time()
            diff_plain = global_model - local_model
            baseline_score = np.sum(diff_plain ** 2)
            calc_time_plain = time.time() - t0

            metrics["Baseline"]["err"].append(0)
            metrics["Baseline"]["time"].append(calc_time_plain)
            metrics["Baseline"]["size"].append(len(pickle.dumps(global_model)) / 1024)

            # ==========================================
            # 实验组: CKKS (严格使用 (A-B)^2 评估)
            # ==========================================
            ctx_ckks = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=POLY_MOD_DEGREE,
                                  coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES)
            ctx_ckks.global_scale = 2 ** 40
            ctx_ckks.generate_galois_keys()

            enc_global_ckks = ts.ckks_vector(ctx_ckks, global_model)
            enc_local_ckks = ts.ckks_vector(ctx_ckks, local_model)

            t0 = time.time()
            # 密态欧氏距离计算: 密文相减后与自身做内积
            diff_ckks = enc_global_ckks - enc_local_ckks
            enc_score_ckks = diff_ckks.dot(diff_ckks)
            calc_time_ckks = time.time() - t0

            score_ckks = enc_score_ckks.decrypt()[0]

            metrics["CKKS"]["err"].append(abs(baseline_score - score_ckks))
            metrics["CKKS"]["time"].append(calc_time_ckks)
            metrics["CKKS"]["size"].append(len(enc_local_ckks.serialize()) / 1024)

            # ==========================================
            # 对照组: BFV (强制整数化后计算欧氏距离)
            # ==========================================
            # 修复 1：使用合法的批处理素数 786433
            # 修复 2：统一 coeff_mod_bit_sizes，保证通信体积绝对公平
            ctx_bfv = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=POLY_MOD_DEGREE,
                plain_modulus=786433,
                coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES
            )
            ctx_bfv.generate_galois_keys()

            # 修复 3：为了防止高维点积在 786433 模数下溢出，被迫降低量化精度
            quant_scale = 50
            global_quant = [int(x * quant_scale) for x in global_model]
            local_quant = [int(x * quant_scale) for x in local_model]

            enc_global_bfv = ts.bfv_vector(ctx_bfv, global_quant)
            enc_local_bfv = ts.bfv_vector(ctx_bfv, local_quant)

            t0 = time.time()
            diff_bfv = enc_global_bfv - enc_local_bfv
            enc_score_bfv = diff_bfv.dot(diff_bfv)
            calc_time_bfv = time.time() - t0

            # 反量化: (scale * scale)
            score_bfv = enc_score_bfv.decrypt()[0] / (quant_scale ** 2)

            metrics["BFV"]["err"].append(abs(baseline_score - score_bfv))
            metrics["BFV"]["time"].append(calc_time_bfv)
            metrics["BFV"]["size"].append(len(enc_local_bfv.serialize()) / 1024)

        # 统计平均值与标准差
        for scheme in ["Baseline", "CKKS", "BFV"]:
            results[size][scheme]["err_mean"] = np.mean(metrics[scheme]["err"])
            results[size][scheme]["err_std"] = np.std(metrics[scheme]["err"])
            results[size][scheme]["time_mean"] = np.mean(metrics[scheme]["time"])
            results[size][scheme]["time_std"] = np.std(metrics[scheme]["time"])
            results[size][scheme]["size_mean"] = np.mean(metrics[scheme]["size"])

        print(
            f"[CKKS] 平均计算耗时: {results[size]['CKKS']['time_mean']:.4f}s, 绝对误差: {results[size]['CKKS']['err_mean']:.7f}")
        print(
            f"[BFV] 平均计算耗时: {results[size]['BFV']['time_mean']:.4f}s, 绝对误差: {results[size]['BFV']['err_mean']:.7f}")

    return vector_sizes, results


def plot_publication_graphs(dims, results):
    schemes = ['Baseline (Plaintext)', 'CKKS (Ours)', 'BFV (Quantized)']
    colors = ['#7f7f7f', '#2ca02c', '#d62728']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Performance of Homomorphic Encryption in FL Contribution Evaluation', fontsize=16, fontweight='bold',
                 y=1.05)

    x = np.arange(len(dims))
    width = 0.25

    # 1. 绝对误差 (对数坐标)
    ax = axes[0]
    for i, scheme in enumerate(["Baseline", "CKKS", "BFV"]):
        means = [results[d][scheme]["err_mean"] for d in dims]
        stds = [results[d][scheme]["err_std"] for d in dims]
        # 给 baseline 添加微小偏移以防对数坐标报错
        if scheme == "Baseline": means = [1e-10] * len(dims)
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=schemes[i], color=colors[i], capsize=5, alpha=0.9)
    ax.set_title('Absolute Evaluation Error (Log Scale)')
    ax.set_ylabel('Error Margin')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Dim {d}" for d in dims])
    ax.legend()

    # 2. 密态计算时间 (对数坐标)
    ax = axes[1]
    for i, scheme in enumerate(["Baseline", "CKKS", "BFV"]):
        means = [results[d][scheme]["time_mean"] for d in dims]
        stds = [results[d][scheme]["time_std"] for d in dims]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, label=schemes[i], color=colors[i], capsize=5, alpha=0.9)
    ax.set_title('Computational Overhead (Log Scale)')
    ax.set_ylabel('Time (Seconds)')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Dim {d}" for d in dims])
    ax.legend()

    # 3. 通信开销 (线性坐标)
    ax = axes[2]
    for i, scheme in enumerate(["Baseline", "CKKS", "BFV"]):
        means = [results[d][scheme]["size_mean"] for d in dims]
        ax.bar(x + (i - 1) * width, means, width, label=schemes[i], color=colors[i], alpha=0.9)
    ax.set_title('Communication Overhead per Model Update')
    ax.set_ylabel('Size (KB)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Dim {d}" for d in dims])
    ax.legend()

    plt.tight_layout()
    plt.savefig('fl_he_evaluation_scisec1.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    dims, exp_results = run_fl_contribution_experiment()
    plot_publication_graphs(dims, exp_results)