import tenseal as ts
import numpy as np
import pandas as pd
import time


def create_context(degree):
    """
    统一 Scale 为 2**40 (4096除外)。
    不同 Degree 的 RNS 分解层级不同，将真实反映在最终的微小底噪差异上。
    """
    if degree == 4096:
        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=degree, coeff_mod_bit_sizes=[40, 20, 40])
        ctx.global_scale = 2 ** 20
    elif degree == 8192:
        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=degree, coeff_mod_bit_sizes=[60, 40, 40, 60])
        ctx.global_scale = 2 ** 40
    elif degree == 16384:
        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=degree, coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60])
        ctx.global_scale = 2 ** 40
    else:
        raise ValueError("不支持的 Degree")

    ctx.generate_galois_keys()
    ctx.generate_relin_keys()  # 生成重线性化密钥
    return ctx


def run_comprehensive_experiment():
    degrees = [4096, 8192, 16384]
    dimensions = [128, 512, 1024, 4096, 16384, 65536, 262144]
    iterations = 20
    all_data = []

    print("=== 开始执行高标准控制变量同态评估实验 ===")

    for degree in degrees:
        print(f"\n>> 初始化环境：poly_modulus_degree = {degree} ...", flush=True)
        context = create_context(degree)
        max_slots = degree // 2

        for dim in dimensions:
            print(f"  测试维度 {dim}...", end="", flush=True)
            for i in range(iterations):
                np.random.seed(dim + i)

                v_local = np.random.uniform(-1, 1, dim)
                v_global = np.random.uniform(-1, 1, dim)

                plain_d_square = np.sum((v_local - v_global) ** 2)

                chunks_l = []
                chunks_g = []
                for c in range(0, dim, max_slots):
                    chunks_l.append(ts.ckks_vector(context, v_local[c: c + max_slots]))
                    chunks_g.append(ts.ckks_vector(context, v_global[c: c + max_slots]))

                start_eval = time.time()
                enc_total_d_square = None

                for enc_l, enc_g in zip(chunks_l, chunks_g):
                    diff = enc_l - enc_g
                    enc_d_square = diff.dot(diff)

                    if enc_total_d_square is None:
                        enc_total_d_square = enc_d_square
                    else:
                        enc_total_d_square += enc_d_square

                end_eval = time.time()
                eval_time = end_eval - start_eval

                cipher_result = enc_total_d_square.decrypt()[0]

                # 【修复核心：采纳 Claude 严谨的数学边界处理】
                if abs(plain_d_square) < 1e-12:
                    # 如果明文极其接近0，回退使用绝对误差防止除零溢出
                    relative_error = abs(plain_d_square - cipher_result)
                else:
                    # 正常使用纯净的相对误差公式
                    relative_error = abs(plain_d_square - cipher_result) / abs(plain_d_square)

                all_data.append({
                    "Degree": degree,
                    "Dimension": dim,
                    "Iteration": i,
                    "Relative_Error": relative_error,
                    "Evaluation_Time": eval_time
                })
            print(" 完成")

    df = pd.DataFrame(all_data)
    df.to_csv("experiment_1c_comprehensive.csv", index=False)
    print("\n所有实验结果已保存至 experiment_1c_comprehensive.csv")
    return df


if __name__ == "__main__":
    run_comprehensive_experiment()