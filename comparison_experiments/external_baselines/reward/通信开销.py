import tenseal as ts
import numpy as np
import sys


def measure_communication_overhead():
    # 初始化 CKKS 环境 (与之前的实验保持参数一致)
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2 ** 40
    context.generate_galois_keys()

    dimensions = [128, 512, 1024, 2048, 4096]

    print(f"{'维度 (N)':<10} | {'明文大小 (KB)':<15} | {'密文大小 (KB)':<15} | {'密文膨胀率 (Expansion)'}")
    print("-" * 65)

    for dim in dimensions:
        # 1. 生成明文向量
        v_plain = np.random.uniform(-1, 1, dim)
        plain_bytes = v_plain.nbytes  # 获取 Numpy 数组的真实内存占用

        # 2. 生成密文向量
        enc_v = ts.ckks_vector(context, v_plain)
        # 在网络中传输同态密文需要序列化，所以测量序列化后的字节数
        cipher_bytes = len(enc_v.serialize())

        plain_kb = plain_bytes / 1024
        cipher_kb = cipher_bytes / 1024
        expansion = cipher_bytes / plain_bytes

        print(f"{dim:<10} | {plain_kb:<15.2f} | {cipher_kb:<15.2f} | {expansion:.1f}x")


measure_communication_overhead()