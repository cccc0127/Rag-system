import os
import urllib.request


def download_mnist_locally():
    # 1. 自动创建所需的多层目录结构 (exist_ok=True 表示如果存在就不报错)
    save_dir = './data/MNIST/raw'
    os.makedirs(save_dir, exist_ok=True)

    # 2. 准备下载链接
    # 官方的 http://yann.lecun.com/exdb/mnist/ 经常超时
    # 这里我们替换为 AWS 上的一个极速、极稳定的官方备用源
    base_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'

    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    print(f"📁 准备将 MNIST 数据集下载到: {os.path.abspath(save_dir)}\n")

    # 3. 循环下载 4 个核心文件
    for file in files:
        file_path = os.path.join(save_dir, file)

        # 智能检测：如果文件已经存在且有大小，就直接跳过，防止重复下载浪费时间
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"✅ 文件已存在，跳过下载: {file}")
        else:
            print(f"⬇️ 正在高速下载: {file} ...")
            download_url = base_url + file
            try:
                # 执行下载
                urllib.request.urlretrieve(download_url, file_path)
                print(f"  👉 {file} 下载完成！")
            except Exception as e:
                print(f"  ❌ 下载 {file} 时失败，错误信息: {e}")
                return  # 遇到错误立刻停止

    print("\n🎉 全部 4 个数据集文件准备完毕！你可以直接去跑你的主程序了。")


if __name__ == '__main__':
    download_mnist_locally()