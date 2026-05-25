from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chunk import chunk_text


test_text = '''深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。深度学习在图像识别、自然语言处理等领域取得了巨大成功。

大语言模型是深度学习的重要应用，如GPT、LLaMA等模型能够理解和生成人类语言。这些模型通过海量文本数据进行预训练，具备强大的上下文理解能力。

隐私保护在AI应用中至关重要，特别是处理敏感数据时需要采取适当的保护措施。差分隐私是一种有效的隐私保护技术，可以在数据发布时添加噪声来保护个人信息。

大模型训练需要大量的计算资源和数据支持。分布式训练技术可以将训练任务分配到多个GPU或服务器上，显著提高训练效率。

模型压缩和量化技术可以减小模型体积，提高推理速度，使得大模型能够在边缘设备上运行。'''

chunks = chunk_text(test_text, chunk_size=200, overlap=50)
print(f'🔹 分块数量: {len(chunks)}')
for i, chunk in enumerate(chunks):
    print(f'--- Chunk {i+1} ({len(chunk)} chars) ---')
    print(chunk)
    print()

print('🔹 完整性检查:')
missing = [sentence for sentence in test_text.replace('\n', '').split('。') if sentence and sentence not in ''.join(chunks).replace('\n', '')]
if missing:
    print('❌ 以下内容未出现在任何分块中:')
    for sentence in missing:
        print(sentence)
else:
    print('✅ 原文主要句子均已覆盖')
