from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from privacy_judge import PrivacyScorer


def test_privacy_scorer():
    # 初始化隐私评估器（启用NLP分类）
    scorer = PrivacyScorer(enable_nlp=True)
    
    # 测试不同类型文本
    test_cases = [
        ('公开信息', '北京是中国的首都，2024年人口约2150万。'),
        ('个人敏感', '张三的身份证号是110101199003071234。'),
        ('金融信息', '我的银行卡号是6228481234567890123，余额10000元。'),
        ('企业机密', '本公司2024年营收目标为50亿元，内部资料请勿外传。'),
        ('混合内容', '会议将于下周一召开，参会人员包括张三。'),
    ]
    
    print('=' * 60)
    print('🔹 隐私评估测试结果')
    print('=' * 60)
    
    for label, text in test_cases:
        eps, delta = scorer.get_privacy_params(text)
        
        # 根据eps值判断隐私等级
        if eps >= 5.0:
            level = '🟢 低敏感'
        elif eps >= 1.0:
            level = '🟡 中敏感'
        else:
            level = '🔴 高敏感'
        
        print(f'\n【{label}】')
        print(f'文本: {text}')
        print(f'ε (隐私预算): {eps:.4f}')
        print(f'δ (动态参数): {delta:.4f}')
        print(f'隐私等级: {level}')


if __name__ == '__main__':
    test_privacy_scorer()
