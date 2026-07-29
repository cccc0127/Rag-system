import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def generate_gas_scalability_plot_pro():
    # 严格的学术规范配置
    plt.rcParams.update({
        "font.family": "serif",  # 衬线字体（如 Times New Roman），契合 LaTeX
        "font.size": 12,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.5
    })

    # 真实的区块链实测数据
    data = {
        "Nodes (N)": [3, 15, 30, 60, 150, 300],
        # 分别对应: 1组, 5组, 10组, 20组, 50组, 100组的实测值
        "Batched Gas (Ours)": [97928, 305286, 564087, 1082532, 2636479, 5227716]
    }
    df = pd.DataFrame(data)

    # 计算传统单笔非批量发送的 Baseline
    df["Non-Batched Gas (Baseline)"] = df["Nodes (N)"] * 55270

    # 适配双栏论文的黄金比例尺寸 (调整为 8x5，避免图表过于空旷)
    fig, ax = plt.subplots(figsize=(8, 5))

    # 【优化项】填充两条线之间的区域，直观展示“节省的Gas”
    ax.fill_between(df["Nodes (N)"], df["Batched Gas (Ours)"], df["Non-Batched Gas (Baseline)"],
                    color='#2ca02c', alpha=0.15, label='Gas Saved Area')

    # 绘制 Baseline (灰色虚线，表示传统的低效方式)
    ax.plot(df["Nodes (N)"], df["Non-Batched Gas (Baseline)"],
            marker='s', markersize=7, color='#7f7f7f', linestyle='--', linewidth=2,
            label='Non-Batched Settlement (Baseline)')

    # 绘制我们优化的批量方案 (红色实线，突出优化效果)
    ax.plot(df["Nodes (N)"], df["Batched Gas (Ours)"],
            marker='o', markersize=7, color='#d62728', linestyle='-', linewidth=2.5,
            label='Batched Settlement (Ours)')

    # 【重要优化】强制 Y 轴和 X 轴从 0 开始，保持对比的诚实性
    ax.set_ylim(bottom=0)
    # 给右侧留出 20 的余量，防止点和线条紧贴边框
    ax.set_xlim(left=0, right=320)

    # 规范 X 轴：只显示我们实际测试的离散点
    ax.set_xticks(df["Nodes (N)"])
    ax.set_xticklabels(df["Nodes (N)"])

    # 规范 Y 轴：明确 Million Gas 单位并适当加粗
    ax.set_ylabel("Smart Contract Gas Consumption", weight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x * 1e-6:.1f}M'))

    ax.set_xlabel("Number of Evaluated Workers ($N$)", weight='bold')
    ax.set_title("System Scalability: Settlement Overhead vs. Baseline", pad=15, weight='bold')

    # 【优化项】更严谨的图例框样式
    ax.legend(loc="upper left", frameon=True, edgecolor='black', fancybox=False, framealpha=0.9)

    # 在 N=300 处添加高光标注
    max_n = 300
    baseline_gas = 300 * 55270
    ours_gas = 9525867
    saved_percent = (baseline_gas - ours_gas) / baseline_gas * 100

    # 【优化项】带有白底黑框的文字说明，防遮挡且更显高级
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.9)

    ax.annotate(f'Saved {saved_percent:.1f}% Gas\n@ $N={max_n}$',
                xy=(max_n, ours_gas),
                xytext=(max_n - 80, ours_gas + 2.5e6),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                fontsize=11, fontweight='bold', color='#2ca02c',
                bbox=bbox_props, ha='center')

    plt.tight_layout()
    output_name = "experiment_3_gas_scalability_pro.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"顶刊级图表已生成: {output_name}")


if __name__ == "__main__":
    generate_gas_scalability_plot_pro()