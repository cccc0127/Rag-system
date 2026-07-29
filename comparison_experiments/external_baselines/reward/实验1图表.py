import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_academic_plots(file_path):
    try:
        # 加上 error_bad_lines 防御机制，无视可能混入的 Markdown 分隔线
        df = pd.read_csv(file_path, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.6
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    df['Degree'] = df['Degree'].astype(str)

    palette = {"4096": "#2ca02c", "8192": "#ff7f0e", "16384": "#1f77b4"}

    # --- 左图：相对误差 ---
    sns.lineplot(data=df, x="Dimension", y="Relative_Error", hue="Degree",
                 style="Degree", markers=True, dashes=False, err_style="band",
                 errorbar=('ci', 95), palette=palette, ax=ax1, linewidth=2.5, markersize=8,
                 legend=False)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    y_min, y_max = df['Relative_Error'].min(), df['Relative_Error'].max()
    ax1.set_ylim(max(y_min * 0.1, 1e-15), y_max * 10)
    ax1.set_title("Precision Fidelity: Relative Error vs. Dimension", pad=15)
    ax1.set_xlabel("Vector Dimension ($N$) - Log Scale")
    ax1.set_ylabel("Relative Error (Log Scale)")

    # --- 右图：服务端评估耗时 (更换了列名) ---
    sns.lineplot(data=df, x="Dimension", y="Evaluation_Time", hue="Degree",
                 style="Degree", markers=True, dashes=False, err_style="band",
                 errorbar=('ci', 95), palette=palette, ax=ax2, linewidth=2.5, markersize=8)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title("Aggregator Overhead: Evaluation Time vs. Dimension", pad=15)
    ax2.set_xlabel("Vector Dimension ($N$) - Log Scale")
    ax2.set_ylabel("Evaluation Time (seconds) - Log Scale")
    ax2.legend(title="Poly Modulus Degree (N)", loc='lower right', framealpha=0.95)

    plt.tight_layout()
    output_name = "experiment_1c_comprehensive_clean.png"
    plt.savefig(output_name, dpi=300)
    print(f"图表已保存为: {output_name}")


if __name__ == "__main__":
    generate_academic_plots("experiment_1c_comprehensive.csv")