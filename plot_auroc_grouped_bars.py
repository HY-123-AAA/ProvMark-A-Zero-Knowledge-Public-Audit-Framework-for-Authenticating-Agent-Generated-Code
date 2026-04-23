# plot_fig5_like_softcolors.py
import matplotlib
matplotlib.use("Agg")  # 稳定输出文件；如果你需要弹窗显示可删掉这两行

import numpy as np
import matplotlib.pyplot as plt

attacks = [
    "0%\nof renamed vars",
    "25%",
    "50%",
    "75%",
    "100%\nof renamed vars",
    "refactoring",
]

data = {
    "Ours (entropy threshold = 0.6)":  [0.94, 0.88, 0.84, 0.79, 0.74, 0.61],
    "Ours (entropy threshold = 1.2)":  [0.96, 0.88, 0.82, 0.76, 0.71, 0.60],
    "SWEET (entropy threshold = 0.6)": [0.96, 0.90, 0.85, 0.80, 0.71, 0.64],
    "SWEET (entropy threshold = 1.2)": [0.95, 0.88, 0.80, 0.74, 0.60, 0.62],
}

# 参考你给的“科研配色”图：低饱和柔和色
color_map = {
    "Ours (entropy threshold = 0.6)":  "#7E98BD",  # 浅蓝
    "Ours (entropy threshold = 1.2)":  "#D58FB1",  # 粉
    "SWEET (entropy threshold = 0.6)": "#9693C2",  # 浅紫
    "SWEET (entropy threshold = 1.2)": "#7C7A7D",  # 灰
}

labels = list(data.keys())
Y = np.array([data[k] for k in labels], dtype=float)

n_methods, n_groups = Y.shape
x = np.arange(n_groups)

total_width = 0.78
bar_w = total_width / n_methods
offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * bar_w

fig, ax = plt.subplots(figsize=(10, 6))
for i, name in enumerate(labels):
    ax.bar(
        x + offsets[i],
        Y[i],
        width=bar_w,
        label=name,
        color=color_map.get(name, None),
        alpha=0.95,          # 再柔和一点
        linewidth=0.0
    )

ax.set_ylabel("AUROC")
ax.set_xlabel("The types of paraphrase attacks")
ax.set_xticks(x)
ax.set_xticklabels(attacks)

# y轴下限稍微低一点，避免 0.60 的柱子贴边“看不见”
ax.set_ylim(0.58, 1.00)

# 让论文风格更干净（可选）
ax.grid(axis="y", alpha=0.18)
ax.set_axisbelow(True)

ax.legend(loc="upper right", frameon=True)
fig.tight_layout()

fig.savefig("auroc_paraphrase_bar_soft.png", dpi=300)
fig.savefig("auroc_paraphrase_bar_soft.pdf")
plt.close(fig)
