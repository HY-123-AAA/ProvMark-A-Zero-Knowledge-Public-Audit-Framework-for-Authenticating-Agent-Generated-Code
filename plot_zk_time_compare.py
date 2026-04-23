# plot_zk_time_compare.py
import matplotlib.pyplot as plt

def add_value_labels(ax, bars, fmt="{:.2f}"):
    ymax = max(b.get_height() for b in bars)
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + 0.03 * ymax,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=10,
        )

def main():
    # 你表中的时间（按截图）
    methods = ["Their", "Ours"]
    proof_time_s = [6.79, 3.62]   # seconds
    verify_time_ms = [120, 4]     # milliseconds

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=300)

    # 左图：证明时间（秒）
    ax0 = axes[0]
    bars0 = ax0.bar(methods, proof_time_s)
    ax0.set_title("Proof Generation Time")
    ax0.set_ylabel("Time (s)")
    ax0.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    add_value_labels(ax0, bars0, fmt="{:.2f}s")

    # 右图：验证时间（毫秒）
    ax1 = axes[1]
    bars1 = ax1.bar(methods, verify_time_ms)
    ax1.set_title("WM Verification Time")
    ax1.set_ylabel("Time (ms)")
    ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    add_value_labels(ax1, bars1, fmt="{:.0f}ms")

    fig.tight_layout()

    out_base = "zk_time_compare"
    fig.savefig(out_base + ".png", bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
