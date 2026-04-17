"""
05_h3_comparison.py — H3 验证：CoT 中途 vs 输入层（硬判据）

比较 AUROC_mid（CoT 40% 位置）和 AUROC_input（t_inst 位置）。
如果 AUROC_mid > AUROC_input → H3 通过，CoT 中途检测有独立价值。

额外分析：strict drift subset = 输入层失败但中途成功的样本比例。

可以完全在本地 Mac 上运行。
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
from pathlib import Path

# ── 路径 ──
DATA_DIR = Path(__file__).parent.parent.parent / "data"
GENERATED_DIR = DATA_DIR / "generated"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

# 加载 v_harm
v_harm = torch.load(RESULTS_DIR / "v_harm.pt").numpy()

# 加载 Tier 1 的 σ 数据（从 04 脚本保存的）
sigma_h = np.load(RESULTS_DIR / "sigma_T1_harmful.npy")  # (n, 20)
sigma_b = np.load(RESULTS_DIR / "sigma_T1_benign.npy")   # (n, 20)
N_CHECKPOINTS = sigma_h.shape[1]


def main():
    print("=" * 60)
    print("H3 验证：CoT 中途 vs 输入层（硬判据）")
    print("=" * 60)

    labels = np.array([1] * len(sigma_h) + [0] * len(sigma_b))

    # ── 输入层 AUROC（t=0，近似 t_inst）──
    scores_input = np.concatenate([sigma_h[:, 0], sigma_b[:, 0]])
    auroc_input = roc_auc_score(labels, scores_input)

    # ── CoT 40% 位置 AUROC ──
    t40 = int(N_CHECKPOINTS * 0.4)
    scores_mid = np.concatenate([sigma_h[:, t40], sigma_b[:, t40]])
    auroc_mid = roc_auc_score(labels, scores_mid)

    # ── CoT 最优位置 AUROC ──
    auroc_per_step = np.array([
        roc_auc_score(labels, np.concatenate([sigma_h[:, t], sigma_b[:, t]]))
        for t in range(N_CHECKPOINTS)
    ])
    best_t = auroc_per_step.argmax()
    auroc_best = auroc_per_step[best_t]

    delta = auroc_mid - auroc_input

    print(f"\n  Tier 1 (Cipher Encoding):")
    print(f"    AUROC_input (t=0):     {auroc_input:.4f}")
    print(f"    AUROC_mid   (t=40%):   {auroc_mid:.4f}")
    print(f"    AUROC_best  (t={best_t/N_CHECKPOINTS:.0%}):  {auroc_best:.4f}")
    print(f"    Delta (mid - input):   {delta:+.4f}")

    if delta > 0.03:
        print(f"    ✅ H3 通过：CoT 中途有独立检测价值")
    elif delta > 0:
        print(f"    ⚠️  H3 部分通过：提升 {delta:.4f} < 0.03 阈值")
    else:
        print(f"    ❌ H3 未通过：中途不比输入层好")

    # ── Tier 1 的特殊检查：AUROC_input 应 ≈ 0.5 ──
    if auroc_input < 0.55:
        print(f"\n    🎯 AUROC_input ≈ {auroc_input:.2f}（接近随机）→ 编码成功隐藏了输入层语义")
        if auroc_mid > 0.80:
            print(f"    🎯 AUROC_mid = {auroc_mid:.2f} → 纯净 drift 证据！")
    elif auroc_input > 0.70:
        print(f"\n    ⚠️ AUROC_input = {auroc_input:.2f}（编码没完全隐藏语义）→ drift claim 削弱")

    # ── Strict drift subset ──
    # 输入层 σ < median（检测不到），但中途 σ > median（检测到了）
    threshold_input = np.median(scores_input)
    threshold_mid = np.median(scores_mid)

    n_harmful = len(sigma_h)
    strict_drift_count = 0
    for i in range(n_harmful):
        if sigma_h[i, 0] < threshold_input and sigma_h[i, t40] > threshold_mid:
            strict_drift_count += 1

    strict_drift_pct = strict_drift_count / n_harmful * 100 if n_harmful > 0 else 0
    print(f"\n    Strict drift subset: {strict_drift_count}/{n_harmful} ({strict_drift_pct:.0f}%)")
    print(f"    （输入层检测不到，但 CoT 中途检测到的样本）")
    if strict_drift_pct > 20:
        print(f"    ✅ drift 是一个独立的、在输入层不可见的现象")
    else:
        print(f"    ⚠️ drift 增量较小，大部分信号在输入层就有了")

    # ── 画图 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：柱状对比
    t_axis = np.linspace(0, 1, N_CHECKPOINTS)
    methods = ['Input\n(t=0)', f'Mid-CoT\n(t=40%)', f'Best\n(t={best_t/N_CHECKPOINTS:.0%})']
    aurocs = [auroc_input, auroc_mid, auroc_best]
    bars = axes[0].bar(methods, aurocs, color=['gray', 'steelblue', 'coral'], alpha=0.8)
    axes[0].axhline(y=0.80, color='red', linestyle='--', alpha=0.6, label='0.80')
    axes[0].axhline(y=0.50, color='gray', linestyle=':', alpha=0.4, label='Random')
    axes[0].set_ylim(0.35, 1.05)
    axes[0].set_ylabel('AUROC')
    axes[0].set_title(f'H3: Input vs Mid-CoT\nΔ = {delta:+.3f}')
    axes[0].legend()
    for bar, val in zip(bars, aurocs):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                     f'{val:.3f}', ha='center', fontsize=11)

    # 右图：AUROC 随 CoT 位置变化 + 输入层基准线
    axes[1].plot(t_axis, auroc_per_step, color='purple', linewidth=2, label='AUROC over CoT')
    axes[1].axhline(y=auroc_input, color='gray', linestyle='--', linewidth=1.5,
                    label=f'Input AUROC = {auroc_input:.3f}')
    axes[1].axhline(y=0.50, color='gray', linestyle=':', alpha=0.4)
    axes[1].axvline(x=0.4, color='blue', linestyle=':', alpha=0.6, label='40% CoT')
    axes[1].set_xlabel('CoT Progress (normalized)')
    axes[1].set_ylabel('AUROC')
    axes[1].set_title('AUROC over CoT Progress')
    axes[1].set_ylim(0.35, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "h3_comparison.png", dpi=150)
    print(f"\n  图已保存 → h3_comparison.png")
    plt.close()


if __name__ == "__main__":
    main()
