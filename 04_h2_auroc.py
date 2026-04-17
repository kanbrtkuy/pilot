"""
04_h2_auroc.py — H2 验证：AUROC vs CoT 位置

在每个归一化时间步 t 计算 AUROC（G1a vs G3a），画出检测能力随 CoT 进度的变化。

判断标准：CoT 40% 位置 AUROC > 0.80 → H2 通过

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

# ── 参数 ──
N_CHECKPOINTS = 20

# 加载 v_harm
v_harm = torch.load(RESULTS_DIR / "v_harm.pt").numpy()

# 自动检测层号
sample_file = list((GENERATED_DIR / "T1_harmful").glob("hs_*.npz"))
available_layers = sorted([int(k.replace("layer_", "")) for k in np.load(sample_file[0]).keys()])
LAYER = available_layers[len(available_layers) // 2]


def load_sigma_trajectories(group_dir, v, layer_idx, n_checkpoints):
    trajectories = []
    for f in sorted(group_dir.glob("hs_*.npz")):
        hs = np.load(f)[f"layer_{layer_idx}"]
        n_tokens = len(hs)
        if n_tokens < 5:
            continue
        indices = np.linspace(0, n_tokens - 1, n_checkpoints).astype(int)
        sampled = hs[indices]
        sigma = sampled @ v
        trajectories.append(sigma)
    return np.array(trajectories) if trajectories else np.zeros((0, n_checkpoints))


def compute_auroc_over_time(sigma_h, sigma_b, n_checkpoints):
    """在每个时间步计算 AUROC"""
    labels = np.array([1] * len(sigma_h) + [0] * len(sigma_b))
    aurocs = []
    for t in range(n_checkpoints):
        scores = np.concatenate([sigma_h[:, t], sigma_b[:, t]])
        try:
            aurocs.append(roc_auc_score(labels, scores))
        except ValueError:
            aurocs.append(0.5)
    return np.array(aurocs)


def main():
    print("=" * 60)
    print("H2 验证：AUROC vs CoT 位置")
    print("=" * 60)

    for tier_name, h_dir, b_dir, fname in [
        ("Tier 1", "T1_harmful", "T1_benign", "h2_tier1_auroc.png"),
        ("Tier 2", "T2_harmful", "T2_benign", "h2_tier2_auroc.png"),
    ]:
        sigma_h = load_sigma_trajectories(GENERATED_DIR / h_dir, v_harm, LAYER, N_CHECKPOINTS)
        sigma_b = load_sigma_trajectories(GENERATED_DIR / b_dir, v_harm, LAYER, N_CHECKPOINTS)

        if len(sigma_h) == 0 or len(sigma_b) == 0:
            print(f"  {tier_name}: 数据不足，跳过")
            continue

        aurocs = compute_auroc_over_time(sigma_h, sigma_b, N_CHECKPOINTS)

        t_axis = np.linspace(0, 1, N_CHECKPOINTS)
        auroc_at_40 = aurocs[int(N_CHECKPOINTS * 0.4)]
        auroc_max = aurocs.max()
        above_08 = t_axis[aurocs >= 0.80]
        first_08 = above_08[0] if len(above_08) > 0 else None

        print(f"\n  {tier_name}:")
        print(f"    AUROC @ 40% CoT : {auroc_at_40:.4f}")
        print(f"    最高 AUROC       : {auroc_max:.4f}")
        if first_08 is not None:
            print(f"    首次超 0.80     : {first_08:.0%} CoT 位置")

        if auroc_at_40 > 0.80:
            print(f"    ✅ H2 通过（{tier_name}）")
        elif auroc_at_40 > 0.65:
            print(f"    ⚠️  H2 部分通过 → Pivot")
        else:
            print(f"    ❌ H2 失败")

        # 画图
        plt.figure(figsize=(10, 5))
        plt.plot(t_axis, aurocs, color='purple', linewidth=2)
        plt.axhline(y=0.80, color='red', linestyle='--', alpha=0.7, label='AUROC = 0.80')
        plt.axhline(y=0.50, color='gray', linestyle=':', alpha=0.5, label='Random')
        plt.axvline(x=0.40, color='gray', linestyle=':', alpha=0.6, label='40% CoT')
        if first_08 is not None:
            plt.axvline(x=first_08, color='green', linewidth=1.5,
                        label=f'First > 0.80 @ {first_08:.0%}')
        plt.xlabel('CoT Progress (normalized)', fontsize=12)
        plt.ylabel('AUROC', fontsize=12)
        plt.title(f'H2: {tier_name} — Detection Ability over CoT Progress\n'
                  f'AUROC@40% = {auroc_at_40:.3f}, max = {auroc_max:.3f}', fontsize=13)
        plt.ylim(0.35, 1.05)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / fname, dpi=150)
        print(f"    图已保存 → {fname}")
        plt.close()

    # 保存 Tier 1 的 AUROC 数据供 H3 使用
    sigma_h = load_sigma_trajectories(GENERATED_DIR / "T1_harmful", v_harm, LAYER, N_CHECKPOINTS)
    sigma_b = load_sigma_trajectories(GENERATED_DIR / "T1_benign", v_harm, LAYER, N_CHECKPOINTS)
    if len(sigma_h) > 0 and len(sigma_b) > 0:
        aurocs = compute_auroc_over_time(sigma_h, sigma_b, N_CHECKPOINTS)
        np.save(RESULTS_DIR / "auroc_per_step_tier1.npy", aurocs)
        np.save(RESULTS_DIR / "sigma_T1_harmful.npy", sigma_h)
        np.save(RESULTS_DIR / "sigma_T1_benign.npy", sigma_b)


if __name__ == "__main__":
    main()
