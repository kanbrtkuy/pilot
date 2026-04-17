"""
03_h1_sigma_curves.py — H1 验证：σ(t) 轨迹分叉

对 Tier 1 (G1a vs G3a) 和 Tier 2 (G1b vs G3b) 分别画 σ(t) 曲线。
σ(t) = cos(h(t), v_harm)

判断标准：CoT 中段（40%-60%）Welch t-test p < 0.01, Cohen's d > 0.5

可以完全在本地 Mac 上运行。
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import stats
from pathlib import Path

# ── 路径 ──
DATA_DIR = Path(__file__).parent.parent.parent / "data"
GENERATED_DIR = DATA_DIR / "generated"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 参数 ──
N_CHECKPOINTS = 20  # 将 CoT 归一化为 20 个时间步
# 使用提取 v_harm 时选定的最佳层（从 02 脚本的输出读取）
# 如果没跑过 02，默认用中间层
DEFAULT_LAYER = 11  # 7B 模型的中间偏后

# 加载 v_harm
v_harm_path = RESULTS_DIR / "v_harm.pt"
if v_harm_path.exists():
    v_harm = torch.load(v_harm_path).numpy()
    print(f"加载 v_harm: shape={v_harm.shape}")
else:
    print("⚠️ v_harm.pt 不存在，请先运行 02_extract_v_harm.py")
    exit(1)

# 自动检测层号
sample_file = list((GENERATED_DIR / "T1_harmful").glob("hs_*.npz"))
if sample_file:
    available_layers = [k.replace("layer_", "") for k in np.load(sample_file[0]).keys()]
    LAYER = int(available_layers[len(available_layers) // 2])  # 取中间层
    print(f"使用层: layer_{LAYER}")
else:
    LAYER = DEFAULT_LAYER


def load_sigma_trajectories(group_dir, v, layer_idx, n_checkpoints):
    """加载一组样本的 σ(t) 轨迹，归一化到固定时间步数"""
    trajectories = []
    for f in sorted(group_dir.glob("hs_*.npz")):
        hs = np.load(f)[f"layer_{layer_idx}"]
        n_tokens = len(hs)
        if n_tokens < 5:
            continue

        indices = np.linspace(0, n_tokens - 1, n_checkpoints).astype(int)
        sampled = hs[indices]
        sigma = sampled @ v  # (n_checkpoints,)
        trajectories.append(sigma)

    return np.array(trajectories) if trajectories else np.zeros((0, n_checkpoints))


def plot_tier(tier_name, harmful_dir, benign_dir, output_name):
    """画一个 Tier 的 σ(t) 曲线"""
    sigma_h = load_sigma_trajectories(harmful_dir, v_harm, LAYER, N_CHECKPOINTS)
    sigma_b = load_sigma_trajectories(benign_dir, v_harm, LAYER, N_CHECKPOINTS)

    print(f"\n{tier_name}: harmful={len(sigma_h)} 条, benign={len(sigma_b)} 条")

    if len(sigma_h) == 0 or len(sigma_b) == 0:
        print("  ⚠️ 数据不足，跳过")
        return

    # 统计检验（CoT 40%-60% 区间）
    mid_s = int(N_CHECKPOINTS * 0.4)
    mid_e = int(N_CHECKPOINTS * 0.6)
    h_mid = sigma_h[:, mid_s:mid_e].mean(axis=1)
    b_mid = sigma_b[:, mid_s:mid_e].mean(axis=1)

    t_stat, p_value = stats.ttest_ind(h_mid, b_mid, equal_var=False)
    pooled_std = np.sqrt((h_mid.std()**2 + b_mid.std()**2) / 2)
    cohen_d = (h_mid.mean() - b_mid.mean()) / pooled_std if pooled_std > 0 else 0

    print(f"  σ 均值（中段）: harmful={h_mid.mean():.4f}, benign={b_mid.mean():.4f}")
    print(f"  Welch t-test: p = {p_value:.4f}")
    print(f"  Cohen's d   : {cohen_d:.4f}")

    if p_value < 0.01 and abs(cohen_d) > 0.5:
        print(f"  ✅ H1 通过（{tier_name}）")
    elif p_value < 0.05:
        print(f"  ⚠️  H1 部分通过（显著但效应量不足）")
    else:
        print(f"  ❌ H1 失败 → Kill")

    # 画图
    t_axis = np.linspace(0, 1, N_CHECKPOINTS)

    plt.figure(figsize=(10, 5))
    for data, label, color in [
        (sigma_h, 'Harmful (drift)', 'red'),
        (sigma_b, 'Benign (control)', 'blue'),
    ]:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        plt.plot(t_axis, mean, color=color, linewidth=2, label=label)
        plt.fill_between(t_axis, mean - std, mean + std, alpha=0.15, color=color)

    plt.axvspan(0.4, 0.6, alpha=0.08, color='gray', label='统计检验窗口')
    plt.xlabel('CoT Progress (normalized)', fontsize=12)
    plt.ylabel('σ(t) = cos(h(t), v_harm)', fontsize=12)
    plt.title(f'H1: {tier_name} — σ(t) Trajectory (Layer {LAYER})\n'
              f'p={p_value:.4f}, d={cohen_d:.2f}', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / output_name, dpi=150)
    print(f"  图已保存 → {output_name}")
    plt.close()


def main():
    print("=" * 60)
    print("H1 验证：σ(t) 轨迹分叉")
    print("=" * 60)

    plot_tier(
        "Tier 1 (Cipher Encoding)",
        GENERATED_DIR / "T1_harmful",
        GENERATED_DIR / "T1_benign",
        "h1_tier1_sigma_curves.png"
    )

    plot_tier(
        "Tier 2 (WildJailbreak)",
        GENERATED_DIR / "T2_harmful",
        GENERATED_DIR / "T2_benign",
        "h1_tier2_sigma_curves.png"
    )


if __name__ == "__main__":
    main()
