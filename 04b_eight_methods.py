"""
04b_eight_methods.py — 8 方案对比

在同一批 cached hidden states 上对比 8 种检测方案。
Go/No-Go 只看 M1, M5, M7 三方案，其余作为参考。

数据切分：GroupKFold by sample index，防止泄漏。

可以完全在本地 Mac 上运行。
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from pathlib import Path
import pandas as pd

# ── 路径 ──
DATA_DIR = Path(__file__).parent.parent.parent / "data"
GENERATED_DIR = DATA_DIR / "generated"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

# ── 参数 ──
N_CHECKPOINTS = 20

# 加载 v_harm 变体
v_harm = torch.load(RESULTS_DIR / "v_harm.pt").numpy()

# 尝试加载 CoT-trained 和 contrastive 变体
v_harm_cot = None
v_harm_contrastive = None
for f in RESULTS_DIR.glob("v_harm_*cot*.pt"):
    v_harm_cot = torch.load(f).numpy()
    break
# contrastive 变体在 pilot 中可能没有（需要 per-prompt 配对数据）
# 暂时跳过 M7b

# 自动检测层号和多层列表
sample_file = list((GENERATED_DIR / "T1_harmful").glob("hs_*.npz"))
if not sample_file:
    print("❌ 没有找到 hidden state 文件。请先运行 generate_mock_data.py 或 01_generate_cot_save_hs.py")
    exit(1)

available_layers = sorted([int(k.replace("layer_", "")) for k in np.load(sample_file[0]).keys()])
LAYER = available_layers[len(available_layers) // 2]
MULTI_LAYERS = [available_layers[i] for i in [1, len(available_layers)//2, -1]]  # 浅/中/深三层

print(f"使用层: 主层={LAYER}, 多层={MULTI_LAYERS}")


def load_checkpoint_hs(group_dir, layer_idx, n_checkpoints):
    """加载归一化时间步的 hidden states"""
    all_hs = []
    for f in sorted(group_dir.glob("hs_*.npz")):
        hs = np.load(f)[f"layer_{layer_idx}"]
        n = len(hs)
        if n < 5:
            continue
        indices = np.linspace(0, n - 1, n_checkpoints).astype(int)
        all_hs.append(hs[indices])
    return np.array(all_hs) if all_hs else None  # (n_samples, n_checkpoints, hidden_dim)


def main():
    print("=" * 60)
    print("8 方案对比 @ CoT 40% 位置")
    print("=" * 60)

    # 加载 Tier 1 数据（主分析）
    hs_h = {l: load_checkpoint_hs(GENERATED_DIR / "T1_harmful", l, N_CHECKPOINTS) for l in available_layers}
    hs_b = {l: load_checkpoint_hs(GENERATED_DIR / "T1_benign", l, N_CHECKPOINTS) for l in available_layers}

    if hs_h[LAYER] is None or hs_b[LAYER] is None:
        print("❌ 数据不足")
        return

    n_h = len(hs_h[LAYER])
    n_b = len(hs_b[LAYER])
    labels = np.array([1] * n_h + [0] * n_b)
    # GroupKFold 分组：harmful 和 benign 必须用不重叠的 group ID
    # 否则 harmful[0] 和 benign[0] 会被放到同一 fold（强制配对而非防泄漏）
    groups = np.concatenate([np.arange(n_h), np.arange(n_h, n_h + n_b)])

    t40 = int(N_CHECKPOINTS * 0.4)

    results = []

    # ── M1: cos(h, v_harm) ──
    scores = np.concatenate([hs_h[LAYER][:, t40, :] @ v_harm,
                              hs_b[LAYER][:, t40, :] @ v_harm])
    auroc = roc_auc_score(labels, scores)
    results.append({'method': 'M1: cos(h, v_harm)', 'auroc': auroc, 'go_nogo': True})

    # ── M2: dual direction ──
    # v_refusal 需要用 Arditi NeurIPS'24 方法从 compliant vs refusal states 提取
    # 在 pilot 阶段尚无 refusal 数据，标记为 NaN（与 M7b 同样处理）
    results.append({'method': 'M2: dual direction', 'auroc': float('nan'), 'go_nogo': False})
    print("  ⚠️ M2 跳过（v_refusal 未提取，需要 compliant vs refusal 配对数据）")

    # ── M3: MLP d→32→1 ──
    X = np.concatenate([hs_h[LAYER][:, t40, :], hs_b[LAYER][:, t40, :]])
    cv = GroupKFold(n_splits=min(5, min(n_h, n_b)))
    aurocs = []
    for train_idx, test_idx in cv.split(X, labels, groups=groups):
        clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=42)
        clf.fit(X[train_idx], labels[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        aurocs.append(roc_auc_score(labels[test_idx], proba))
    results.append({'method': 'M3: MLP d→32→1', 'auroc': np.mean(aurocs), 'go_nogo': False})

    # ── M4: multi-layer MLP ──
    X_multi = np.concatenate([
        np.concatenate([hs_h[l][:, t40, :] for l in MULTI_LAYERS], axis=1),
        np.concatenate([hs_b[l][:, t40, :] for l in MULTI_LAYERS], axis=1)
    ])
    aurocs = []
    for train_idx, test_idx in cv.split(X_multi, labels, groups=groups):
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42)
        clf.fit(X_multi[train_idx], labels[train_idx])
        proba = clf.predict_proba(X_multi[test_idx])[:, 1]
        aurocs.append(roc_auc_score(labels[test_idx], proba))
    results.append({'method': 'M4: multi-layer MLP', 'auroc': np.mean(aurocs), 'go_nogo': False})

    # ── M5: full linear probe（Chan 上界）──
    aurocs = []
    for train_idx, test_idx in cv.split(X, labels, groups=groups):
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X[train_idx], labels[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        aurocs.append(roc_auc_score(labels[test_idx], proba))
    results.append({'method': 'M5: full linear probe', 'auroc': np.mean(aurocs), 'go_nogo': True})

    # ── M6: multi-layer linear ──
    aurocs = []
    for train_idx, test_idx in cv.split(X_multi, labels, groups=groups):
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_multi[train_idx], labels[train_idx])
        proba = clf.predict_proba(X_multi[test_idx])[:, 1]
        aurocs.append(roc_auc_score(labels[test_idx], proba))
    results.append({'method': 'M6: multi-layer linear', 'auroc': np.mean(aurocs), 'go_nogo': False})

    # ── M7: CoT-trained 1D ──
    if v_harm_cot is not None:
        scores = np.concatenate([hs_h[LAYER][:, t40, :] @ v_harm_cot,
                                  hs_b[LAYER][:, t40, :] @ v_harm_cot])
        auroc = roc_auc_score(labels, scores)
        results.append({'method': 'M7: CoT-trained 1D', 'auroc': auroc, 'go_nogo': True})
    else:
        results.append({'method': 'M7: CoT-trained 1D', 'auroc': float('nan'), 'go_nogo': True})
        print("  ⚠️ M7 跳过（v_harm_cot 未生成）")

    # ── M7b: contrastive ──
    results.append({'method': 'M7b: contrastive 1D', 'auroc': float('nan'), 'go_nogo': False})
    print("  ⚠️ M7b 跳过（需要 per-prompt 配对数据，Tier 1 暂无）")

    # ── 打印结果 ──
    print(f"\n{'方案':<30} {'AUROC':>8} {'Go/No-Go':>10}")
    print("-" * 50)
    for r in results:
        marker = " ★" if r['go_nogo'] else ""
        auroc_str = f"{r['auroc']:.4f}" if not np.isnan(r['auroc']) else "  N/A "
        print(f"  {r['method']:<28} {auroc_str:>8}{marker}")

    print(f"\n★ = Go/No-Go 决策方案（M1, M5, M7）")
    print(f"Pre-registered rule: 选 AUROC 最高的；差距 <0.02 选最简单的")

    # 保存
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "eight_methods_comparison.csv", index=False)
    print(f"\n✅ 保存到: {RESULTS_DIR / 'eight_methods_comparison.csv'}")


if __name__ == "__main__":
    main()
