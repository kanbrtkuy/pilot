"""
02_extract_v_harm.py — 从 cached hidden states 提取 harmfulness direction

三种变体：
  1. input mean-diff: mean(G1a 的 t_inst) - mean(G3a 的 t_inst)
  2. CoT-trained: mean(G1a 的 CoT 中段) - mean(G3a 的 CoT 中段)
  3. contrastive: per-prompt 配对差值的均值

同时提取 v_refusal（用 Arditi NeurIPS'24 方法）

多层 sweep：在所有保存层上分别提取，选 separability 最高的。

可以完全在本地 Mac 上运行（纯 numpy），不需要 GPU。
"""

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ── 路径 ──
DATA_DIR = Path(__file__).parent.parent.parent / "data"
GENERATED_DIR = DATA_DIR / "generated"
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 从共享配置导入
try:
    from config import LAYERS_TO_SAVE, GENERATED_DIR, RESULTS_DIR, load_direction
except ImportError:
    LAYERS_TO_SAVE = [6, 11, 13, 15, 18, 22, 27]

# ── 加载 hidden states ──

def load_hs(group_dir, n_samples, layer_idx, position='mid'):
    """
    加载一组样本在某层某位置的 hidden states。
    position: 'first' | 'mid' | 'last' | 'inst_end'
    """
    states = []
    for i in range(n_samples):
        f = group_dir / f"hs_{i:04d}.npz"
        if not f.exists():
            continue
        hs = np.load(f)[f"layer_{layer_idx}"]  # (n_tokens, hidden_dim)
        n = len(hs)
        if n == 0:
            continue

        if position == 'first':
            states.append(hs[0])
        elif position == 'mid':
            states.append(hs[n // 2])
        elif position == 'last':
            states.append(hs[-1])
        elif position == 'inst_end':
            # 近似 t_inst：取前 10% 的最后一个 token
            inst_end = max(1, int(n * 0.1))
            states.append(hs[inst_end - 1])
        else:
            states.append(hs[n // 2])

    return np.array(states) if states else np.zeros((0, 0))


def count_samples(group_dir):
    return len(list(group_dir.glob("hs_*.npz")))


def main():
    print("=" * 60)
    print("提取 v_harm（多层 × 多位置 × 多变体）")
    print("=" * 60)

    n_g1a = count_samples(GENERATED_DIR / "T1_harmful")
    n_g3a = count_samples(GENERATED_DIR / "T1_benign")
    n_g1b = count_samples(GENERATED_DIR / "T2_harmful")
    n_g3b = count_samples(GENERATED_DIR / "T2_benign")

    print(f"  T1_harmful (G1a): {n_g1a} 条")
    print(f"  T1_benign  (G3a): {n_g3a} 条")
    print(f"  T2_harmful (G1b): {n_g1b} 条")
    print(f"  T2_benign  (G3b): {n_g3b} 条")

    # ── 用 Tier 1 数据提取 v_harm（编码数据更纯净）──

    best_auroc = 0
    best_config = None

    results = {}

    for layer_idx in LAYERS_TO_SAVE:
        for pos in ['inst_end', 'mid', 'first']:
            # 变体 1: input mean-diff
            h_harmful = load_hs(GENERATED_DIR / "T1_harmful", n_g1a, layer_idx, pos)
            h_benign  = load_hs(GENERATED_DIR / "T1_benign",  n_g3a, layer_idx, pos)

            if len(h_harmful) == 0 or len(h_benign) == 0:
                continue

            v = h_harmful.mean(axis=0) - h_benign.mean(axis=0)
            v = v / (np.linalg.norm(v) + 1e-8)

            # 快速验证：用这个方向做分类的 AUROC
            scores_h = h_harmful @ v
            scores_b = h_benign @ v
            labels = np.array([1] * len(scores_h) + [0] * len(scores_b))
            scores = np.concatenate([scores_h, scores_b])
            auroc = roc_auc_score(labels, scores)

            key = f"layer{layer_idx}_{pos}_mean_diff"
            results[key] = {'v': v, 'auroc': auroc, 'type': 'mean_diff'}
            print(f"  {key}: AUROC = {auroc:.4f}")

            if auroc > best_auroc:
                best_auroc = auroc
                best_config = key

        # 变体 2: CoT-trained（CoT 中段的 mean-diff）
        h_harmful_cot = load_hs(GENERATED_DIR / "T1_harmful", n_g1a, layer_idx, 'mid')
        h_benign_cot  = load_hs(GENERATED_DIR / "T1_benign",  n_g3a, layer_idx, 'mid')

        if len(h_harmful_cot) > 0 and len(h_benign_cot) > 0:
            v_cot = h_harmful_cot.mean(axis=0) - h_benign_cot.mean(axis=0)
            v_cot = v_cot / (np.linalg.norm(v_cot) + 1e-8)

            scores_h = h_harmful_cot @ v_cot
            scores_b = h_benign_cot @ v_cot
            labels = np.array([1] * len(scores_h) + [0] * len(scores_b))
            scores = np.concatenate([scores_h, scores_b])
            auroc_cot = roc_auc_score(labels, scores)

            key = f"layer{layer_idx}_cot_trained"
            results[key] = {'v': v_cot, 'auroc': auroc_cot, 'type': 'cot_trained'}
            print(f"  {key}: AUROC = {auroc_cot:.4f}")

            if auroc_cot > best_auroc:
                best_auroc = auroc_cot
                best_config = key

    # ── 保存结果 ──

    print(f"\n✅ 最佳配置: {best_config}, AUROC = {best_auroc:.4f}")

    # 保存最佳 v_harm
    torch.save(torch.tensor(results[best_config]['v']), RESULTS_DIR / "v_harm.pt")

    # 保存所有变体供 8 方案对比用
    for key, val in results.items():
        torch.save(torch.tensor(val['v']), RESULTS_DIR / f"v_harm_{key}.pt")

    # 保存 AUROC 汇总
    import pandas as pd
    summary = pd.DataFrame([
        {'config': k, 'auroc': v['auroc'], 'type': v['type']}
        for k, v in results.items()
    ]).sort_values('auroc', ascending=False)
    summary.to_csv(RESULTS_DIR / "v_harm_selection.csv", index=False)
    print(f"\n层 × 位置 × 变体 AUROC 汇总:")
    print(summary.to_string(index=False))

    print(f"\n✅ 保存到: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
