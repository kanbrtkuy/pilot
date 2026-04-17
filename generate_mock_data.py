"""
generate_mock_data.py — 生成 mock hidden states 用于本地测试分析脚本

模拟 DeepSeek-R1-Distill-Qwen-7B 的输出：
  - 28 层，hidden_dim = 3584
  - 保存关键层 {11, 19, 23, 27, 32, 39, 47} 的 subset = {5, 9, 11, 13, 15, 19, 27}
    （7B 只有 28 层，所以层号要调整）
  - 每条 CoT 约 100-300 tokens

Mock 数据的 shape 和 dtype 与真实数据完全一致，
保证分析脚本在 RunPod 上无需修改即可运行。

⚠️ 注意：7B 模型只有 28 层（0-27），不是 14B 的 48 层。
层选择相应调整为 {5, 9, 11, 13, 15, 19, 27}。
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path

np.random.seed(42)

# 从共享配置导入（确保和分析脚本使用同一层号）
try:
    from config import N_LAYERS, HIDDEN_DIM, LAYERS_TO_SAVE, SAVE_DTYPE
except ImportError:
    N_LAYERS = 28
    HIDDEN_DIM = 3584
    LAYERS_TO_SAVE = list(range(N_LAYERS))  # 全部 28 层
    SAVE_DTYPE = "float16"

# ── 数据参数 ──
N_SAMPLES_PER_GROUP = 10  # mock 数据每组 10 条（测试用，正式版 50+）
COT_LEN_RANGE = (80, 250)  # CoT token 数范围

# ── 输出目录 ──
DATA_DIR = Path(__file__).parent.parent.parent / "data"
GENERATED_DIR = DATA_DIR / "generated"

GROUPS = ['T1_harmful', 'T1_benign', 'T2_harmful', 'T2_benign']

def make_mock_trajectory(n_tokens, label, tier):
    """
    生成一条 mock hidden state 轨迹。

    G1a/G1b (harmful): σ(t) 从低逐步升高（模拟 drift）
    G3a/G3b (benign):  σ(t) 全程低位
    """
    hs_by_layer = {}

    for layer_idx in LAYERS_TO_SAVE:
        # 基础 hidden states：随机
        hs = np.random.randn(n_tokens, HIDDEN_DIM).astype(np.float16) * 0.1

        if label == 'harmful':
            # 模拟 drift：后半段在某个方向上逐渐增大
            drift_direction = np.random.randn(HIDDEN_DIM).astype(np.float16)
            drift_direction = drift_direction / np.linalg.norm(drift_direction)

            for t in range(n_tokens):
                progress = t / n_tokens  # 0 → 1
                if tier == 'T1':
                    # Tier 1：前 30% 是解码阶段（无 drift），之后开始 drift
                    drift_strength = max(0, (progress - 0.3) / 0.7) * 2.0
                else:
                    # Tier 2：从一开始就有微弱信号，逐步增强
                    drift_strength = progress * 1.5

                hs[t] += drift_direction * drift_strength

        hs_by_layer[f"layer_{layer_idx}"] = hs

    return hs_by_layer


def main():
    print("=" * 60)
    print("生成 mock hidden states")
    print(f"  模型: R1-Distill-Qwen-7B ({N_LAYERS} 层, dim={HIDDEN_DIM})")
    print(f"  保存层: {LAYERS_TO_SAVE}")
    print(f"  每组: {N_SAMPLES_PER_GROUP} 条")
    print("=" * 60)

    for group in GROUPS:
        group_dir = GENERATED_DIR / group
        group_dir.mkdir(parents=True, exist_ok=True)

        tier = 'T1' if 'T1' in group else 'T2'
        label = 'harmful' if 'harmful' in group else 'benign'

        cot_texts = []

        for i in range(N_SAMPLES_PER_GROUP):
            n_tokens = np.random.randint(*COT_LEN_RANGE)
            hs_by_layer = make_mock_trajectory(n_tokens, label, tier)

            # 保存 hidden states
            np.savez_compressed(
                group_dir / f"hs_{i:04d}.npz",
                **hs_by_layer
            )

            # 保存 mock CoT 文本
            if label == 'harmful' and tier == 'T1':
                mock_cot = (
                    "<think>\nLet me decode this Base64... "
                    "decoded to ROT13... applying ROT13 decoding... "
                    f"the result is: [mock harmful content sample {i}]... "
                    "the steps would involve...</think>\n"
                    f"[Mock harmful response {i}]"
                )
            elif label == 'harmful' and tier == 'T2':
                mock_cot = (
                    f"<think>\nThe user is asking about [topic {i}]... "
                    "I should help with this creative writing exercise... "
                    "the detailed process involves...</think>\n"
                    f"[Mock harmful response {i}]"
                )
            else:
                mock_cot = (
                    f"<think>\nLet me decode/analyze this... "
                    f"this appears to be about [benign topic {i}]... "
                    "I can help with this.</think>\n"
                    f"[Mock benign response {i}]"
                )

            cot_texts.append({
                "id": i,
                "prompt": f"[Mock {group} prompt {i}]",
                "cot": mock_cot
            })

        # 保存 CoT 文本
        with open(group_dir / "cot_texts.jsonl", "w") as f:
            for item in cot_texts:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"  ✅ {group}: {N_SAMPLES_PER_GROUP} 条, "
              f"tokens={COT_LEN_RANGE}, "
              f"layers={len(LAYERS_TO_SAVE)}")

    # 验证
    print(f"\n验证:")
    test_file = GENERATED_DIR / "T1_harmful" / "hs_0000.npz"
    hs = np.load(test_file)
    print(f"  文件: {test_file.name}")
    print(f"  层: {sorted(hs.keys())}")
    for k in sorted(hs.keys()):
        print(f"    {k}: shape={hs[k].shape}, dtype={hs[k].dtype}")

    print(f"\n✅ Mock 数据生成完毕: {GENERATED_DIR}")
    print(f"   共 {len(GROUPS)} 组 × {N_SAMPLES_PER_GROUP} 条 = {len(GROUPS) * N_SAMPLES_PER_GROUP} 条")


if __name__ == "__main__":
    main()
