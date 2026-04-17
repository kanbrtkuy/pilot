"""
01_generate_cot_save_hs.py — GPU 推理：生成 CoT + 保存全部层 hidden states

模型：DeepSeek-R1-Distill-Qwen-7B
输入：data/all_prompts.csv（由 00_prepare_data.py 生成）
输出：data/generated/{T1_harmful,T1_benign,T2_harmful,T2_benign}/
      每条样本保存 hs_{id:04d}.npz（全部 28 层 hidden states，float16）
      + cot_texts.jsonl（CoT 文本）

一轮推理，无 system prompt（R1-7B 官方建议不加 system prompt）。

必须在 GPU 上运行（RunPod A100 80GB）。
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 配置 ──

try:
    from config import (
        MODEL_NAME, N_LAYERS, LAYERS_TO_SAVE, SAVE_DTYPE,
        DATA_DIR, GENERATED_DIR
    )
except ImportError:
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    N_LAYERS = 28
    LAYERS_TO_SAVE = list(range(28))
    SAVE_DTYPE = "float16"
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    GENERATED_DIR = DATA_DIR / "generated"

# RunPod 上模型可能在本地路径
MODEL_PATH = Path("/workspace/models/DeepSeek-R1-Distill-Qwen-7B")
if not MODEL_PATH.exists():
    MODEL_PATH = MODEL_NAME  # fallback to HuggingFace download

# numpy dtype
NP_DTYPE = np.float16 if SAVE_DTYPE == "float16" else np.float32


def load_model():
    """加载模型和 tokenizer"""
    print(f"加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    model.eval()
    print(f"  ✅ 模型加载完成，{N_LAYERS} 层，device={model.device}")
    return model, tokenizer


def make_capture_hook(cache, layer_idx):
    """
    生成 forward hook，捕获每个生成 token 最后位置的 hidden state。

    挂在 model.model.layers[layer_idx] 上（residual stream output）。
    """
    def hook_fn(module, input, output):
        # output[0] shape: (batch=1, seq_len, hidden_dim)
        h = output[0][0, -1, :].detach().cpu().to(torch.float32).numpy()
        cache[layer_idx].append(h)
    return hook_fn


def generate_with_hs(model, tokenizer, prompt, max_new_tokens=2048):
    """
    生成 CoT + 捕获全部层的 hidden states。

    返回:
        text: str — 完整生成文本（含 <think>...</think>）
        hs_by_layer: dict — {layer_idx: np.array(n_tokens, hidden_dim)}
    """
    # 构建输入（无 system prompt）
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 准备 cache
    cache = {layer_idx: [] for layer_idx in LAYERS_TO_SAVE}

    # 注册 hooks
    hooks = []
    for layer_idx in LAYERS_TO_SAVE:
        hook = model.model.layers[layer_idx].register_forward_hook(
            make_capture_hook(cache, layer_idx)
        )
        hooks.append(hook)

    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    # 移除 hooks
    for hook in hooks:
        hook.remove()

    # 解码文本
    generated_ids = output_ids[0][input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # 组装 hidden states
    hs_by_layer = {}
    for layer_idx in LAYERS_TO_SAVE:
        if cache[layer_idx]:
            arr = np.array(cache[layer_idx])  # (n_tokens, hidden_dim)
            hs_by_layer[layer_idx] = arr.astype(NP_DTYPE)
        else:
            hs_by_layer[layer_idx] = np.zeros((0, 0), dtype=NP_DTYPE)

    return text, hs_by_layer


def save_sample(output_dir, sample_id, prompt, text, hs_by_layer):
    """保存一条样本的 hidden states + CoT 文本"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hidden states
    np.savez_compressed(
        output_dir / f"hs_{sample_id:04d}.npz",
        **{f"layer_{k}": v for k, v in hs_by_layer.items()}
    )

    # CoT 文本（append 模式）
    with open(output_dir / "cot_texts.jsonl", "a") as f:
        f.write(json.dumps({
            "id": sample_id,
            "prompt": prompt[:200],  # 截断保存，节省空间
            "cot": text
        }, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只跑前 N 条")
    parser.add_argument("--n", type=int, default=5, help="dry-run 条数")
    parser.add_argument("--start", type=int, default=0, help="从第几条开始（断点续跑）")
    args = parser.parse_args()

    # 加载数据
    prompts_file = DATA_DIR / "all_prompts.csv"
    if not prompts_file.exists():
        print(f"❌ {prompts_file} 不存在。请先运行 00_prepare_data.py")
        sys.exit(1)

    df = pd.read_csv(prompts_file)
    print(f"加载 {len(df)} 条 prompt")

    if args.dry_run:
        df = df.head(args.n)
        print(f"⚠️ Dry run 模式：只跑 {args.n} 条")

    if args.start > 0:
        df = df.iloc[args.start:]
        print(f"从第 {args.start} 条开始（断点续跑）")

    # 加载模型
    model, tokenizer = load_model()

    # 生成
    print(f"\n{'='*60}")
    print(f"开始生成 {len(df)} 条 CoT（全部 {len(LAYERS_TO_SAVE)} 层，{SAVE_DTYPE}）")
    print(f"{'='*60}\n")

    for idx, row in df.iterrows():
        tier = row['tier']
        label = row['label']
        group = row.get('group', f"{tier}_{label}")
        out_dir = GENERATED_DIR / f"{tier}_{label}"

        text, hs = generate_with_hs(model, tokenizer, row['prompt'])

        # 检查 </think> 是否存在
        has_think = "</think>" in text
        n_tokens = len(list(hs.values())[0]) if hs else 0

        save_sample(out_dir, idx, row['prompt'], text, hs)

        print(f"  [{idx+1}/{len(df)}] {tier}/{label} "
              f"tokens={n_tokens} "
              f"{'✅' if has_think else '⚠️ no </think>'}")

    print(f"\n{'='*60}")
    print(f"✅ 全部完成。数据保存到 {GENERATED_DIR}")
    print(f"   目录: {', '.join(f'{t}_{l}' for t in ['T1','T2'] for l in ['harmful','benign'])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
