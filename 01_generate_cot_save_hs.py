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
    _dir = Path(__file__).resolve().parent
    _root = _dir.parent.parent if _dir.parent.name == "code" else _dir
    DATA_DIR = _root / "data"
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
    # Batch generation 需要 padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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
    在 GPU 上累积（clone），生成结束后统一转移到 CPU。
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output
        # activation: (batch, seq_len, hidden_dim) — 保留在 GPU
        h = activation[:, -1, :].detach().clone()
        cache[layer_idx].append(h)
    return hook_fn


def generate_with_hs(model, tokenizer, prompts, max_new_tokens=2048):
    """
    批量生成 CoT + 捕获全部层的 hidden states（GPU 上累积，一次转 CPU）。

    Args:
        prompts: list[str] — 一批 prompt
        max_new_tokens: int

    Returns:
        list of (text, hs_by_layer) tuples
    """
    batch_size = len(prompts)

    # 逐条 apply_chat_template，手动左填充（保证与单条 tokenization 一致）
    input_ids_list = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        input_ids_list.append(ids[0])

    max_prompt_len = max(len(ids) for ids in input_ids_list)
    padded_ids, attn_masks = [], []
    for ids in input_ids_list:
        pad_len = max_prompt_len - len(ids)
        padded_ids.append(torch.cat([
            torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype),
            ids
        ]))
        attn_masks.append(torch.cat([
            torch.zeros(pad_len, dtype=torch.long),
            torch.ones(len(ids), dtype=torch.long)
        ]))

    input_ids = torch.stack(padded_ids).to(model.device)
    attention_mask = torch.stack(attn_masks).to(model.device)

    # GPU cache
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
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    # 移除 hooks
    for hook in hooks:
        hook.remove()

    # GPU → CPU 一次性转移
    hs_all = {}
    for layer_idx in LAYERS_TO_SAVE:
        if cache[layer_idx]:
            stacked = torch.stack(cache[layer_idx])  # (n_steps, batch, hidden_dim)
            hs_all[layer_idx] = stacked.cpu().to(torch.float32).numpy()
        else:
            hs_all[layer_idx] = None

    del cache
    torch.cuda.empty_cache()

    # 分离每条 sample 的结果
    results = []
    for i in range(batch_size):
        gen_ids = output_ids[i, max_prompt_len:]

        # 找到实际生成长度（第一个 EOS 位置，用于裁剪 batch 内的 padding）
        eos_mask = (gen_ids == tokenizer.eos_token_id)
        if eos_mask.any():
            gen_len = eos_mask.nonzero(as_tuple=False)[0].item() + 1
        else:
            gen_len = len(gen_ids)

        text = tokenizer.decode(gen_ids[:gen_len], skip_special_tokens=False)

        # 提取 hidden states（N 个 token = N 次 hook fire，含 prefill）
        actual_steps = gen_len
        hs_by_layer = {}
        for layer_idx in LAYERS_TO_SAVE:
            if hs_all[layer_idx] is not None:
                total_steps = hs_all[layer_idx].shape[0]
                trim = min(actual_steps, total_steps)
                arr = hs_all[layer_idx][:trim, i, :]  # (n_tokens, hidden_dim)
                hs_by_layer[layer_idx] = arr.astype(NP_DTYPE)
            else:
                hs_by_layer[layer_idx] = np.zeros((0, 0), dtype=NP_DTYPE)

        results.append((text, hs_by_layer))

    return results


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
    parser.add_argument("--batch-size", type=int, default=1, help="batch 大小")
    parser.add_argument("--max-tokens", type=int, default=2048, help="最大生成 token 数")
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
    total = len(df)
    bs = args.batch_size
    print(f"\n{'='*60}")
    print(f"开始生成 {total} 条 CoT（batch={bs}，max_tokens={args.max_tokens}，"
          f"{len(LAYERS_TO_SAVE)} 层，{SAVE_DTYPE}）")
    print(f"{'='*60}\n")

    rows = list(df.iterrows())
    done = 0
    for batch_start in range(0, len(rows), bs):
        batch_rows = rows[batch_start:batch_start + bs]
        prompts = [row['prompt'] for _, row in batch_rows]

        results = generate_with_hs(model, tokenizer, prompts, args.max_tokens)

        for (idx, row), (text, hs) in zip(batch_rows, results):
            tier = row['tier']
            label = row['label']
            out_dir = GENERATED_DIR / f"{tier}_{label}"

            has_think = "</think>" in text
            n_tokens = len(list(hs.values())[0]) if hs else 0

            save_sample(out_dir, idx, row['prompt'], text, hs)
            done += 1

            print(f"  [{done}/{total}] {tier}/{label} "
                  f"tokens={n_tokens} "
                  f"{'✅' if has_think else '⚠️ no </think>'}")

    print(f"\n{'='*60}")
    print(f"✅ 全部完成。数据保存到 {GENERATED_DIR}")
    print(f"   目录: {', '.join(f'{t}_{l}' for t in ['T1','T2'] for l in ['harmful','benign'])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
