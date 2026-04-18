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
import hashlib
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
        DATA_DIR, GENERATED_DIR,
        TARGET_MAX_TOKENS, N_SAVE_CHECKPOINTS, DECODE_MARKERS,
    )
except ImportError:
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    N_LAYERS = 48
    LAYERS_TO_SAVE = list(range(48))
    SAVE_DTYPE = "float16"
    _dir = Path(__file__).resolve().parent
    _root = _dir.parent.parent if _dir.parent.name == "code" else _dir
    DATA_DIR = _root / "data"
    GENERATED_DIR = DATA_DIR / "generated"
    TARGET_MAX_TOKENS = 4096
    N_SAVE_CHECKPOINTS = 100
    DECODE_MARKERS = ["decoded", "result is", "the instruction is", "解码结果", "解码后"]

# 本地模型路径（RunPod 上已下载）；若不存在则 fallback 到 HF 下载
_local_14b = Path("/workspace/models/DeepSeek-R1-Distill-Qwen-14B")
_local_7b = Path("/workspace/models/DeepSeek-R1-Distill-Qwen-7B")
if _local_14b.exists() and "14B" in MODEL_NAME:
    MODEL_PATH = _local_14b
elif _local_7b.exists() and "7B" in MODEL_NAME:
    MODEL_PATH = _local_7b
else:
    MODEL_PATH = MODEL_NAME

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

        # Hook 步数语义（HF generate 实际行为，不是 N+1 次 forward）：
        #   N 个新 token 总共触发 N 次 forward —— prefill 那次已经产生了第 1 个 token 的 logits。
        #   step 0         → prefill：prompt 最后一个 token 的 h（= 输入级信号，H3 用）
        #   step k (1..N-1) → 生成 token k-1 产出后、预测 token k 之前的 h
        #   最后一个生成 token（index N-1）的 h 捕获不到 —— 没有后续 forward。
        # 因此 n_positions = gen_len（不是 gen_len+1）。
        n_positions = gen_len if gen_len > 0 else 1  # 至少保留 prefill
        n_save = min(N_SAVE_CHECKPOINTS, n_positions)
        if n_save >= 2:
            save_indices = np.linspace(0, n_positions - 1, n_save).astype(int)
        elif n_save == 1:
            save_indices = np.array([0], dtype=int)
        else:
            save_indices = np.array([], dtype=int)

        # 从 hs_all 取出 sample i 的 checkpoint；batch>1 时 total_steps = max_gen_len+1，
        # 对 gen_len 较短的 sample，save_indices 仍在合法范围内（clip 作为保险）。
        n_hook_steps = None
        hs_by_layer = {}
        for layer_idx in LAYERS_TO_SAVE:
            if hs_all[layer_idx] is not None and n_save > 0:
                total_steps = hs_all[layer_idx].shape[0]
                if n_hook_steps is None:
                    n_hook_steps = int(total_steps)
                safe_indices = np.clip(save_indices, 0, total_steps - 1)
                arr = hs_all[layer_idx][safe_indices, i, :]  # (n_save, hidden_dim)
                hs_by_layer[layer_idx] = arr.astype(NP_DTYPE)
            else:
                hs_by_layer[layer_idx] = np.zeros((0, 0), dtype=NP_DTYPE)

        results.append((text, hs_by_layer, gen_len, save_indices, n_hook_steps))

    return results


def _find_decode_marker(text):
    """在 CoT 文本中查找最早的解码标志词。返回 (marker_name, char_pos) 或 (None, None)。"""
    text_lower = text.lower()
    best_pos = None
    best_name = None
    for m in DECODE_MARKERS:
        p = text_lower.find(m.lower())
        if p >= 0 and (best_pos is None or p < best_pos):
            best_pos = p
            best_name = m
    return best_name, best_pos


def save_sample(output_dir, sample_id, prompt, text, hs_by_layer, gen_len, save_indices, n_hook_steps=None):
    """保存一条样本的 hidden states + CoT 文本 + 解码标志词元数据

    安全约束（见 memory/feedback_harmful_payload_terminal.md）：
    - 不把 prompt 原文写进 metadata（T2_harmful 可能是真实对抗文本）；改写长度 + SHA256 前缀。
      原文如需回查，用 id 在 data/all_prompts.csv 里 join。
    - `cot` 文本保留在磁盘以便后续分析；inspector 脚本禁止打印内容。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Hidden states（已下采样到 N_SAVE_CHECKPOINTS）
    np.savez_compressed(
        output_dir / f"hs_{sample_id:04d}.npz",
        **{f"layer_{k}": v for k, v in hs_by_layer.items()}
    )

    # 解码标志词定位（char 位置 → 生成 token 位置 → checkpoint 索引）
    # save_indices 索引 0 = prefill（prompt 最后 token 的 h），
    # save_indices 索引 k (1..gen_len-1) = 产出生成 token k-1 之后的 h。
    # 生成 token i 的对应 hook step = i + 1，但最后一个生成 token (i=gen_len-1)
    # 没有后续 forward，h 未捕获 —— 钳到最近的已捕获 hook step (gen_len-1)。
    marker_name, marker_char_pos = _find_decode_marker(text)
    marker_token_pos = None
    marker_token_pct = None
    marker_checkpoint_idx = None
    if marker_char_pos is not None and len(text) > 0 and gen_len > 0:
        marker_token_pos = int(marker_char_pos / len(text) * gen_len)
        marker_token_pct = float(marker_token_pos / gen_len) if gen_len > 0 else None
        if len(save_indices) > 0:
            marker_hook_step = min(marker_token_pos + 1, gen_len - 1)
            marker_checkpoint_idx = int(np.searchsorted(save_indices, marker_hook_step))
            marker_checkpoint_idx = min(marker_checkpoint_idx, len(save_indices) - 1)

    prompt_str = prompt if isinstance(prompt, str) else str(prompt)
    prompt_bytes = prompt_str.encode("utf-8", errors="replace")
    prompt_sha8 = hashlib.sha256(prompt_bytes).hexdigest()[:8]

    meta = {
        "id": int(sample_id),
        "prompt_len": len(prompt_str),
        "prompt_sha8": prompt_sha8,
        "cot": text,  # on-disk only; inspector must not print this
        "cot_chars": len(text),
        "n_gen_tokens": int(gen_len),
        "n_hook_steps": int(n_hook_steps) if n_hook_steps is not None else None,
        "n_checkpoints_saved": int(len(save_indices)),
        "has_closed_think": "</think>" in text,
        "decode_marker_found": marker_name is not None,
        "decode_marker_name": marker_name,
        "decode_marker_char_pos": marker_char_pos,
        "decode_marker_token_pos": marker_token_pos,
        "decode_marker_token_pct": marker_token_pct,
        "decode_marker_checkpoint_idx": marker_checkpoint_idx,
    }
    with open(output_dir / "cot_texts.jsonl", "a") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只跑前 N 条")
    parser.add_argument("--n", type=int, default=5, help="dry-run 条数")
    parser.add_argument("--start", type=int, default=0, help="从第几条开始（断点续跑）")
    parser.add_argument("--batch-size", type=int, default=1, help="batch 大小")
    parser.add_argument("--max-tokens", type=int, default=TARGET_MAX_TOKENS,
                        help=f"最大生成 token 数（默认 {TARGET_MAX_TOKENS}，对齐工业部署预算）")
    parser.add_argument("--data-file", type=str, default=None,
                        help="CSV 输入文件（默认 data/all_prompts.csv；用于 capability/isolation 测试切换数据源）")
    args = parser.parse_args()

    # 加载数据
    prompts_file = Path(args.data_file) if args.data_file else (DATA_DIR / "all_prompts.csv")
    if not prompts_file.exists():
        print(f"❌ {prompts_file} 不存在。请先运行 00_prepare_data.py")
        sys.exit(1)

    df = pd.read_csv(prompts_file)
    print(f"加载 {len(df)} 条 prompt")

    # 先应用 --start（断点续跑 / 跳组），再应用 --dry-run 的 --n（取前 N 条）
    # 顺序关键：先 start 后 head，才能实现 "从第 50 条起取 5 条" 的跳组 dry-run
    if args.start > 0:
        df = df.iloc[args.start:]
        print(f"从第 {args.start} 条开始（断点续跑 / 跳组）")

    if args.dry_run:
        df = df.head(args.n)
        print(f"⚠️ Dry run 模式：只跑 {args.n} 条")

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
    invariant_logged = False
    for batch_start in range(0, len(rows), bs):
        batch_rows = rows[batch_start:batch_start + bs]
        prompts = [row['prompt'] for _, row in batch_rows]

        results = generate_with_hs(model, tokenizer, prompts, args.max_tokens)

        for (idx, row), (text, hs, gen_len, save_indices, n_hook_steps) in zip(batch_rows, results):
            tier = row['tier']
            label = row['label']
            out_dir = GENERATED_DIR / f"{tier}_{label}"

            has_think = "</think>" in text
            n_ckpts = len(save_indices)

            # One-shot hook-step invariant (HF generate: N forwards for N new tokens).
            # For batch=1, n_hook_steps == gen_len; for batch>1, n_hook_steps == max(gen_len).
            if args.dry_run and not invariant_logged and n_hook_steps is not None:
                expected = gen_len
                ok = "✅" if n_hook_steps == expected else ("⚠️" if n_hook_steps < expected else "ℹ️")
                print(f"  [invariant] {ok} n_hook_steps={n_hook_steps} "
                      f"vs gen_len={expected} (batch=1 expects equal, batch>1 expects >=)")
                invariant_logged = True

            save_sample(out_dir, idx, row['prompt'], text, hs,
                        gen_len=gen_len, save_indices=save_indices,
                        n_hook_steps=n_hook_steps)
            done += 1

            print(f"  [{done}/{total}] {tier}/{label} "
                  f"gen_tokens={gen_len} ckpts={n_ckpts} "
                  f"{'✅ closed' if has_think else '⚠️ no </think>'}")

    print(f"\n{'='*60}")
    print(f"✅ 全部完成。数据保存到 {GENERATED_DIR}")
    print(f"   目录: {', '.join(f'{t}_{l}' for t in ['T1','T2'] for l in ['harmful','benign'])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
