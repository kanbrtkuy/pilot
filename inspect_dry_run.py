"""Dry-run inspector: only emits tensor metadata + numeric statistics.

Never prints raw prompt / CoT text to stdout. If you need to inspect raw
strings, write them to a local file and open it outside the agent context.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


DEFAULT_GROUP_DIR = Path("/workspace/pilot_code/data/generated/T1_harmful")


def inspect_cot_texts(group_dir: Path) -> None:
    path = group_dir / "cot_texts.jsonl"
    if not path.exists():
        print(f"[cot_texts.jsonl] MISSING at {path}")
        return

    print(f"[cot_texts.jsonl] {path}")
    n_records = 0
    n_closed = 0
    cot_lens: list[int] = []
    prompt_lens: list[int] = []

    n_malformed = 0
    marker_hits = 0
    marker_token_pcts: list[float] = []
    gen_tokens_list: list[int] = []
    invariant_violations: list[tuple[int, int, int]] = []  # (id, n_hook_steps, n_gen_tokens)
    with path.open() as f:
        for lineno, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                n_malformed += 1
                print(f"  line {lineno}: malformed JSON (skipped)")
                continue
            if not isinstance(d, dict):
                n_malformed += 1
                print(f"  line {lineno}: not a JSON object (skipped)")
                continue

            n_records += 1
            # ⚠️ Safety invariant: NEVER read or print the raw `cot` or `prompt` strings.
            # Always prefer explicit metadata fields (cot_chars, prompt_len, prompt_sha8,
            # has_closed_think). Only fall back to len()/substring checks on raw values
            # if metadata is absent (legacy records) — and even then, never print content.
            cot_len = d.get("cot_chars")
            if cot_len is None:
                cot_raw = d.get("cot", "")
                cot_len = len(cot_raw) if isinstance(cot_raw, str) else 0
                has_think_fallback = isinstance(cot_raw, str) and "</think>" in cot_raw
            else:
                has_think_fallback = None

            prompt_len = d.get("prompt_len")
            prompt_sha8 = d.get("prompt_sha8")
            if prompt_len is None:
                prompt_raw = d.get("prompt", "")
                prompt_len = len(prompt_raw) if isinstance(prompt_raw, str) else 0

            cot_lens.append(cot_len)
            prompt_lens.append(prompt_len)

            has_think = d.get("has_closed_think")
            if has_think is None:
                has_think = has_think_fallback
            n_closed += int(bool(has_think))
            sample_id = d.get("id", "?")
            n_gen_tokens = d.get("n_gen_tokens")
            n_hook_steps = d.get("n_hook_steps")
            n_ckpts_saved = d.get("n_checkpoints_saved")

            print(f"  id={sample_id}")
            if prompt_sha8:
                print(f"    prompt chars : {prompt_len} (sha8={prompt_sha8})")
            else:
                print(f"    prompt chars : {prompt_len}")
            print(f"    cot chars    : {cot_len}")
            if n_gen_tokens is not None:
                print(f"    gen tokens   : {n_gen_tokens}")
            if n_hook_steps is not None:
                # HF generate fires N forwards for N new tokens (prefill produces first token).
                # Invariant: n_hook_steps == gen_len for batch=1; n_hook_steps >= gen_len for batch>1.
                floor = n_gen_tokens if isinstance(n_gen_tokens, int) else None
                if floor is None:
                    tag = "?"
                elif n_hook_steps < floor:
                    tag = "VIOLATION"
                elif n_hook_steps == floor:
                    tag = "OK (batch=1 or longest)"
                else:
                    tag = f"OK (batch>1, slack={n_hook_steps - floor})"
                print(f"    hook steps   : {n_hook_steps} (floor gen_len={floor}) [{tag}]")
                if floor is not None and n_hook_steps < floor:
                    invariant_violations.append((int(sample_id) if isinstance(sample_id, int) else -1,
                                                 int(n_hook_steps), int(n_gen_tokens)))
            if n_ckpts_saved is not None:
                print(f"    ckpts saved  : {n_ckpts_saved}")
            print(f"    has </think> : {has_think}")

            # Decode marker — 只用显式 metadata 字段，永不回读 cot 文本
            marker_found = d.get("decode_marker_found")
            if marker_found is True:
                marker_hits += 1
                mname = d.get("decode_marker_name")
                mtok = d.get("decode_marker_token_pos")
                mpct = d.get("decode_marker_token_pct")
                mckpt = d.get("decode_marker_checkpoint_idx")
                if mpct is not None:
                    marker_token_pcts.append(mpct)
                mpct_str = f"{100*mpct:.1f}%" if mpct is not None else "?"
                print(f"    decode marker: '{mname}' @ tok {mtok} ({mpct_str}), ckpt #{mckpt}")
            elif marker_found is False:
                print(f"    decode marker: NOT FOUND")

            if n_gen_tokens is not None:
                gen_tokens_list.append(n_gen_tokens)

    if n_records:
        print(
            f"  summary: n={n_records}, malformed={n_malformed}, "
            f"closed={n_closed}/{n_records}, "
            f"cot_chars[min/median/max]="
            f"{min(cot_lens)}/{int(np.median(cot_lens))}/{max(cot_lens)}, "
            f"prompt_chars[min/median/max]="
            f"{min(prompt_lens)}/{int(np.median(prompt_lens))}/{max(prompt_lens)}"
        )
        if gen_tokens_list:
            print(
                f"  gen_tokens: min={min(gen_tokens_list)}, "
                f"median={int(np.median(gen_tokens_list))}, "
                f"max={max(gen_tokens_list)}"
            )
        print(f"  decode markers: {marker_hits}/{n_records} hit")
        if marker_token_pcts:
            pcts = np.array(marker_token_pcts) * 100
            print(
                f"    marker token%: min={pcts.min():.1f}%, "
                f"median={np.median(pcts):.1f}%, max={pcts.max():.1f}%"
            )
        if invariant_violations:
            print(f"  ⚠️ prefill invariant violations: {len(invariant_violations)}/{n_records} "
                  f"(first few: {invariant_violations[:5]})")
        else:
            print(f"  prefill invariant: OK (n_hook_steps >= n_gen_tokens for all records)")
    else:
        print(f"  summary: no records (malformed={n_malformed})")


KEY_PATTERN = re.compile(r"^layer_(\d+)$")


def _parse_key(k: str) -> int | None:
    m = KEY_PATTERN.match(k)
    return int(m.group(1)) if m else None


def inspect_hs_file(path: Path) -> None:
    print(f"[{path.name}] {path}")
    hs = np.load(path)
    raw_keys = list(hs.keys())
    indexed = [(_parse_key(k), k) for k in raw_keys]
    malformed_keys = [k for idx, k in indexed if idx is None]
    valid = sorted([(idx, k) for idx, k in indexed if idx is not None])
    keys = [k for _, k in valid]
    indices = [idx for idx, _ in valid]

    if malformed_keys:
        print(f"  WARNING: {len(malformed_keys)} key(s) did not match layer_<int> pattern (names suppressed)")

    if not keys:
        print("  empty archive (no valid layer_<int> keys)")
        return

    first = hs[keys[0]]
    print(f"  layers       : {len(keys)} (layer {indices[0]} .. layer {indices[-1]})")
    print(f"  shape        : {first.shape}")
    print(f"  dtype        : {first.dtype}")
    print(f"  tokens (dim0): {first.shape[0] if first.ndim >= 1 else 'n/a'}")

    mismatched: list[int] = []
    nonfinite_layers: list[tuple[int, int]] = []
    layer0_stats: tuple[float, float, float, float] | None = None

    for idx, k in valid:
        arr = hs[k]
        if arr.shape != first.shape:
            mismatched.append(idx)
        finite = np.isfinite(arr)
        n_nonfinite = int(finite.size - finite.sum())
        if n_nonfinite:
            nonfinite_layers.append((idx, n_nonfinite))
        if idx == indices[0] and finite.all():
            layer0_stats = (
                float(arr.min()), float(arr.max()),
                float(arr.mean()), float(arr.std()),
            )

    if mismatched:
        print(f"  WARNING: {len(mismatched)} layer(s) shape-mismatched (indices: {mismatched[:10]}{'...' if len(mismatched) > 10 else ''})")
    else:
        print("  shape consistency: OK (all layers match)")

    total_nonfinite = sum(n for _, n in nonfinite_layers)
    if nonfinite_layers:
        print(f"  non-finite   : {total_nonfinite} across {len(nonfinite_layers)} layer(s) (first few: {nonfinite_layers[:5]})")
    else:
        print(f"  non-finite   : 0 across all {len(keys)} layer(s)")

    if layer0_stats is not None:
        mn, mx, mean, std = layer0_stats
        print(f"  stats(layer{indices[0]}): min={mn:.4f} max={mx:.4f} mean={mean:.4f} std={std:.4f}")


def inspect_hs_dir(group_dir: Path) -> None:
    files = sorted(group_dir.glob("hs_*.npz"))
    if not files:
        print(f"[hs_*.npz] no files found in {group_dir}")
        return
    for f in files:
        inspect_hs_file(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--group-dir",
        type=Path,
        default=DEFAULT_GROUP_DIR,
        help="Group directory containing cot_texts.jsonl and hs_*.npz",
    )
    args = ap.parse_args()

    group_dir: Path = args.group_dir
    if not group_dir.exists():
        raise SystemExit(f"group dir not found: {group_dir}")

    print(f"=== inspect_dry_run :: {group_dir} ===")
    inspect_cot_texts(group_dir)
    inspect_hs_dir(group_dir)


if __name__ == "__main__":
    main()
