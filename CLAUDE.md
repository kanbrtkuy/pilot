# Project: Pilot Experiment — Online CoT Safety Monitor

## What This Project Does

This is a **pilot experiment** testing whether **reasoning drift** is detectable in LLM hidden states during Chain-of-Thought (CoT) generation.

**Reasoning drift** = when a disguised input (cipher-encoded or jailbreak-wrapped) looks benign at the surface, but the model's internal hidden state gradually drifts toward harmful directions as it reasons through the CoT.

**The pilot answers ONE question**: Does this drift signal exist? Yes → invest 6-8 weeks in full experiments. No → kill the project.

## Model

**DeepSeek-R1-Distill-Qwen-7B**
- Architecture: Qwen2ForCausalLM
- 28 layers (0-27), hidden_dim = 3584
- All 28 layers saved as float16
- No system prompt (official recommendation)

## Data Design (Two-Tier)

**Tier 1 (pure drift test)**: HarmBench prompts encoded with Base64+ROT13. Input is opaque ciphertext with zero harmful semantics. Harmful intent only emerges when model decodes during CoT.
- G1a: 50 encoded harmful prompts
- G3a: 50 encoded benign prompts (identical format, only decoded content differs)

**Tier 2 (ecological validity)**: WildJailbreak adversarial prompts using real-world jailbreak tactics.
- G1b: 50 adversarial_harmful
- G3b: 50 adversarial_benign (matched by tactics count and length)

**No Group 2** — no safety system prompt control (model officially recommends not using system prompt).

Target: 50 **valid** samples per group (not 50 inputs). Gemini API scores validity, refill from pool if needed.

## Decision Rules

| Hypothesis | Pass Condition | Fail Action |
|---|---|---|
| H1: σ(t) trajectory divergence | p < 0.01, Cohen's d > 0.5 | **Kill** |
| H2: AUROC > 0.80 at 40% CoT | Best of 8 methods > 0.80 | **Pivot** or Kill |
| H3: AUROC_mid > AUROC_input | Tier 1 mid-CoT beats input-level | **Pivot** (narrow claim) |

Go/No-Go based on M1 (simplest), M5 (Chan upper bound), M7 (transfer diagnostic) only.

## Scripts (Execution Order on RunPod)

```
Phase 1: GPU inference
  00_prepare_data.py      → Download WildJailbreak + construct Tier 1 cipher data
  01_generate_cot_save_hs.py → Generate CoT + save all 28 layers (float16)

Phase 2: CPU analysis (can run locally after downloading results)
  02_extract_v_harm.py    → Extract harmfulness direction (multi-layer sweep)
  03_h1_sigma_curves.py   → H1: trajectory divergence test
  04_h2_auroc.py          → H2: AUROC vs CoT position
  04b_eight_methods.py    → 8 detection methods comparison
  05_h3_comparison.py     → H3: mid-CoT vs input-level

Supporting:
  config.py               → Shared configuration (paths, model params, helpers)
  gemini_scorer.py        → Gemini API validity scoring
  generate_mock_data.py   → Mock data for local testing
```

## IMPORTANT: Known Issues to Fix Before Running

See `KNOWN_ISSUES.md` for 3 bugs identified by code review that MUST be fixed during dry run.

## Key References

| Paper | Level | How Used |
|---|:--:|---|
| Zhao et al. "Latent Guard" (NeurIPS 2025) | A | v_harm extraction position (t_inst) |
| "Safety Layers" (ICLR 2025) | A | Layer selection (middle layers strongest) |
| Chan et al. (FoRLM @ NeurIPS 2025) | B | M5 full probe reference |
| Arditi et al. "Refusal Direction" (NeurIPS 2024) | A | v_refusal extraction, mean-diff method |
| SEAL "Three Minds" (arXiv 2505.16241) | C | Tier 1 cipher encoding method, R1 100% ASR |
| andyrdt/refusal_direction (GitHub) | — | Hook implementation reference |

## Documentation

- `docs/pilot_experiment.md` — Experiment design (decision rules, data, analysis)
- `docs/data_generation_pipeline.md` — Data construction details
- `docs/runpod_setup.md` — RunPod environment setup guide
