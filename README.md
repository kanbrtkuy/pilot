# Pilot Experiment: Online CoT Safety Monitor

Testing whether reasoning drift signal exists in hidden states during CoT generation.

## Model
DeepSeek-R1-Distill-Qwen-7B (28 layers, hidden_dim=3584)

## Scripts (execution order)

| Script | Where | Purpose |
|--------|-------|---------|
| `config.py` | — | Shared config (paths, model params, helpers) |
| `00_prepare_data.py` | RunPod | Download WildJailbreak + construct Tier 1 cipher data |
| `01_generate_cot_save_hs.py` | RunPod (GPU) | Generate CoT + save all 28 layers hidden states |
| `gemini_scorer.py` | RunPod/Local | Judge CoT validity via Gemini API |
| `generate_mock_data.py` | Local | Create mock data for testing analysis scripts |
| `02_extract_v_harm.py` | Local (CPU) | Extract v_harm direction (multi-layer sweep) |
| `03_h1_sigma_curves.py` | Local (CPU) | H1: trajectory divergence test |
| `04_h2_auroc.py` | Local (CPU) | H2: AUROC vs CoT position |
| `04b_eight_methods.py` | Local (CPU) | 8 detection methods comparison |
| `05_h3_comparison.py` | Local (CPU) | H3: mid-CoT vs input-level (hard gate) |

## Data Design
- **Tier 1**: Base64+ROT13 cipher encoded (pure drift test)
- **Tier 2**: WildJailbreak adversarial (ecological validity)
- **Groups**: G1a (T1 harmful), G3a (T1 benign), G1b (T2 harmful), G3b (T2 benign)
