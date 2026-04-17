# Known Issues — Must Fix Before Running

These 3 bugs were identified by Codex (GPT-5.4) code review. Fix them during the dry run phase on RunPod.

---

## Issue 1 (Critical): Hook output format check

**File**: `01_generate_cot_save_hs.py`, function `make_capture_hook`

**Problem**: The hook assumes `output[0]` is valid, but Qwen2DecoderLayer may return a Tensor directly (not a tuple). If output is a plain Tensor, `output[0]` takes the first token instead of the full sequence.

**Fix**: Add isinstance check, following andyrdt/refusal_direction's pattern:

```python
def make_capture_hook(cache, layer_idx):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output
        # activation shape: (batch=1, seq_len, hidden_dim)
        h = activation[0, -1, :].detach().cpu().to(torch.float32).numpy()
        cache[layer_idx].append(h)
    return hook_fn
```

**Verification**: During dry run, print `type(output)` and `output.shape` (or `output[0].shape`) for the first sample to confirm.

---

## Issue 2 (Medium): Prefill phase hidden state

**File**: `01_generate_cot_save_hs.py`

**Problem**: During `model.generate()`, the first forward pass is the prefill phase — it processes the entire prompt at once. The hook fires and captures a hidden state with `seq_len = prompt_length`, not `seq_len = 1`. The `[-1]` indexing still gives the correct position (last token of prompt), but this first entry in the cache is the prompt's last token state, not a generated token's state.

**Fix options**:
- Option A: Skip the first hook call (set a flag, ignore first capture)
- Option B: Keep it but mark it as "prompt state" — this is actually useful for H3 (input-level AUROC)

**Recommended**: Option B — keep the first entry but be aware it represents the prompt, not generated content. In analysis scripts, `sigma[0]` corresponds to the input-level signal (useful for H3 comparison).

**Verification**: During dry run, check `len(cache[0])` after generation. It should be `n_generated_tokens + 1` (the +1 is the prefill). Check `cache[0][0].shape` to confirm it matches expected hidden_dim.

---

## Issue 3 (Medium): Gemini response parsing fragility

**File**: `gemini_scorer.py`, function `score_one`

**Problem**: Parsing relies on splitting by "Q1"/"Q2" string markers. If Gemini doesn't follow the exact format (e.g., uses "Question 1:" or a list format), parsing fails silently.

**Fix**: Add regex fallback and parse-failure logging:

```python
import re

def parse_yes_no(text, marker):
    """Robust YES/NO extraction with fallback"""
    # Try exact marker split first
    if marker in text:
        segment = text.split(marker)[1].split('\n')[0]
        if 'YES' in segment.upper():
            return True
        if 'NO' in segment.upper():
            return False
    
    # Fallback: regex for "Q1: YES" or "1. YES" or "1) YES"
    pattern = rf'{marker}[:\s)]+\s*(YES|NO)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper() == 'YES'
    
    # Last resort: if only one YES/NO in the text
    yes_count = text.upper().count('YES')
    no_count = text.upper().count('NO')
    if yes_count == 1 and no_count == 0:
        return True
    if no_count == 1 and yes_count == 0:
        return False
    
    return None  # Parse failure — log this

# In score_one, handle parse failures:
result = parse_yes_no(text, 'Q1')
if result is None:
    return {'valid': False, 'parse_error': True, 'raw_response': text}
```

**Verification**: During Gemini scoring, count and log parse failures. If > 10%, the prompt needs redesign.

---

## Quick Verification Checklist (Dry Run)

```bash
# Run 3 samples to verify all fixes
python 01_generate_cot_save_hs.py --dry-run --n 3

# Check 1: Hook output type
# → Should print "output type: <class 'tuple'>" or "<class 'torch.Tensor'>"

# Check 2: Cache length
# → Should be n_generated_tokens + 1 (prefill + generation steps)

# Check 3: Hidden state shape
# → Should be (3584,) for each entry

# Check 4: </think> presence
# → CoT should contain </think> marker
```
