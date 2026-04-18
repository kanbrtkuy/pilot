"""
config.py — 共享配置：路径、模型参数、常量

所有 pilot 脚本从这里导入，避免重复定义和不一致。

模型：DeepSeek-R1-Distill-Qwen-14B  (pilot pivot 2026-04-18)
  - num_hidden_layers = 48 (layers 0-47)
  - hidden_size = 5120
  - num_key_value_heads = 8  (GQA)
  - 来源：https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

Pivot 依据（见 docs/ 中的 pilot pivot 记录）：
  7B distill 在 n=20 base64-only T1_harmful 上 decode 失败率 = 100%
  （"decoded" 关键词命中 90%，但 claimed decode 与真实 plaintext
   lexical overlap 均值仅 7-8%，0/20 达到 ≥50% 阈值）。
  H1 假设"R1 会在 CoT 解密后产生 drift"在 7B 上前提不成立。
  升级到 14B 以验证 capability 阈值是否在这个档位达标。

7B legacy 值（pivot 前）：N_LAYERS=28, HIDDEN_DIM=3584, kv_heads=4
"""

from pathlib import Path
import numpy as np

# ══════════════════════════════════════════════════
# 路径
# ══════════════════════════════════════════════════

# 自动检测项目根目录
# Mac 本地: scripts in code/pilot/ → parent.parent = code → .parent.name == "code"
# RunPod:   scripts at repo root  → parent = pilot_code/
_script_dir = Path(__file__).resolve().parent
if _script_dir.parent.name == "code":
    PROJECT_ROOT = _script_dir.parent.parent
else:
    PROJECT_ROOT = _script_dir
DATA_DIR = PROJECT_ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
RESULTS_DIR = PROJECT_ROOT / "results"

# 确保目录存在
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════
# 模型参数（R1-Distill-Qwen-14B）
# ══════════════════════════════════════════════════

N_LAYERS = 48          # 14B 模型总层数（0-47）
HIDDEN_DIM = 5120      # hidden_size
NUM_KV_HEADS = 8       # GQA
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

# 保存全部 48 层的 hidden states
# 存储估算（下采样到 100 checkpoint 后）：
#   200 条 × 48 层 × 100 checkpoint × 5120 dim × float16 ≈ 10 MB/样本 × 200 = 2 GB
LAYERS_TO_SAVE = list(range(N_LAYERS))  # [0, 1, 2, ..., 47]

# 中间层子集（harmfulness 信号预期最强的区间，占总层数 40%-60%）
# 48 层的 40%-60% = [19, 29]，取中心窗口
MID_LAYERS = [22, 23, 24, 25, 26]

# 多层 MLP 用的三层（浅/中/深）—— 按 7B 的相对位置 (6/28, 13/28, 27/28) 等比映射
MULTI_LAYERS = [10, 22, 47]

# hidden state 保存精度
SAVE_DTYPE = "float16"  # 节省一半存储，精度损失可忽略

# ══════════════════════════════════════════════════
# 分析参数
# ══════════════════════════════════════════════════

N_CHECKPOINTS = 20     # CoT 归一化为 20 个时间步（分析阶段子采样用）
MIN_TOKENS = 20        # 少于此数的 CoT 跳过（避免退化采样）

# 生成阶段的 HS 下采样（对齐工业部署 budget）
TARGET_MAX_TOKENS = 2048    # 工业实际 cap ~2-3K；pilot 用 2K 做 ecological-validity 主测
N_SAVE_CHECKPOINTS = 100    # 每样本保存 100 个 checkpoint（1% 相对精度，磁盘 ~20 MB/样本）
DECODE_MARKERS = [
    "decoded",
    "result is",
    "the instruction is",
    "解码结果",
    "解码后",
    "decoded message",
    "decoded text",
]

# ══════════════════════════════════════════════════
# 数据分组
# ══════════════════════════════════════════════════

GROUPS = ['T1_harmful', 'T1_benign', 'T2_harmful', 'T2_benign']

# ══════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════

def load_direction(name="v_harm.pt"):
    """加载方向向量为 numpy array，带 weights_only=True 避免 FutureWarning"""
    import torch
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在。请先运行 02_extract_v_harm.py")
    return torch.load(path, weights_only=True).numpy()


def detect_layers(group_dir=None):
    """从 .npz 文件自动检测可用层号，返回排序的整数列表"""
    if group_dir is None:
        group_dir = GENERATED_DIR / "T1_harmful"
    sample_files = sorted(group_dir.glob("hs_*.npz"))
    if not sample_files:
        raise FileNotFoundError(f"在 {group_dir} 中没有找到 hs_*.npz 文件")
    keys = np.load(sample_files[0]).keys()
    return sorted([int(k.replace("layer_", "")) for k in keys])


def load_hidden_states(group_dir, layer_idx, mode='trajectory', position='mid', n_checkpoints=N_CHECKPOINTS):
    """
    统一的 hidden state 加载函数。

    mode='trajectory': 返回 (n_samples, n_checkpoints, hidden_dim) — 用于轨迹分析
    mode='single':     返回 (n_samples, hidden_dim) — 用于 v_harm 提取

    position 参数（mode='single' 时使用）:
      'first'    — 第一个 token
      'mid'      — CoT 中间 token
      'last'     — 最后一个 token
      'inst_end' — 近似 t_inst（前 10% 的最后一个 token）
    """
    results = []
    for f in sorted(Path(group_dir).glob("hs_*.npz")):
        data = np.load(f)
        key = f"layer_{layer_idx}"
        if key not in data:
            continue
        hs = data[key]  # (n_tokens, hidden_dim)
        n = len(hs)
        if n < MIN_TOKENS:
            continue

        if mode == 'trajectory':
            indices = np.linspace(0, n - 1, n_checkpoints).astype(int)
            results.append(hs[indices])
        elif mode == 'single':
            if position == 'first':
                results.append(hs[0])
            elif position == 'mid':
                results.append(hs[n // 2])
            elif position == 'last':
                results.append(hs[-1])
            elif position == 'inst_end':
                inst_end = max(1, int(n * 0.1))
                results.append(hs[inst_end - 1])
            else:
                results.append(hs[n // 2])

    if not results:
        return None
    return np.array(results)


def load_sigma_trajectories(group_dir, v, layer_idx, n_checkpoints=N_CHECKPOINTS):
    """加载 σ(t) = h(t) @ v 的轨迹，归一化到 n_checkpoints 个时间步"""
    hs = load_hidden_states(group_dir, layer_idx, mode='trajectory', n_checkpoints=n_checkpoints)
    if hs is None:
        return np.zeros((0, n_checkpoints))
    # hs shape: (n_samples, n_checkpoints, hidden_dim)
    # v shape: (hidden_dim,)
    return hs @ v  # (n_samples, n_checkpoints)


def count_samples(group_dir):
    """计算一个目录下 hs_*.npz 文件的数量"""
    return len(list(Path(group_dir).glob("hs_*.npz")))
