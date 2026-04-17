# RunPod A100 环境配置指南

**目标**：在 RunPod A100 80GB 上完整运行 Pilot 实验
**预计配置时间**：30-40 分钟
**预计实验时间**：2-3 小时（Phase 1 GPU 推理）+ 1 小时（Phase 2 CPU 分析）
**模型**：DeepSeek-R1-Distill-Qwen-7B（~15GB 显存）
**数据**：Tier 1 SEAL 风格编码（HarmBench → Base64+ROT13）+ Tier 2 WildJailbreak adversarial（HuggingFace）

---

## 总体工作流

```
【本地 Mac，提前做，免费】
──────────────────────────────────────────────
Step A：写所有脚本代码
Step B：用 mock 数据测试 Phase 2 分析脚本跑通
Step C：推到 GitHub
        ↓

【RunPod，配置阶段，约 1 小时，$1.5】
──────────────────────────────────────────────
Step 1-8：配置环境 + 下载模型 + 同步代码
        ↓

【RunPod，GPU 生成阶段，约 2-3 小时，$3-4】
──────────────────────────────────────────────
Step 9：dry run 5 条样本验证代码跑通
        ↓
Step 10a：00_prepare_data.py
          （构造 Tier 1 编码数据 50+50 + 下载 Tier 2 WildJailbreak 50+50）
        ↓
Step 10b：01_generate_cot_save_hs.py
          （一轮推理：全部 200 条 prompt → 不加 system prompt）
          （G1a 编码有害 + G3a 编码无害 + G1b 越狱有害 + G3b 越狱无害）
          保存所有 hidden states 到 /workspace
        ↓

【下载到本地 Mac】
──────────────────────────────────────────────
Step 11：下载 hidden states（约 5-8GB，200 条 × 7 层）+ CoT 文本
         → 停掉 RunPod Pod，不再产生费用
        ↓

【本地 Mac，分析阶段，免费】
──────────────────────────────────────────────
02_extract_v_harm.py    → 提取 v_harm（多层 × 多位置策略）
03_h1_sigma_curves.py   → H1：σ(t) 轨迹分叉检验
04_h2_auroc.py          → H2：AUROC vs CoT position
04b_eight_methods.py    → 8 方案对比（M1-M7b）
05_h3_comparison.py     → H3：CoT 中途 vs 输入层
        ↓

【决策】
──────────────────────────────────────────────
H1 + H2 + H3 结果 → Go / Pivot / Kill
```

---

## 【本地 Mac】Step A：写代码并用 mock 数据测试

**原则**：把所有能在本地验证的逻辑先在本地跑通，再上 RunPod，节省 GPU 费用。

### 代码分工

| 文件 | 能否本地测试 | 说明 |
|------|-----------|------|
| `00_prepare_data.py` | ✅ 完全可以 | 下载 WildJailbreak + 采样 |
| `01_generate_cot_save_hs.py` | ⚠️ 只能 mock | 需要 GPU 加载模型 |
| `02_extract_v_harm.py` | ✅ 完全可以 | 纯 numpy 计算 |
| `03_h1_sigma_curves.py` | ✅ 完全可以 | 纯分析 + matplotlib |
| `04_h2_auroc.py` | ✅ 完全可以 | 纯 sklearn |
| `04b_eight_methods.py` | ✅ 完全可以 | 纯 sklearn（GroupKFold） |
| `05_h3_comparison.py` | ✅ 完全可以 | 纯分析 |

### 本地测试方法

```bash
# 在本地 Mac 上，先安装分析依赖（不需要 torch）
pip install scikit-learn matplotlib numpy pandas datasets

# 生成 mock hidden states（模拟 7B 模型，28 层，hidden_dim=3584）
python pilot/generate_mock_data.py

# 测试所有分析脚本是否跑通
python pilot/02_extract_v_harm.py
python pilot/03_h1_sigma_curves.py
python pilot/04_h2_auroc.py
python pilot/04b_eight_methods.py
python pilot/05_h3_comparison.py
# 检查 pilot/results/ 下是否生成了图片
```

### 推到 GitHub

```bash
git add 0_pilot/
git commit -m "feat: add pilot experiment scripts"
git push origin main
```

---

## 【RunPod】Step 1：创建 Pod

### 配置参数

| 参数 | 选择 |
|------|------|
| GPU | A100 80GB PCIe（$1.39/hr）或 A100 SXM（$1.49/hr）|
| 基础镜像 | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| Container Disk | 20 GB |
| Volume（持久化）| 40 GB（模型 14GB + hidden states 8GB + conda 环境 10GB + 余量）|

---

## 【RunPod】Step 2-4：SSH + Node.js + Claude Code

```bash
# SSH 连接
ssh root@<POD_IP> -p <PORT>

# 安装 Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

# 安装 Claude Code
npm install -g @anthropic-ai/claude-code
claude  # 登录
```

---

## 【RunPod】Step 5：安装 ARIS

```bash
cd /workspace
git clone https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git aris
mkdir -p ~/.claude/skills
cp -r /workspace/aris/skills/* ~/.claude/skills/
```

---

## 【RunPod】Step 6：Conda 环境

```bash
# 安装 Miniforge 到 Volume（持久化）
cd /tmp
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3
/workspace/miniforge3/bin/conda init bash
source ~/.bashrc

# 创建环境
conda create -n pilot python=3.11 -y
conda activate pilot

# 安装依赖
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install transformers>=4.51.0 accelerate>=1.1.0
pip install flash-attn>=2.6.0 --no-build-isolation
pip install scikit-learn matplotlib numpy pandas datasets

# 验证
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 【RunPod】Step 7：下载模型

```bash
pip install huggingface_hub
huggingface-cli login  # 输入 hf_ token

huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir /workspace/models/DeepSeek-R1-Distill-Qwen-7B

ls /workspace/models/DeepSeek-R1-Distill-Qwen-7B/
```

**模型选择说明**：使用 R1-Distill-Qwen-7B 标准对齐版本。
- Chan et al. [B FoRLM'25] 在此模型上做了最接近的 CoT activation 安全预测实验
- 同模型 = 最少变量，pilot 结果可与 Chan 直接对比
- 7B 比 14B 更快（生成速度约 2 倍），显存更省（15GB vs 30GB）

---

## 【RunPod】Step 8：同步代码

```bash
cd /workspace
git clone https://github.com/<你的账号>/<repo>.git pilot_code

# 或者 scp：
# scp -r -P <PORT> ./0_pilot root@<POD_IP>:/workspace/pilot_code
```

---

## 【RunPod】Step 9：dry run

```bash
cd /workspace/pilot_code
conda activate pilot

# 只跑 3 条样本，验证全流程
python 01_generate_cot_save_hs.py --dry-run --n 3

# 人工检查
python -c "
import numpy as np
hs = np.load('data/generated/group1_drift/hs_0000.npz')
print('layers:', list(hs.keys()))
for k in hs.keys():
    print(f'  {k}: shape={hs[k].shape}')
# 预期：layer_X: (n_tokens, 3584)
"
```

检查点：
- `</think>` 解析是否正确
- hidden state shape 是否符合预期
- CoT 文本是否正常生成

---

## 【RunPod】Step 10：正式运行

### 睡觉前检查清单

```
□ dry run 3 条样本跑通（Step 9 完成）
□ 模型已下载到 /workspace/models/
□ WildJailbreak 数据可以正常下载
□ df -h 确认 /workspace 有足够空间（至少 20GB 剩余）
□ 设置 RunPod 最大运行时长（建议 6 小时，防止睡过头计费）
```

### 交给 Claude Code 自动执行

```bash
cd /workspace/pilot_code
claude
```

> "请按顺序运行 pilot 实验：
> 1. python 00_prepare_data.py（构造 Tier 1 编码数据 + 下载 Tier 2 WildJailbreak，约 10 分钟）
> 2. python 01_generate_cot_save_hs.py（一轮推理生成 200 条 CoT + 保存 hidden states，约 2-3 小时）
>
> 每步完成后检查输出文件是否正常。全部完成后记录结果到 results/run_log.txt。"

---

## 【本地 Mac】Step 11：下载结果 + 分析

```bash
# 下载 hidden states 和 CoT 文本
scp -r -P <PORT> \
  root@<POD_IP>:/workspace/pilot_code/data/generated \
  ./data/generated

# 停掉 RunPod Pod
# → 从此不再产生 GPU 费用

# 本地跑分析（纯 CPU）
python 02_extract_v_harm.py
python 03_h1_sigma_curves.py
python 04_h2_auroc.py
python 04b_eight_methods.py
python 05_h3_comparison.py

# 查看结果
open results/sigma_curves.png
open results/auroc_curve.png
open results/h3_comparison.png
```

---

## 省钱总结

| 阶段 | 在哪跑 | GPU 费用 |
|------|--------|:--:|
| 写代码 + mock 测试 | 本地 Mac | $0 |
| 环境配置（Step 1-8）| RunPod | ~$1 |
| Phase 1 生成 CoT（Step 10）| RunPod | ~$3-4 |
| Phase 2 分析（Step 11）| 本地 Mac | $0 |
| **总计** | | **~$4-5** |

---

## 常见问题

**Q：Pod 重启后环境还在吗？**
A：`/workspace` Volume 里的数据都在（模型、代码、hidden states、conda）。需要重跑 `conda init bash && source ~/.bashrc`。

**Q：WildJailbreak 下载慢怎么办？**
A：数据集约 200MB，RunPod 北美节点通常很快。如果慢，可以在本地下载后 scp 传到 RunPod。

**Q：组 1 有效样本太少（模型对伪装 prompt 也一直拒绝）怎么办？**
A：WildJailbreak 有 82K 条 adversarial_harmful，再采样 200 条重跑。如果仍然太少，用 JailbreakBench 的 PAIR 攻击变体。最后手段：用直接有害 prompt 做 fallback pilot。

**Q：hidden state 文件太大下载不了？**
A：7B 模型保存 7 层 × 300 条 × ~500 tokens × 3584 dim × float16 ≈ 7-8GB。如果网速慢，先在 RunPod 上跑完 Phase 2 分析，只下载结果图片（几 MB）。

---

## 最终目录结构

```
/workspace/
├── miniforge3/                          ← conda（Volume 持久化）
├── models/
│   └── DeepSeek-R1-Distill-Qwen-7B/    ← 模型（~14GB）
├── aris/                                ← ARIS skills
└── pilot_code/                          ← 实验代码
    ├── 00_prepare_data.py               ← 构造 Tier 1 编码 + 下载 Tier 2 WildJailbreak
    ├── 01_generate_cot_save_hs.py       ← GPU 生成 CoT + 保存 hidden states（一轮，200 条）
    ├── 02_extract_v_harm.py             ← 提取 v_harm（多层 × 多位置 × contrastive）
    ├── 03_h1_sigma_curves.py            ← H1 验证（Tier 1 + Tier 2 分别画）
    ├── 04_h2_auroc.py                   ← H2 验证
    ├── 04b_eight_methods.py             ← 8 方案对比（GroupKFold）
    ├── 05_h3_comparison.py              ← H3 验证（AUROC_mid vs AUROC_input 硬判据）
    ├── data/
    │   ├── all_prompts.csv              ← 全部 200 条 prompt（含 tier/label 标记）
    │   └── generated/                   ← hidden states（~5-8GB）
    │       ├── T1_harmful/              ← Tier 1 编码有害
    │       ├── T1_benign/               ← Tier 1 编码无害
    │       ├── T2_harmful/              ← Tier 2 越狱有害
    │       └── T2_benign/               ← Tier 2 越狱无害
    └── results/                         ← 分析结果（图片 + CSV）
```
