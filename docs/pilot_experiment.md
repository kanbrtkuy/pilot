# Pilot 实验设计

**目的**：验证 reasoning drift 信号是否存在——伪装输入下，CoT 中途的 hidden state 是否出现可检测的 harmfulness 漂移
**模型**：DeepSeek-R1-Distill-Qwen-7B
**目标**：每组 50 条**有效**样本（不是 50 条输入）
**预计时间**：单卡 4-5 小时 / 四卡 2-3 小时
**关联文件**：`data_generation_pipeline.md`（数据构造细节）、`runpod_setup.md`（环境配置）

---

## 1. 决策规则（先看这里）

| 假设 | 判断标准 | 失败处理 |
|------|---------|---------|
| **H1** | Tier 1 中 G1a 和 G3a 的 σ(t) 轨迹在 CoT 中段可见分叉，p < 0.01，Cohen's d > 0.5 | **Kill** |
| **H2** | 8 方案中最佳方案在 CoT 40% 位置 AUROC > 0.80 | **Pivot**：若 M5 也 <0.70 → Kill；否则升级 probe |
| **H3** | Tier 1 中 AUROC_mid > AUROC_input（硬判据） | **Pivot**：信号存在但不优于输入层 → 收窄 claim |

### Kill 条件

以下**任一条件**触发即 Kill 或 Pivot：
- 所有 8 方案最佳 AUROC < 0.70
- M5（full linear probe）也 < 0.70 → 信号不存在
- Tier 1 的 AUROC_input > 0.70 → 编码没挡住输入层检测，不是 drift

### Go/No-Go 决策

```
H1 通过 + H2 通过 + H3 通过 → Go（投入 6-8 周跑 full plan）
H1 通过 + H2 通过 + H3 失败 → Pivot（收窄 claim）
M1 失败但 M5 成功           → Pivot（升级 probe，1D cosine 不够）
M5 也失败                    → Kill（换方向）
```

### Pre-registered Selection Rule

- Go/No-Go 只看 **M1（最简）、M5（上界）、M7（迁移诊断）** 三方案
- 其余 5 方案（M2/M3/M4/M6/M7b）作为参考，帮助理解信号特性
- 主指标：CoT 40% 位置的 AUROC
- 全部 8 方案结果必须完整报告

---

## 2. 数据设计

### 两层四组

| 组 | 层 | 输入 | 格式 | 来源 | 目标有效数 |
|:--:|:--:|------|------|------|:--:|
| **G1a** | Tier 1 | 编码的有害请求 | "Decode Base64+ROT13: [密文]" | HarmBench → 编码 | 50 |
| **G3a** | Tier 1 | 编码的无害请求 | "Decode Base64+ROT13: [密文]" | JailbreakBench benign → 编码 | 50 |
| **G1b** | Tier 2 | 伪装有害（真实越狱策略） | 各种越狱风格 | WildJailbreak adversarial_harmful | 50 |
| **G3b** | Tier 2 | 伪装无害（看起来像越狱） | 各种越狱风格 | WildJailbreak adversarial_benign | 50 |

### 对比逻辑

```
Tier 1（纯净 drift 证据）：
  G1a vs G3a → 格式 100% 相同，唯一区别 = 解码后语义
  预期：AUROC_input ≈ 0.5（密文无语义），AUROC_mid > 0.80（解码后 drift）

Tier 2（生态效度）：
  G1b vs G3b → 格式相似（都是越狱风格）
  预期：AUROC_mid > AUROC_input（CoT 中途检测有增量价值）
```

### 有效样本定义

| 组 | "有效" = | 由 Gemini 判断 |
|:--:|------|:--:|
| G1a | 模型成功解码 + 产出有害 CoT | Q1: 解码成功？ Q2: 有害内容？ |
| G3a | 模型成功解码 + 正常回答 | Q1: 解码成功？ |
| G1b | 模型产出有害 CoT | Q3: 有害内容？ |
| G3b | 模型正常回答 | 不需要筛选 |

投入量根据成功率动态调整（"补到够为止"），详见 `data_generation_pipeline.md`。

---

## 3. 执行流程

```
Phase 0：本地 Mac（免费）
  写代码 + mock 测试 + 预注册规则 + 推 GitHub

Phase 1：RunPod GPU（约 $4-5）
  ① Dry run 5 条（确认 </think> 解析、hidden state shape）
  ② 生成 CoT + 保存 hidden states（一轮推理，无 system prompt）
  ③ Gemini 批量打分 + 不够就补采样
  → 每组达到 50 条有效后停止
  → 下载结果到本地，停掉 RunPod

Phase 2：本地 Mac CPU（免费，约 1 小时）
  ① 提取 v_harm
  ② H1 验证
  ③ H2 验证 + 8 方案对比
  ④ H3 验证

Phase 3：决策（30 分钟）
  Go / Pivot / Kill
```

---

## 4. v_harm 提取

### 提取位置

**首选 `t_inst`**（用户指令最后一个 token）——Zhao et al. [A NeurIPS'25] 发现 harmfulness 编码在此位置。

对比策略：`t_inst` / `t_post_inst` / `last-k pooling`，选 separability 最高的。

### 提取层

重点扫描中间层 {19, 23, 27}（"How Alignment and Jailbreak Work" EMNLP 2024 Findings [A] 发现中间层 harmfulness→emotion 关联最强），加浅/深两端 {11, 32, 39, 47} 做对照。

### 三种 v_harm 变体

```python
# 变体 1: input mean-diff（最简单）
v_harm = mean(G1a 的 t_inst hidden states) - mean(G3a 的 t_inst hidden states)

# 变体 2: CoT-trained（避免 input→CoT 迁移问题）
v_harm_cot = mean(G1a 的 CoT 中段 hidden states) - mean(G3a 的 CoT 中段 hidden states)

# 变体 3: contrastive（借鉴 R2D [A EMNLP'25]，per-prompt 配对消除域差）
# 注：Tier 1 的 G1a 和 G3a 不是同 prompt 配对，所以这个变体主要用于 v_harm_cot
```

---

## 5. H1 验证：σ(t) 轨迹分叉

**问题**：G1a 和 G3a 的 hidden state 轨迹是否在 CoT 中段出现可检测的分叉？

```python
# 对每条 CoT，在 N 个归一化时间步计算 σ(t) = cos(h(t), v_harm)
# 分别画 G1a（编码有害）和 G3a（编码无害）的均值 ± std 曲线
# 在 CoT 40%-60% 区间做 Welch t-test

# 判断标准：p < 0.01 且 Cohen's d > 0.5 → H1 通过
```

**Tier 1 和 Tier 2 分别画**——Tier 1 是主证据，Tier 2 是补充。

### 时序分析（Tier 1 专属）

区分解码阶段 vs 推理阶段：
- 在 CoT 文本中搜索解码结束标志词（"decoded", "result is", "解码结果"）
- 分界点之前 = 解码阶段，之后 = 推理阶段
- 分别计算两个阶段的 AUROC

---

## 6. H2 验证：AUROC vs CoT 位置

**问题**：在 CoT 进行到多少百分比时，检测能力就足够强？

```python
# 在每个归一化时间步 t，用 σ(t) 做二分类（G1a vs G3a），计算 AUROC
# 画 AUROC vs t 的曲线

# 判断标准：CoT 40% 位置 AUROC > 0.80 → H2 通过
```

---

## 7. 8 方案对比

在同一批 cached hidden states 上（一次推理，八种 CPU 后处理）：

| # | 方案 | 目的 |
|:--:|------|------|
| M1 | cos(h, v_harm) | 最简，Go/No-Go 决策用 |
| M2 | cos(h, v_harm) + cos(h, v_refusal) | 双方向 |
| M3 | MLP(h): d→32→1 | 非线性增益 |
| M4 | MLP([h_l23; h_l32; h_l47]): 3d→64→1 | 多层 + 非线性 |
| M5 | Full linear probe | Chan [B] 上界，Go/No-Go 决策用 |
| M6 | Multi-layer linear probe | 区分多层增益 vs 非线性增益 |
| M7 | cos(h, v_harm_cot) | 迁移诊断，Go/No-Go 决策用 |
| M7b | cos(h, v_harm_contrastive) | R2D [A] 风格 per-prompt 配对 |

**数据切分**：按 prompt_id 分组（GroupKFold），防止同一编码 prompt 的 hidden states 泄漏到不同 fold。

---

## 8. H3 验证：CoT 中途 vs 输入层（硬判据）

**问题**：CoT 中途检测是否优于输入层检测？

```python
# 对 Tier 1 数据（G1a vs G3a）：
#   AUROC_input = 用 t_inst 位置的 σ 做分类
#   AUROC_mid   = 用 CoT 40% 位置的 σ 做分类
# 
# 判断标准：AUROC_mid > AUROC_input → H3 通过
#
# 额外分析：strict drift subset
#   = 输入层 σ < threshold 但 CoT 中途 σ > threshold 的样本
#   如果 > 20% → drift 是一个独立的、在输入层不可见的现象
```

**Tier 1 的预期**：
- AUROC_input ≈ 0.5（密文无语义，输入层完全检测不到）
- AUROC_mid > 0.80（解码后 drift 信号暴露）
- 如果这个预期成立 → 最强 drift 证据

---

## 9. 预注册规则汇总

在跑实验**之前**必须固定的规则：

| 规则 | 内容 |
|------|------|
| **有效样本定义** | G1a: 解码成功 + 有害; G3a: 解码成功; G1b: 有害; G3b: 无条件 |
| **时间窗边界** | 解码结束标志词：`"decoded"`, `"result is"`, `"解码结果"`, `"the instruction is"` |
| **主指标** | CoT 40% 位置 AUROC |
| **Go/No-Go 方案** | 只看 M1, M5, M7 |
| **数据切分** | GroupKFold by prompt_id |
| **Kill 阈值** | M5 AUROC < 0.70 → Kill |

---

## 10. 参考文献（pilot 核心依赖）

| 论文 | 级别 | pilot 中怎么用 |
|------|:--:|------|
| Zhao et al. "Latent Guard" (NeurIPS 2025) | A | v_harm 提取位置（t_inst） |
| "Safety Layers" (ICLR 2025) | A | 层选择（中间层） |
| Chan et al. (FoRLM @ NeurIPS 2025) | B | M5 full probe 的参考 + 数据效率证据 |
| Arditi et al. "Refusal Direction" (NeurIPS 2024) | A | v_refusal 提取 |
| ICR Probe (ACL 2025) | A | 跨层动态特征 |
| SEAL "Three Minds" (arXiv 2505.16241) | C | Tier 1 编码方式参考 + R1 100% ASR 动机 |
| R2D (EMNLP 2025) | A | M7b contrastive 方向的灵感 |
| "False Sense of Security" (NeurIPS MechInterp Workshop 2025) | B | G3a/G3b 受控对照的动机（格式 vs 语义） |
