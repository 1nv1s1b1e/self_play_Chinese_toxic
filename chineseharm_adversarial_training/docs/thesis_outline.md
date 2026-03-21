# 毕业设计：基于对抗博弈的中文有毒语言检测方法

## 一、选题背景与研究意义

### 1.1 研究背景
- 中文互联网有毒语言（仇恨言论、歧视性言论）日益严重
- 现有检测模型对隐式仇恨（反讽、暗语、阴阳怪气）识别能力不足
- ToxiCN 数据集提供了中文有毒语言的标准化基准（12011条，6分类）

### 1.2 研究意义
- 提出一种基于 Self-Play 对抗博弈的检测能力增强方法
- 通过 Challenger-Reviewer 双模型对抗迭代，持续提升 Reviewer 的分类准确率
- 特别关注隐式仇恨言论的检测能力提升

### 1.3 国内外研究现状
- 有毒语言检测方法综述（规则→机器学习→预训练模型→LLM）
- ToxiCN 数据集与中文有毒语言分类体系
- Self-Play 在 NLP 中的应用（AlphaGo → RLHF → 对抗训练）
- GRPO (Group Relative Policy Optimization) 算法

---

## 二、相关理论与技术基础

### 2.1 ToxiCN 数据集与分类体系
- **二分类**: toxic / non-toxic
- **6 类标签**: 性别歧视、种族歧视、地域偏见、LGBTQ歧视、其他仇恨、无毒
- **多标签处理**: 842条多标签样本的处理策略
- 数据分布：有毒6461条 / 无毒5550条

### 2.2 大语言模型微调
- LoRA (Low-Rank Adaptation) 参数高效微调
- SFT (Supervised Fine-Tuning) 冷启动训练
- Qwen2.5 系列模型（0.5B / 1.5B / 3B / 7B）

### 2.3 强化学习与 GRPO
- GRPO 算法原理：组内相对优势估计
- 与 PPO/RLHF 的区别：无需 value model，直接从 K 条采样中计算 advantage
- KL 散度惩罚防止策略崩塌

### 2.4 Self-Play 对抗博弈框架
- Stackelberg 博弈模型：Challenger（攻击者）vs Reviewer（防御者）
- 1-acc 奖励设计：Challenger 的奖励 = 1 - Reviewer 的准确率
- 纳什均衡与对抗平衡

---

## 三、方法设计

### 3.1 整体框架

```
┌─────────────────────────────────────────────────────────┐
│                    Self-Play 对抗训练框架                   │
│                                                           │
│  ┌───────────┐    生成对抗文本    ┌───────────┐          │
│  │ Challenger │ ──────────────→ │  Reviewer  │          │
│  │  (生成器)  │ ←────────────── │  (检测器)  │          │
│  └───────────┘   1-acc 奖励信号   └───────────┘          │
│       ↑ GRPO                         ↑ GRPO              │
│       │                              │                    │
│  "骗过 Reviewer"              "正确分类"                  │
│  的文本获得奖励               的输出获得奖励               │
└─────────────────────────────────────────────────────────┘
```

### 3.2 阶段一：SFT 冷启动

#### 3.2.1 Challenger SFT
- **训练数据**: 8161条，messages 格式
- **Prompt 格式**:
  ```
  [system]: 你是一个文本生成助手。根据用户给定的类别和表达方式，
            生成一条对应的真实互联网文本。只输出文本本身，不要任何解释。
  [user]:   类别：{category}\n表达方式：{expression}\n只输出文本本身，不要解释：
  ```
- **训练配置**: LoRA rank=32, alpha=64, lr=2e-4, 3 epochs
- **设计理由**: 不在 prompt 中注入分类规则，让模型通过 SFT 内化类别知识

#### 3.2.2 Reviewer SFT
- **训练数据**: 8167条，messages 格式
- **Prompt 格式**:
  ```
  [system]: You are a helpful assistant.
  [user]:   作为中文有毒语言检测专家，请对以下文本进行分析。
            有毒类别：性别歧视、种族歧视、地域偏见、LGBTQ歧视、其他仇恨。
            注意：隐式仇恨（反讽、暗语、谐音、阴阳怪气）也属有害；
            转述仇恨事件但立场中立的属无害。
            请严格按以下格式输出：
            判断：有害（或：判断：无害）
            类别：<类别名>（仅有害时输出）
            文本：{text}
  ```
- **输出格式**: 仅判断 + 类别（无毒性类型/表达方式）
- **设计理由**: 简化输出降低任务难度，聚焦核心分类能力

### 3.3 阶段二：Self-Play 对抗训练

#### 3.3.1 每步流程
```
Step N:
  1. Datagen: Challenger 生成 189 条对抗文本 (5类×20 + 89无毒)
  2. Reviewer 评估: 对生成文本分类，计算 1-acc 信号
  3. Challenger GRPO: 在线 Reviewer 实时评估 K=4 条 completion
  4. Reviewer GRPO: 对抗样本 + 种子混合 + 困难样本回放
  5. 评估: 在测试集 (2403条) 上计算 acc / macro-F1
```

#### 3.3.2 Challenger 奖励函数（多级 1-acc）
```
quality_gate < 0.01              → -1.0 (垃圾文本)
Reviewer binary 被骗 (有害判无害) → gate × 1.0 (最高奖励)
Reviewer category 判错            → gate × 0.6 (中等奖励)
Reviewer 完全正确                → gate × 0.0 → -1.0
```
- **在线 Reviewer 推理**: GRPO 训练时冻结 Reviewer 对每条新 completion 实时评估
- **质量门控**: 长度/重复/格式/多样性四维检测

#### 3.3.3 Reviewer 奖励函数（多级准确率）
```
binary + category 全对 → +1.0
binary 对 category 错  → +0.4
无毒正确判无害        → +0.6
binary 错误           → -0.5
parse 失败            → -1.0
```

#### 3.3.4 训练数据策略
- **种子数据混合**: 30% 原始数据防止灾难性遗忘
- **困难样本回放**: 最近 5 轮错判样本 ×2 重采样（上限 1000）
- **Prompt 一致性**: 训练、推理、评估使用完全相同的 prompt 格式

#### 3.3.5 GRPO 超参数

| 参数 | Challenger | Reviewer |
|------|-----------|----------|
| LoRA rank | 32 | 32 |
| Learning rate | 5e-7 | 5e-7 |
| per_device_bs | 2 | 4 |
| num_generations (K) | 4 | 4 |
| max_completion_length | 128 | 64 |
| temperature | 0.8 | 1.0 |
| KL beta | 0.02 | 0.02 |
| DeepSpeed | ZeRO-2 | ZeRO-2 |
| Epochs/step | 1 | 1 |

### 3.4 关键设计决策

#### 3.4.1 LoRA vs 全量微调
- 全量微调在 GRPO 阶段 OOM（Adam 优化器 36GB）
- LoRA 仅 ~0.5% 参数可训练，优化器 ~0.2GB
- 允许同时加载 Challenger + 冻结 Reviewer 在同一 NPU

#### 3.4.2 在线 vs 静态 Reviewer 评估
- 静态方式：所有 K 条 completion 得到相同预计算 reward → advantage=0 → 无梯度信号
- 在线方式：每条 completion 实时评估 → 不同 reward → 有效 advantage

#### 3.4.3 Best Model 保留策略
- 每步评估后与历史最优对比
- 未超过 best 时丢弃本步模型，下步从 best 继续
- 保证 best accuracy 单调不降

---

## 四、实验设计

### 4.1 实验环境
- **硬件**: 华为昇腾 910B NPU × 4（32GB/卡 或 64GB/卡）
- **软件**: PyTorch + torch_npu, TRL (GRPO), DeepSpeed, vLLM-Ascend
- **基础模型**: Qwen2.5-3B-Instruct / Qwen2.5-7B-Instruct

### 4.2 数据集划分
| 划分 | 条数 | 用途 |
|------|------|------|
| Train | 8167 | SFT 训练 + Self-play 种子数据 |
| Val | 1441 | SFT 阶段验证 |
| Test | 2403 | Self-play 每步评估 + 最终评估 |

### 4.3 评估指标
- **Overall Accuracy**: 6 分类整体准确率
- **Macro F1**: 各类别 F1 的宏平均（关注小类别）
- **Per-category Accuracy**: 每类别准确率
- **ASR (Attack Success Rate)**: Challenger 骗过 Reviewer 的比率
- **多标签评估**: 预测命中 all_labels 中任一即算正确

### 4.4 实验方案

#### 实验 1：基线对比
- Qwen2.5-3B base（未微调）
- Qwen2.5-3B + SFT（冷启动后）
- Qwen2.5-3B + SFT + Self-Play（对抗训练后）

#### 实验 2：模型规模消融
- 0.5B / 1.5B / 3B / 7B 在 SFT 和 Self-Play 上的表现

#### 实验 3：Self-Play 步数分析
- 训练曲线：acc / F1 / ASR 随步数变化
- 最优步数选择

#### 实验 4：消融实验
- 有/无在线 Reviewer（验证在线评估的必要性）
- 有/无困难样本回放（验证错题回放的效果）
- 有/无种子数据混合（验证防遗忘策略）
- 不同 K 值（2/4/8）对 GRPO 的影响

#### 实验 5：类别级分析
- 各类别的提升幅度（特别关注"其他仇恨"和隐式仇恨）
- Challenger 生成文本的质量分析

---

## 五、实验结果与分析

### 5.1 SFT 基线结果
（待实验填充：baseline vs SFT 的 acc/F1 对比表）

### 5.2 Self-Play 训练曲线
（待实验填充：metrics.jsonl 中 acc/F1/ASR 随步数变化的折线图）

### 5.3 最终模型对比
（待实验填充：各模型在测试集上的 per-category 结果表）

### 5.4 消融实验结果
（待实验填充）

### 5.5 案例分析
- Challenger 生成的典型对抗样本
- Reviewer 从错误到正确的分类变化
- 困难样本的类别分布

---

## 六、总结与展望

### 6.1 主要贡献
1. 提出基于 Challenger-Reviewer Self-Play 的中文有毒语言检测增强框架
2. 设计多级 1-acc 奖励函数（binary fooling + category confusion）
3. 提出困难样本动态回放 + 种子数据混合的防遗忘训练策略
4. 在 ToxiCN 数据集上验证了方法的有效性

### 6.2 创新点
- **对抗博弈驱动**: 不依赖人工标注新数据，通过对抗自动发现检测盲区
- **在线奖励评估**: GRPO 训练中实时 Reviewer 推理，确保有效梯度信号
- **训练-推理一致性**: SFT、Self-play、评估全链路 prompt 统一

### 6.3 不足与展望
- 探索更大规模模型（14B+）的效果
- 引入外部 API Verifier 提供更强的对抗信号
- 将方法推广到其他安全检测任务（谣言检测、情感操控等）
- 多轮对话场景下的有毒语言检测

---

## 参考文献

1. ToxiCN: A Holistic Chinese Toxic Language Benchmark (Lu et al., 2025)
2. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
3. Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (Chen et al., 2024)
4. LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
5. Group Relative Policy Optimization (GRPO)
6. Qwen2.5 Technical Report (Alibaba, 2024)

---

## 附录

### A. 项目代码结构
```
chineseharm_adversarial_training/
├── corrected_data/          # 数据源（标签修正后）
├── prepared_data/           # 部署后的训练数据
├── split_data/              # 评估数据
├── scripts/
│   ├── run_pipeline/        # 流水线脚本
│   ├── integrated_selfplay/ # Self-Play 核心代码
│   │   ├── run_selfplay.sh
│   │   ├── generate_dynamic_data.py
│   │   ├── adversarial_trl_grpo.py
│   │   ├── build_parquet.py
│   │   ├── constants.py
│   │   ├── challenger_reward.py
│   │   ├── reward_functions/reviewer_reward.py
│   │   └── quality_gate.py
│   ├── model_lora/          # SFT 训练脚本
│   └── model_eval/          # 评估脚本
└── setup_corrected_data.sh  # 数据部署脚本
```

### B. 完整超参数表
（见第三章 3.3.5 节）

### C. 硬件配置与运行时间
（待实验填充）
