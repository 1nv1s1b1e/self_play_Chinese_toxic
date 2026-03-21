#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一版 GRPO 训练脚本 — TRL + DeepSpeed (昇腾 910B)
===================================================
合并 v1 的 adversarial_trl_grpo.py + adversarial_trl_grpo_fixed.py

核心改进:
  - Challenger: 始终使用逐样本真对抗奖励 (challenger_reward.py, Plan 1)
  - Reviewer:   使用多维奖励 (reviewer_reward.py)
  - 不再需要 monkey-patch，直接导入统一模块

用法:
  python -m torch.distributed.run --nproc_per_node=4 \\
      adversarial_trl_grpo.py \\
      --role challenger \\
      --model_path /path/to/model \\
      --dataset_path /path/to/grpo_round1.parquet \\
      --output_dir /path/to/output \\
      --max_steps 100 \\
      --deepspeed ds_zero2.json
"""

import os
import sys
import json
import argparse
from collections import Counter
from typing import List, Optional

import torch

try:
    import torch_npu
except ImportError:
    pass

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# ── 统一导入 ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "reward_functions"))

from challenger_reward import compute_score as challenger_adversarial_reward
from reward_functions.reviewer_reward import compute_score as reviewer_reward_func
from reward_functions.reviewer_reward import flush_batch_summary as reviewer_flush_batch
from constants import (
    REVIEWER_SYSTEM_PROMPT, HARMFUL_CATEGORIES,
    format_reviewer_user_content, parse_classification_output,
)
from quality_gate import quality_gate as _quality_gate

print("   ✅ 奖励函数加载成功 (在线 Reviewer 1-acc 版)")


# ── 在线 Reviewer 推理奖励计算器 ──────────────────────────────────────────────────

class OnlineReviewerReward:
    """
    冻结 Reviewer 模型，在 GRPO rollout 时对 Challenger 每条新生成文本实时评分。

    奖励公式（Search Self-Play 1-acc）：
      - quality_gate 过滤低质生成（垃圾文本直接 -1.0）
      - Reviewer 推理：被骗 → adversarial_signal=1.0，未被骗 → 0.0
      - parse 失败 → 视为"未被骗"（不奖励垃圾输出）
      - final reward = quality_gate × adversarial_signal × 2 - 1  ∈ [-1, 1]
    """

    def __init__(
        self,
        model_path: str,
        local_rank: int = 0,
        batch_size: int = 8,
        max_new_tokens: int = 128,
    ):
        self.model_path     = model_path
        self.batch_size     = batch_size
        self.max_new_tokens = max_new_tokens
        self.local_rank     = local_rank

        # 确定设备
        try:
            import torch_npu  # noqa: F401
            self._device = f"npu:{local_rank}"
        except ImportError:
            if torch.cuda.is_available():
                self._device = f"cuda:{local_rank}"
            else:
                self._device = "cpu"

        print(f"[OnlineReviewerReward] 正在加载冻结 Reviewer: {model_path} → {self._device}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self._device},
            trust_remote_code=True,
        )
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        print(f"[OnlineReviewerReward] ✅ Reviewer 加载完成，已完全冻结")

    def _build_prompt(self, text: str) -> str:
        msgs = [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user",   "content": format_reviewer_user_content(text)},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"{REVIEWER_SYSTEM_PROMPT}\n用户：{format_reviewer_user_content(text)}\n助手："

    def _batch_infer(self, texts: List[str]) -> List[str]:
        """批量推理，返回 Reviewer 原始输出文本列表。"""
        results: List[str] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start: start + self.batch_size]
            raw_prompts = [self._build_prompt(t) for t in batch]
            enc = self._tokenizer(
                raw_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            input_ids      = enc["input_ids"].to(self._device)
            attention_mask = enc["attention_mask"].to(self._device)
            with torch.no_grad():
                out = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            prompt_len = input_ids.shape[1]
            for j in range(len(batch)):
                gen = out[j][prompt_len:]
                results.append(
                    self._tokenizer.decode(gen, skip_special_tokens=True).strip()
                )
        return results

    def compute_rewards(
        self,
        completion_texts: List[str],
        true_categories: List[str],
    ) -> List[float]:
        """
        批量计算多级对抗奖励 (binary fooling + category confusion)。

        奖励层次 (对有害类别目标):
          quality_gate < 0.01                          → -1.0 (垃圾)
          parse 失败                                   → -1.0 (无效)
          Reviewer binary 被骗 (有害判无害)            → gate × 1.0 → 映射 [+高]
          Reviewer binary 对, 但 category 判错          → gate × 0.6 → 映射 [+中]
          Reviewer 完全正确 (binary + category)          → gate × 0.0 → 映射 [-1]

        对无毒类别目标:
          Reviewer 误判为有害                            → gate × 0.8
          Reviewer 正确判无害                            → gate × 0.0

        设计原理:
          旧版只有 binary fooling 信号 → Challenger 只学生成"模糊文本"
          新版加入 category confusion 信号 → Challenger 被激励生成
          "让 Reviewer 在有害子类别间混淆"的文本 → 推动 Reviewer
          提升 category 区分能力。
        """
        gates        = [_quality_gate(t) for t in completion_texts]
        valid_idx    = [i for i, g in enumerate(gates) if g > 0.01]
        valid_texts  = [completion_texts[i] for i in valid_idx]

        reviewer_raw = self._batch_infer(valid_texts) if valid_texts else []

        rewards = [-1.0] * len(completion_texts)  # gate 失败默认 -1.0

        for j, i in enumerate(valid_idx):
            gate     = gates[i]
            true_cat = true_categories[i]
            true_is_harmful = true_cat in HARMFUL_CATEGORIES

            parsed = parse_classification_output(reviewer_raw[j])

            # parse 失败 → 不奖励
            if parsed.get("binary") is None:
                adversarial_signal = 0.0
            else:
                reviewer_says_harmful = (parsed["binary"] == "有害")
                if true_is_harmful:
                    if not reviewer_says_harmful:
                        # binary 被骗 → 最高奖励
                        adversarial_signal = 1.0
                    elif parsed.get("category") != true_cat:
                        # binary 对但 category 错 → 中等奖励
                        # 这推动 Challenger 生成"让 Reviewer 分错类"的文本
                        adversarial_signal = 0.6
                    else:
                        # Reviewer 完全正确 → 不奖励
                        adversarial_signal = 0.0
                else:
                    # 无毒目标
                    if reviewer_says_harmful:
                        adversarial_signal = 0.8
                    else:
                        adversarial_signal = 0.0

            raw         = gate * adversarial_signal
            rewards[i]  = raw * 2.0 - 1.0

        return rewards


# ── 奖励函数包装器 ────────────────────────────────────────────────────────────

def get_challenger_reward(reviewer: Optional[OnlineReviewerReward] = None):
    """
    Challenger GRPO 奖励函数包装器。

    - reviewer 不为 None（默认）: 在线 1-acc — 对每条新生成文本实时调用 Reviewer 推理
    - reviewer 为 None (fallback): 使用 parquet 中预存的 reviewer_fooled 静态信号
    """
    def reward_fn(prompts, completions, **kwargs):
        extra_infos = kwargs.get("extra_info", [])

        # 提取所有 completion 文本
        texts: List[str] = []
        for i in range(len(completions)):
            texts.append(completions[i][0]["content"] if completions[i] else "")

        if reviewer is not None:
            # ── 在线模式：实时 Reviewer 推理 ──
            true_categories: List[str] = []
            for i in range(len(prompts)):
                extra = (
                    extra_infos[i]
                    if i < len(extra_infos) and isinstance(extra_infos[i], dict)
                    else {}
                )
                true_categories.append(extra.get("category", ""))
            return reviewer.compute_rewards(texts, true_categories)

        # ── 静态 fallback：使用 parquet 中预存的 reviewer_fooled ──
        reward_models = kwargs.get("reward_model", [])
        scores: List[float] = []
        for i, text in enumerate(texts):
            extra = (
                extra_infos[i]
                if i < len(extra_infos) and isinstance(extra_infos[i], dict)
                else {}
            )
            gt = reward_models[i] if i < len(reward_models) else {}
            try:
                score = challenger_adversarial_reward(
                    data_source="toxicn_challenger",
                    solution_str=text,
                    ground_truth=gt,
                    extra_info=extra,
                )
            except Exception as ex:
                print(f"[Reward Warning] challenger fallback error at index {i}: {ex}")
                score = 0.0
            scores.append(float(score))
        return scores

    return reward_fn


def get_reviewer_reward():
    """Reviewer GRPO 奖励函数包装器。"""
    def reward_fn(prompts, completions, **kwargs):
        scores = []
        reward_models = kwargs.get("reward_model", [])
        extra_infos   = kwargs.get("extra_info",   [])

        for i in range(len(prompts)):
            solution_str = completions[i][0]["content"] if completions[i] else ""

            gt    = ""
            extra = {}
            if i < len(reward_models):
                rm = reward_models[i]
                if isinstance(rm, dict):
                    gt = rm
                elif isinstance(rm, str):
                    gt = rm

            if i < len(extra_infos) and isinstance(extra_infos[i], dict):
                extra = extra_infos[i]

            try:
                score = reviewer_reward_func(
                    data_source="toxicn_reviewer",
                    solution_str=solution_str,
                    ground_truth=gt,
                    extra_info=extra,
                )
            except Exception as ex:
                print(f"[Reward Warning] reviewer error at index {i}: {ex}")
                score = 0.0

            scores.append(float(score))

        # 批次级别奖励汇总日志 (仅 reviewer 有批汇总)
        try:
            reviewer_flush_batch()
        except Exception:
            pass

        return scores

    return reward_fn


def get_reward_wrapper(role: str, reviewer: Optional[OnlineReviewerReward] = None):
    """统一入口：根据 role 返回对应的奖励函数。"""
    if role == "challenger":
        return get_challenger_reward(reviewer=reviewer)
    elif role == "reviewer":
        return get_reviewer_reward()
    else:
        raise ValueError(f"未知 role: {role}")


# ── 数据加载 ─────────────────────────────────────────────────────────────────

def load_grpo_dataset(dataset_path: str) -> Dataset:
    """
    加载 GRPO 训练数据集。
    支持 parquet (generate_dynamic_data 输出) 和 json/jsonl。
    """
    if dataset_path.endswith(".parquet"):
        df = pd.read_parquet(dataset_path)
    elif dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
        df = pd.read_json(dataset_path, lines=dataset_path.endswith(".jsonl"))
    else:
        raise ValueError(f"不支持的数据格式: {dataset_path}")

    if len(df) == 0:
        raise ValueError(f"数据集为空: {dataset_path}")

    print(f"   数据集列名: {list(df.columns)}")
    print(f"   数据集大小: {len(df)} 行")

    if "prompt" in df.columns:
        if isinstance(df["prompt"].iloc[0], str):
            df["prompt"] = df["prompt"].apply(json.loads)
        if "reward_model" in df.columns and isinstance(df["reward_model"].iloc[0], str):
            df["reward_model"] = df["reward_model"].apply(json.loads)
        if "extra_info" in df.columns and isinstance(df["extra_info"].iloc[0], str):
            df["extra_info"] = df["extra_info"].apply(json.loads)
        return Dataset.from_pandas(df)

    # ── 自动转换原始种子数据格式 ──
    print("   [转换] 检测到原始种子格式，自动转换为 GRPO 格式...")
    col_text = "文本"  if "文本"  in df.columns else "original_text"
    col_cat  = "标签"  if "标签"  in df.columns else "category"
    col_tt   = "toxic_type_label" if "toxic_type_label" in df.columns else "toxic_type"
    col_expr = "expression_label" if "expression_label" in df.columns else "expression"

    CHALLENGER_DEFAULT_PROMPT = "类别：{category}\n只输出文本本身，不要解释："

    rows = []
    for _, row in df.iterrows():
        original_text = str(row.get(col_text, "")) if col_text in df.columns else ""
        cat  = str(row.get(col_cat, "无毒"))
        tt   = str(row.get(col_tt, "无毒"))
        expr = str(row.get(col_expr, "非仇恨"))
        rows.append({
            "prompt": [
                {"role": "user", "content": CHALLENGER_DEFAULT_PROMPT.format(category=cat)}
            ],
            "reward_model": {
                "ground_truth": original_text if original_text else cat,
                "target_category": cat,
            },
            "extra_info": {
                "category": cat, "expression": expr, "toxic_type": tt,
                "original_text": original_text,
                "reviewer_fooled": False,
                "reviewer_binary_ok": True,
            },
        })

    return Dataset.from_pandas(pd.DataFrame(rows))


def summarize_dataset(dataset: Dataset) -> None:
    if len(dataset) == 0:
        print("   [警告] 空数据集，跳过统计。")
        return

    category_counter = Counter()
    has_extra = "extra_info" in dataset.column_names
    if has_extra:
        for item in dataset["extra_info"]:
            if isinstance(item, dict):
                cat = item.get("category")
                if cat:
                    category_counter[str(cat)] += 1

    if category_counter:
        print("   数据类别分布:")
        for cat, cnt in sorted(category_counter.items(), key=lambda x: (-x[1], x[0])):
            print(f"     - {cat}: {cnt}")
    else:
        print("   [提示] 无 extra_info.category，跳过类别统计。")


# ── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="统一版 GRPO 训练 (TRL + DeepSpeed)")
    parser.add_argument("--role",         required=True,  choices=["challenger", "reviewer"])
    parser.add_argument("--model_path",   required=True,  type=str)
    parser.add_argument("--dataset_path", required=True,  type=str)
    parser.add_argument("--output_dir",   required=True,  type=str)
    parser.add_argument("--max_steps",    default=0,      type=int,
                        help="GRPO 训练步数 (0=使用 num_epochs 模式)")
    parser.add_argument("--num_epochs",   default=1,      type=int,
                        help="GRPO 训练 epoch 数 (仅 max_steps=0 时生效)")
    parser.add_argument("--save_steps",   default=9999,   type=int)
    parser.add_argument("--learning_rate", default=1e-6,  type=float)
    parser.add_argument("--per_device_batch_size", default=2, type=int)
    parser.add_argument("--num_generations", default=4,   type=int)
    parser.add_argument("--max_completion_length", default=256, type=int)
    parser.add_argument("--grad_accum",   default=4,      type=int)
    parser.add_argument("--selfplay_step", default=0,     type=int,
                        help="当前自对弈步数 (用于元数据记录)")
    parser.add_argument("--seed",         default=42,     type=int)
    # ── 在线 Reviewer 推理（Challenger 角色专用）──
    parser.add_argument(
        "--reviewer_model_path", default="", type=str,
        help=(
            "冻结 Reviewer 模型路径（仅 role=challenger 时生效）。"
            "提供后使用实时 Reviewer 推理计算 1-acc 奖励；"
            "不提供则 fallback 到 parquet 中预存的静态信号。"
        ),
    )
    parser.add_argument(
        "--reviewer_batch_size", default=8, type=int,
        help="Reviewer 推理 batch size（默认 8）",
    )
    args, unknown = parser.parse_known_args()

    # 提取 deepspeed 配置路径
    ds_config = None
    if "--deepspeed" in unknown:
        ds_idx = unknown.index("--deepspeed")
        ds_config = unknown[ds_idx + 1]

    print(f"\n{'='*60}")
    print(f"  GRPO 训练 (在线 Reviewer 1-acc)")
    print(f"  角色         : {args.role}")
    print(f"  自对弈步数   : {args.selfplay_step}")
    print(f"  模型路径     : {args.model_path}")
    if args.role == "challenger":
        print(f"  封冻 Reviewer : {args.reviewer_model_path or '(未提供，使用静态 fallback)'}")
    print(f"  数据路径     : {args.dataset_path}")
    print(f"  训练模式     : {'max_steps=' + str(args.max_steps) if args.max_steps > 0 else 'epochs=' + str(args.num_epochs)}")
    print(f"  输出目录     : {args.output_dir}")
    print(f"  DeepSpeed    : {ds_config}")
    print(f"{'='*60}\n")

    torch.manual_seed(args.seed)
    try:
        torch.npu.manual_seed_all(args.seed)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("1. [校验] 检查数据集路径...")
    if not os.path.exists(args.dataset_path):
        print(f"   ❌ 找不到数据集: {args.dataset_path}")
        sys.exit(1)

    print(f"2. [初始化] 挂载奖励函数 — Role: {args.role}")
    # ── 在线 Reviewer 初始化（仅 challenger 角色）──
    online_reviewer: Optional[OnlineReviewerReward] = None
    if args.role == "challenger" and args.reviewer_model_path:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"   [在线 Reviewer] 加载中 (local_rank={local_rank}, "
              f"batch_size={args.reviewer_batch_size})...")
        online_reviewer = OnlineReviewerReward(
            model_path  = args.reviewer_model_path,
            local_rank  = local_rank,
            batch_size  = args.reviewer_batch_size,
            max_new_tokens = 128,
        )
        print("   ✅ 在线 Reviewer 就绪，将使用实时 1-acc 奖励")
    elif args.role == "challenger":
        print("   ⚠️  未提供 --reviewer_model_path，使用 parquet 静态信号 (fallback)")

    reward_wrapper = get_reward_wrapper(args.role, reviewer=online_reviewer)

    print("3. [加载] 训练数据集...")
    dataset = load_grpo_dataset(args.dataset_path)
    print(f"   ✅ {len(dataset)} 条")
    summarize_dataset(dataset)

    print("4. [加载] Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("5. [加载] 模型 (device_map=None, DeepSpeed 管理)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    )

    # ── LoRA: 只训练少量参数，大幅降低优化器显存 ──
    from peft import LoraConfig, get_peft_model, TaskType
    _model_name = args.model_path.lower()
    if "7b" in _model_name or "14b" in _model_name:
        lora_rank = 32
    elif "3b" in _model_name:
        lora_rank = 32
    else:
        lora_rank = 16
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print("   ✅ 模型加载完成 (LoRA)")

    max_comp_len = args.max_completion_length
    if args.max_completion_length == 256:
        max_comp_len = 256 if args.role == "challenger" else 80

    # ── 训练模式: epochs 或 max_steps ──
    if args.max_steps > 0:
        effective_max_steps = args.max_steps
        effective_epochs = 100  # 大数，由 max_steps 控制
        effective_save_steps = args.max_steps
    else:
        effective_max_steps = -1  # TRL 使用 num_train_epochs
        effective_epochs = args.num_epochs
        effective_save_steps = 9999  # 足够大，训练结束自动保存

    print(f"6. [配置] GRPOConfig ({args.role}, epochs={effective_epochs}, DeepSpeed: {ds_config})...")
    training_args = GRPOConfig(
        output_dir                  = args.output_dir,
        learning_rate               = args.learning_rate,
        per_device_train_batch_size = args.per_device_batch_size,
        gradient_accumulation_steps = args.grad_accum,
        num_generations             = args.num_generations,
        max_completion_length       = max_comp_len,
        bf16                        = True,
        logging_steps               = 1,
        max_steps                   = effective_max_steps,
        num_train_epochs            = effective_epochs,
        save_steps                  = effective_save_steps,
        save_total_limit            = 2,
        gradient_checkpointing      = True,
        use_vllm                    = False,
        deepspeed                   = ds_config,
        beta                        = 0.02,
        # Reviewer 温度设为 1.0:
        # 分类任务输出极短(2行), 0.3~0.7 几乎无多样性 (实测 unique_cat=1.0~1.2)
        # 1.0 时才有可观的输出差异 (unique_cat=1.4), 给 GRPO 提供梯度信号
        temperature                 = 0.8 if args.role == "challenger" else 1.0,
        report_to                   = "tensorboard",
        logging_dir                 = os.path.join(args.output_dir, "tb_logs"),
    )

    print("7. [启动] 初始化 GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_wrapper],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"8. 🚀 开始 {args.role} GRPO 更新！")
    trainer.train()

    print("9. [保存] 合并 LoRA → 完整模型（vllm 兼容）...")
    _save_model = trainer.model
    if hasattr(_save_model, "merge_and_unload"):
        _save_model = _save_model.merge_and_unload()
        print("   ✅ LoRA 已合并到基础模型")
    _save_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ── 确保 config.json 存在于输出目录根（vllm 下一轮加载必需）──
    # DeepSpeed ZeRO 环境下 save_model 有时不写 config.json 到根目录
    _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if _local_rank == 0:
        import shutil
        _cfg_src = os.path.join(args.model_path, "config.json")
        _cfg_dst = os.path.join(args.output_dir, "config.json")
        if os.path.exists(_cfg_src) and not os.path.exists(_cfg_dst):
            shutil.copy2(_cfg_src, _cfg_dst)
            print(f"   ✅ config.json 补充复制: {_cfg_dst}")
        # 同样补充 generation_config.json（部分模型需要）
        for _fname in ["generation_config.json"]:
            _src = os.path.join(args.model_path, _fname)
            _dst = os.path.join(args.output_dir, _fname)
            if os.path.exists(_src) and not os.path.exists(_dst):
                shutil.copy2(_src, _dst)

    done_file = os.path.join(args.output_dir, "training_done.txt")
    with open(done_file, "w") as f:
        f.write(f"role={args.role}\nsteps={effective_max_steps}\nepochs={effective_epochs}\n")

    metadata = {
        "role": args.role,
        "selfplay_step": args.selfplay_step,
        "model_path": args.model_path,
        "reviewer_model_path": args.reviewer_model_path,
        "reward_mode": "online_1acc" if online_reviewer else "static_fallback",
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "max_steps": effective_max_steps,
        "num_epochs": effective_epochs,
        "learning_rate": args.learning_rate,
        "per_device_batch_size": args.per_device_batch_size,
        "num_generations": args.num_generations,
        "max_completion_length": max_comp_len,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        "deepspeed": ds_config,
        "dataset_size": len(dataset),
    }
    with open(os.path.join(args.output_dir, "train_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"🎉 [{args.role}] GRPO 训练完成！模型已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
