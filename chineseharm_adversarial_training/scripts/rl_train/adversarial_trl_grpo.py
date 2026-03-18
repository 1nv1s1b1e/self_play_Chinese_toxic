#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对抗博弈 GRPO 训练脚本 — TRL + DeepSpeed (昇腾 910B)
======================================================
支持两种模式:
  --role challenger : 训练 Challenger，使用 selfplay 奖励函数 (含 ASR 信号)
  --role reviewer   : 训练 Reviewer，使用二分类 + 多维奖励函数

数据 parquet 格式 (由 generate_dynamic_data.py 生成):
  prompt       : list[dict]  — chat messages
  reward_model : dict        — ground_truth / category / etc.
  extra_info   : dict        — ASR stats / original_text / etc.

用法:
  python -m torch.distributed.run --nproc_per_node=2 \\
      adversarial_trl_grpo.py \\
      --role challenger \\
      --model_path /path/to/model \\
      --dataset_path /path/to/challenger_grpo_round1.parquet \\
      --output_dir /path/to/challenger_output \\
      --max_steps 100 \\
      --deepspeed ds_zero2.json
"""

import os
import sys
import json
import argparse
from collections import Counter
import torch
try:
    import torch_npu
except ImportError:
    pass
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# ── 添加 reward_functions 目录到 sys.path ────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "reward_functions"))

try:
    from challenger_reward_selfplay import compute_score as challenger_selfplay_reward
    from challenger_reward_v7       import compute_score as challenger_static_reward
    from reviewer_reward            import compute_score as reviewer_reward_func
    print("   ✅ 奖励函数加载成功！")
except Exception as e:
    print(f"   ❌ 奖励函数加载失败: {e}")
    sys.exit(1)


# ── 奖励函数包装器 ────────────────────────────────────────────────────────────

def get_challenger_reward(use_selfplay: bool):
    """
    返回 Challenger GRPO 的奖励函数。
    - use_selfplay=True:  使用含 ASR 信号的 selfplay 奖励
    - use_selfplay=False: 使用静态 v7 奖励（首轮冷启动用）
    """
    def reward_fn(prompts, completions, **kwargs):
        scores = []
        reward_models = kwargs.get("reward_model", [])
        extra_infos   = kwargs.get("extra_info",   [])

        for i in range(len(prompts)):
            # TRL GRPOTrainer: completions[i] 是 list[dict]，第一个元素是生成消息
            solution_str = completions[i][0]["content"] if completions[i] else ""

            gt = ""
            gt_for_reward = ""
            extra = {}
            if i < len(reward_models):
                rm = reward_models[i]
                if isinstance(rm, dict):
                    gt = rm.get("ground_truth", "")
                    gt_for_reward = gt
                    # reference_texts: 传给奖励函数用于 topic_signal 计算
                    extra = dict(extra_infos[i]) if i < len(extra_infos) and isinstance(extra_infos[i], dict) else {}
                    ref_texts = rm.get("reference_texts", [])
                    extra["reference_texts"] = ref_texts
                    # 兼容 challenger_reward_v7/selfplay：优先提供 original_text 作为 reference
                    if not extra.get("original_text") and isinstance(ref_texts, list) and len(ref_texts) > 0:
                        ref_idx = i % len(ref_texts)
                        ref_text = str(ref_texts[ref_idx]).strip()
                        if ref_text:
                            extra["original_text"] = ref_text
                    # 若 ground_truth 是短标签（如“性别歧视”），则改用 reference 文本作为 reward 的参考
                    if isinstance(ref_texts, list) and len(ref_texts) > 0:
                        if not isinstance(gt_for_reward, str) or len(gt_for_reward.strip()) < 6:
                            gt_for_reward = str(ref_texts[i % len(ref_texts)])
                elif isinstance(rm, str):
                    gt = rm
                    gt_for_reward = rm

            if i < len(extra_infos) and isinstance(extra_infos[i], dict) and not extra:
                extra = extra_infos[i]

            reward_func = challenger_selfplay_reward if use_selfplay else challenger_static_reward
            try:
                score = reward_func(
                    data_source="toxicn_challenger",
                    solution_str=solution_str,
                    ground_truth=gt_for_reward if gt_for_reward else gt,
                    extra_info=extra,
                )
            except Exception as ex:
                print(f"[Reward Warning] challenger reward error at index {i}: {ex}")
                score = 0.0

            scores.append(float(score))

        return scores

    return reward_fn


def get_reviewer_reward():
    """
    返回 Reviewer GRPO 的奖励函数。
    reward_model 中包含 Verifier 验证后的 ground_truth，
    比原始数据集标签更可靠。
    """
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
                    # reward_model 包含 Verifier 验证的标签
                    # ground_truth 字段已由 build_reviewer_parquet 用 verified_cat 填充
                    gt = rm   # reviewer_reward 接受完整 dict 作为 ground_truth
                elif isinstance(rm, str):
                    gt = rm

            if i < len(extra_infos) and isinstance(extra_infos[i], dict):
                extra = extra_infos[i]
                # 将 verifier 验证结果透传到 reward function
                # reviewer_reward.py 会根据 extra_info 中的额外字段做附加评分

            try:
                score = reviewer_reward_func(
                    data_source="toxicn_reviewer",
                    solution_str=solution_str,
                    ground_truth=gt,
                    extra_info=extra,
                )
            except Exception as ex:
                print(f"[Reward Warning] reviewer reward error at index {i}: {ex}")
                score = 0.0

            scores.append(float(score))

        return scores

    return reward_fn


def get_reward_wrapper(role: str, use_selfplay: bool = True):
    """统一入口：根据 role 返回对应的奖励函数。"""
    if role == "challenger":
        return get_challenger_reward(use_selfplay=use_selfplay)
    elif role == "reviewer":
        return get_reviewer_reward()
    else:
        raise ValueError(f"未知 role: {role}")


# ── 数据加载 ─────────────────────────────────────────────────────────────────

def load_grpo_dataset(dataset_path: str) -> Dataset:
    """
    加载 GRPO 训练数据集。
    支持两种来源:
      1. generate_dynamic_data.py 生成的 parquet (含 prompt/reward_model/extra_info 列)
      2. 原始 train_seed.parquet (含 文本/标签 列，自动转换)
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

    # ── 判断是否已是 GRPO 格式 ───────────────────────────────────────────────
    if "prompt" in df.columns:
        # 已是 generate_dynamic_data.py 输出的格式，直接使用
        # 确保 prompt 列是 list[dict] 而非 JSON 字符串
        if isinstance(df["prompt"].iloc[0], str):
            df["prompt"] = df["prompt"].apply(json.loads)
        if "reward_model" in df.columns and isinstance(df["reward_model"].iloc[0], str):
            df["reward_model"] = df["reward_model"].apply(json.loads)
        if "extra_info" in df.columns and isinstance(df["extra_info"].iloc[0], str):
            df["extra_info"] = df["extra_info"].apply(json.loads)
        return Dataset.from_pandas(df)

    # ── 自动转换原始种子数据格式 ─────────────────────────────────────────────
    print("   [转换] 检测到原始种子格式，自动转换为 GRPO 格式...")
    col_text = "文本"   if "文本"   in df.columns else "original_text"
    col_cat  = "标签"   if "标签"   in df.columns else "category"
    col_tt   = "toxic_type_label"  if "toxic_type_label"  in df.columns else "toxic_type"
    col_expr = "expression_label" if "expression_label" in df.columns else "expression"

    CHALLENGER_DEFAULT_PROMPT = (
        "类别：{category}\n只输出文本本身，不要解释："
    )

    rows = []
    for _, row in df.iterrows():
        original_text = str(row.get(col_text, "")) if col_text in df.columns else ""
        cat  = str(row.get(col_cat,  "无毒"))
        tt   = str(row.get(col_tt,   "无毒"))
        expr = str(row.get(col_expr, "非仇恨"))
        rows.append({
            "prompt": [
                {"role": "user", "content": CHALLENGER_DEFAULT_PROMPT.format(category=cat)}
            ],
            "reward_model": {
                "ground_truth": original_text if original_text else cat,
                "target_category": cat,
            },
            "extra_info":   {"category": cat, "expression": expr, "toxic_type": tt,
                              "original_text": original_text,
                              "cat_adversarial_success_rate": 0.5,
                              "cat_label_verified_rate": 0.5},
        })

    return Dataset.from_pandas(pd.DataFrame(rows))


def summarize_dataset(dataset: Dataset) -> None:
    """
    输出训练数据概览，便于快速检查类别分布。
    """
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
        print("   [提示] 数据中未找到 extra_info.category，跳过类别统计。")


# ── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="对抗博弈 GRPO 训练 (TRL + DeepSpeed)")
    parser.add_argument("--role",         required=True,  choices=["challenger", "reviewer"],
                        help="训练角色: challenger 或 reviewer")
    parser.add_argument("--model_path",   required=True,  type=str,  help="模型路径")
    parser.add_argument("--dataset_path", required=True,  type=str,  help="训练数据 parquet 路径")
    parser.add_argument("--output_dir",   required=True,  type=str,  help="输出目录")
    parser.add_argument("--max_steps",    default=100,    type=int,  help="最大训练步数")
    parser.add_argument("--save_steps",   default=100,    type=int,  help="保存间隔步数")
    parser.add_argument("--learning_rate", default=1e-6,  type=float, help="学习率")
    parser.add_argument("--per_device_batch_size", default=2, type=int, help="每卡 batch size")
    parser.add_argument("--num_generations", default=4,   type=int,  help="每 prompt 采样数量")
    parser.add_argument("--max_completion_length", default=256, type=int, help="最大生成长度")
    parser.add_argument("--grad_accum",   default=4,      type=int,  help="梯度累积步数")
    parser.add_argument("--seed",         default=42,     type=int,  help="随机种子")
    parser.add_argument("--use_selfplay", action="store_true", default=True,
                        help="使用 selfplay 奖励 (challenger 含 ASR 信号)")
    parser.add_argument("--no_selfplay",  action="store_true", default=False,
                        help="禁用 selfplay，使用静态 v7 奖励 (冷启动用)")
    # parse_known_args 允许 transformers/TRL 的 --deepspeed 等参数透传
    args, unknown = parser.parse_known_args()

    # 处理 selfplay 标志
    use_selfplay = args.use_selfplay and not args.no_selfplay

    # 提取 deepspeed 配置路径
    ds_config = None
    if "--deepspeed" in unknown:
        ds_idx = unknown.index("--deepspeed")
        ds_config = unknown[ds_idx + 1]

    print(f"\n{'='*60}")
    print(f"  对抗博弈 GRPO 训练")
    print(f"  角色       : {args.role}")
    print(f"  模型路径   : {args.model_path}")
    print(f"  数据路径   : {args.dataset_path}")
    print(f"  输出目录   : {args.output_dir}")
    print(f"  Selfplay   : {use_selfplay}")
    print(f"  DeepSpeed  : {ds_config}")
    print(f"{'='*60}\n")

    torch.manual_seed(args.seed)
    try:
        torch.npu.manual_seed_all(args.seed)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── 检查数据集路径 ──────────────────────────────────────────────────────────
    print("1. [校验] 检查数据集路径...")
    if not os.path.exists(args.dataset_path):
        print(f"   ❌ 找不到数据集: {args.dataset_path}")
        sys.exit(1)

    # ── 加载奖励函数 ─────────────────────────────────────────────────────────────
    print(f"2. [初始化] 正在挂载奖励函数 — Role: {args.role}, Selfplay: {use_selfplay}")
    reward_wrapper = get_reward_wrapper(args.role, use_selfplay=use_selfplay)

    # ── 加载数据集 ───────────────────────────────────────────────────────────────
    print("3. [加载] 正在加载训练数据集...")
    dataset = load_grpo_dataset(args.dataset_path)
    print(f"   ✅ 数据集加载完成，共 {len(dataset)} 条。")
    summarize_dataset(dataset)

    # ── 加载 Tokenizer ────────────────────────────────────────────────────────────
    print("4. [加载] 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 生成任务需要 left padding

    # ── 加载模型 (不使用 device_map，完全交给 DeepSpeed 管理) ─────────────────────
    print("5. [加载] 正在从磁盘读取模型权重 (device_map=None, 交由 DeepSpeed 管理)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,         # ← 关键: 让 DeepSpeed 完全掌控设备分配
        trust_remote_code=True,
    )
    print("   ✅ 模型加载完成。")

    # ── 设置 GRPO 训练参数 ────────────────────────────────────────────────────────
    # max_completion_length: challenger 生成较长文本, reviewer 生成简短分类
    max_comp_len = args.max_completion_length
    if args.max_completion_length == 256:
        max_comp_len = 256 if args.role == "challenger" else 80

    print(f"6. [配置] 设定 TRL GRPOConfig for {args.role} (DeepSpeed: {ds_config})...")
    training_args = GRPOConfig(
        output_dir                  = args.output_dir,
        learning_rate               = args.learning_rate,
        per_device_train_batch_size = args.per_device_batch_size,
        gradient_accumulation_steps = args.grad_accum,
        num_generations             = args.num_generations,   # 每 prompt 采样 K 个候选
        max_completion_length       = max_comp_len,
        bf16                        = True,
        logging_steps               = 1,
        max_steps                   = args.max_steps,
        save_steps                  = args.save_steps,
        save_total_limit            = 2,                      # 只保留最近 2 个检查点
        gradient_checkpointing      = True,
        use_vllm                    = False,                  # 昇腾 910B 不使用 vLLM 加速推理
        deepspeed                   = ds_config,
        # 抑制 GRPO 中 KL penalty，由奖励函数自身控制对抗强度
        kl_coef                     = 0.02,
        # 温度控制采样多样性
        temperature                 = 0.8 if args.role == "challenger" else 0.3,
        # 日志到 tensorboard
        report_to                   = "tensorboard",
        logging_dir                 = os.path.join(args.output_dir, "tb_logs"),
    )

    print("7. [启动] 正在初始化 GRPOTrainer...")
    trainer = GRPOTrainer(
        model         = model,
        reward_funcs  = [reward_wrapper],
        args          = training_args,
        train_dataset = dataset,
        processing_class = tokenizer,   # TRL ≥ 0.12 推荐写法
    )

    print(f"8. 🚀 [训练] 开始 {args.role} GRPO 更新！")
    trainer.train()

    print("9. [保存] 保存最终模型与 Tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 保存训练完成标记，供 shell 脚本检测
    done_file = os.path.join(args.output_dir, "training_done.txt")
    with open(done_file, "w") as f:
        f.write(f"role={args.role}\nsteps={args.max_steps}\n")

    # 训练元信息，便于回溯实验配置
    metadata = {
        "role": args.role,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "learning_rate": args.learning_rate,
        "per_device_batch_size": args.per_device_batch_size,
        "num_generations": args.num_generations,
        "max_completion_length": max_comp_len,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        "use_selfplay": use_selfplay,
        "deepspeed": ds_config,
        "dataset_size": len(dataset),
    }
    meta_file = os.path.join(args.output_dir, "train_metadata.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"🎉 [{args.role}] GRPO 训练完成！模型已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()
