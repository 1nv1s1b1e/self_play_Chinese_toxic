#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan 3 专用: 修改版 generate_dynamic_data — 集成拒绝采样
========================================================

与 v1 generate_dynamic_data.py 的区别:
  Step 3.5 (新增): Challenger 生成后，用 rejection_sampler 过滤低质量文本
  同步过滤 tasks / generated_texts，保证索引对齐

这是正确的集成方式——在 Phase 0 内部过滤生成文本，
而非在 parquet 级别后处理（parquet 只有 prompt，没有生成文本）。
"""

import os
import sys

PLAN3_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(os.path.dirname(PLAN3_DIR), "rl_train")
sys.path.insert(0, V1_DIR)
sys.path.insert(0, PLAN3_DIR)


def main():
    """Phase 0 主流程，Step 3 后插入拒绝采样。"""
    import random
    import logging
    import json
    from pathlib import Path

    import torch
    import torch_npu
    import pandas as pd

    from generate_dynamic_data import (
        parse_args, build_sampling_tasks,
        load_model_and_tokenizer, batch_generate, free_model,
        build_reviewer_parquet,
        REVIEWER_USER_TEMPLATE, REVIEWER_SYSTEM_PROMPT,
        HARMFUL_CATEGORIES, ALL_CATEGORIES,
        _build_challenger_system_prompt,
    )
    from verifier import Verifier
    from rejection_sampler import filter_low_quality_samples
    from build_parquet_adversarial import build_challenger_parquet_adversarial

    logger = logging.getLogger("plan3_datagen")

    args = parse_args()

    # Plan 3 额外参数: 拒绝采样阈值 (从环境变量读取)
    rejection_threshold = float(os.environ.get("REJECTION_THRESHOLD", "0.3"))

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info(f"║  [Plan 3] Phase 0: 数据生成 + 拒绝采样 (第 {args.round_idx} 轮)  ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"  [Plan 3a] 拒绝采样阈值 = {rejection_threshold}")

    # ── Step 1: 加载种子数据 ──
    logger.info("\n[Step 1] 加载种子数据...")
    if args.seed_data.endswith(".parquet"):
        seed_df = pd.read_parquet(args.seed_data)
    else:
        seed_df = pd.read_json(args.seed_data)
    logger.info(f"   种子数据共 {len(seed_df)} 条")

    # ── Step 2: 构建采样任务 ──
    logger.info("\n[Step 2] 构建采样任务...")
    tasks = build_sampling_tasks(seed_df, args.samples_per_cat)
    logger.info(f"   共构建 {len(tasks)} 个生成任务")

    # ── Step 3: Challenger 推理 ──
    logger.info("\n[Step 3] Challenger 推理，生成对抗文本...")
    challenger_model, challenger_tokenizer = load_model_and_tokenizer(args.challenger_model)
    challenger_messages = [
        [
            {"role": "system", "content": _build_challenger_system_prompt(t["category"])},
            {"role": "user",   "content": t["challenger_instruction"]},
        ]
        for t in tasks
    ]
    generated_texts = batch_generate(
        model=challenger_model, tokenizer=challenger_tokenizer,
        prompts_msgs=challenger_messages,
        max_new_tokens=args.max_gen_tokens, batch_size=args.batch_size,
        temperature=args.temperature, top_p=0.9, do_sample=True,
    )
    logger.info(f"   Challenger 生成完成，共 {len(generated_texts)} 条")
    free_model(challenger_model)
    del challenger_tokenizer

    # ════════════════════════════════════════════════════════════
    # Step 3.5 [Plan 3a 核心]: 拒绝采样过滤低质量生成
    # 在构建 parquet 之前过滤，保证后续所有步骤的索引对齐
    # ════════════════════════════════════════════════════════════
    logger.info(f"\n[Step 3.5] [Plan 3a] 拒绝采样 (threshold={rejection_threshold})...")
    filter_result = filter_low_quality_samples(
        tasks=tasks,
        generated_texts=generated_texts,
        threshold=rejection_threshold,
    )

    n_original = filter_result["n_original"]
    n_kept = filter_result["n_kept"]
    n_filtered = filter_result["n_filtered"]
    rejection_rate = filter_result["rejection_rate"]

    tasks = filter_result["tasks"]
    generated_texts = filter_result["generated_texts"]

    logger.info(f"   原始样本: {n_original} → 保留: {n_kept} → 过滤: {n_filtered}")
    logger.info(f"   拒绝率: {rejection_rate:.1%}")

    if n_kept == 0:
        logger.error("   ❌ 所有样本被过滤！降低 threshold 或检查 Challenger 输出质量")
        logger.info("   回退: 跳过拒绝采样，使用全部样本")
        # 重新加载未过滤的数据
        tasks_bak = build_sampling_tasks(seed_df, args.samples_per_cat)
        # 重新生成...这里简单起见直接报错退出
        raise RuntimeError("All samples rejected, please lower REJECTION_THRESHOLD")

    # ── Step 4: Verifier 推理 ──
    logger.info("\n[Step 4] Verifier 评估生成文本...")
    verifier = Verifier(
        model_path=args.verifier_model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_rev_tokens,
    )

    # ── Step 5: Reviewer 推理 ──
    logger.info("\n[Step 5] 当前 Reviewer 模型推理...")
    reviewer_model, reviewer_tokenizer = load_model_and_tokenizer(args.reviewer_model)
    reviewer_messages = []
    for gen_text in generated_texts:
        text_clean = (gen_text or "").strip()[:500]
        user_content = REVIEWER_USER_TEMPLATE.format(text=text_clean)
        reviewer_messages.append([
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ])
    reviewer_outputs = batch_generate(
        model=reviewer_model, tokenizer=reviewer_tokenizer,
        prompts_msgs=reviewer_messages,
        max_new_tokens=args.max_rev_tokens, batch_size=args.batch_size,
        temperature=0.1, top_p=0.9, do_sample=False,
    )
    free_model(reviewer_model)
    del reviewer_tokenizer

    # ── Step 6: 计算奖励信号 ──
    logger.info("\n[Step 6] 计算各类别奖励信号...")
    true_categories = [t["category"] for t in tasks]
    verifier_results = verifier.batch_verify(generated_texts)

    sample_rewards = verifier.compute_rewards_from_results(
        true_categories=true_categories,
        verifier_results=verifier_results,
        reviewer_outputs=reviewer_outputs,
    )
    verifier_stats = verifier.compute_category_stats_from_rewards(
        true_categories=true_categories,
        sample_rewards=sample_rewards,
    )
    evaluation_report = verifier.build_evaluation_report(
        true_categories=true_categories,
        sample_rewards=sample_rewards,
        category_stats=verifier_stats,
    )

    for cat, stat in verifier_stats.items():
        logger.info(f"   {cat:12s} | ASR={stat['verifier_asr']:.3f} | BinAcc={stat['reviewer_binary_acc']:.3f}")

    verifier.unload()

    # ── Step 7: 构建 parquet (修复版: 注入逐样本对抗信号) ──
    logger.info("\n[Step 7] [修复版] 构建含逐样本对抗信号的 GRPO parquet (过滤后数据)...")
    challenger_df = build_challenger_parquet_adversarial(
        tasks=tasks,
        sample_rewards=sample_rewards,
        verifier_stats=verifier_stats,
        build_system_prompt_fn=_build_challenger_system_prompt,
    )
    reviewer_df = build_reviewer_parquet(tasks, generated_texts, verifier_results)

    challenger_out = output_dir / f"challenger_grpo_round{args.round_idx}.parquet"
    reviewer_out = output_dir / f"reviewer_grpo_round{args.round_idx}.parquet"
    challenger_df.to_parquet(str(challenger_out), index=False)
    reviewer_df.to_parquet(str(reviewer_out), index=False)

    logger.info(f"   Challenger: {challenger_out} ({len(challenger_df)} 行)")
    logger.info(f"   Reviewer:   {reviewer_out} ({len(reviewer_df)} 行)")

    # ── Step 8: 统计报告 ──
    overall_asr = evaluation_report["overall"]["overall_verifier_asr"]
    stats_out = output_dir / f"selfplay_stats_round{args.round_idx}.json"
    report = {
        "round": args.round_idx,
        "plan": "plan3_data_pipeline",
        "rejection_sampling": {
            "threshold": rejection_threshold,
            "n_original": n_original,
            "n_kept": n_kept,
            "n_filtered": n_filtered,
            "rejection_rate": rejection_rate,
        },
        "total_generated": len(generated_texts),
        "overall_verifier_asr": overall_asr,
        "overall_metrics": evaluation_report["overall"],
        "verifier_stats_by_category": verifier_stats,
    }
    with open(str(stats_out), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✅ [Plan 3] Phase 0 完成！ASR={overall_asr:.3f}")
    logger.info(f"   拒绝采样: {n_filtered}/{n_original} 被过滤 ({rejection_rate:.1%})")
    print(f"CHALLENGER_GRPO_DATA={challenger_out}")
    print(f"REVIEWER_GRPO_DATA={reviewer_out}")
    print(f"SELFPLAY_STATS={stats_out}")


if __name__ == "__main__":
    main()
