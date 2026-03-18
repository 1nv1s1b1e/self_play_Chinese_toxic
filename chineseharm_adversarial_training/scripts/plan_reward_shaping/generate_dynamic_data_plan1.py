#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan 1 专用: 修改版 generate_dynamic_data — 注入逐样本对抗信号
==============================================================

与 v1 generate_dynamic_data.py 的区别:
  Step 7 中调用 build_challenger_parquet_adversarial 替代 build_challenger_parquet,
  将逐样本的 reviewer_fooled / reviewer_binary_ok 注入 Challenger GRPO parquet。

实现方式: 完整重写 Phase 0 主流程，Step 7 使用 build_challenger_parquet_adversarial 注入逐样本信号。
"""

import os
import sys

# ── 路径设置 ──
PLAN1_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(os.path.dirname(PLAN1_DIR), "rl_train")
sys.path.insert(0, V1_DIR)
sys.path.insert(0, PLAN1_DIR)


def main():
    """
    完整重写 main()，与 v1 逻辑相同，但 Step 7 使用 Plan 1 的 parquet builder。
    """
    import argparse
    import random
    import logging
    from pathlib import Path

    import torch
    import torch_npu
    import pandas as pd

    from build_parquet_adversarial import build_challenger_parquet_adversarial

    logger = logging.getLogger("plan1_datagen")

    # 复用 v1 的所有工具函数
    from generate_dynamic_data import (
        parse_args, build_sampling_tasks,
        load_model_and_tokenizer, batch_generate, free_model,
        build_reviewer_parquet,
        REVIEWER_USER_TEMPLATE, REVIEWER_SYSTEM_PROMPT,
        HARMFUL_CATEGORIES, ALL_CATEGORIES,
        _build_challenger_system_prompt,
    )
    from verifier import Verifier

    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info(f"║  [Plan 1] Phase 0: 动态对抗数据生成 (第 {args.round_idx} 轮)     ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"  [Plan 1 改动] Challenger parquet 将注入逐样本 reviewer_fooled 信号")

    # ── Step 1: 加载种子数据 ──
    logger.info("\n[Step 1] 加载种子数据...")
    if args.seed_data.endswith(".parquet"):
        seed_df = pd.read_parquet(args.seed_data)
    elif args.seed_data.endswith(".json"):
        seed_df = pd.read_json(args.seed_data)
    else:
        raise ValueError(f"不支持的种子数据格式: {args.seed_data}")
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

    # ── Step 6: 计算逐样本奖励 ──
    logger.info("\n[Step 6] 计算逐样本奖励信号...")
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

    # 打印统计
    for cat, stat in verifier_stats.items():
        fn_or_fp = stat.get("reviewer_fn_rate", 0) if cat in HARMFUL_CATEGORIES else stat.get("reviewer_fp_rate", 0)
        logger.info(f"   {cat:12s} | ASR={stat['verifier_asr']:.3f} | BinAcc={stat['reviewer_binary_acc']:.3f} | FN/FP={fn_or_fp:.3f}")

    # 统计逐样本 fooled 数量
    n_fooled = sum(1 for sr in sample_rewards if sr.get("reviewer_was_fooled", False))
    logger.info(f"\n   [Plan 1] 逐样本统计: {n_fooled}/{len(sample_rewards)} 条被 Reviewer 漏检 (fooled)")

    verifier.unload()

    # ════════════════════════════════════════════════════════════
    # Step 7: 构建 parquet — Plan 1 核心改动
    # 使用 build_challenger_parquet_adversarial 注入逐样本信号
    # ════════════════════════════════════════════════════════════
    logger.info("\n[Step 7] [Plan 1] 构建含逐样本对抗信号的 Challenger GRPO parquet...")

    challenger_df = build_challenger_parquet_adversarial(
        tasks=tasks,
        sample_rewards=sample_rewards,
        verifier_stats=verifier_stats,
        build_system_prompt_fn=_build_challenger_system_prompt,
    )

    # Reviewer parquet (与 v1 完全相同)
    reviewer_df = build_reviewer_parquet(tasks, generated_texts, verifier_results)

    challenger_out = output_dir / f"challenger_grpo_round{args.round_idx}.parquet"
    reviewer_out   = output_dir / f"reviewer_grpo_round{args.round_idx}.parquet"
    challenger_df.to_parquet(str(challenger_out), index=False)
    reviewer_df.to_parquet(str(reviewer_out), index=False)

    logger.info(f"   Challenger GRPO: {challenger_out}  ({len(challenger_df)} 行)")
    logger.info(f"   Reviewer   GRPO: {reviewer_out}   ({len(reviewer_df)} 行)")

    # 验证逐样本信号确实被注入
    if len(challenger_df) > 0:
        sample_extra = challenger_df.iloc[0]["extra_info"]
        if isinstance(sample_extra, dict):
            logger.info(f"   [Plan 1 验证] extra_info 含 reviewer_fooled={sample_extra.get('reviewer_fooled', 'MISSING')}")
            logger.info(f"   [Plan 1 验证] extra_info 含 reviewer_binary_ok={sample_extra.get('reviewer_binary_ok', 'MISSING')}")

    # ── Step 7.5: 样本级明细 (与 v1 兼容) ──
    import json
    sample_rows = []
    for i, sr in enumerate(sample_rewards):
        task = tasks[i]
        vr = sr["verifier_result"]
        rr = sr["reviewer_result"]
        sample_rows.append({
            "idx": i,
            "target_category": task["category"],
            "generated_text": (generated_texts[i] or "").strip(),
            "reviewer_was_fooled": sr.get("reviewer_was_fooled", False),
            "reviewer_binary_correct": sr.get("reviewer_binary_correct", True),
            "r_challenger": sr["r_challenger"],
            "r_reviewer": sr["r_reviewer"],
        })
    sample_df = pd.DataFrame(sample_rows)
    sample_out = output_dir / f"sample_rewards_round{args.round_idx}.parquet"
    sample_df.to_parquet(str(sample_out), index=False)

    # ── Step 8: 统计报告 ──
    stats_out = output_dir / f"selfplay_stats_round{args.round_idx}.json"
    overall_asr = evaluation_report["overall"]["overall_verifier_asr"]
    report = {
        "round": args.round_idx,
        "plan": "plan1_reward_shaping",
        "challenger_model": args.challenger_model,
        "reviewer_model": args.reviewer_model,
        "total_generated": len(generated_texts),
        "n_reviewer_fooled": n_fooled,
        "overall_verifier_asr": overall_asr,
        "overall_metrics": evaluation_report["overall"],
        "verifier_stats_by_category": verifier_stats,
    }
    with open(str(stats_out), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✅ [Plan 1] Phase 0 完成！ ASR={overall_asr:.3f}, Fooled={n_fooled}/{len(sample_rewards)}")
    print(f"CHALLENGER_GRPO_DATA={challenger_out}")
    print(f"REVIEWER_GRPO_DATA={reviewer_out}")
    print(f"SELFPLAY_STATS={stats_out}")


if __name__ == "__main__":
    main()
