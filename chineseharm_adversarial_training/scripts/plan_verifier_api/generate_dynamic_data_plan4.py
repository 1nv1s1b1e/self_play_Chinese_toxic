#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan 4 专用: 修改版 generate_dynamic_data — 使用 API Verifier
==============================================================

与 v1 的区别:
  Step 4: 使用 APIVerifier 替代本地冻结 7B Verifier
  其他步骤完全相同

好处:
  1. 无需加载 7B 模型到 NPU → Phase 0 更快、显存更低
  2. 72B+ API 模型标签更准确 → ground truth 更可靠
  3. NPU 显存全部留给 Challenger + Reviewer 推理
"""

import os
import sys

PLAN4_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.path.join(os.path.dirname(PLAN4_DIR), "rl_train")
sys.path.insert(0, V1_DIR)
sys.path.insert(0, PLAN4_DIR)


def main():
    """
    Phase 0 主流程，Step 4 使用 API Verifier。
    """
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
    from build_parquet_adversarial import build_challenger_parquet_adversarial

    # Plan 4: API Verifier
    from api_verifier import APIVerifier, APIVerifierConfig

    logger = logging.getLogger("plan4_datagen")

    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info(f"║  [Plan 4] Phase 0: API Verifier (第 {args.round_idx} 轮)          ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

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
    # Step 4: API Verifier (Plan 4 核心改动)
    # 替代 v1 的本地 7B Verifier，无需加载模型到 NPU
    # ════════════════════════════════════════════════════════════
    logger.info("\n[Step 4] [Plan 4] API Verifier 评估生成文本...")
    logger.info(f"   API: {os.environ.get('VERIFIER_API_BASE', 'default')}")
    logger.info(f"   模型: {os.environ.get('VERIFIER_API_MODEL', 'qwen-plus')}")

    try:
        api_verifier = APIVerifier()
        verifier_results = api_verifier.batch_verify(generated_texts)
        use_api = True
    except Exception as e:
        logger.warning(f"   ⚠️  API Verifier 失败: {e}")
        logger.warning(f"   ⚠️  降级到本地 7B Verifier...")
        from verifier import Verifier
        local_verifier = Verifier(
            model_path=args.verifier_model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_rev_tokens,
        )
        verifier_results = local_verifier.batch_verify(generated_texts)
        use_api = False

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
    logger.info("\n[Step 6] 计算奖励信号...")
    true_categories = [t["category"] for t in tasks]

    if use_api:
        sample_rewards = api_verifier.compute_rewards_from_results(
            true_categories, verifier_results, reviewer_outputs
        )
        verifier_stats = api_verifier.compute_category_stats_from_rewards(
            true_categories, sample_rewards
        )
        evaluation_report = api_verifier.build_evaluation_report(
            true_categories, sample_rewards, verifier_stats
        )
    else:
        sample_rewards = local_verifier.compute_rewards_from_results(
            true_categories, verifier_results, reviewer_outputs
        )
        verifier_stats = local_verifier.compute_category_stats_from_rewards(
            true_categories, sample_rewards
        )
        evaluation_report = local_verifier.build_evaluation_report(
            true_categories, sample_rewards, verifier_stats
        )

    for cat, stat in verifier_stats.items():
        logger.info(f"   {cat:12s} | ASR={stat['verifier_asr']:.3f} | BinAcc={stat['reviewer_binary_acc']:.3f}")

    if use_api:
        api_verifier.unload()
    else:
        local_verifier.unload()

    # ── Step 7: 构建 parquet (修复版: 注入逐样本对抗信号) ──
    logger.info("\n[Step 7] [修复版] 构建含逐样本对抗信号的 GRPO parquet...")
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
    stats_out = output_dir / f"selfplay_stats_round{args.round_idx}.json"
    overall_asr = evaluation_report["overall"]["overall_verifier_asr"]
    report = {
        "round": args.round_idx,
        "plan": "plan4_verifier_api",
        "verifier_type": "api" if use_api else "local_7b_fallback",
        "api_model": os.environ.get("VERIFIER_API_MODEL", "qwen-plus") if use_api else args.verifier_model,
        "total_generated": len(generated_texts),
        "overall_verifier_asr": overall_asr,
        "overall_metrics": evaluation_report["overall"],
        "verifier_stats_by_category": verifier_stats,
    }
    with open(str(stats_out), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✅ [Plan 4] Phase 0 完成！ASR={overall_asr:.3f}")
    logger.info(f"   Verifier: {'API (' + os.environ.get('VERIFIER_API_MODEL', 'qwen-plus') + ')' if use_api else '本地 7B (降级)'}")
    print(f"CHALLENGER_GRPO_DATA={challenger_out}")
    print(f"REVIEWER_GRPO_DATA={reviewer_out}")
    print(f"SELFPLAY_STATS={stats_out}")


if __name__ == "__main__":
    main()
