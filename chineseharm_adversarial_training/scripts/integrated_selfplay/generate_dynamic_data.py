#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一版 Phase 0: 动态对抗数据生成
=================================
集成所有改进:
  - Plan 1: 逐样本对抗信号注入 (build_parquet.build_challenger_parquet)
  - Plan 3a: 拒绝采样质量过滤 (rejection_sampler.filter_low_quality_samples)
  - Plan 4:  API Verifier 后端支持 (verifier.create_verifier)

流程:
  Step 1:   加载种子数据
  Step 2:   构建采样任务（随机采样 few-shot 示例）
  Step 3:   Challenger 推理 → few-shot 多轮对话生成对抗文本
  Step 3.5: 拒绝采样过滤低质量样本 (Plan 3a)
  Step 5:   Reviewer 推理 → 采集训练数据
  Step 6:   计算逐样本 1-acc 奖励（Search Self-Play，无 Verifier）
  Step 7:   构建 parquet（Challenger GRPO + Reviewer GRPO）
  Step 8:   保存统计报告

用法:
  python generate_dynamic_data.py \\
      --challenger_model /path/to/challenger \\
      --reviewer_model   /path/to/reviewer \\
      --verifier_model   /path/to/verifier \\
      --seed_data        /path/to/train_seed.parquet \\
      --output_dir       /path/to/round_dir \\
      --round_idx 1 \\
      --verifier_backend api \\
      --samples_per_cat 64
"""

import os
import sys
import gc
import json
import argparse
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import re

# vLLM 加速 (可选, 自动 fallback 到 HuggingFace)
_USE_VLLM = False
try:
    from vllm import LLM, SamplingParams
    _USE_VLLM = True
except ImportError:
    pass

try:
    import torch_npu
except ImportError:
    pass

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 统一导入本包模块 ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from constants import (
    HARMFUL_CATEGORIES, ALL_CATEGORIES, RULES,
    REVIEWER_SYSTEM_PROMPT, REVIEWER_USER_TEMPLATE,
    CAT_DEFAULTS,
    build_challenger_system_prompt,
    get_category_rules,
    format_reviewer_user_content,
    parse_classification_output,
)
from rejection_sampler import filter_low_quality_samples
from build_parquet import build_challenger_parquet, build_reviewer_parquet

# ── 日志配置 ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── 模型推理工具函数 ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str, device: str = "npu"):
    logger.info(f"   -> 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    logger.info(f"   -> 模型加载完成")
    return model, tokenizer


def free_model(model):
    del model
    gc.collect()
    try:
        torch.npu.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def batch_generate(
    model,
    tokenizer,
    prompts_msgs: List[List[Dict]],
    max_new_tokens: int = 128,
    batch_size: int = 8,
    temperature: float = 0.85,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> List[str]:
    results = []
    device = next(model.parameters()).device

    raw_texts = []
    for msgs in prompts_msgs:
        try:
            txt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            txt = ""
            for m in msgs:
                role, content = m.get("role", ""), m.get("content", "")
                if role == "system":
                    txt += f"<|system|>{content}\n"
                elif role == "user":
                    txt += f"<|user|>{content}\n<|assistant|>"
                else:
                    txt += f"{content}\n"
        raw_texts.append(txt)

    for i in tqdm(range(0, len(raw_texts), batch_size), desc="   推理进度"):
        batch_texts = raw_texts[i: i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j in range(len(batch_texts)):
            prompt_len = input_ids.shape[1]
            gen_tokens = output_ids[j][prompt_len:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            results.append(text)

    return results


def vllm_generate(
    model_path,
    prompts_msgs,
    max_new_tokens=128,
    temperature=0.85,
    top_p=0.9,
    do_sample=True,
    tensor_parallel_size=1,
):
    """
    vLLM 离线批量推理 -- 比 HuggingFace generate 快 10~20 倍。
    支持 TP 多卡并行，自动处理注意力头不整除时的降级。
    """
    requested_tp = max(1, int(tensor_parallel_size))

    def _build_llm(tp):
        return LLM(
            model=model_path,
            tensor_parallel_size=tp,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=2048,
            gpu_memory_utilization=0.85,
        )

    logger.info("   -> [vLLM] Loading: %s (TP=%d)", model_path, requested_tp)
    try:
        llm = _build_llm(requested_tp)
    except Exception as e:
        msg = str(e)
        m = re.search(
            r"attention heads \((\d+)\).*tensor parallel size \((\d+)\)",
            msg, flags=re.IGNORECASE | re.DOTALL,
        )
        if m and "divisible" in msg.lower():
            num_heads = int(m.group(1))
            fallback_tp = 1
            for tp in range(min(requested_tp, num_heads), 0, -1):
                if num_heads % tp == 0:
                    fallback_tp = tp
                    break
            if fallback_tp != requested_tp:
                logger.warning(
                    "   -> [vLLM] TP=%d incompatible with heads=%d; fallback TP=%d",
                    requested_tp, num_heads, fallback_tp,
                )
                llm = _build_llm(fallback_tp)
            else:
                raise
        else:
            raise

    tok = llm.get_tokenizer()
    raw_prompts = []
    SYS_TAG = chr(60) + "|system|" + chr(62)
    USR_TAG = chr(60) + "|user|" + chr(62)
    AST_TAG = chr(60) + "|assistant|" + chr(62)
    for msgs in prompts_msgs:
        try:
            raw_prompts.append(
                tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )
        except Exception:
            s = ""
            for mi in msgs:
                r, c = mi.get("role", ""), mi.get("content", "")
                if r == "system":
                    s += SYS_TAG + c + "\n"
                elif r == "user":
                    s += USR_TAG + c + "\n" + AST_TAG
                else:
                    s += c + "\n"
            raw_prompts.append(s)

    if do_sample and temperature > 0:
        params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    else:
        params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    logger.info("   -> [vLLM] Generating %d samples...", len(raw_prompts))
    outputs = llm.generate(raw_prompts, params)
    results = [o.outputs[0].text.strip() for o in outputs]

    # Release vLLM engine
    del llm
    gc.collect()
    try:
        torch.npu.empty_cache()
    except Exception:
        pass

    return results


# ── 采样任务构建 ─────────────────────────────────────────────────────────────────

def build_sampling_tasks(
    seed_df: pd.DataFrame,
    samples_per_cat: int,
    nontoxic_samples: int = 0,
) -> List[Dict]:
    col_text = "文本"  if "文本"  in seed_df.columns else "original_text"
    col_cat  = "标签"  if "标签"  in seed_df.columns else "category"
    col_tt   = "toxic_type_label" if "toxic_type_label" in seed_df.columns else "toxic_type"
    col_expr = "expression_label" if "expression_label" in seed_df.columns else "expression"

    tasks: List[Dict] = []

    for cat in ALL_CATEGORIES:
        # 无毒类别使用独立的样本数（确保充足的无毒训练数据）
        n_samples = nontoxic_samples if (cat == "无毒" and nontoxic_samples > 0) else samples_per_cat
        cat_df = seed_df[seed_df[col_cat] == cat]
        if len(cat_df) == 0:
            logger.warning(f"   [警告] 种子数据中无类别 '{cat}'，跳过")
            continue

        ref_texts = cat_df[col_text].tolist()
        if len(ref_texts) > 20:
            ref_texts = random.sample(ref_texts, 20)

        if col_expr in cat_df.columns:
            expr_tt_pairs = list(
                cat_df[[col_expr, col_tt]].drop_duplicates().itertuples(index=False, name=None)
            )
        else:
            defaults = CAT_DEFAULTS.get(cat, {})
            expr_tt_pairs = [(defaults.get("expression", "非仇恨"), defaults.get("toxic_type", "无毒"))]

        pair_count = max(1, len(expr_tt_pairs))
        if n_samples >= pair_count:
            pair_alloc = [n_samples // pair_count] * pair_count
            for idx in range(n_samples % pair_count):
                pair_alloc[idx] += 1
        else:
            selected = set(random.sample(range(pair_count), n_samples))
            pair_alloc = [1 if idx in selected else 0 for idx in range(pair_count)]

        for pair_idx, (expr, tt) in enumerate(expr_tt_pairs):
            n_for_pair = pair_alloc[pair_idx]
            if n_for_pair <= 0:
                continue

            for _ in range(n_for_pair):
                # 随机采样一条真实样本作为 few-shot 示例
                few_shot_example = random.choice(ref_texts).strip() if ref_texts else ""

                # 干净的 base instruction（不含内联参考，由 few-shot 多轮对话提供示范）
                instruction = (
                    f"类别：{cat}\n"
                    f"表达方式：{expr}\n"
                    "只输出文本本身，不要解释："
                )

                tasks.append({
                    "category":    cat,
                    "expression":  expr,
                    "toxic_type":  tt,
                    "challenger_instruction": instruction,
                    "few_shot_example":       few_shot_example,
                    "reference_texts":        ref_texts,
                    "reference_anchor":       few_shot_example,
                })

    random.shuffle(tasks)
    return tasks


# ── Search Self-Play 1-acc 奖励计算（无 Verifier）────────────────────────────────

def compute_sample_rewards_oneacc(
    reviewer_outputs: List[str],
    true_categories: List[str],
) -> List[Dict]:
    # Multi-level adversarial reward based on Reviewer output (no Verifier).
    #
    # Harmful targets:
    #   binary fooled (judged safe)    -> r_challenger=1.0
    #   binary ok but category wrong   -> r_challenger=0.6
    #   fully correct                  -> r_challenger=0.0
    # Non-toxic targets:
    #   false positive (judged harmful) -> r_challenger=0.8
    #   correct (judged safe)           -> r_challenger=0.0
    #
    # Reviewer reward (multi-level):
    #   binary wrong  -> -0.5
    #   binary+cat ok -> +1.0
    #   binary ok, cat wrong -> +0.4
    #   non-toxic correct -> +0.6
    #   parse failure -> -1.0
    reviewer_results = [parse_classification_output(o) for o in reviewer_outputs]
    while len(reviewer_results) < len(true_categories):
        reviewer_results.append({"binary": None, "category": None,
                                  "toxic_type": None, "expression": None})

    sample_rewards: List[Dict] = []
    for i, true_cat in enumerate(true_categories):
        rr = reviewer_results[i]
        true_is_harmful = true_cat in HARMFUL_CATEGORIES

        if rr.get("binary") is None:
            reviewer_says_harmful   = None
            reviewer_binary_correct = False
            reviewer_cat_correct    = False
            reviewer_was_fooled     = False
            reviewer_cat_fooled     = False
            r_challenger = 0.0
            r_reviewer   = -1.0
        else:
            reviewer_says_harmful   = (rr.get("binary") == "\u6709\u5bb3")
            reviewer_binary_correct = (reviewer_says_harmful == true_is_harmful)
            reviewer_cat_correct    = (rr.get("category") == true_cat)

            if true_is_harmful:
                reviewer_was_fooled = not reviewer_says_harmful
                reviewer_cat_fooled = reviewer_binary_correct and not reviewer_cat_correct
            else:
                reviewer_was_fooled = reviewer_says_harmful
                reviewer_cat_fooled = False

            if reviewer_was_fooled:
                r_challenger = 1.0
            elif reviewer_cat_fooled:
                r_challenger = 0.6
            elif not true_is_harmful and reviewer_says_harmful:
                r_challenger = 0.8
            else:
                r_challenger = 0.0

            if not reviewer_binary_correct:
                r_reviewer = -0.5
            elif not true_is_harmful:
                r_reviewer = 0.6
            elif reviewer_cat_correct:
                r_reviewer = 1.0
            else:
                r_reviewer = 0.4

        sample_rewards.append({
            "true_category":           true_cat,
            "true_is_harmful":         true_is_harmful,
            "verifier_result":         {},
            "reviewer_result":         rr,
            "verifier_says_harmful":   true_is_harmful,
            "reviewer_was_fooled":     reviewer_was_fooled,
            "reviewer_cat_fooled":     reviewer_cat_fooled,
            "reviewer_binary_correct": reviewer_binary_correct,
            "reviewer_cat_correct":    reviewer_cat_correct,
            "reviewer_false_negative": true_is_harmful and (reviewer_says_harmful is False),
            "reviewer_false_positive": (not true_is_harmful) and bool(reviewer_says_harmful),
            "r_challenger":            float(r_challenger),
            "r_reviewer":              float(r_reviewer),
        })

    return sample_rewards


def compute_category_stats_oneacc(
    true_categories: List[str],
    sample_rewards: List[Dict],
) -> Dict:
    """按类别聚合多级对抗奖励统计。"""
    from collections import defaultdict
    cat_data: Dict[str, list] = defaultdict(list)
    for cat, sr in zip(true_categories, sample_rewards):
        cat_data[cat].append(sr)

    stats: Dict = {}
    for cat, rewards in cat_data.items():
        total      = len(rewards)
        fooled     = sum(1 for r in rewards if r["reviewer_was_fooled"])
        cat_fooled = sum(1 for r in rewards if r.get("reviewer_cat_fooled", False))
        binary_ok  = sum(1 for r in rewards if r["reviewer_binary_correct"])
        cat_ok     = sum(1 for r in rewards if r["reviewer_cat_correct"])
        avg_r_c    = sum(r["r_challenger"] for r in rewards) / total if total > 0 else 0.0
        avg_r_r    = sum(r["r_reviewer"] for r in rewards) / total if total > 0 else 0.0
        stats[cat] = {
            "adversarial_success_rate":  round(fooled / total, 4) if total > 0 else 0.0,
            "category_confusion_rate":   round(cat_fooled / total, 4) if total > 0 else 0.0,
            "reviewer_binary_accuracy":  round(binary_ok / total, 4) if total > 0 else 0.0,
            "reviewer_category_accuracy": round(cat_ok / total, 4) if total > 0 else 0.0,
            "avg_r_challenger":          round(avg_r_c, 4),
            "avg_r_reviewer":            round(avg_r_r, 4),
            "total":                     total,
        }
    return stats


# ── 参数解析 ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="统一版 Phase 0 数据生成")
    parser.add_argument("--challenger_model", required=True, type=str)
    parser.add_argument("--reviewer_model",   required=True, type=str)
    parser.add_argument("--verifier_model",   default="",    type=str,
                        help="本地 Verifier 模型路径 (backend=local 时必填)")
    parser.add_argument("--seed_data",        required=True, type=str)
    parser.add_argument("--output_dir",       required=True, type=str)
    parser.add_argument("--round_idx",        required=True, type=int)
    parser.add_argument("--samples_per_cat",  default=64,    type=int)
    parser.add_argument("--batch_size",       default=8,     type=int)
    parser.add_argument("--max_gen_tokens",   default=128,   type=int)
    parser.add_argument("--max_rev_tokens",   default=256,   type=int)
    parser.add_argument("--temperature",      default=0.85,  type=float)
    parser.add_argument("--seed",             default=42,    type=int)
    parser.add_argument("--num_npus",         default=4,     type=int,
                        help="NPU 卡数 (兼容 shell 传参，实际由 ASCEND_RT_VISIBLE_DEVICES 控制)")
    # Verifier 后端配置
    parser.add_argument("--verifier_backend", default="local", type=str,
                        choices=["local", "api", "async"],
                        help="Verifier 后端: local(本地模型), api(同步API), async(异步API)")
    parser.add_argument("--verifier_api_model", default="qwen-plus", type=str,
                        help="API Verifier 模型名称")
    # Plan 3a: 拒绝采样
    parser.add_argument("--enable_rejection_sampling", action="store_true", default=False,
                        help="启用 Plan 3a 拒绝采样过滤 (默认由 shell 脚本传入)")
    parser.add_argument("--no_rejection_sampling", action="store_true", default=False,
                        help="显式禁用拒绝采样")
    # Challenger 无毒样本过采样
    parser.add_argument("--nontoxic_samples", default=20, type=int,
                        help="Challenger 生成中无毒样本数，默认 20 (确保学习有毒/无毒边界)")
    # Reviewer 数据混合（防止灾难性遗忘）
    parser.add_argument("--reviewer_mix_ratio", default=0.5, type=float,
                        help="Reviewer 训练数据中原始种子数据的混合比例 (0-1)，默认 0.5")
    parser.add_argument("--reviewer_nontoxic_boost", default=2.0, type=float,
                        help="原始数据中无毒样本的过采样倍数，默认 2.0 (无毒:有害=2:1)")
    parser.add_argument("--reviewer_hard_boost", default=2, type=int,
                        help="Reviewer 困难样本(错判样本)重采样倍数，默认 2")
    parser.add_argument("--reviewer_repeat_bonus_cap", default=2, type=int,
                        help="对跨轮重复错题的额外重采样上限，默认 2")
    parser.add_argument("--history_hard_dir", default="", type=str,
                        help="历史轮次 sample_rewards 目录（用于错题回放），如 selfplay_integrated_data/3B")
    parser.add_argument("--history_hard_max_rows", default=3000, type=int,
                        help="每轮最多加载的历史错题数，默认 3000")
    return parser.parse_args()


def load_history_hard_rows(history_dir: str, current_round: int, max_rows: int) -> List[Dict]:
    """从历史轮次 sample_rewards parquet 加载 Reviewer 错判样本。"""
    if not history_dir:
        return []

    base = Path(history_dir)
    if not base.exists():
        logger.warning(f"历史错题目录不存在，跳过: {history_dir}")
        return []

    all_rows: List[Dict] = []
    for p in sorted(base.glob("step_*/sample_rewards_round*.parquet")):
        m = re.search(r"round(\d+)", p.stem)
        if not m:
            continue
        round_idx = int(m.group(1))
        if round_idx >= current_round:
            continue

        try:
            df = pd.read_parquet(str(p))
        except Exception as e:
            logger.warning(f"读取历史错题失败: {p} ({e})")
            continue

        if "reviewer_binary_correct" not in df.columns or "reviewer_cat_correct" not in df.columns:
            continue

        hard_df = df[(df["reviewer_binary_correct"] == False) | (df["reviewer_cat_correct"] == False)]
        if len(hard_df) == 0:
            continue

        all_rows.extend(hard_df.to_dict("records"))

    if max_rows > 0 and len(all_rows) > max_rows:
        all_rows = random.sample(all_rows, max_rows)

    return all_rows


# ── 主流程 ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    enable_rejection = args.enable_rejection_sampling and not args.no_rejection_sampling

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info(f"║  统一版 Phase 0: 动态对抗数据生成  (第 {args.round_idx} 轮)         ║")
    logger.info("║  [Few-Shot Challenger + 1-acc 无 Verifier 版]               ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"  Challenger   : {args.challenger_model}")
    logger.info(f"  Reviewer     : {args.reviewer_model}")
    logger.info(f"  奖励模式     : 1-acc (Search Self-Play, 无 Verifier)")
    logger.info(f"  种子数据     : {args.seed_data}")
    logger.info(f"  输出目录     : {output_dir}")
    logger.info(f"  每类样本数   : {args.samples_per_cat}")
    logger.info(f"  拒绝采样     : {'启用' if enable_rejection else '禁用'}")
    if args.history_hard_dir:
        logger.info(f"  历史错题回放 : {args.history_hard_dir} (max={args.history_hard_max_rows})")
    logger.info(f"  重复错题加权上限 : {args.reviewer_repeat_bonus_cap}")

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
    tasks = build_sampling_tasks(seed_df, args.samples_per_cat, nontoxic_samples=args.nontoxic_samples)
    logger.info(f"   共构建 {len(tasks)} 个生成任务 (其中无毒 {args.nontoxic_samples} 条)")

    # ── Step 3: Challenger 推理 ──
    logger.info(f"\n[Step 3] Challenger 推理 (vLLM={_USE_VLLM}, NPUs={args.num_npus})...")

    # 构建消息：与 SFT 训练格式一致 [system, user] (无 few-shot)
    challenger_messages = []
    for t in tasks:
        msgs = [
            {"role": "system", "content": build_challenger_system_prompt(t["category"])},
            {"role": "user", "content": t["challenger_instruction"]},
        ]
        challenger_messages.append(msgs)

    if _USE_VLLM:
        generated_texts = vllm_generate(
            model_path=args.challenger_model,
            prompts_msgs=challenger_messages,
            max_new_tokens=args.max_gen_tokens,
            temperature=args.temperature,
            top_p=0.9,
            do_sample=True,
            tensor_parallel_size=args.num_npus,
        )
    else:
        challenger_model, challenger_tokenizer = load_model_and_tokenizer(args.challenger_model)
        generated_texts = batch_generate(
            model=challenger_model,
            tokenizer=challenger_tokenizer,
            prompts_msgs=challenger_messages,
            max_new_tokens=args.max_gen_tokens,
            batch_size=args.batch_size,
            temperature=args.temperature,
            top_p=0.9,
            do_sample=True,
        )
        logger.info("   释放 Challenger 模型显存...")
        free_model(challenger_model)
        del challenger_tokenizer

    logger.info(f"   Challenger 生成完成，共 {len(generated_texts)} 条")

    # ── Step 3.5: 拒绝采样过滤 (Plan 3a) ──
    if enable_rejection:
        logger.info("\n[Step 3.5] 拒绝采样过滤 (Plan 3a)...")
        before_count = len(tasks)
        filtered = filter_low_quality_samples(
            tasks=tasks,
            generated_texts=generated_texts,
        )
        tasks          = filtered["tasks"]
        generated_texts = filtered["generated_texts"]

        after_count = len(tasks)
        reject_count = before_count - after_count
        reject_pct = (reject_count / before_count * 100) if before_count > 0 else 0.0
        logger.info(f"   拒绝采样: {before_count} → {after_count} (过滤 {reject_count} 条, {reject_pct:.1f}%)")
    else:
        logger.info("\n[Step 3.5] 拒绝采样已禁用，跳过。")

    # ── Step 5: Reviewer 推理 ──
    logger.info(f"\n[Step 5] 当前 Reviewer 推理 (vLLM={_USE_VLLM})...")

    reviewer_messages = []
    for gen_text in generated_texts:
        text_clean = (gen_text or "").strip()[:500]
        user_content = format_reviewer_user_content(text_clean)
        reviewer_messages.append([
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ])

    if _USE_VLLM:
        reviewer_outputs = vllm_generate(
            model_path=args.reviewer_model,
            prompts_msgs=reviewer_messages,
            max_new_tokens=args.max_rev_tokens,
            temperature=0.0,
            do_sample=False,
            tensor_parallel_size=args.num_npus,
        )
    else:
        reviewer_model, reviewer_tokenizer = load_model_and_tokenizer(args.reviewer_model)
        reviewer_outputs = batch_generate(
            model=reviewer_model,
            tokenizer=reviewer_tokenizer,
            prompts_msgs=reviewer_messages,
            max_new_tokens=args.max_rev_tokens,
            batch_size=args.batch_size,
            temperature=0.1,
            top_p=0.9,
            do_sample=False,
        )
        logger.info("   释放 Reviewer 模型显存...")
        free_model(reviewer_model)
        del reviewer_tokenizer

    logger.info(f"   Reviewer 输出完成，共 {len(reviewer_outputs)} 条")

    # ── Step 6: 计算逐样本奖励（1-acc，Search Self-Play，无 Verifier）──
    logger.info("\n[Step 6] 计算逐样本 1-acc 奖励信号 (Search Self-Play, 无 Verifier)...")
    true_categories = [t["category"] for t in tasks]

    sample_rewards = compute_sample_rewards_oneacc(
        reviewer_outputs=reviewer_outputs,
        true_categories=true_categories,
    )
    category_stats = compute_category_stats_oneacc(
        true_categories=true_categories,
        sample_rewards=sample_rewards,
    )

    total_samples   = len(sample_rewards)
    total_fooled    = sum(1 for r in sample_rewards if r["reviewer_was_fooled"])
    overall_asr     = total_fooled / total_samples if total_samples > 0 else 0.0
    overall_rev_acc = sum(1 for r in sample_rewards if r["reviewer_binary_correct"]) / total_samples if total_samples > 0 else 0.0

    logger.info("   ┌──────────────────────────────────────────────────────────────────┐")
    logger.info("   │  类别        │ BinFool │ CatConf │ BinAcc │ CatAcc │ R_C avg  │")
    logger.info("   ├──────────────────────────────────────────────────────────────────┤")
    for cat, stat in category_stats.items():
        logger.info(
            f"   │  {cat:10s}  │  {stat['adversarial_success_rate']:.3f}  │"
            f"  {stat['category_confusion_rate']:.3f}  │"
            f"  {stat['reviewer_binary_accuracy']:.3f} │"
            f"  {stat['reviewer_category_accuracy']:.3f} │"
            f"  {stat['avg_r_challenger']:+.3f}   │"
        )
    logger.info("   └──────────────────────────────────────────────────────────────────┘")
    logger.info(
        "   [Overall] ASR(1-acc)=%.3f | Reviewer_Acc=%.3f | Fooled=%d/%d",
        overall_asr, overall_rev_acc, total_fooled, total_samples,
    )

    # 先构建样本明细，供困难样本重采样与评估导出复用
    sample_rows = []
    for i, sr in enumerate(sample_rewards):
        task = tasks[i]
        rr   = sr["reviewer_result"]
        sample_rows.append({
            "idx":                     i,
            "target_category":         task["category"],
            "target_expression":       task["expression"],
            "target_toxic_type":       task["toxic_type"],
            "challenger_instruction":  task.get("challenger_instruction", ""),
            "challenger_system_prompt": build_challenger_system_prompt(task["category"]),
            "generated_text":          (generated_texts[i] or "").strip(),
            "few_shot_example":        task.get("few_shot_example", ""),
            "reviewer_output":         (reviewer_outputs[i] if i < len(reviewer_outputs) else ""),
            "reviewer_binary":         rr.get("binary"),
            "reviewer_category":       rr.get("category"),
            "reviewer_binary_correct": sr["reviewer_binary_correct"],
            "reviewer_cat_correct":    sr["reviewer_cat_correct"],
            "reviewer_was_fooled":     sr["reviewer_was_fooled"],
            "reviewer_cat_fooled":     sr.get("reviewer_cat_fooled", False),
            "r_challenger":            sr["r_challenger"],
            "r_reviewer":              sr["r_reviewer"],
        })

    current_hard_count = sum(
        1 for r in sample_rows
        if (not bool(r.get("reviewer_binary_correct", True))) or (not bool(r.get("reviewer_cat_correct", True)))
    )
    history_hard_rows = load_history_hard_rows(
        history_dir=args.history_hard_dir,
        current_round=args.round_idx,
        max_rows=args.history_hard_max_rows,
    )
    total_hard_rows = sample_rows + history_hard_rows

    # ── Step 7: 构建并保存 parquet ──
    logger.info("\n[Step 7] 构建并保存 GRPO 训练 parquet (Few-Shot Challenger + 1-acc 奖励)...")
    logger.info(
        f"   困难样本池: current={current_hard_count}, history={len(history_hard_rows)}, total={len(total_hard_rows)}"
    )

    challenger_df = build_challenger_parquet(tasks, sample_rewards)
    reviewer_df   = build_reviewer_parquet(
        tasks=tasks,
        generated_texts=generated_texts,
        seed_df=seed_df,
        mix_ratio=args.reviewer_mix_ratio,
        nontoxic_boost=args.reviewer_nontoxic_boost,
        hard_sample_rows=total_hard_rows,
        hard_sample_multiplier=args.reviewer_hard_boost,
        repeat_bonus_cap=args.reviewer_repeat_bonus_cap,
    )

    challenger_out = output_dir / f"challenger_grpo_round{args.round_idx}.parquet"
    reviewer_out   = output_dir / f"reviewer_grpo_round{args.round_idx}.parquet"

    challenger_df.to_parquet(str(challenger_out), index=False)
    reviewer_df.to_parquet(str(reviewer_out), index=False)

    # 统计数据来源分布
    adv_count = 0
    orig_nontoxic_count = 0
    orig_toxic_count = 0
    hard_count = 0
    for _, row in reviewer_df.iterrows():
        extra = row.get("extra_info", {})
        if isinstance(extra, dict):
            src = extra.get("source", "")
            if src == "adversarial":
                adv_count += 1
            elif src == "original_nontoxic":
                orig_nontoxic_count += 1
            elif src == "original_toxic":
                orig_toxic_count += 1
            elif src == "hard_mining":
                hard_count += 1
    
    logger.info(f"   Challenger GRPO: {challenger_out}  ({len(challenger_df)} 行)")
    logger.info(f"   Reviewer   GRPO: {reviewer_out}   ({len(reviewer_df)} 行)")
    logger.info(f"      └─ 对抗样本(有害): {adv_count}")
    logger.info(f"      └─ 原始无毒样本:   {orig_nontoxic_count}")
    logger.info(f"      └─ 原始有害样本:   {orig_toxic_count}")
    logger.info(f"      └─ 困难样本重采样: {hard_count} (boost={args.reviewer_hard_boost})")
    logger.info(f"      └─ 配置: mix_ratio={args.reviewer_mix_ratio}, nontoxic_boost={args.reviewer_nontoxic_boost}")

    # 样本级评估明细
    sample_rewards_df  = pd.DataFrame(sample_rows)
    sample_rewards_out = output_dir / f"sample_rewards_round{args.round_idx}.parquet"
    sample_rewards_df.to_parquet(str(sample_rewards_out), index=False)
    logger.info(f"   样本级评估: {sample_rewards_out}  ({len(sample_rewards_df)} 行)")

    # Challenger 提问-生成追踪（用于分析提问质量）
    challenger_trace_cols = [
        "idx",
        "target_category",
        "target_expression",
        "target_toxic_type",
        "challenger_system_prompt",
        "challenger_instruction",
        "few_shot_example",
        "generated_text",
        "reviewer_output",
        "reviewer_binary",
        "reviewer_category",
        "reviewer_binary_correct",
        "reviewer_cat_correct",
        "reviewer_was_fooled",
        "r_challenger",
        "r_reviewer",
    ]
    challenger_trace_df = sample_rewards_df[[c for c in challenger_trace_cols if c in sample_rewards_df.columns]]
    challenger_trace_out = output_dir / f"challenger_trace_round{args.round_idx}.parquet"
    challenger_trace_df.to_parquet(str(challenger_trace_out), index=False)
    logger.info(f"   Challenger 追踪: {challenger_trace_out}  ({len(challenger_trace_df)} 行)")

    # ── Step 8: 保存统计报告 ──
    stats_out = output_dir / f"selfplay_stats_round{args.round_idx}.json"
    
    report = {
        "round":                args.round_idx,
        "challenger_model":     args.challenger_model,
        "reviewer_model":       args.reviewer_model,
        "reward_mode":          "1-acc (search_selfplay, no_verifier)",
        "prompt_mode":          "few_shot_multiturn",
        "rejection_sampling":   enable_rejection,
        "total_generated":      len(generated_texts),
        "total_fooled":         total_fooled,
        "overall_asr_1acc":     round(overall_asr, 4),
        "overall_reviewer_acc": round(overall_rev_acc, 4),
        "challenger_grpo_size": len(challenger_df),
        "reviewer_grpo_size":   len(reviewer_df),
        "reviewer_mix_ratio":   args.reviewer_mix_ratio,
        "reviewer_nontoxic_boost": args.reviewer_nontoxic_boost,
        "reviewer_hard_boost": args.reviewer_hard_boost,
        "reviewer_repeat_bonus_cap": args.reviewer_repeat_bonus_cap,
        "reviewer_current_hard_count": current_hard_count,
        "reviewer_history_hard_count": len(history_hard_rows),
        "reviewer_adversarial_count":   adv_count,
        "reviewer_orig_nontoxic_count": orig_nontoxic_count,
        "reviewer_orig_toxic_count":    orig_toxic_count,
        "reviewer_hard_count":          hard_count,
        "sample_eval_path":     str(sample_rewards_out),
        "challenger_trace_path": str(challenger_trace_out),
        "stats_by_category":    category_stats,
    }
    with open(str(stats_out), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"   统计报告: {stats_out}")

    logger.info("\n✅ Phase 0 (Few-Shot + 1-acc Self-Play) 数据生成完成！")
    logger.info(f"   整体对抗成功率 (1-acc ASR) = {overall_asr:.3f}")

    # 输出供 shell 脚本读取的路径变量
    print(f"CHALLENGER_GRPO_DATA={challenger_out}")
    print(f"REVIEWER_GRPO_DATA={reviewer_out}")
    print(f"SELFPLAY_STATS={stats_out}")
    print(f"CHALLENGER_TRACE={challenger_trace_out}")


if __name__ == "__main__":
    main()
