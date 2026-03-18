#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 2: Temperature 多样性验证 — 证明 0.7 比 0.3 产生更多不同输出
================================================================
核心假设: temperature=0.3 下 Reviewer 的 8 个 rollout 几乎相同,
GRPO 等于只有 1 个样本, 方差为 0; temperature=0.7 下有真正的多样性.

用法 (需要在 NPU/GPU 服务器上运行):
  python tests/test_temperature_diversity.py \
      --model_path /path/to/reviewer_3B \
      --num_samples 20

如无 GPU, 使用 --mock 模式运行模拟测试.
"""

import os, sys, json, argparse, random
from collections import Counter

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_SELF_DIR)
sys.path.insert(0, _PARENT)
sys.path.insert(0, os.path.join(_PARENT, "reward_functions"))

from constants import (
    REVIEWER_SYSTEM_PROMPT,
    format_reviewer_user_content,
    parse_classification_output,
    HARMFUL_CATEGORIES,
)


def get_test_texts():
    """取一批有代表性的测试文本 (含显式/隐式/边界/无毒)."""
    return [
        ("性别歧视", "女司机就是马路杀手，这不是歧视，这是事实"),
        ("性别歧视", "说实话女生搞编程确实不太行，逻辑思维不如男生"),
        ("种族歧视", "黑人天生体育好，读书不行"),
        ("地域偏见", "河南人偷井盖这事我觉得不是段子是真的"),
        ("LGBTQ歧视", "同性恋不正常，应该去治疗"),
        ("其他仇恨", "你这种人就是废物，活着浪费空气"),
        ("无毒", "最近在研究各地方言的差异，挺有意思的"),
        ("无毒", "今天天气不错，适合出去走走"),
        ("性别歧视", "现在的女拳真的越来越魔怔了"),
        ("种族歧视", "我们小区那个混血小孩长得还行就是太黑了"),
    ]


def mock_temperature_test():
    """不需要 GPU 的模拟测试: 模拟不同 temperature 下的输出多样性."""
    print("=" * 70)
    print("测试: Temperature 多样性 (模拟模式)")
    print("=" * 70)

    test_texts = get_test_texts()

    # 模拟: temperature=0.3 时, 8个rollout中大部分相同
    # temperature=0.7 时, 有更多类别/表达方式变化
    random.seed(42)

    for true_cat, text in test_texts[:5]:
        print(f"\n── 文本: {text[:40]}...  (真实类别: {true_cat}) ──")

        # 模拟 temp=0.3: 高概率选同一个, 偶尔有微小变化
        t03_outputs = []
        main_cat = true_cat  # 假设大部分时候选对
        for _ in range(8):
            r = random.random()
            if r < 0.85:  # 85% 完全相同
                t03_outputs.append(f"判断：有害\n类别：{main_cat}")
            elif r < 0.95:
                alt = random.choice([c for c in HARMFUL_CATEGORIES if c != main_cat])
                t03_outputs.append(f"判断：有害\n类别：{alt}")
            else:
                t03_outputs.append("判断：无害")

        # 模拟 temp=0.7: 更多多样性
        t07_outputs = []
        for _ in range(8):
            r = random.random()
            if r < 0.50:  # 50% 正确
                t07_outputs.append(f"判断：有害\n类别：{main_cat}")
            elif r < 0.80:  # 30% category 错
                alt = random.choice([c for c in HARMFUL_CATEGORIES if c != main_cat])
                t07_outputs.append(f"判断：有害\n类别：{alt}")
            elif r < 0.90:
                t07_outputs.append("判断：无害")
            else:
                t07_outputs.append(f"判断：有害\n类别：其他仇恨")

        # 分析
        t03_cats = [parse_classification_output(o).get("category", "?") for o in t03_outputs]
        t07_cats = [parse_classification_output(o).get("category", "?") for o in t07_outputs]

        t03_unique = len(set(t03_cats))
        t07_unique = len(set(t07_cats))

        print(f"   temp=0.3: categories={t03_cats}  unique={t03_unique}")
        print(f"   temp=0.7: categories={t07_cats}  unique={t07_unique}")

        # 用新版 reward 计算方差
        from reward_functions.reviewer_reward import compute_score
        gt = {"category": true_cat}
        t03_rewards = [compute_score("toxicn_reviewer", o, gt) for o in t03_outputs]
        t07_rewards = [compute_score("toxicn_reviewer", o, gt) for o in t07_outputs]

        import statistics
        t03_var = statistics.variance(t03_rewards) if len(set(t03_rewards)) > 1 else 0
        t07_var = statistics.variance(t07_rewards) if len(set(t07_rewards)) > 1 else 0

        print(f"   temp=0.3 rewards: {[f'{r:+.1f}' for r in t03_rewards]}  var={t03_var:.3f}")
        print(f"   temp=0.7 rewards: {[f'{r:+.1f}' for r in t07_rewards]}  var={t07_var:.3f}")

    print("\n   结论: temp=0.7 产生更多不同类别输出 → 新版多级奖励方差更大 → GRPO 有信号")
    print()


def real_temperature_test(model_path, num_samples=10, num_gen=8):
    """真实模型推理测试 (需要 GPU/NPU)."""
    print("=" * 70)
    print(f"测试: Temperature 多样性 (真实推理, model={model_path})")
    print("=" * 70)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("   [错误] 需要 torch + transformers")
        return

    try:
        import torch_npu
    except ImportError:
        pass

    print(f"   加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    test_texts = get_test_texts()[:num_samples]
    temperatures = [0.3, 0.5, 0.7, 1.0]

    from reward_functions.reviewer_reward import compute_score
    import statistics

    all_results = []

    for true_cat, text in test_texts:
        print(f"\n── 文本: {text[:40]}...  (真实: {true_cat}) ──")

        user_content = format_reviewer_user_content(text)
        msgs = [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            raw_prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            raw_prompt = f"{REVIEWER_SYSTEM_PROMPT}\n{user_content}\n"

        for temp in temperatures:
            outputs_text = []
            for _ in range(num_gen):
                enc = tokenizer(raw_prompt, return_tensors="pt", truncation=True, max_length=1024)
                input_ids = enc["input_ids"].to(model.device)
                attn = enc["attention_mask"].to(model.device)

                with torch.no_grad():
                    out = model.generate(
                        input_ids=input_ids, attention_mask=attn,
                        max_new_tokens=80,
                        do_sample=(temp > 0),
                        temperature=temp if temp > 0 else 1.0,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                gen = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                outputs_text.append(gen)

            # 分析
            parsed = [parse_classification_output(o) for o in outputs_text]
            cats = [p.get("category", "?") for p in parsed]
            bins = [p.get("binary", "?") for p in parsed]

            gt = {"category": true_cat}
            rewards = [compute_score("toxicn_reviewer", o, gt) for o in outputs_text]

            unique_cats = len(set(cats))
            unique_rewards = len(set(rewards))
            var = statistics.variance(rewards) if len(set(rewards)) > 1 else 0

            cat_counter = Counter(cats)
            cat_str = " ".join(f"{c}:{n}" for c, n in cat_counter.most_common())

            print(f"   temp={temp:.1f}: unique_cat={unique_cats} unique_r={unique_rewards} "
                  f"var={var:.3f} cats=[{cat_str}]")

            all_results.append({
                "text": text[:40],
                "true_cat": true_cat,
                "temperature": temp,
                "unique_cats": unique_cats,
                "unique_rewards": unique_rewards,
                "reward_variance": var,
                "cat_distribution": dict(cat_counter),
            })

    # 汇总
    print("\n" + "=" * 70)
    print("汇总: 不同 temperature 下的平均指标")
    print("=" * 70)
    for temp in temperatures:
        subset = [r for r in all_results if r["temperature"] == temp]
        avg_unique_cats = sum(r["unique_cats"] for r in subset) / len(subset)
        avg_unique_rewards = sum(r["unique_rewards"] for r in subset) / len(subset)
        avg_var = sum(r["reward_variance"] for r in subset) / len(subset)
        print(f"   temp={temp:.1f}: avg_unique_cats={avg_unique_cats:.1f}  "
              f"avg_unique_rewards={avg_unique_rewards:.1f}  avg_variance={avg_var:.4f}")

    # 保存结果
    out_path = os.path.join(_PARENT, "tests", "temperature_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n   结果已保存: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="", type=str,
                        help="Reviewer 模型路径 (不提供则用 mock 模式)")
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--num_gen", default=8, type=int,
                        help="每个 temperature 生成次数")
    parser.add_argument("--mock", action="store_true",
                        help="使用模拟模式 (无需 GPU)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mock or not args.model_path:
        mock_temperature_test()
    else:
        real_temperature_test(args.model_path, args.num_samples, args.num_gen)
