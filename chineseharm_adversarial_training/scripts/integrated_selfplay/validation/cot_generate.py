#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoT data generator - call API to generate Chain-of-Thought reasoning.

Supports two dataset formats:
  Format A (split_data/train.json): keys = 文本, 标签, toxic, expression, ...
  Format B (root train.json):       keys = content, target (one-hot), toxic, expression, ...

Usage:
  python cot_generate.py --mode pilot --train_data /path/to/train.json --output_dir /path/to/out --api_key sk-xxx
  python cot_generate.py --mode full  --train_data /path/to/train.json --output_dir /path/to/out --api_key sk-xxx
"""

import os, sys, json, time, argparse, random, logging
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

COT_SYSTEM_PROMPT = """你是中文有毒语言分析专家。用户会给你一段文本和它的正确标签，请你生成一段逐步推理分析，解释为什么该文本属于这个标签类别。

分析要求：
1. 识别文本中支持该标签的关键语言特征（词汇、语气、修辞手法等）
2. 解释为什么这些特征符合该类别的定义
3. 如果是隐式仇恨，指出其隐晦手法（反讽、阴阳怪气、谐音、刻板印象等）
4. 如果是无毒，解释为什么该文本虽然涉及敏感话题但不构成歧视或攻击

输出格式（严格遵守）：
分析：[你的逐步推理，2-4句话，解释为什么文本属于给定标签]
判断：有害（或：判断：无害）
类别：[仅有害时输出] 性别歧视/种族歧视/地域偏见/LGBTQ歧视/其他仇恨
毒性类型：[仅有害时输出] 一般攻击性/仇恨言论
表达方式：[仅有害时输出] 显式仇恨/隐式仇恨/转述"""

TARGET_NAMES = ['性别歧视', '种族歧视', '地域偏见', 'LGBTQ歧视', '其他仇恨']
EXPR_NAMES = {0: '非仇恨(无毒)', 1: '显式仇恨', 2: '隐式仇恨', 3: '转述'}
REVIEWER_SYSTEM_PROMPT = "You are a helpful assistant."

RULES = """
性别歧视：
1. **对特定性别的贬低与攻击**：使用贬低性词汇攻击特定性别群体，包含"女拳"、"直男癌"、"普信男/女"等网络用语，或通过"女人就是..."、"男的没一个好东西"等表述否定特定性别的能力、价值与人格。
2. **性别刻板印象与偏见**：强化对特定性别的刻板认知，如"女人不适合做领导"、"男人就该..."等性别角色规训，将个体行为上升到性别群体层面进行评价。
3. **隐含性别歧视**：通过看似客观或调侃的表述传递性别偏见，使用阴阳怪气、反讽、段子等方式暗示或贬低特定性别群体。

种族歧视：
1. **种族性侮辱与攻击**：使用带有歧视性质的词汇指代特定种族群体，包含侮辱性的简称、别称、谐音或外号，对特定种族的外貌、文化、能力进行恶意攻击。
2. **种族偏见与刻板印象**：基于种族对个人或群体的能力、品德、行为进行偏颇的负面概括，宣扬种族优越论或劣等论。
3. **隐含种族歧视**：以阴阳怪气、看似客观分析或"开玩笑"的方式传递种族偏见，将个别事件上升为种族特征。

地域偏见：
1. **地域性侮辱与攻击**：针对特定地区人群使用贬义性称呼或进行侮辱，如"xx人都是骗子"等典型地域黑言论，使用"偷"、"骗"、"穷"、"懒"等刻板标签。
2. **地域刻板印象**：将个别事件或个别人物的负面行为上升为对某一地区全体居民的负面评价，制造地域对立与矛盾。
3. **隐含地域歧视**：以段子、调侃、"说实话"等伪客观形式传播地域偏见，煽动地域对立情绪。

LGBTQ歧视：
1. **对LGBTQ群体的恶意攻击**：使用歧视性词汇如"变态"、"恶心"、"不正常"等攻击LGBTQ群体，否定其身份认同、性取向或合法权利。
2. **恐同/恐跨偏见**：表达对LGBTQ群体的厌恶、排斥或恐惧，将同性恋、跨性别等视为"疾病"或"道德败坏"，主张对LGBTQ群体进行"矫正"。
3. **隐含LGBTQ歧视**：以"正常/不正常"的二元框架暗示否定特定性取向或性别认同，用隐语或调侃方式进行隐晦歧视。

其他仇恨：
1. **一般性人身攻击**：不针对特定身份群体但具有显著恶意的侮辱性语言，使用脏话、污言秽语对个人进行攻击谩骂。
2. **群体性恶意言论**：针对非上述特定分类但可识别的群体（如特定职业、年龄段、宗教信仰等群体）发起的恶意攻击与污名化。
3. **泛化仇恨表达**：对社会群体进行泛化的恶意评价，煽动对立情绪，但不明确指向上述特定受害群体。
"""

COT_USER_TEMPLATE = (
    "作为中文有毒语言检测专家，请对以下文本进行逐步分析和分类。\n"
    "先分析文本的语言特征和潜在意图，再给出判断。\n\n"
    "请按以下格式输出：\n"
    "分析：[2-4句话的逐步推理]\n"
    "判断：有害/无害\n"
    "（以下仅当有害时输出）\n"
    "类别：性别歧视/种族歧视/地域偏见/LGBTQ歧视/其他仇恨\n"
    "毒性类型：一般攻击性/仇恨言论\n"
    "表达方式：显式仇恨/隐式仇恨/转述\n\n"
    "注意：隐式仇恨包括反讽、暗语、谐音、阴阳怪气等手法；"
    "转述指报道/讨论仇恨事件但自身立场模糊的内容。\n\n"
    f"规则: {RULES}\n"
    "文本：{text}"
)


def get_text(d):
    """Extract text from either dataset format."""
    for key in ['文本', 'content', 'text', 'original_text']:
        if key in d and d[key]:
            return str(d[key]).strip()
    return ""


def get_category(d):
    """Extract category label from either dataset format."""
    # Format A: has 标签 directly
    if '标签' in d and d['标签']:
        return str(d['标签'])

    # Format B: has target one-hot vector
    if 'target' in d:
        target = d['target']
        if isinstance(target, str):
            import ast
            target = ast.literal_eval(target)
        if isinstance(target, list):
            active = [TARGET_NAMES[i] for i, v in enumerate(target) if v == 1]
            return active[0] if active else "无毒"

    # Fallback: use toxic field
    if 'toxic' in d:
        return "有毒" if d['toxic'] == 1 else "无毒"

    return "无毒"


def get_toxic(d):
    """Extract toxic flag."""
    if 'toxic' in d:
        return int(d['toxic'])
    cat = get_category(d)
    return 0 if cat == "无毒" else 1


def get_expression(d):
    """Extract expression type."""
    if 'expression' in d:
        return int(d['expression'])
    return 0


def get_toxic_type(d):
    """Extract toxic_type."""
    if 'toxic_type' in d:
        return int(d['toxic_type'])
    return 0


def build_api_user_prompt(text, category, expression, toxic_type):
    """Build API user prompt with the GIVEN human label."""
    EXPR_MAP = {0: '非仇恨', 1: '显式仇恨', 2: '隐式仇恨', 3: '转述'}
    TT_MAP = {0: '无毒', 1: '一般攻击性', 2: '仇恨言论'}

    expr_str = EXPR_MAP.get(expression, str(expression))
    tt_str = TT_MAP.get(toxic_type, str(toxic_type))

    if category == "无毒":
        label_info = f"正确标签：无毒（该文本虽涉及敏感话题但不构成歧视或攻击）"
    else:
        label_info = f"正确标签：有害\n类别：{category}\n毒性类型：{tt_str}\n表达方式：{expr_str}"

    return f"文本：{text.strip()[:500]}\n\n{label_info}\n\n请根据以上正确标签，生成逐步推理分析："


def call_api(text, category, expression, toxic_type, api_key, api_base, model, max_retries=3):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=api_base)
    user_prompt = build_api_user_prompt(text, category, expression, toxic_type)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": COT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1, max_tokens=512,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                logger.warning(f"API failed: {e}")
                return None


def validate_cot(cot_output, true_category):
    result = {"binary_match": False, "category_match": False,
              "cot_binary": None, "cot_category": None}
    if not cot_output:
        return result
    true_is_harmful = true_category in TARGET_NAMES

    for m in ["判断：有害", "判断:有害", "判断: 有害"]:
        if m in cot_output:
            result["cot_binary"] = "有害"
            break
    if result["cot_binary"] is None:
        for m in ["判断：无害", "判断:无害", "判断: 无害"]:
            if m in cot_output:
                result["cot_binary"] = "无害"
                break

    if result["cot_binary"] is not None:
        result["binary_match"] = ((result["cot_binary"] == "有害") == true_is_harmful)

    for cat in TARGET_NAMES:
        if f"类别：{cat}" in cot_output or f"类别:{cat}" in cot_output:
            result["cot_category"] = cat
            break

    if true_category == "无毒":
        result["category_match"] = (result["cot_binary"] == "无害")
    else:
        result["category_match"] = (result["cot_category"] == true_category)
    return result


def stratified_sample(data, n):
    """Stratified sample by category label (6 classes)."""
    groups = defaultdict(list)
    for d in data:
        cat = get_category(d)  # 性别歧视/种族歧视/地域偏见/LGBTQ歧视/其他仇恨/无毒
        groups[cat].append(d)

    selected = []
    total = len(data)
    for cat, pool in groups.items():
        budget = max(10, int(n * len(pool) / total))
        budget = min(budget, len(pool))
        selected.extend(random.sample(pool, budget))

    random.shuffle(selected)
    seen = set()
    unique = []
    for d in selected:
        k = get_text(d)[:200]
        if k and k not in seen:
            seen.add(k)
            unique.append(d)
    return unique[:n]


def process_one(args_tuple):
    idx, d, api_key, api_base, model = args_tuple
    text = get_text(d)
    true_cat = get_category(d)
    expr = get_expression(d)
    tt = get_toxic_type(d)
    cot = call_api(text, true_cat, expr, tt, api_key, api_base, model)
    v = validate_cot(cot, true_cat) if cot else {
        "binary_match": False, "category_match": False,
        "cot_binary": None, "cot_category": None}
    return {
        "idx": idx,
        "text": text,
        "true_category": true_cat,
        "true_toxic": get_toxic(d),
        "expression": expr,
        "toxic_type": tt,
        "cot_output": cot or "",
        **v,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["pilot", "full"])
    p.add_argument("--train_data", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--api_key", required=True)
    p.add_argument("--api_base", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--api_model", default="qwen-plus")
    p.add_argument("--num_pilot", default=1000, type=int)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()

    random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    logger.info(f"[1] Loading {args.train_data}")
    with open(args.train_data, "r", encoding="utf-8") as f:
        data = json.load(f)
    for i, d in enumerate(data):
        d['_idx'] = i

    # Auto-detect format
    sample = data[0]
    if '文本' in sample:
        text_field = '文本'
        label_field = '标签'
        fmt = "Format A (文本/标签)"
    elif 'content' in sample:
        text_field = 'content'
        label_field = 'target'
        fmt = "Format B (content/target)"
    else:
        logger.error(f"Unknown format. Keys: {list(sample.keys())}")
        sys.exit(1)

    logger.info(f"    {len(data)} rows, {fmt}")
    logger.info(f"    Sample text: \"{get_text(data[0])[:60]}\"")
    logger.info(f"    Sample category: {get_category(data[0])}")

    # Select
    if args.mode == "pilot":
        texts = stratified_sample(data, args.num_pilot)
        logger.info(f"[2] Pilot: stratified sample {len(texts)} rows")
    else:
        texts = data
        logger.info(f"[2] Full: all {len(texts)} rows")

    # Show distribution by category
    dist = Counter()
    for d in texts:
        dist[get_category(d)] += 1
    for cat in ['性别歧视', '种族歧视', '地域偏见', 'LGBTQ歧视', '其他仇恨', '无毒']:
        if cat in dist:
            logger.info(f"    {cat}: {dist[cat]}")

    # Resume
    ckpt = out / f"cot_{args.mode}_checkpoint.jsonl"
    done_idx = set()
    if ckpt.exists():
        with open(str(ckpt), "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_idx.add(json.loads(line)['idx'])
                except Exception:
                    pass
        logger.info(f"    Resume: {len(done_idx)} already done")

    work = [(d['_idx'], d, args.api_key, args.api_base, args.api_model)
            for d in texts if d['_idx'] not in done_idx]
    logger.info(f"    Remaining: {len(work)}")

    # API calls
    if work:
        logger.info(f"[3] Calling API (workers={args.workers}, model={args.api_model})")
        done_total = len(done_idx)
        all_total = len(texts)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, w): w[0] for w in work}
            for fut in as_completed(futs):
                r = fut.result()
                done_total += 1
                with open(str(ckpt), "a", encoding="utf-8") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                if done_total % 100 == 0 or done_total == all_total:
                    logger.info(f"    Progress: {done_total}/{all_total}")

    # Collect
    logger.info(f"[4] Collecting results")
    results = []
    with open(str(ckpt), "r", encoding="utf-8") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except Exception:
                pass

    total = len(results)
    ok_api = sum(1 for r in results if r['cot_output'])
    ok_bin = sum(1 for r in results if r['binary_match'])
    ok_cat = sum(1 for r in results if r['category_match'])

    # Build SFT jsonl — use ALL samples (no filtering)
    sft = []
    for r in results:
        if not r['cot_output']:
            continue  # skip only API failures (empty output)
        sft.append({
            "messages": [
                {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                {"role": "user", "content": COT_USER_TEMPLATE.format(text=r['text'])},
                {"role": "assistant", "content": r["cot_output"]},
            ],
            "category": r["true_category"],
            "toxic": r["true_toxic"],
            "expression": r["expression"],
        })

    sft_path = out / f"cot_{args.mode}_sft.jsonl"
    with open(str(sft_path), "w", encoding="utf-8") as f:
        for s in sft:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    full_path = out / f"cot_{args.mode}_results.json"
    with open(str(full_path), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Report
    print("\n" + "=" * 60)
    print(f"  CoT {args.mode} done!")
    print(f"  API success: {ok_api}/{total}")
    print(f"  Binary match: {ok_bin}/{ok_api} = {ok_bin/max(ok_api,1)*100:.1f}%")
    print(f"  Category match: {ok_cat}/{ok_api} = {ok_cat/max(ok_api,1)*100:.1f}%")
    print(f"  Valid SFT: {len(sft)} rows")
    print(f"  SFT file: {sft_path}")
    print("=" * 60)

    # Per category
    cat_total = Counter()
    cat_ok = Counter()
    for r in results:
        c = r['true_category']
        cat_total[c] += 1
        if r['binary_match']:
            cat_ok[c] += 1
    print(f"\n  By category:")
    for c in ['性别歧视', '种族歧视', '地域偏见', 'LGBTQ歧视', '其他仇恨', '无毒']:
        if c in cat_total:
            t, o = cat_total[c], cat_ok.get(c, 0)
            print(f"    {c}: {o}/{t} = {o/t*100:.0f}%")

    rate = ok_bin / max(ok_api, 1)
    print()
    if rate > 0.85:
        print(f"  OK: match rate {rate*100:.0f}% > 85%")
        if args.mode == "pilot":
            print(f"  Next: run --mode full")
    elif rate > 0.70:
        print(f"  WARNING: match rate {rate*100:.0f}%, check mismatched samples")
    else:
        print(f"  BAD: match rate {rate*100:.0f}% < 70%, adjust prompt or use stronger model")


if __name__ == "__main__":
    main()
