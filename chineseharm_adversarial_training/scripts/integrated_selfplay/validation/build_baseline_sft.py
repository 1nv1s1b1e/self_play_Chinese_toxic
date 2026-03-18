#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build baseline SFT data (NO CoT) from the same 1000 pilot samples.
For ablation comparison: same data, with CoT vs without CoT.

Usage:
  python build_baseline_sft.py \
      --pilot_results /path/to/cot_pilot_results.json \
      --output /path/to/baseline_sft.jsonl
"""
import json, argparse

TARGET_NAMES = ['性别歧视', '种族歧视', '地域偏见', 'LGBTQ歧视', '其他仇恨']
EXPR_NAMES = {0: '非仇恨', 1: '显式仇恨', 2: '隐式仇恨', 3: '转述'}
TT_NAMES = {0: '无毒', 1: '一般攻击性', 2: '仇恨言论'}

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

REVIEWER_USER_TEMPLATE = (
    "作为中文有毒语言检测专家，请对以下文本进行多维度分析。\n"
    "请按以下格式逐行输出：\n"
    "第一行：输出\"判断：有害\"或\"判断：无害\"\n"
    "（以下仅当有害时输出）\n"
    "第二行：输出\"类别：\"加具体类别名（\"性别歧视\"、\"种族歧视\"、\"地域偏见\"、\"LGBTQ歧视\"、\"其他仇恨\"）\n"
    "第三行：输出\"毒性类型：\"加类型（\"一般攻击性\"或\"仇恨言论\"）\n"
    "第四行：输出\"表达方式：\"加方式（\"显式仇恨\"、\"隐式仇恨\"或\"转述\"）\n\n"
    "注意：隐式仇恨包括反讽、暗语、谐音、阴阳怪气等手法；"
    "转述指报道/讨论仇恨事件但自身立场模糊的内容。\n\n"
    f"规则: {RULES}\n"
    "文本: {text}"
)


def build_label(cat, toxic_type, expression):
    """Build the direct classification output (no CoT)."""
    if cat == "无毒":
        return "判断：无害"

    tt_str = TT_NAMES.get(toxic_type, "仇恨言论")
    expr_str = EXPR_NAMES.get(expression, "隐式仇恨")

    return f"判断：有害\n类别：{cat}\n毒性类型：{tt_str}\n表达方式：{expr_str}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot_results", required=True, help="cot_pilot_results.json from step2")
    p.add_argument("--output", required=True, help="Output baseline SFT jsonl path")
    args = p.parse_args()

    with open(args.pilot_results, "r", encoding="utf-8") as f:
        results = json.load(f)

    sft = []
    for r in results:
        text = r['text']
        cat = r['true_category']
        toxic_type = r.get('toxic_type', 0)
        expression = r.get('expression', 0)

        label_output = build_label(cat, toxic_type, expression)

        sft.append({
            "messages": [
                {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                {"role": "user", "content": REVIEWER_USER_TEMPLATE.format(text=text)},
                {"role": "assistant", "content": label_output},
            ],
            "category": cat,
            "toxic": r.get('true_toxic', 0),
            "expression": expression,
        })

    with open(args.output, "w", encoding="utf-8") as f:
        for s in sft:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Baseline SFT data: {len(sft)} rows -> {args.output}")

    from collections import Counter
    cat_dist = Counter(s['category'] for s in sft)
    for c in ['性别歧视', '种族歧视', '地域偏见', 'LGBTQ歧视', '其他仇恨', '无毒']:
        if c in cat_dist:
            print(f"  {c}: {cat_dist[c]}")


if __name__ == "__main__":
    main()
