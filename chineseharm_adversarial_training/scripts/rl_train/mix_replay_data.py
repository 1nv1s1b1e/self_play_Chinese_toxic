#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据混合与经验回放脚本 (Phase B 之前执行)
=============================================
作用:
  为了防止 Reviewer 出现灾难性遗忘 (只学新生成的 1500 条攻击文本),
  将本轮生成的动态对抗 GRPO 数据与原始种子数据 (Seed Data) 进行混合。

用法:
  python mix_replay_data.py \
      --dynamic_data /path/to/reviewer_grpo_roundX.parquet \
      --seed_data /path/to/train_seed.parquet \
      --output_data /path/to/reviewer_mixed_roundX.parquet \
      --seed_ratio 2.0
"""

import os
import argparse
import pandas as pd
import json

def parse_args():
    parser = argparse.ArgumentParser(description="混合经验回放数据")
    parser.add_argument("--dynamic_data", required=True, type=str, help="本轮动态生成的 Reviewer parquet数据")
    parser.add_argument("--seed_data",    required=True, type=str, help="原始种子数据 parquet/json")
    parser.add_argument("--output_data",  required=True, type=str, help="混合后的输出 parquet 数据")
    parser.add_argument("--seed_ratio",   default=2.0, type=float, help="混入种子数据的倍数 (默认2.0: 1份新数据 + 2份旧数据)")
    parser.add_argument("--seed",         default=42, type=int, help="随机抽样种子")
    return parser.parse_args()

def convert_seed_to_grpo_format(df_seed: pd.DataFrame) -> pd.DataFrame:
    """将原始格式的 test/train seed 数据转为 Reviewer GRPO 所需的 prompt/reward_model 格式"""
    REVIEWER_SYSTEM_PROMPT = "You are a helpful assistant."

    # ToxiCN 分类规则知识 (与 prepare_reviewer_sft.py / generate_dynamic_data.py 保持一致)
    REVIEWER_RULES = """性别歧视：
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
3. **泛化仇恨表达**：对社会群体进行泛化的恶意评价，煽动对立情绪，但不明确指向上述特定受害群体。"""

    REVIEWER_USER_TEMPLATE = (
        """作为中文有毒语言检测专家，请对以下文本进行多维度分析。
有毒类别："性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"。如果无毒请回答"无毒"。
注意：隐含歧视（如反讽、暗喻、阴阳怪气）也属于有毒内容；转述仇恨事件但本身不有毒的内容应判为无毒。
请按以下格式逐行输出：
第一行：输出"判断：有害"或"判断：无害"
（以下仅当有害时输出）
第二行：输出"类别："加具体类别名（"性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"）
第三行：输出"毒性类型："加类型（"一般攻击性"或"仇恨言论"）
第四行：输出"表达方式："加方式（"显式仇恨"、"隐式仇恨"或"转述"）
规则:
"""
        + REVIEWER_RULES
        + """
文本: {text}"""
    )

    rows = []
    # 尝试找到对应的列名
    col_text  = "文本"   if "文本"   in df_seed.columns else "original_text"
    col_cat   = "标签"   if "标签"   in df_seed.columns else "category"
    col_tt    = "toxic_type_label"  if "toxic_type_label"  in df_seed.columns else "toxic_type"
    col_expr  = "expression_label" if "expression_label" in df_seed.columns else "expression"

    for _, row in df_seed.iterrows():
        text = str(row.get(col_text, ""))
        cat  = str(row.get(col_cat, "无毒"))
        tt   = str(row.get(col_tt, "无毒"))
        expr = str(row.get(col_expr, "非仇恨"))

        if not text:
            continue

        user_content = REVIEWER_USER_TEMPLATE.format(text=text.strip()[:500])
        rows.append({
            "prompt": [
                {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_content}
            ],
            "reward_model": {
                "ground_truth": cat,
                "category":     cat,
                "toxic_type":   tt,
                "expression":   expr,
            },
            "extra_info": {
                "original_text": text.strip()[:200],
                "category":      cat,
                "is_replay":     True  # 标记此条为经验回放数据
            }
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    print(f"🔄 开始混合数据...")
    print(f"   - 动态新数据  : {args.dynamic_data}")
    print(f"   - 历史种子数据: {args.seed_data}")
    
    # 1. 读入本轮动态生成数据
    df_dynamic = pd.read_parquet(args.dynamic_data)
    dynamic_size = len(df_dynamic)
    print(f"   👉 当前轮次动态对抗样本数: {dynamic_size}")

    if dynamic_size == 0:
        raise ValueError("动态数据为空，无需混合！")

    # 2. 读入原始种子数据
    if args.seed_data.endswith('.parquet'):
        df_seed_raw = pd.read_parquet(args.seed_data)
    else:
        df_seed_raw = pd.read_json(args.seed_data)
    print(f"   👉 种子数据池总量: {len(df_seed_raw)}")

    # 3. 按比例抽样
    target_seed_size = int(dynamic_size * args.seed_ratio)
    if target_seed_size > len(df_seed_raw):
        target_seed_size = len(df_seed_raw)
        
    df_seed_sampled = df_seed_raw.sample(n=target_seed_size, random_state=args.seed)
    print(f"   👉 从历史池中抽取了: {target_seed_size} 条 (比例 {args.seed_ratio}x)")

    # 4. 转换历史数据格式到 GRPO 需要的一致格式
    df_seed_grpo = convert_seed_to_grpo_format(df_seed_sampled)

    # 5. 合并并打乱
    df_mixed = pd.concat([df_dynamic, df_seed_grpo], ignore_index=True)
    df_mixed = df_mixed.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # 保存
    os.makedirs(os.path.dirname(args.output_data), exist_ok=True)
    df_mixed.to_parquet(args.output_data, index=False)
    print(f"✅ 混合完成！总数据量: {len(df_mixed)} -> 保存至: {args.output_data}")

if __name__ == "__main__":
    main()
