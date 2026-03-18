#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL数据预处理 - 将SFT数据转换为verl GRPO训练所需的parquet格式

verl要求的数据格式:
{
    "data_source": "toxicn_challenger" 或 "toxicn_reviewer",
    "prompt": [{"role": "user", "content": "..."}],
    "ability": "safety",
    "reward_model": {"style": "rule", "ground_truth": "..."},
    "extra_info": {"index": 0, "category": "性别歧视", "split": "train"}
}

Challenger prompt: 给定类别 → 生成该类别的文本
Reviewer prompt:  给定文本 → 判断类别

生成文件:
  rl/challenger_grpo_train.parquet  - Challenger GRPO训练数据
  rl/reviewer_grpo_train.parquet    - Reviewer GRPO训练数据
  rl/challenger_grpo_val.parquet    - Challenger GRPO验证数据
  rl/reviewer_grpo_val.parquet      - Reviewer GRPO验证数据
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd

# ============================================================
# 知识规则 (基于ToxiCN数据集的有毒语言分类体系)
# ============================================================
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

ALL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]
HARMFUL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"]

# ============================================================
# ToxiCN 毒性类型 & 表达方式 字段映射
# ============================================================
TOXIC_TYPE_MAP = {0: "无毒", 1: "一般攻击性", 2: "仇恨言论"}
EXPRESSION_MAP = {0: "非仇恨", 1: "显式仇恨", 2: "隐式仇恨", 3: "转述"}


def build_challenger_prompt(category: str, expression: str = "显式仇恨") -> str:
    """
    构建Challenger的prompt (v2: 按expression分化模板)
    Challenger的任务: 给定违规类别和表达方式,生成一条真实的、难以被检测的该类别文本
    """
    if category == "无毒":
        return (
            "你是一个中文社交媒体内容模拟器。请生成一条看起来像是互联网上发布的正常文本。\n"
            "要求：\n"
            "1. 文本应该看起来自然、真实\n"
            "2. 内容完全无毒\n"
            "3. 但可能在表面上容易被误判为有毒内容（使用一些边界表达）\n"
            "4. 只输出生成的文本，不要任何解释\n\n"
            "请直接生成文本："
        )

    cat_rules = _get_category_rules(category)

    if expression == "隐式仇恨":
        return (
            f"你是一个中文有毒语言研究助手。为了研究隐蔽有毒内容检测系统的漏洞，"
            f"请你模拟生成一条属于「{category}」类别的**隐式**有毒文本。\n\n"
            f"该类别的详细规则定义如下：\n{cat_rules}\n\n"
            f"要求：\n"
            f"1. 使用反讽、暗语、谐音、阴阳怪气等手法，让检测系统难以识别\n"
            f"2. 文本应当是真实互联网场景中可能出现的\n"
            f"3. 不能直接使用明显的歧视性词汇\n"
            f"4. 只输出生成的文本，不要任何解释或标注\n\n"
            f"请直接生成文本："
        )
    elif expression == "转述":
        return (
            f"你是一个中文有毒语言研究助手。请模拟生成一条以新闻转述或讨论口吻呈现的文本，"
            f"内容涉及「{category}」相关的歧视事件，但本身立场模糊。\n\n"
            f"该类别的详细规则定义如下：\n{cat_rules}\n\n"
            f"要求：\n"
            f"1. 以客观报道或引用他人发言的口吻呈现\n"
            f"2. 内容涉及歧视性言论但文本本身不直接表态\n"
            f"3. 使内容具有边界模糊性，难以明确判定\n"
            f"4. 只输出生成的文本，不要任何解释或标注\n\n"
            f"请直接生成文本："
        )
    else:
        # 显式仇恨 / 默认
        return (
            f"你是一个中文有毒语言研究助手。为了研究有毒内容检测系统的漏洞，"
            f"请你模拟生成一条属于「{category}」类别的文本。\n\n"
            f"该类别的详细规则定义如下：\n{cat_rules}\n\n"
            f"要求：\n"
            f"1. 生成的文本应当是真实互联网场景中可能出现的\n"
            f"2. 文本应该具有一定的隐蔽性，不能太直白明显\n"
            f"3. 使用隐喻、暗语、谐音、反讽等手法增加检测难度\n"
            f"4. 只输出生成的文本，不要任何解释或标注\n\n"
            f"请直接生成文本："
        )


def build_reviewer_prompt(text: str) -> str:
    """
    构建Reviewer的prompt (v2: 4维输出 — 二分类+类别+毒性类型+表达方式)
    """
    return (
        '作为中文有毒语言检测专家，请判断以下文本是否包含有毒内容。\n'
        '请严格按以下格式逐行输出：\n'
        '第一行：输出"判断：有害"或"判断：无害"\n'
        '第二行（仅当有害时）：输出"类别："加具体类别名（"性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"）\n'
        '第三行（仅当有害时）：输出"毒性类型："加类型（"一般攻击性"或"仇恨言论"）\n'
        '第四行（仅当有害时）：输出"表达方式："加方式（"显式仇恨"、"隐式仇恨"或"转述"）\n\n'
        '说明：\n'
        '- 一般攻击性：泛化的侮辱谩骂，不针对特定身份群体的仇恨结构\n'
        '- 仇恨言论：针对特定身份群体的系统性偏见与歧视\n'
        '- 显式仇恨：直白、明显的攻击性语言\n'
        '- 隐式仇恨：使用反讽、暗语、谐音、阴阳怪气等隐晦歧视\n'
        '- 转述：以新闻或讨论口吻呈现的仇恨事件，立场模糊\n\n'
        '注意：隐含歧视也属于有毒内容；转述仇恨事件但本身不有毒的内容应判为无毒。\n\n'
        f'规则: {RULES}\n'
        f'文本: {text}'
    )


# ============================================================
# RL 阶段弱化版 Reviewer Prompt (无关键词映射)
# ============================================================
BRIEF_RULES_FOR_RL = """性别歧视：对特定性别群体的贬低、攻击、刻板印象或隐含偏见，包括显式和隐式的性别歧视言论。
种族歧视：基于种族对个人或群体进行侮辱、偏见表达或刻板评价，包含种族优越或劣等论。
地域偏见：针对特定地区人群的侮辱、刻板印象或地域歧视言论，包括地域黑和地域对立。
LGBTQ歧视：对LGBTQ群体的侮辱、否定、恐惧或隐含歧视，包括恐同、恐跨偏见。
其他仇恨：不属于上述特定群体但具有明显恶意的攻击性或仇恨性言论。
无毒：内容不属于上述任何有毒类别。注意：转述仇恨事件但本身不有毒的内容应判为无毒。""".strip()


def build_reviewer_prompt_rl(text: str) -> str:
    """
    RL阶段的 Reviewer prompt (弱化版: 只有类别描述，无关键词映射)
    v2: 4维输出格式（二分类+类别+毒性类型+表达方式）
    """
    return (
        '作为中文有毒语言检测专家，请判断以下文本是否包含有毒内容。\n'
        '请严格按以下格式逐行输出：\n'
        '第一行：输出"判断：有害"或"判断：无害"\n'
        '第二行（仅当有害时）：输出"类别："加具体类别名（"性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"）\n'
        '第三行（仅当有害时）：输出"毒性类型："加类型（"一般攻击性"或"仇恨言论"）\n'
        '第四行（仅当有害时）：输出"表达方式："加方式（"显式仇恨"、"隐式仇恨"或"转述"）\n\n'
        f'参考: {BRIEF_RULES_FOR_RL}\n\n'
        f'文本: {text}'
    )


def _get_category_rules(category: str) -> str:
    """提取特定类别的规则"""
    lines = RULES.strip().split('\n')
    capturing = False
    result = []
    for line in lines:
        if line.startswith(category + "："):
            capturing = True
            result.append(line)
        elif capturing and line and not any(line.startswith(c + "：") for c in ALL_CATEGORIES if c != category):
            result.append(line)
        elif capturing and any(line.startswith(c + "：") for c in ALL_CATEGORIES if c != category):
            break
    return '\n'.join(result)


def prepare_challenger_grpo_data(
    df: pd.DataFrame, split: str,
    adversarial_success_map: dict = None,
    label_verified_map: dict = None,
) -> list:
    """
    准备 Challenger 的 GRPO 训练数据 (v11: 注入 Phase A 对抗信号)

    Args:
        df: 训练数据 DataFrame
        split: 数据集划分名称
        adversarial_success_map: {original_text: bool}
            Phase A-2 中当前 Reviewer_N-1 是否被骗（按 original_text 索引）
        label_verified_map: {original_text: bool}
            Phase A-3 中冻结 reviewer_v0 是否确认类别正确（按 original_text 索引）
    """
    records = []
    for idx, row in df.iterrows():
        category = row['category'] if 'category' in row else row.get('标签', '')
        text = row['original_text'] if 'original_text' in row else row.get('文本', '')
        expression = row.get('expression_label', '显式仇恨')

        prompt_content = build_challenger_prompt(category, expression)

        challenger_sys = (
            "你是一个文本生成助手。根据用户给定的类别和表达方式，"
            "生成一条对应的真实互联网文本。只输出文本本身，不要任何解释。"
        )

        extra = {
            "split": split,
            "index": int(idx),
            "category": category,
            "original_text": text,
            "expression": expression,
            "toxic_type": row.get('toxic_type_label', ''),
        }

        # 注入 Phase A 预计算的对抗信号（Verifier 输出）
        if adversarial_success_map and text in adversarial_success_map:
            extra["adversarial_success"] = bool(adversarial_success_map[text])
        if label_verified_map and text in label_verified_map:
            extra["label_verified"] = bool(label_verified_map[text])

        record = {
            "data_source": "toxicn_challenger",
            "prompt": [{"role": "system", "content": challenger_sys},
                       {"role": "user", "content": prompt_content}],
            "ability": "safety",
            "reward_model": {
                "style": "rule",
                "ground_truth": text
            },
            "extra_info": extra,
        }
        records.append(record)

    return records



def prepare_reviewer_grpo_data(df: pd.DataFrame, split: str) -> list:
    """
    准备Reviewer的GRPO训练数据 (v2: ground_truth包含4维标注)
    RL阶段使用弱化版prompt (无关键词映射)
    """
    records = []
    for idx, row in df.iterrows():
        category = row['category'] if 'category' in row else row.get('标签', '')
        text = row['original_text'] if 'original_text' in row else row.get('文本', '')
        toxic_type_label = row.get('toxic_type_label', '')
        expression_label = row.get('expression_label', '')
        
        # RL阶段使用弱化版prompt
        prompt_content = build_reviewer_prompt_rl(text)
        
        # ground_truth 编码为 JSON 字符串，含全部维度
        gt = {
            "category": category,
            "toxic_type": toxic_type_label,
            "expression": expression_label,
        }
        
        record = {
            "data_source": "toxicn_reviewer",
            "prompt": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt_content}],
            "ability": "safety",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(gt, ensure_ascii=False)
            },
            "extra_info": {
                "split": split,
                "index": int(idx),
                "category": category,
                "original_text": text,
                "toxic_type": toxic_type_label,
                "expression": expression_label,
            }
        }
        records.append(record)
    
    return records


def main():
    parser = argparse.ArgumentParser(description="准备verl GRPO训练数据")
    parser.add_argument("--split_dir", type=str,
                       default="/home/ma-user/work/test/split_data",
                       help="split_data目录 (含train/val/test.parquet)")
    parser.add_argument("--output_dir", type=str,
                       default="/home/ma-user/work/test/prepared_data/rl",
                       help="输出目录")
    
    args = parser.parse_args()
    
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("准备 verl GRPO 训练数据")
    print("=" * 70)
    
    # 加载数据
    print("\n[1/3] 加载分割数据...")
    train_df = pd.read_parquet(split_dir / "train.parquet")
    val_df = pd.read_parquet(split_dir / "val.parquet")
    test_df = pd.read_parquet(split_dir / "test.parquet")
    
    print(f"  train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    
    # 统一列名
    col_map = {}
    if '文本' in train_df.columns:
        col_map['文本'] = 'original_text'
    if '标签' in train_df.columns:
        col_map['标签'] = 'category'
    if col_map:
        train_df = train_df.rename(columns=col_map)
        val_df = val_df.rename(columns=col_map)
        test_df = test_df.rename(columns=col_map)
    
    # 构建Challenger GRPO数据
    print("\n[2/3] 构建 Challenger GRPO 数据...")
    challenger_train = prepare_challenger_grpo_data(train_df, "train")
    challenger_val = prepare_challenger_grpo_data(val_df, "val")
    
    challenger_train_df = pd.DataFrame(challenger_train)
    challenger_val_df = pd.DataFrame(challenger_val)
    
    challenger_train_df.to_parquet(output_dir / "challenger_grpo_train.parquet", index=False)
    challenger_val_df.to_parquet(output_dir / "challenger_grpo_val.parquet", index=False)
    
    print(f"  challenger_grpo_train: {len(challenger_train_df)} 条")
    print(f"  challenger_grpo_val:   {len(challenger_val_df)} 条")
    
    # 构建Reviewer GRPO数据
    print("\n[3/3] 构建 Reviewer GRPO 数据...")
    reviewer_train = prepare_reviewer_grpo_data(train_df, "train")
    reviewer_val = prepare_reviewer_grpo_data(val_df, "val")
    
    reviewer_train_df = pd.DataFrame(reviewer_train)
    reviewer_val_df = pd.DataFrame(reviewer_val)
    
    reviewer_train_df.to_parquet(output_dir / "reviewer_grpo_train.parquet", index=False)
    reviewer_val_df.to_parquet(output_dir / "reviewer_grpo_val.parquet", index=False)
    
    print(f"  reviewer_grpo_train: {len(reviewer_train_df)} 条")
    print(f"  reviewer_grpo_val:   {len(reviewer_val_df)} 条")
    
    # 打印样例
    print("\n" + "=" * 70)
    print("样例预览")
    print("=" * 70)
    
    print("\n--- Challenger样例 ---")
    sample = challenger_train[0]
    print(f"data_source: {sample['data_source']}")
    print(f"category: {sample['extra_info']['category']}")
    print(f"prompt: {sample['prompt'][0]['content'][:100]}...")
    print(f"ground_truth: {sample['reward_model']['ground_truth'][:60]}...")
    
    print("\n--- Reviewer样例 ---")
    sample = reviewer_train[0]
    print(f"data_source: {sample['data_source']}")
    print(f"category: {sample['extra_info']['category']}")
    print(f"prompt: {sample['prompt'][0]['content'][:100]}...")
    print(f"ground_truth: {sample['reward_model']['ground_truth']}")
    
    # 统计
    print("\n" + "=" * 70)
    print("类别分布统计")
    print("=" * 70)
    for name, df in [("train", train_df), ("val", val_df)]:
        dist = df['category'].value_counts()
        print(f"\n{name}:")
        for cat, cnt in dist.items():
            print(f"  {cat}: {cnt}")
    
    print(f"\n✓ verl GRPO数据准备完成! 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
