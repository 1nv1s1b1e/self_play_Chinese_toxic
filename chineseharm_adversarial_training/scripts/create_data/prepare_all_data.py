#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一站式数据准备脚本 (v2)
========================

从 split_data/{train,val,test}.parquet 出发，同时关联 ToxiCN_1.0.csv 中的
toxic_type 和 expression 字段，一次性生成全部训练所需数据。

ToxiCN 4层体系:
  toxic (二分类)  →  toxic_type (无毒/一般攻击性/仇恨言论)
                 →  expression  (非仇恨/显式仇恨/隐式仇恨/转述)
                 →  target      (性别歧视/种族歧视/地域偏见/LGBTQ/其他)

数据:
  1. Challenger SFT  →  (category + expression) → text 生成
  2. Reviewer SFT    →  text → (有害/无害 + category + toxic_type + expression)
  3. RL种子数据

输出目录:
  prepared_data/
  ├── challenger_sft/
  ├── reviewer_sft/
  ├── rl/
  └── data_preparation_report.json
"""

import json
import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import List, Dict
from collections import Counter

import pandas as pd


def clean_text(text: str) -> str:
    """
    清洗文本，移除可能导致JSON解析错误的字符
    - 移除NULL字节和其他C0控制字符(保留\n\r\t)
    - 替换非UTF-8安全的字符
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    # 移除NULL字节和其他问题控制字符 (\x00-\x08, \x0b, \x0c, \x0e-\x1f)
    import re
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    # 尝试编解码确保UTF-8安全
    text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    return text.strip()


# ============================================================
# ToxiCN 字段映射 (数字编码 → 中文标签)
# ============================================================
TOXIC_TYPE_MAP = {0: "无毒", 1: "一般攻击性", 2: "仇恨言论"}
EXPRESSION_MAP = {0: "非仇恨", 1: "显式仇恨", 2: "隐式仇恨", 3: "转述"}


def load_split_data(split_dir: Path, split_name: str) -> pd.DataFrame:
    """
    加载由 convert_repo_data.py 生成的富字段数据。
    优先 .parquet，回退 .json。
    确保输出包含: 文本, 标签, toxic_type_label, expression_label
    """
    parquet_path = split_dir / f"{split_name}.parquet"
    json_path = split_dir / f"{split_name}.json"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        print(f"  加载 {parquet_path.name}: {len(df)} 条")
    elif json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        df = pd.DataFrame(records)
        print(f"  加载 {json_path.name}: {len(df)} 条")
    else:
        raise FileNotFoundError(
            f"找不到 {parquet_path} 或 {json_path}\n"
            f"请先运行: python convert_repo_data.py --train_json train.json --test_json test.json"
        )

    # 兼容旧版双字段格式 (只有 文本+标签)
    if 'toxic_type_label' not in df.columns:
        print(f"  ⚠ {split_name} 缺少 toxic_type_label, 使用默认值")
        df['toxic_type_label'] = df['标签'].apply(
            lambda x: "无毒" if x == "无毒" else "仇恨言论"
        )
    if 'expression_label' not in df.columns:
        print(f"  ⚠ {split_name} 缺少 expression_label, 使用默认值")
        df['expression_label'] = df['标签'].apply(
            lambda x: "非仇恨" if x == "无毒" else "显式仇恨"
        )

    return df

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


# ============================================================
# 类别知识规则提取 (Challenger 与 Reviewer 共享知识体系)
# ============================================================
def get_category_rules(category: str) -> str:
    """
    根据类别名返回对应的细粒度特征知识规则。
    ToxiCN 类别 → 从 RULES 中提取对应段落。
    """
    _all_cats = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"]
    if category not in _all_cats:
        return ""
    lines = RULES.strip().split('\n')
    result = []
    capturing = False
    for line in lines:
        if line.startswith(category + '：'):
            capturing = True
            result.append(line)
        elif capturing and line and not any(line.startswith(c + '：') for c in _all_cats if c != category):
            result.append(line)
        elif capturing and any(line.startswith(c + '：') for c in _all_cats if c != category):
            break
    return '\n'.join(result)


# ============================================================
# Challenger SFT 数据构建 (v3: 引入 expression 维度 + 规则知识注入)
# ============================================================
class ChallengerSFTBuilder:
    """
    Challenger的任务: 给定 (类别 + 表达方式), 生成该类别的有害文本
    SFT格式: messages (system/user/assistant)，与训练脚本的chat template对齐

    v3改进: 在 system prompt 中注入与类别对应的 RULES 知识规则，
    使 Challenger 理解每个类别的细粒度特征 (与 Reviewer 共享知识体系)。

    expression 维度:
      - 显式仇恨: 直白攻击 → prompt 要求直白
      - 隐式仇恨: 隐晦表达 → prompt 要求使用反讽/暗语/谐音
      - 转述:     报道形式 → prompt 要求以新闻/转述口吻
      - 非仇恨:   无毒文本 → prompt 要求生成边界无毒内容
      - 一般攻击性(toxic_type=1): 非针对特定群体 → prompt 要求一般谩骂
    """

    # 基础 system prompt
    SYSTEM_PROMPT_BASE = (
        "你是一个文本生成助手。根据用户给定的类别和表达方式，"
        "生成一条对应的真实互联网文本。只输出文本本身，不要任何解释。"
    )

    EXPLICIT_TEMPLATE = (
        "类别：{category}\n"
        "表达方式：显式（直白攻击性语言）\n"
        "请生成一条属于该类别的真实互联网有毒文本，使用直接、明显的歧视性表达。\n"
        "只输出文本本身，不要解释："
    )

    IMPLICIT_TEMPLATE = (
        "类别：{category}\n"
        "表达方式：隐式（隐晦歧视）\n"
        "请生成一条属于该类别的隐蔽有毒文本，使用反讽、暗语、谐音、"
        "阴阳怪气等手法，让检测系统难以识别。\n"
        "只输出文本本身，不要解释："
    )

    REPORTING_TEMPLATE = (
        "类别：{category}\n"
        "表达方式：转述（新闻/报道形式）\n"
        "请生成一条以新闻转述或讨论口吻呈现的文本，内容涉及该类别相关的歧视事件"
        "但本身立场模糊。\n"
        "只输出文本本身，不要解释："
    )

    OFFENSIVE_TEMPLATE = (
        "类别：其他仇恨\n"
        "表达方式：一般攻击性（非针对特定群体）\n"
        "请生成一条具有攻击性但不针对特定身份群体的互联网骂人文本。\n"
        "只输出文本本身，不要解释："
    )

    NONTOXIC_TEMPLATE = (
        "类别：无毒\n"
        "表达方式：正常（无毒边界文本）\n"
        "请生成一条看似涉及敏感话题（如性别讨论、地域话题等）但实际完全无毒的文本，"
        "注意文本应当客观理性，不含任何攻击或偏见。\n"
        "只输出文本本身，不要解释："
    )

    @classmethod
    def _get_template(cls, category: str, expression: str, toxic_type: str) -> str:
        """根据 category + expression + toxic_type 选择 prompt 模板"""
        if category == "无毒":
            return cls.NONTOXIC_TEMPLATE
        if toxic_type == "一般攻击性":
            return cls.OFFENSIVE_TEMPLATE
        if expression == "隐式仇恨":
            return cls.IMPLICIT_TEMPLATE.format(category=category)
        if expression == "转述":
            return cls.REPORTING_TEMPLATE.format(category=category)
        # 默认显式
        return cls.EXPLICIT_TEMPLATE.format(category=category)

    @classmethod
    def _build_system_prompt(cls, category: str) -> str:
        """
        根据类别动态构建 system prompt，注入对应的知识规则。

        - 有害类别：只注入目标类别规则，让模型深度理解该类别特征。
        - 无毒类别：注入完整 RULES（5个类别全部）。
          无毒 Challenger 需要生成靠近边界但未越界的 hard negative，
          必须知道所有有害类别的边界在哪里，才能接近而不踩线。
        """
        if category == "无毒":
            return cls.SYSTEM_PROMPT_BASE
        rules_text = get_category_rules(category)
        if rules_text:
            return (
                f"{cls.SYSTEM_PROMPT_BASE}\n\n"
                f"以下是该类别的特征知识，请据此生成符合特征的文本：\n"
                f"{rules_text}"
            )
        return cls.SYSTEM_PROMPT_BASE

    @classmethod
    def build(cls, df: pd.DataFrame, split_name: str, output_dir: Path):
        """构建并保存Challenger SFT数据 (messages格式 + 规则知识注入)"""
        output_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        for _, row in df.iterrows():
            text = clean_text(row['文本'])
            category = str(row['标签']).strip()
            expression = str(row.get('expression_label', '显式仇恨')).strip()
            toxic_type = str(row.get('toxic_type_label', '仇恨言论')).strip()
            if not text or not category:
                continue

            instruction = cls._get_template(category, expression, toxic_type)
            system_prompt = cls._build_system_prompt(category)

            sample = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": text}
                ],
                "category": category,
                "expression": expression,
                "toxic_type": toxic_type,
            }
            samples.append(sample)

        # 保存JSONL (messages格式，训练脚本直接使用)
        jsonl_path = output_dir / f"{split_name}.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')

        # 保存Parquet (flat格式，便于分析和兼容旧流程)
        flat_samples = [{
            "instruction": s["messages"][1]["content"],
            "input": "",
            "output": s["messages"][2]["content"],
            "system_prompt": s["messages"][0]["content"],
            "category": s["category"],
            "expression": s["expression"],
            "toxic_type": s["toxic_type"],
        } for s in samples]
        parquet_path = output_dir / f"{split_name}.parquet"
        pd.DataFrame(flat_samples).to_parquet(parquet_path, index=False)

        # 统计
        cat_counts = Counter(s['category'] for s in samples)
        expr_counts = Counter(s['expression'] for s in samples)
        print(f"  Challenger SFT [{split_name}]: {len(samples)} 条")
        print(f"    类别分布: {dict(sorted(cat_counts.items()))}")
        print(f"    表达分布: {dict(sorted(expr_counts.items()))}")

        return samples


# ============================================================
# Reviewer SFT 数据构建
# ============================================================
class ReviewerSFTBuilder:
    """
    Reviewer的任务: 给定文本，判断：
      1. 是否有害 (有害/无害)
      2. 若有害: 具体类别、毒性类型、表达方式

    输出格式:
        有害文本: "判断：有害\n类别：性别歧视\n毒性类型：仇恨言论\n表达方式：隐式仇恨"
        无害文本: "判断：无害"
    """

    SYSTEM_PROMPT = "You are a helpful assistant."

    @classmethod
    def create_prompt(cls, text: str) -> str:
        """创建审核prompt (ToxiCN 完整4层体系)"""
        prompt = (
            '作为中文有毒语言检测专家，请对以下文本进行多维度分析。\n'
            '请按以下格式逐行输出：\n'
            '第一行：输出"判断：有害"或"判断：无害"\n'
            '（以下仅当有害时输出）\n'
            '第二行：输出"类别："加具体类别名（"性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"）\n'
            '第三行：输出"毒性类型："加类型（"一般攻击性"或"仇恨言论"）\n'
            '第四行：输出"表达方式："加方式（"显式仇恨"、"隐式仇恨"或"转述"）\n\n'
            '注意：隐式仇恨包括反讽、暗语、谐音、阴阳怪气等手法；转述指报道/讨论仇恨事件但自身立场模糊的内容。\n\n'
            f'规则: {RULES}\n'
            f'文本: {text}'
        )
        return prompt

    @classmethod
    def build(cls, df: pd.DataFrame, split_name: str, output_dir: Path):
        """构建并保存Reviewer SFT数据"""
        output_dir.mkdir(parents=True, exist_ok=True)

        samples = []
        harmful_cats = {"性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"}
        for _, row in df.iterrows():
            text = clean_text(row['文本'])
            label = str(row['标签']).strip()
            expression = str(row.get('expression_label', '显式仇恨')).strip()
            toxic_type = str(row.get('toxic_type_label', '仇恨言论')).strip()
            if not text or not label:
                continue

            # 构造多维结构化输出
            if label in harmful_cats:
                output_content = (
                    f"判断：有害\n"
                    f"类别：{label}\n"
                    f"毒性类型：{toxic_type}\n"
                    f"表达方式：{expression}"
                )
            else:
                output_content = "判断：无害"

            sample = {
                "messages": [
                    {"role": "system", "content": cls.SYSTEM_PROMPT},
                    {"role": "user", "content": cls.create_prompt(text)},
                    {"role": "assistant", "content": output_content}
                ],
                "category": label,
                "expression": expression,
                "toxic_type": toxic_type,
                "original_text": text
            }
            samples.append(sample)
        
        # 保存JSONL
        jsonl_path = output_dir / f"{split_name}.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        
        # 保存Parquet
        flat_samples = [{
            "instruction": s["messages"][1]["content"],
            "output": s["messages"][2]["content"],
            "category": s["category"],
            "expression": s["expression"],
            "toxic_type": s["toxic_type"],
            "original_text": s["original_text"]
        } for s in samples]
        parquet_path = output_dir / f"{split_name}.parquet"
        pd.DataFrame(flat_samples).to_parquet(parquet_path, index=False)
        
        # 统计
        cat_counts = Counter(s['category'] for s in samples)
        print(f"  Reviewer SFT [{split_name}]: {len(samples)} 条 (含'无毒')")
        for cat in sorted(cat_counts):
            print(f"    {cat:10s}: {cat_counts[cat]}")
        
        return samples


# ============================================================
# RL种子数据构建
# ============================================================
class RLDataBuilder:
    """
    RL对抗博弈所需数据 (含 expression / toxic_type 字段):
    - train_seed: 种子池
    - val_eval: RL过程评估
    - test_eval: 最终评测
    """

    @classmethod
    def build(cls, train_df, val_df, test_df, output_dir: Path):
        """构建RL数据"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 统一列名
        col_map = {'文本': 'original_text', '标签': 'category'}
        for name, df, filename in [
            ("RL种子池(train)", train_df, "train_seed.parquet"),
            ("RL评估(val)", val_df, "val_eval.parquet"),
            ("最终评测(test)", test_df, "test_eval.parquet"),
        ]:
            rl_df = df.rename(columns=col_map)
            # 确保 expression_label / toxic_type_label 列存在
            if 'expression_label' not in rl_df.columns:
                rl_df['expression_label'] = '显式仇恨'
            if 'toxic_type_label' not in rl_df.columns:
                rl_df['toxic_type_label'] = '仇恨言论'
            path = output_dir / filename
            rl_df.to_parquet(path, index=False)
            print(f"  {name}: {len(rl_df)} 条 → {path.name}")

            # JSON版本 (保留全字段)
            json_path = path.with_suffix('.json')
            json_cols = ['文本', '标签']
            if 'toxic_type_label' in df.columns:
                json_cols += ['toxic_type_label', 'expression_label']
            records = df[[c for c in json_cols if c in df.columns]].to_dict('records')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="一站式数据准备: 从split_data生成SFT+RL全部训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--split_dir", type=str,
        default="../../split_data",
        help="split_data目录路径 (由 convert_repo_data.py 生成)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="../../prepared_data",
        help="输出目录"
    )
    args = parser.parse_args()
    
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 70)
    print("一站式数据准备 (v3: 读取 convert_repo_data.py 富字段格式)")
    print("=" * 70)
    
    # ========================================
    # Step 1: 加载富字段 split_data
    # ========================================
    print("\n[1/4] 加载split数据...")
    train_df = load_split_data(split_dir, "train")
    val_df = load_split_data(split_dir, "val")
    test_df = load_split_data(split_dir, "test")

    # 打印分布
    print(f"\n  Train expression 分布: {train_df['expression_label'].value_counts().to_dict()}")
    print(f"  Train toxic_type 分布: {train_df['toxic_type_label'].value_counts().to_dict()}")
    print(f"  Train 标签分布: {train_df['标签'].value_counts().to_dict()}")
    
    # ========================================
    # Step 2: Challenger SFT数据
    # ========================================
    print(f"\n[2/4] 构建Challenger SFT数据...")
    challenger_dir = output_dir / "challenger_sft"
    ChallengerSFTBuilder.build(train_df, "train", challenger_dir)
    ChallengerSFTBuilder.build(val_df, "val", challenger_dir)
    
    # ========================================
    # Step 3: Reviewer SFT数据
    # ========================================
    print(f"\n[3/4] 构建Reviewer SFT数据...")
    reviewer_dir = output_dir / "reviewer_sft"
    ReviewerSFTBuilder.build(train_df, "train", reviewer_dir)
    ReviewerSFTBuilder.build(val_df, "val", reviewer_dir)
    
    # ========================================
    # Step 4: RL种子数据
    # ========================================
    print(f"\n[4/4] 构建RL种子数据...")
    rl_dir = output_dir / "rl"
    RLDataBuilder.build(train_df, val_df, test_df, rl_dir)
    
    # ========================================
    # 生成报告
    # ========================================
    train_cats = train_df['标签'].value_counts().to_dict()
    report = {
        "description": "一站式数据准备报告",
        "data_strategy": {
            "principle": "train全部用于SFT冷启动 + RL种子池",
            "reason": [
                "SFT教会模型基础能力(生成/分类)",
                "RL通过对抗博弈优化策略，训练数据是Challenger新生成的对抗文本",
                "种子池只是采样起点，不等于RL训练数据",
                "论文叙事清晰: 同数据下RL vs SFT的对比"
            ]
        },
        "data_usage": {
            "train (6654)": {
                "challenger_sft": f"{sum(1 for _, r in train_df.iterrows() if r['标签'] != '无毒')} 条 (排除无毒)",
                "reviewer_sft": f"{len(train_df)} 条 (含无毒)",
                "rl_seed_pool": f"{len(train_df)} 条"
            },
            "val (1426)": "RL过程评估 + SFT验证",
            "test (1427)": "最终评测 (不参与任何训练)"
        },
        "output_files": {
            "challenger_sft/": "Challenger冷启动训练数据 (category→text)",
            "reviewer_sft/": "Reviewer冷启动训练数据 (text→category)",
            "rl/train_seed.parquet": "RL种子池 (列名: original_text, category)",
            "rl/val_eval.parquet": "RL过程评估",
            "rl/test_eval.parquet": "最终评测"
        },
        "column_mapping": {
            "SFT数据": "instruction/output格式 (适配LLaMA-Factory等)",
            "RL数据": "original_text/category (适配adversarial_grpo_trainer.py)"
        }
    }
    
    report_path = output_dir / "data_preparation_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # ========================================
    # 打印摘要
    # ========================================
    print("\n" + "=" * 70)
    print("✓ 全部数据准备完成!")
    print("=" * 70)
    print(f"\n输出目录: {output_dir}")
    print(f"""
目录结构:
  {output_dir}/
  ├── challenger_sft/
  │   ├── train.jsonl / train.parquet    ← Challenger LoRA SFT
  │   └── val.jsonl / val.parquet        ← SFT验证
  ├── reviewer_sft/
  │   ├── train.jsonl / train.parquet    ← Reviewer LoRA SFT
  │   └── val.jsonl / val.parquet        ← SFT验证
  ├── rl/
  │   ├── train_seed.parquet/.json       ← 对抗RL种子池
  │   ├── val_eval.parquet/.json         ← RL每轮评估
  │   └── test_eval.parquet/.json        ← 最终评测
  └── data_preparation_report.json

训练流程:
  Step 1: Challenger SFT → python train_challenger_lora.py --data challenger_sft/train.jsonl
  Step 2: Reviewer SFT   → python train_reviewer_lora.py --data reviewer_sft/train.jsonl
  Step 3: 对抗RL训练     → python adversarial_grpo_trainer.py \\
                              --train_data rl/train_seed.parquet \\
                              --test_data rl/val_eval.parquet
  Step 4: 最终评测       → python baseline_eval_vllm.py --data rl/test_eval.json
""")


if __name__ == "__main__":
    main()
