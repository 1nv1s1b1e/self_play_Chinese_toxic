#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Challenger SFT数据准备脚本
将 split_data/{train,val}.parquet/json 转换为 few-shot 多轮指令微调格式，
用于训练恶意生成模型（Challenger）的冷启动 SFT。

格式与 generate_dynamic_data.py (GRPO Phase 0) 完全对齐：
  [system(规则知识), user(指令), assistant(同类随机示例), user(指令), assistant(目标文本)]

指令模板与 GRPO 保持一致：
  "类别：{category}\n表达方式：{expression}\n只输出文本本身，不要解释："
"""

import json
import random
import argparse
import re
from pathlib import Path
from collections import Counter
import pandas as pd
from typing import List, Dict
from tqdm import tqdm


def clean_text(text: str) -> str:
    """
    清洗文本，移除可能导致JSON解析错误的字符。
    - 移除 NULL 字节和其他 C0 控制字符 (保留 \\n\\r\\t)
    - 替换非 UTF-8 安全的字符
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    return text.strip()

# ============================================================
# 知识规则 (基于ToxiCN数据集的有毒语言分类体系，与 Reviewer 共享)
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

def get_category_rules(category: str) -> str:
    """根据类别名返回对应的细粒度特征知识规则"""
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


class ChallengerDataBuilder:
    """构建用于训练Challenger的SFT数据集

    格式与 GRPO Phase 0 完全对齐 (few-shot 多轮对话)：
      [system(规则知识), user(指令), assistant(同类few-shot示例), user(指令), assistant(目标文本)]

    指令格式与 generate_dynamic_data.py 保持一致：
      "类别：{cat}\n表达方式：{expr}\n只输出文本本身，不要解释："
    """

    # 基础 system prompt
    SYSTEM_PROMPT_BASE = (
        "你是一个文本生成助手。根据用户给定的类别和表达方式，"
        "生成一条对应的真实互联网文本。只输出文本本身，不要任何解释。"
    )

    # 指令模板（与 generate_dynamic_data.py 完全相同）
    TEMPLATE = "类别：{category}\n表达方式：{expression}\n只输出文本本身，不要解释："
    
    # 有效的有害类别（不包括无毒）
    HARMFUL_CATEGORIES = {"性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"}

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self, input_file: str) -> List[Dict]:
        """
        加载 split 数据（全量，含无毒样本）。
        优先 .parquet，回退 .json（与 prepare_all_data.py 保持一致）。
        """
        input_path = Path(input_file)
        parquet_path = input_path.with_suffix('.parquet')
        json_path    = input_path.with_suffix('.json')

        if input_path.suffix == '.parquet' and input_path.exists():
            df = pd.read_parquet(input_path)
            print(f"加载 {input_path.name}: {len(df)} 条")
            data = df.to_dict('records')
        elif parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            print(f"加载 {parquet_path.name}: {len(df)} 条")
            data = df.to_dict('records')
        elif json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"加载 {json_path.name}: {len(data)} 条")
        else:
            raise FileNotFoundError(f"找不到数据文件: {input_path}")

        # 打印分布（含无毒）
        cat_counts  = Counter(str(x.get("标签", "")) for x in data)
        expr_counts = Counter(
            str(x.get("expression_label") or x.get("表达方式") or x.get("表达类型") or "非仇恨")
            for x in data
        )
        print(f"  类别分布: {dict(sorted(cat_counts.items()))}")
        print(f"  表达分布: {dict(sorted(expr_counts.items()))}")
        return data
    
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
            return (
                f"{cls.SYSTEM_PROMPT_BASE}\n\n"
                f"你需要生成看似涉及敏感话题但实际完全无毒的文本（hard negative）。"
                f"以下是各类有害语言的识别边界，请确保生成内容不触碰这些边界：\n"
                f"{RULES.strip()}"
            )
        rules_text = get_category_rules(category)
        if rules_text:
            return (
                f"{cls.SYSTEM_PROMPT_BASE}\n\n"
                f"以下是该类别的特征知识，请据此生成符合特征的文本：\n"
                f"{rules_text}"
            )
        return cls.SYSTEM_PROMPT_BASE
        
    def transform_to_sft(self, data: List[Dict]) -> List[Dict]:
        """
        转换为 few-shot 多轮 SFT 格式，与 GRPO Phase 0 完全对齐。

        消息结构：
          [system(规则), user(指令), assistant(同类随机示例), user(指令), assistant(目标文本)]
        """
        # 按类别+表达方式建立索引，方便随机采样 few-shot
        from collections import defaultdict
        cat_expr_pool: dict = defaultdict(list)
        for item in data:
            cat  = str(item.get("标签", "")).strip()
            # 兼容 bench.json ("表达方式"/"表达类型") 和 split_data ("expression_label")
            expr = (item.get("expression_label") or
                    item.get("表达方式") or
                    item.get("表达类型") or "显式仇恨")
            expr = str(expr).strip()
            text = clean_text(item.get("文本", ""))
            if cat and text:
                cat_expr_pool[(cat, expr)].append(text)

        sft_data = []

        for item in tqdm(data, desc="转换数据"):
            text     = clean_text(item.get("文本", ""))
            category = str(item.get("标签", "")).strip()
            expr     = str(item.get("expression_label") or
                           item.get("表达方式") or
                           item.get("表达类型") or "显式仇恨").strip()

            if not text or not category:
                continue

            instruction   = self.TEMPLATE.format(category=category, expression=expr)
            system_prompt = self._build_system_prompt(category)

            # 从同类别同表达方式的池中随机选一条不同的文本作为 few-shot 示例
            pool = [t for t in cat_expr_pool[(category, expr)] if t != text]
            if not pool:
                pool = cat_expr_pool[(category, expr)]  # fallback：允许自身
            few_shot = random.choice(pool) if pool else ""

            # few-shot 多轮消息（与 generate_dynamic_data.py 完全一致）
            msgs = [{"role": "system", "content": system_prompt}]
            if few_shot:
                msgs.append({"role": "user",      "content": instruction})
                msgs.append({"role": "assistant", "content": few_shot})
            msgs.append({"role": "user",      "content": instruction})
            msgs.append({"role": "assistant", "content": text})

            sft_data.append({
                "messages":       msgs,
                "original_label": category,
                "expression":     expr,
            })

        # 打印分布统计（与 prepare_all_data.py 风格一致）
        cat_counts  = Counter(s["original_label"] for s in sft_data)
        expr_counts = Counter(s["expression"]     for s in sft_data)
        print(f"  转换完成: {len(sft_data)} 条")
        print(f"  类别分布: {dict(sorted(cat_counts.items()))}")
        print(f"  表达分布: {dict(sorted(expr_counts.items()))}")

        return sft_data
        
    def save_data(self, sft_data: List[Dict], split_name: str = "train"):
        """保存为JSONL和Parquet格式，split_name 为 train / val / test"""
        # 1. 保存JSONL (messages格式，训练脚本直接使用)
        jsonl_path = self.output_dir / f"{split_name}.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in sft_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✓ JSONL已保存: {jsonl_path}")

        # 2. 保存Parquet (flat格式，便于分析，与 prepare_all_data.py 字段一致)
        flat_samples = [{
            "instruction":  s["messages"][1]["content"],
            "input":        "",
            "output":       s["messages"][-1]["content"],  # 最后一轮 assistant = 目标文本
            "system_prompt": s["messages"][0]["content"],
            "category":     s["original_label"],
            "expression":   s["expression"],
        } for s in sft_data]
        parquet_path = self.output_dir / f"{split_name}.parquet"
        pd.DataFrame(flat_samples).to_parquet(parquet_path, index=False)
        print(f"✓ Parquet已保存: {parquet_path}")

        # 打印示例（仅 train）
        if split_name == "train":
            print("\n=== 数据示例（few-shot 多轮格式）===")
            for i in range(min(2, len(sft_data))):
                sample = sft_data[i]
                for msg in sample["messages"]:
                    role = msg["role"].upper().ljust(9)
                    print(f"  [{role}] {str(msg['content'])[:100]}")
                print("-" * 50)

def main():
    parser = argparse.ArgumentParser(
        description="生成Challenger SFT训练数据 (few-shot多轮 + 规则知识注入)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train_file", type=str,
                        default="../../split_data/train.json",
                        help="训练集split文件路径，支持 .json / .parquet")
    parser.add_argument("--val_file", type=str,
                        default="../../split_data/val.json",
                        help="验证集split文件路径，支持 .json / .parquet")
    parser.add_argument("--output_dir", type=str,
                        default="../../prepared_data/challenger_sft",
                        help="输出目录")

    args = parser.parse_args()

    builder = ChallengerDataBuilder(args.output_dir)

    for split_name, file_path in [
        ("train", args.train_file),
        ("val",   args.val_file),
    ]:
        if not Path(file_path).with_suffix('.json').exists() \
                and not Path(file_path).with_suffix('.parquet').exists() \
                and not Path(file_path).exists():
            print(f"⚠️  跳过 {split_name}: 文件不存在 {file_path}")
            continue
        print(f"\n{'='*60}")
        print(f"处理 {split_name} split ...")
        print(f"{'='*60}")
        data = builder.load_data(file_path)
        sft_data = builder.transform_to_sft(data)
        builder.save_data(sft_data, split_name)
        print(f"✓ {split_name}: {len(sft_data)} 条SFT数据")

    print(f"\n{'='*60}")
    print("✓ 全部完成！(messages格式 + few-shot多轮 + 规则知识注入)")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
