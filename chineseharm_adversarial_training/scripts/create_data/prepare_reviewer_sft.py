#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reviewer SFT数据准备 (v2: Rationale 增强版)
将ToxiCN数据转换为有毒语言分类任务的SFT格式
基于ToxiCN数据集的有毒语言分类体系

v2改进: 支持读取 rationale 数据，输出格式为 "分析 + 判断标签"
  当提供 --rationale_file 时：
    output = "分析：{rationale}\n判断：有害\n类别：..."
  当不提供时：
    output = "判断：有害\n类别：..."  (向后兼容)
"""

import json
import argparse
from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional
from tqdm import tqdm

# ToxiCN有毒语言分类规则
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

# 类别默认 toxic_type / expression（与 constants.py 的 CAT_DEFAULTS 一致）
CAT_DEFAULTS = {
    "性别歧视": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "种族歧视": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "地域偏见": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "LGBTQ歧视": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "其他仇恨":  {"toxic_type": "一般攻击性", "expression": "显式仇恨"},
}


class ReviewerDataBuilder:
    """构建Reviewer分类模型的SFT数据"""
    
    def __init__(self, input_file: str, output_dir: str, rationale_map: Optional[Dict] = None):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rationale_map = rationale_map or {}
    
    @staticmethod
    def create_official_prompt(text: str) -> str:
        """创建与 constants.py REVIEWER_USER_TEMPLATE 一致的 prompt"""
        prompt = (
            '作为中文有毒语言检测专家，请对以下文本进行分析和判断。\n\n'
            '请按以下格式输出：\n'
            '第一步：输出"分析："，然后用2-4句话分析文本中的关键表达、修辞手法及其对应的分类规则特征。\n'
            '第二步：输出"判断：有害"或"判断：无害"\n'
            '（以下仅当有害时输出）\n'
            '第三步：输出"类别："加具体类别名（"性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"）\n'
            '第四步：输出"毒性类型："加类型（"一般攻击性"或"仇恨言论"）\n'
            '第五步：输出"表达方式："加方式（"显式仇恨"、"隐式仇恨"或"转述"）\n\n'
            '注意：隐式仇恨包括反讽、暗语、谐音、阴阳怪气等手法；'
            '转述指报道/讨论仇恨事件但自身立场模糊的内容。\n\n'
            f'规则: {RULES}\n'
            f'文本: {text}'
        )
        return prompt
    
    @staticmethod
    def build_assistant_output(category: str, rationale: str = "",
                               toxic_type: str = "", expression: str = "") -> str:
        """
        构建 assistant 的输出内容（分析 + 判断标签）。
        与 convert_grpo_to_sft.py 的格式保持一致。
        """
        parts = []
        if rationale:
            parts.append(f"分析：{rationale}")
        
        if category in ["无毒", ""]:
            parts.append("判断：无害")
        else:
            parts.append("判断：有害")
            parts.append(f"类别：{category}")
            if toxic_type and toxic_type != "无毒":
                parts.append(f"毒性类型：{toxic_type}")
            if expression and expression != "非仇恨":
                parts.append(f"表达方式：{expression}")
        
        return "\n".join(parts)
    
    def load_data(self) -> List[Dict]:
        """加载数据"""
        print(f"加载数据: {self.input_file}")
        
        if str(self.input_file).endswith('.json'):
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif str(self.input_file).endswith('.parquet'):
            df = pd.read_parquet(self.input_file)
            data = df.to_dict('records')
        else:
            raise ValueError("不支持的文件格式")
        
        print(f"✓ 加载 {len(data)} 条样本")
        return data
    
    def transform_to_sft(self, data: List[Dict]) -> List[Dict]:
        """转换为SFT格式（使用与 constants.py 一致的 prompt + 分析+判断 输出）"""
        sft_data = []
        rationale_hit = 0
        
        for item in tqdm(data, desc="转换数据"):
            text = item.get("文本", "")
            label = item.get("标签", "")
            
            if not text or not label:
                continue
            
            # 在 rationale_map 中查找
            text_key = text.strip()[:200]
            r_info = self.rationale_map.get(text_key, {})
            rationale = r_info.get("rationale", "")
            if rationale:
                rationale_hit += 1
            
            # 获取 toxic_type 和 expression
            # 优先从 rationale 文件，其次从类别默认值
            if label in CAT_DEFAULTS:
                defaults = CAT_DEFAULTS[label]
                toxic_type = r_info.get("toxic_type", "") or defaults["toxic_type"]
                expression = r_info.get("expression", "") or defaults["expression"]
            elif label == "无毒":
                toxic_type = "无毒"
                expression = "非仇恨"
            else:
                toxic_type = r_info.get("toxic_type", "")
                expression = r_info.get("expression", "")
            
            # 构建 assistant 输出
            output = self.build_assistant_output(
                category=label, rationale=rationale,
                toxic_type=toxic_type, expression=expression
            )
            
            # 使用 messages 格式（与 self-play pipeline 一致）
            instruction = self.create_official_prompt(text)
            
            sample = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": output},
                ],
                "category": label,
                "toxic_type": toxic_type,
                "expression": expression,
                "original_text": text,
            }
            sft_data.append(sample)
        
        if self.rationale_map:
            print(f"  → Rationale 命中: {rationale_hit}/{len(data)} ({rationale_hit/max(1,len(data))*100:.1f}%)")
        
        return sft_data
    
    def save_data(self, sft_data: List[Dict], split_name: str):
        """保存数据"""
        # Parquet
        parquet_path = self.output_dir / f"reviewer_{split_name}.parquet"
        pd.DataFrame(sft_data).to_parquet(parquet_path, index=False)
        print(f"✓ Parquet: {parquet_path}")
        
        # JSONL
        jsonl_path = self.output_dir / f"reviewer_{split_name}.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in sft_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✓ JSONL: {jsonl_path}")
        
        return len(sft_data)

def load_rationale_file(rationale_path: str) -> Dict[str, Dict]:
    """
    加载 rationale 数据文件，构建 text -> {rationale, toxic_type, expression} 的映射。
    支持 generate_rationales.py 生成的 JSONL 格式。
    """
    if not rationale_path or not Path(rationale_path).exists():
        return {}
    
    rationale_map = {}
    loaded = 0
    skipped = 0
    
    with open(rationale_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                text = item.get("text", "").strip()[:200]
                rationale = item.get("rationale", "")
                is_valid = item.get("rationale_valid", True)
                
                if text and rationale and is_valid:
                    rationale_map[text] = {
                        "rationale": rationale,
                        "toxic_type": item.get("toxic_type", ""),
                        "expression": item.get("expression", ""),
                    }
                    loaded += 1
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1
    
    print(f"  ✓ 加载 rationale: {loaded} 条有效, {skipped} 条跳过")
    return rationale_map


def main():
    parser = argparse.ArgumentParser(description="Reviewer SFT数据准备 (v2: Rationale 增强)")
    parser.add_argument("--train_file", type=str,
                       default="./data/splits/train.json",
                       help="训练集文件")
    parser.add_argument("--val_file", type=str,
                       default="./data/splits/val.json",
                       help="验证集文件")
    parser.add_argument("--test_file", type=str,
                       default="./data/splits/test.json",
                       help="测试集文件")
    parser.add_argument("--output_dir", type=str,
                       default="./data/reviewer_sft",
                       help="输出目录")
    parser.add_argument("--rationale_file", type=str, default="",
                       help="rationale 数据文件 (generate_rationales.py 输出的 .jsonl)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Reviewer SFT数据准备 (v2: Rationale 增强)")
    print("=" * 60)
    
    # 加载 rationale 数据
    rationale_map = {}
    if args.rationale_file:
        print(f"\n加载 rationale: {args.rationale_file}")
        rationale_map = load_rationale_file(args.rationale_file)
    else:
        print("\n⚠️  未提供 --rationale_file，将使用无 rationale 的基线格式")
    
    stats = {}
    
    for split_name, file_path in [
        ('train', args.train_file),
        ('val', args.val_file),
        ('test', args.test_file)
    ]:
        if not Path(file_path).exists():
            print(f"⚠️  跳过 {split_name}: 文件不存在 {file_path}")
            continue
        
        print(f"\n处理 {split_name} ...")
        # 只对训练集注入 rationale，验证/测试集不注入（防止泄漏）
        r_map = rationale_map if split_name == 'train' else {}
        builder = ReviewerDataBuilder(file_path, args.output_dir, rationale_map=r_map)
        data = builder.load_data()
        sft_data = builder.transform_to_sft(data)
        count = builder.save_data(sft_data, split_name)
        stats[split_name] = count
    
    print("\n" + "=" * 60)
    print("总结:")
    for split_name, count in stats.items():
        print(f"  {split_name}: {count} 样本")
    if rationale_map:
        print(f"  Rationale 数据: {len(rationale_map)} 条可用")
    print("=" * 60)
    print("\n可以使用这些数据训练Reviewer LoRA模型")

if __name__ == "__main__":
    main()
