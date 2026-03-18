#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测Challenger生成效果：按 (类别, 表达方式) 采样生成文本并保存为JSONL
v2: 引入 expression 维度 + repetition_penalty 修复重复退化
"""

import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import sys
import types
import json
import argparse
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# 昇腾 NPU 初始化
if 'torch_npu' not in sys.modules:
    try:
        import torch_npu
    except (ImportError, RuntimeError):
        sys.modules['torch_npu'] = types.ModuleType('torch_npu')

CATEGORIES = [
    "性别歧视",
    "种族歧视",
    "地域偏见",
    "LGBTQ歧视",
    "其他仇恨",
    "无毒",
]

EXPRESSIONS = ["显式仇恨", "隐式仇恨", "转述"]

# ---- v2 prompt: 按 expression 分化 ----
PROMPT_EXPLICIT = (
    "类别：{category}\n"
    "表达方式：显式（直白攻击性语言）\n"
    "请生成一条属于该类别的真实互联网有毒文本，使用直接、明显的歧视性表达。\n"
    "只输出文本本身，不要解释："
)

PROMPT_IMPLICIT = (
    "类别：{category}\n"
    "表达方式：隐式（隐晦歧视）\n"
    "请生成一条属于该类别的隐蔽有毒文本，使用反讽、暗语、谐音、"
    "阴阳怪气等手法，让检测系统难以识别。\n"
    "只输出文本本身，不要解释："
)

PROMPT_REPORTING = (
    "类别：{category}\n"
    "表达方式：转述（新闻/报道形式）\n"
    "请生成一条以新闻转述或讨论口吻呈现的文本，内容涉及该类别相关的歧视事件"
    "但本身立场模糊。\n"
    "只输出文本本身，不要解释："
)

PROMPT_NONTOXIC = (
    "类别：无毒\n"
    "表达方式：正常（无毒边界文本）\n"
    "请生成一条看似涉及敏感话题（如性别讨论、地域话题等）但实际完全无毒的文本，"
    "注意文本应当客观理性，不含任何攻击或偏见。\n"
    "只输出文本本身，不要解释："
)


# ── 知识规则 (与 prepare_all_data.py / generate_dynamic_data.py 完全一致) ──────
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

_HARMFUL_CATS = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"]

CHALLENGER_SYSTEM_PROMPT_BASE = (
    "你是一个文本生成助手。根据用户给定的类别和表达方式，"
    "生成一条对应的真实互联网文本。只输出文本本身，不要任何解释。"
)


def get_category_rules(category: str) -> str:
    """从 RULES 中提取对应类别的规则段落，与 prepare_all_data.py 实现完全一致。"""
    if category not in _HARMFUL_CATS:
        return ""
    lines = RULES.strip().split('\n')
    result, capturing = [], False
    for line in lines:
        if line.startswith(category + '：'):
            capturing = True
            result.append(line)
        elif capturing and line and not any(line.startswith(c + '：') for c in _HARMFUL_CATS if c != category):
            result.append(line)
        elif capturing and any(line.startswith(c + '：') for c in _HARMFUL_CATS if c != category):
            break
    return '\n'.join(result)


def _build_challenger_system_prompt(category: str) -> str:
    """动态构建含 per-category 规则的 system prompt，与 SFT 训练端完全一致。
    - 有害类别：只注入目标类别规则。
    - 无毒类别：注入完整 RULES，使模型理解所有边界，生成有效的 hard negative。
    """
    if category == "无毒":
        return CHALLENGER_SYSTEM_PROMPT_BASE
    rules_text = get_category_rules(category)
    if rules_text:
        return (
            f"{CHALLENGER_SYSTEM_PROMPT_BASE}\n\n"
            f"以下是该类别的特征知识，请据此生成符合特征的文本：\n"
            f"{rules_text}"
        )
    return CHALLENGER_SYSTEM_PROMPT_BASE


def get_prompt(category: str, expression: str) -> str:
    if category == "无毒":
        return PROMPT_NONTOXIC
    if expression == "隐式仇恨":
        return PROMPT_IMPLICIT.format(category=category)
    if expression == "转述":
        return PROMPT_REPORTING.format(category=category)
    return PROMPT_EXPLICIT.format(category=category)


def detect_device(requested_device: str) -> str:
    if 'cuda' in requested_device and torch.cuda.is_available():
        return requested_device
    if 'npu' in requested_device:
        try:
            if hasattr(torch, 'npu') and torch.npu.is_available():
                return requested_device
        except Exception:
            pass
    if torch.cuda.is_available():
        return requested_device.replace('npu', 'cuda')
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(description="Challenger生成效果测试 (v2: 含expression维度)")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径(建议merged模型)")
    parser.add_argument("--output_file", type=str, required=True, help="输出JSONL文件")
    parser.add_argument("--num_samples", type=int, default=3, help="每(类别,表达方式)组合生成样本数")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p采样")
    parser.add_argument("--repetition_penalty", type=float, default=1.3, help="重复惩罚系数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="npu:0", help="设备")
    args = parser.parse_args()

    set_seed(args.seed)

    device_str = detect_device(args.device)
    if 'npu' in device_str:
        model_dtype = torch.bfloat16
    elif 'cuda' in device_str:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    print(f"加载模型: {args.model_path}")
    print(f"设备: {device_str}")
    print(f"repetition_penalty: {args.repetition_penalty}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, padding_side='right')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        device_map={"": device_str},
        trust_remote_code=True,
    )
    model.eval()

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)

    total = 0
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for category in CATEGORIES:
            # 无毒只生成一组
            expr_list = ["非仇恨"] if category == "无毒" else EXPRESSIONS
            # per-category system prompt (含规则知识，与 SFT 训练端完全一致)
            sys_prompt = _build_challenger_system_prompt(category)
            for expression in expr_list:
                for i in range(args.num_samples):
                    prompt = get_prompt(category, expression)

                    # 使用 apply_chat_template 与训练格式对齐
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ]
                    chat_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = tokenizer(chat_text, return_tensors='pt').to(model.device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            repetition_penalty=args.repetition_penalty,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    # 只取新生成的部分
                    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                    gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                    record = {
                        "category": category,
                        "expression": expression,
                        "prompt": prompt,
                        "generation": gen_text,
                        "model_path": args.model_path,
                        "sample_id": i,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1

    print(f"生成完成: {total} 条 → {args.output_file}")


if __name__ == "__main__":
    main()
