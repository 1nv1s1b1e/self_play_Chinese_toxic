#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToxiCN有毒语言检测 + vLLM加速
- 基于ToxiCN有毒语言分类体系
- vLLM批量推理加速
- 支持多NPU/GPU并行
"""

import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import json
import argparse
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from collections import defaultdict

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM未安装")

# ToxiCN有毒语言分类规则
RULES = """
性别歧视：
1. **对特定性别的贬低与攻击**：使用贬低性词汇攻击特定性别群体，包含"女拳"、"直男癌"、"普信男/女"等网络用语。
2. **性别刻板印象与偏见**：强化对特定性别的刻板认知，将个体行为上升到性别群体层面。
3. **隐含性别歧视**：通过调侃、反讽、段子等方式暗示贬低特定性别群体。

种族歧视：
1. **种族性侮辱与攻击**：使用歧视性词汇指代特定种族群体，对其外貌、文化、能力进行恶意攻击。
2. **种族偏见与刻板印象**：基于种族进行偏颇的负面概括，宣扬种族优越或劣等论。
3. **隐含种族歧视**：以看似客观或"开玩笑"的方式传递种族偏见。

地域偏见：
1. **地域性侮辱与攻击**：针对特定地区人群使用贬义性称呼或进行侮辱，如"xx人都是骗子"等地域黑言论。
2. **地域刻板印象**：将个别负面行为上升为对地区全体居民的评价。
3. **隐含地域歧视**：以段子、调侃等形式传播地域偏见。

LGBTQ歧视：
1. **对LGBTQ群体的恶意攻击**：使用"变态"、"不正常"等词汇攻击LGBTQ群体。
2. **恐同/恐跨偏见**：将同性恋、跨性别视为"疾病"或"道德败坏"。
3. **隐含LGBTQ歧视**：以二元框架暗示否定特定性取向或性别认同。

其他仇恨：
1. **一般性人身攻击**：具有显著恶意的侮辱性语言，使用脏话污言攻击人。
2. **群体性恶意言论**：针对特定职业、年龄等群体的恶意攻击与污名化。
3. **泛化仇恨表达**：对社会群体泛化恶意评价，煽动对立情绪。
"""

SYSTEM_PROMPT = "You are a helpful assistant."

def create_prompt(text: str) -> str:
    """创建ToxiCN格式prompt（4维输出：二分类+类别+毒性类型+表达方式）"""
    prompt = (
        '作为中文有毒语言检测专家，请判断以下文本是否包含有毒内容。\n'
        '请严格按以下格式逐行输出：\n'
        '第一行：输出"判断：有害"或"判断：无害"\n'
        '第二行（仅当有害时）：输出"类别："加具体类别名（"性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"）\n'
        '第三行（仅当有害时）：输出"毒性类型："加类型（"一般攻击性"或"仇恨言论"）\n'
        '第四行（仅当有害时）：输出"表达方式："加方式（"显式仇恨"、"隐式仇恨"或"转述"）\n\n'
        '说明：\n'
        '- 一般攻击性：泛化的侮辱谩骂，不针对特定身份群体的仇恨结构\n'
        '- 仇恨言论：针对特定身份群体（性别、种族、地域、LGBTQ等）的系统性偏见与歧视\n'
        '- 显式仇恨：直白、明显的攻击性语言\n'
        '- 隐式仇恨：使用反讽、暗语、谐音、阴阳怪气等隐晦歧视\n'
        '- 转述：以新闻或讨论口吻呈现的仇恨事件，立场模糊\n\n'
        '注意：隐含歧视也属于有毒内容；转述仇恨事件但本身不有毒的内容应判为无毒。\n\n'
        f'规则: {RULES}\n'
        f'文本: {text}'
    )
    return prompt

HARMFUL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"]
TOXIC_TYPES = ["一般攻击性", "仇恨言论"]
EXPRESSIONS = ["显式仇恨", "隐式仇恨", "转述"]

def extract_prediction_full(output_text: str) -> dict:
    """提取4维预测：二分类 + 类别 + 毒性类型 + 表达方式"""
    all_categories = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]
    result = {"binary": None, "category": None, "toxic_type": None, "expression": None}
    if not output_text:
        return result

    text = output_text.strip()

    # 解析二分类
    has_harmful = ("判断：有害" in text or "判断:有害" in text
                   or "判断: 有害" in text or "判断 ：有害" in text)
    has_safe = ("判断：无害" in text or "判断:无害" in text
                or "判断: 无害" in text or "判断 ：无害" in text)

    if has_harmful:
        result["binary"] = "有害"
        for cat in HARMFUL_CATEGORIES:
            if cat in text:
                result["category"] = cat
                break
        if result["category"] is None:
            result["category"] = "其他仇恨"
        # 解析毒性类型
        for tt in TOXIC_TYPES:
            if tt in text:
                result["toxic_type"] = tt
                break
        # 解析表达方式
        for expr in EXPRESSIONS:
            if expr in text:
                result["expression"] = expr
                break
        return result

    if has_safe:
        result["binary"] = "无害"
        result["category"] = "无毒"
        result["toxic_type"] = "无毒"
        result["expression"] = "非仇恨"
        return result

    # 兼容旧格式（无"判断："前缀）
    for cat in all_categories:
        if cat in text[:30]:
            result["category"] = cat
            result["binary"] = "有害" if cat in HARMFUL_CATEGORIES else "无害"
            return result
    for cat in all_categories:
        if cat in text:
            result["category"] = cat
            result["binary"] = "有害" if cat in HARMFUL_CATEGORIES else "无害"
            return result

    return result

def extract_prediction(output_text: str) -> str:
    """提取预测类别（向后兼容接口）"""
    return extract_prediction_full(output_text)["category"]

def evaluate_with_vllm(
    model_path: str,
    data: List[Dict],
    tensor_parallel_size: int = 1,
    num_debug_samples: int = 20
):
    """
    使用vLLM评测（官方格式）
    """
    print(f"\n{'='*80}")
    print(f"vLLM评测: {model_path}")
    print(f"并行数: {tensor_parallel_size}")
    print(f"{'='*80}")
    
    # 初始化vLLM
    print("初始化vLLM...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",  # NPU推荐bfloat16
        max_model_len=4096,
        tensor_parallel_size=tensor_parallel_size,
    )
    print("✓ vLLM初始化完成")
    
    # vLLM采样参数（对齐官方）
    sampling_params = SamplingParams(
        max_tokens=1024,  # 官方用1024
        temperature=0.0,   # do_sample=False等价
        top_p=1.0,
    )
    
    # 准备prompts（使用apply_chat_template格式）
    print("\n准备prompts...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    formatted_prompts = []
    true_labels = []
    
    for item in data:
        text = item['文本']
        true_label = item['标签']
        
        # 官方格式prompt
        prompt_content = create_prompt(text)
        
        # 使用chat template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_content}
        ]
        
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        formatted_prompts.append(formatted_text)
        true_labels.append(true_label)
    
    print(f"✓ 准备完成: {len(formatted_prompts)} prompts")
    
    # 批量推理
    print("\nvLLM批量推理中...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    # 处理结果
    print("\n处理结果...")
    correct = 0
    total = 0
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    failed_samples = []
    debug_samples = []
    all_predictions = []  # 保存全部预测 (用于expression分层分析)
    
    for i, output in enumerate(tqdm(outputs, desc="评估")):
        true_label = true_labels[i]
        response = output.outputs[0].text.strip()
        
        # 提取预测 (新格式: 二分类 + 类别)
        pred_full = extract_prediction_full(response)
        pred = pred_full["category"]
        pred_binary = pred_full["binary"]
        
        # 判断正确性
        is_correct = (pred == true_label) if pred else False
        
        if is_correct:
            correct += 1
            category_stats[true_label]["correct"] += 1
        
        total += 1
        category_stats[true_label]["total"] += 1
        
        # 保存全部预测
        all_predictions.append({
            "文本": data[i]['文本'],
            "标签": true_label,
            "predict_label": pred,
            "predict_binary": pred_binary,
            "predict_toxic_type": pred_full.get("toxic_type"),
            "predict_expression": pred_full.get("expression"),
            "correct": is_correct,
        })
        
        # 保存调试样本
        if len(debug_samples) < num_debug_samples:
            debug_samples.append({
                "sample_id": i + 1,
                "text": data[i]['文本'],
                "true_label": true_label,
                "predicted_label": pred,
                "correct": is_correct,
                "response": response
            })
        
        # 提取失败样本
        if pred is None:
            failed_samples.append({
                "text": data[i]['文本'][:100],
                "true_label": true_label,
                "response": response[:300]
            })
    
    # 计算准确率
    accuracy = correct / total * 100
    
    # 各类别准确率 + F1
    category_accuracy = {}
    label_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    
    for i, output in enumerate(outputs):
        true_label = true_labels[i]
        response = output.outputs[0].text.strip()
        pred = extract_prediction(response)
        if pred is None:
            label_stats[true_label]["FN"] += 1
        elif pred == true_label:
            label_stats[true_label]["TP"] += 1
        else:
            label_stats[pred]["FP"] += 1
            label_stats[true_label]["FN"] += 1
    
    all_cats = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]
    f1_scores = []
    category_metrics = {}
    
    for cat in all_cats:
        s = label_stats[cat]
        cat_total = category_stats[cat]["total"]
        cat_correct = category_stats[cat]["correct"]
        cat_acc = cat_correct / cat_total * 100 if cat_total > 0 else 0
        
        p = s["TP"] / (s["TP"] + s["FP"]) if (s["TP"] + s["FP"]) > 0 else 0
        r = s["TP"] / (s["TP"] + s["FN"]) if (s["TP"] + s["FN"]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_scores.append(f1)
        
        category_accuracy[cat] = {
            "accuracy": cat_acc,
            "correct": cat_correct,
            "total": cat_total
        }
        category_metrics[cat] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "TP": s["TP"],
            "FP": s["FP"],
            "FN": s["FN"],
            "sample_count": s["TP"] + s["FN"]
        }
    
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    # Weighted F1 (ToxiCN原始论文使用weighted平均)
    total_samples = sum(label_stats[c]["TP"] + label_stats[c]["FN"] for c in all_cats)
    weighted_f1 = sum(
        f1_scores[i] * (label_stats[all_cats[i]]["TP"] + label_stats[all_cats[i]]["FN"])
        for i in range(len(all_cats))
    ) / total_samples if total_samples > 0 else 0
    weighted_precision = sum(
        category_metrics.get(all_cats[i], {}).get("precision", 0) * (label_stats[all_cats[i]]["TP"] + label_stats[all_cats[i]]["FN"])
        for i in range(len(all_cats))
    ) / total_samples if total_samples > 0 else 0
    weighted_recall = sum(
        category_metrics.get(all_cats[i], {}).get("recall", 0) * (label_stats[all_cats[i]]["TP"] + label_stats[all_cats[i]]["FN"])
        for i in range(len(all_cats))
    ) / total_samples if total_samples > 0 else 0
    
    # 二分类指标: toxic vs non-toxic (ToxiCN Task I)
    # 使用模型显式输出的二分类判断 (判断：有害/无害)
    binary_tp = 0  # 真有害 & 预测有害
    binary_tn = 0  # 真无害 & 预测无害
    binary_fp = 0  # 真无害 & 预测有害
    binary_fn = 0  # 真有害 & 预测无害
    binary_unparsed = 0
    for ap in all_predictions:
        true_is_harmful = ap["标签"] in HARMFUL_CATEGORIES
        pred_binary = ap.get("predict_binary")
        if pred_binary is None:
            if true_is_harmful:
                binary_fn += 1
            else:
                binary_tn += 1  # 保守: 无法解析视为未判有害
            binary_unparsed += 1
            continue
        if true_is_harmful and pred_binary == "有害":
            binary_tp += 1
        elif true_is_harmful and pred_binary == "无害":
            binary_fn += 1
        elif not true_is_harmful and pred_binary == "有害":
            binary_fp += 1
        else:
            binary_tn += 1
    binary_precision = binary_tp / (binary_tp + binary_fp) if (binary_tp + binary_fp) > 0 else 0
    binary_recall = binary_tp / (binary_tp + binary_fn) if (binary_tp + binary_fn) > 0 else 0
    binary_f1 = 2 * binary_precision * binary_recall / (binary_precision + binary_recall) if (binary_precision + binary_recall) > 0 else 0
    binary_acc = (binary_tp + binary_tn) / total if total > 0 else 0
    
    binary_metrics = {
        "accuracy": round(binary_acc * 100, 2),
        "precision": round(binary_precision, 4),
        "recall": round(binary_recall, 4),
        "f1_score": round(binary_f1, 4),
    }
    
    print(f"\n总体准确率: {accuracy:.2f}% ({correct}/{total})")
    print(f"Macro-F1: {macro_f1:.4f}  |  Weighted-F1: {weighted_f1:.4f}")
    print(f"Weighted-P: {weighted_precision:.4f}  |  Weighted-R: {weighted_recall:.4f}")
    print(f"\n二分类(toxic/non-toxic): Acc={binary_metrics['accuracy']:.1f}% P={binary_precision:.3f} R={binary_recall:.3f} F1={binary_f1:.3f}")
    print(f"\n各类别指标:")
    for cat in all_cats:
        ca = category_accuracy.get(cat, {})
        cm = category_metrics.get(cat, {})
        print(f"  {cat:12s}: Acc={ca.get('accuracy',0):5.1f}% P={cm.get('precision',0):.3f} R={cm.get('recall',0):.3f} F1={cm.get('f1_score',0):.3f} ({ca.get('correct',0)}/{ca.get('total',0)})")
    
    if failed_samples:
        print(f"\n⚠️  提取失败: {len(failed_samples)} 样本")
        print("失败样本预览（前3个）:")
        for i, sample in enumerate(failed_samples[:3]):
            print(f"\n  [{i+1}] {sample['true_label']}")
            print(f"      文本: {sample['text']}...")
            print(f"      响应: {sample['response'][:150]}...")
    
    # 调试样本
    print(f"\n{'='*80}")
    print(f"调试样本（前3个）:")
    for i, sample in enumerate(debug_samples[:3]):
        print(f"\n样本 {sample['sample_id']}:")
        print(f"  文本: {sample['text'][:50]}...")
        print(f"  真实: {sample['true_label']} | 预测: {sample['predicted_label']} | {'✓' if sample['correct'] else '✗'}")
        print(f"  响应: {sample['response'][:100]}...")
    print(f"{'='*80}")
    
    # 清理显存
    del llm
    import gc
    gc.collect()
    
    import torch
    if torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "model_path": model_path,
        "overall_accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "binary_metrics": binary_metrics,
        "correct": correct,
        "total": total,
        "category_accuracy": category_accuracy,
        "category_metrics": category_metrics,
        "failed_extractions": len(failed_samples),
        "failed_samples": failed_samples[:10],
        "debug_samples": debug_samples,
        "all_predictions": all_predictions
    }

def load_eval_data(data_path: str):
    """加载评测数据，支持json/jsonl/parquet格式，兼容多种字段命名"""

    TOPIC_TO_CATEGORY = {
        "gender": "性别歧视", "race": "种族歧视",
        "region": "地域偏见", "lgbt": "LGBTQ歧视",
    }

    def normalize_record(rec: dict) -> dict:
        """统一字段名: 确保有 文本 和 标签"""
        if '文本' not in rec and 'content' in rec:
            rec['文本'] = rec['content']
        if '标签' not in rec and 'topic' in rec:
            if rec.get('toxic', 0) == 0:
                rec['标签'] = '无毒'
            elif rec.get('toxic_type', 2) == 1:
                rec['标签'] = '其他仇恨'
            else:
                rec['标签'] = TOPIC_TO_CATEGORY.get(rec['topic'], '其他仇恨')
        return rec

    if data_path.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(data_path)
        data = []
        for _, row in df.iterrows():
            data.append(normalize_record({
                "文本": row.get("original_text", row.get("文本", row.get("content", ""))),
                "标签": row.get("category", row.get("标签", ""))
            }))
        return data
    elif data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(normalize_record(json.loads(line.strip())))
        return data
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return [normalize_record(r) for r in raw]


def main():
    parser = argparse.ArgumentParser(description="官方格式 + vLLM加速")
    parser.add_argument("--model_path", type=str,
                       default="/home/ma-user/work/test/models_base/Qwen/Qwen2.5-3B-Instruct",
                       help="模型路径")
    parser.add_argument("--data_path", type=str,
                       default="/home/ma-user/work/test/split_data/test.parquet",
                       help="评测数据路径 (json/jsonl/parquet)")
    parser.add_argument("--output_dir", type=str,
                       default="/home/ma-user/work/test/eval_results",
                       help="评测结果输出目录")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="样本数")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="并行数（NPU/GPU）")
    parser.add_argument("--num_debug_samples", type=int, default=20,
                       help="调试样本数")
    parser.add_argument("--tag", type=str, default="",
                       help="结果文件标签 (如 base, lora)")
    
    args = parser.parse_args()
    
    if not VLLM_AVAILABLE:
        print("错误: vLLM未安装")
        return
    
    print("=" * 80)
    print("ToxiCN有毒语言检测 + vLLM加速")
    print("=" * 80)
    print(f"模型: {args.model_path}")
    print(f"数据: {args.data_path}")
    print(f"并行: {args.tensor_parallel_size}")
    print()
    
    # 加载数据
    print("加载数据...")
    data = load_eval_data(args.data_path)
    
    if args.num_samples and args.num_samples < len(data):
        data = data[:args.num_samples]
    
    print(f"✓ 数据: {len(data)} 样本\n")
    
    # 评估
    results = evaluate_with_vllm(
        args.model_path,
        data,
        args.tensor_parallel_size,
        args.num_debug_samples
    )
    
    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 自动生成输出文件名
    model_name = Path(args.model_path).name
    tag = f"_{args.tag}" if args.tag else ""
    output_path = os.path.join(args.output_dir, f"eval_vllm_{model_name}{tag}.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # 保存时分离all_predictions以减小主文件体积
        all_preds = results.pop("all_predictions", [])
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 单独保存全部预测 (用于expression_analysis.py)
    preds_path = os.path.join(args.output_dir, f"predictions_{model_name}{tag}.json")
    with open(preds_path, 'w', encoding='utf-8') as f:
        json.dump(all_preds, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 结果已保存: {output_path}")
    print(f"✓ 全部预测: {preds_path} ({len(all_preds)} 条)")
    print(f"✓ 调试样本: {len(results.get('debug_samples', []))} 个")

if __name__ == "__main__":
    main()
