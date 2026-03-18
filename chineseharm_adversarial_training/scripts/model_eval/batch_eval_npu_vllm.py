#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于ToxiCN有毒语言检测的NPU适配版本 —— vLLM加速版
- 使用vLLM离线推理引擎，大幅提升吞吐
- 通过tensor_parallel_size支持NPU多卡并行
- 避免手动多进程/多线程，由vLLM内部管理并行
- 设置VLLM_WORKER_MULTIPROC_METHOD=spawn防止NPU多进程错误
"""

import os
# ===== 多进程安全：必须在所有其他import之前设置 =====
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json
import argparse
from collections import defaultdict
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ToxiCN有毒语言分类规则
rules = """
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

HARMFUL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"]
TOXIC_TYPES = ["一般攻击性", "仇恨言论"]
EXPRESSIONS = ["显式仇恨", "隐式仇恨", "转述"]

SYSTEM_PROMPT = "You are a helpful assistant."


def build_messages(text: str) -> list[dict]:
    """构造单条样本的对话消息列表"""
    prompt = (
        '作为中文有毒语言检测专家，请判断以下文本是否包含有毒内容。\n'
        '请严格按以下格式逐行输出：\n'
        '第一行：输出"判断：有害"或"判断：无害"\n'
        '第二行（仅当有害时）：输出"类别："加具体类别名（"性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"）\n'
        '第三行（仅当有害时）：输出"毒性类型："加类型（"一般攻击性"或"仇恨言论"）\n'
        '第四行（仅当有害时）：输出"表达方式："加方式（"显式仇恨"、"隐式仇恨"或"转述"）\n\n'
        '注意：隐含歧视（如反讽、暗喻、阴阳怪气）也属于有毒内容；转述仇恨事件但本身不有毒的内容应判为无毒。\n\n'
        f'规则: {rules}\n'
        f'文本: {text}'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def extract_prediction_full(output_text: str) -> dict:
    """提取4维预测：二分类 + 类别 + 毒性类型 + 表达方式"""
    all_categories = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]
    result = {"binary": None, "category": None, "toxic_type": None, "expression": None}
    if not output_text:
        return result

    text = output_text.strip()

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
        for tt in TOXIC_TYPES:
            if tt in text:
                result["toxic_type"] = tt
                break
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

    # 兼容旧格式
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


def calculate_metrics(results):
    """
    计算准确率和F1指标（ToxiCN有毒语言检测格式）
    包含：Precision、Recall、F1-Score
    """
    label_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})

    correct = 0
    total = 0
    failed_extractions = 0

    for item in results:
        true_label = item["标签"]
        pred_label = item["predict_label"]

        total += 1

        if pred_label is None:
            failed_extractions += 1
            label_stats[true_label]['FN'] += 1
        elif pred_label == true_label:
            correct += 1
            label_stats[true_label]['TP'] += 1
        else:
            label_stats[pred_label]['FP'] += 1
            label_stats[true_label]['FN'] += 1

    accuracy = correct / total * 100 if total > 0 else 0

    label_order = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]

    category_metrics = {}
    total_f1 = 0

    print(f"\n{'='*80}")
    print(f"评测结果")
    print(f"{'='*80}")
    print(f"总体准确率: {accuracy:.2f}% ({correct}/{total})")

    if failed_extractions > 0:
        print(f"⚠️  提取失败: {failed_extractions} 样本")

    print(f"\n{'='*80}")
    print(f"各类别详细指标:")
    print(f"{'='*80}")
    print(f"{'类别':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'样本数':<8}")
    print(f"{'-'*80}")

    for label in label_order:
        stats = label_stats[label]
        TP = stats['TP']
        FP = stats['FP']
        FN = stats['FN']

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        total_f1 += f1_score
        sample_count = TP + FN

        category_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "sample_count": sample_count
        }

        print(f"{label:<12} {precision:<10.4f} {recall:<10.4f} {f1_score:<10.4f} {sample_count:<8}")

    macro_f1 = total_f1 / len(label_order)

    total_samples_w = sum(label_stats[lb]['TP'] + label_stats[lb]['FN'] for lb in label_order)
    weighted_f1 = sum(
        category_metrics[lb]['f1_score'] * (label_stats[lb]['TP'] + label_stats[lb]['FN'])
        for lb in label_order
    ) / total_samples_w if total_samples_w > 0 else 0
    weighted_precision = sum(
        category_metrics[lb]['precision'] * (label_stats[lb]['TP'] + label_stats[lb]['FN'])
        for lb in label_order
    ) / total_samples_w if total_samples_w > 0 else 0
    weighted_recall = sum(
        category_metrics[lb]['recall'] * (label_stats[lb]['TP'] + label_stats[lb]['FN'])
        for lb in label_order
    ) / total_samples_w if total_samples_w > 0 else 0

    # 二分类指标: toxic vs non-toxic
    binary_tp = 0
    binary_tn = 0
    binary_fp = 0
    binary_fn = 0
    for item in results:
        true_is_harmful = item["标签"] in HARMFUL_CATEGORIES
        pred_binary = item.get("predict_binary")
        if pred_binary is None:
            if true_is_harmful:
                binary_fn += 1
            else:
                binary_tn += 1
            continue
        if true_is_harmful and pred_binary == "有害":
            binary_tp += 1
        elif true_is_harmful and pred_binary == "无害":
            binary_fn += 1
        elif not true_is_harmful and pred_binary == "有害":
            binary_fp += 1
        else:
            binary_tn += 1
    binary_p = binary_tp / (binary_tp + binary_fp) if (binary_tp + binary_fp) > 0 else 0
    binary_r = binary_tp / (binary_tp + binary_fn) if (binary_tp + binary_fn) > 0 else 0
    binary_f1 = 2 * binary_p * binary_r / (binary_p + binary_r) if (binary_p + binary_r) > 0 else 0
    binary_acc = (binary_tp + binary_tn) / total * 100 if total > 0 else 0

    binary_metrics = {
        "accuracy": round(binary_acc, 2),
        "precision": round(binary_p, 4),
        "recall": round(binary_r, 4),
        "f1_score": round(binary_f1, 4),
    }

    print(f"{'-'*80}")
    print(f"{'Macro-F1':<12} {' '*10} {' '*10} {macro_f1:<10.4f}")
    print(f"{'Weighted-F1':<12} {' '*10} {' '*10} {weighted_f1:<10.4f}")
    print(f"{'Weighted-P':<12} {weighted_precision:<10.4f}")
    print(f"{'Weighted-R':<12} {weighted_recall:<10.4f}")
    print(f"\n二分类(toxic/non-toxic): Acc={binary_acc:.1f}% P={binary_p:.3f} R={binary_r:.3f} F1={binary_f1:.3f}")
    print(f"{'='*80}")

    return {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "failed_extractions": failed_extractions,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "binary_metrics": binary_metrics,
        "category_metrics": category_metrics
    }


def load_eval_data(data_path: str):
    """加载评测数据，支持json/jsonl/parquet格式，兼容多种字段命名"""

    TOPIC_TO_CATEGORY = {
        "gender": "性别歧视", "race": "种族歧视",
        "region": "地域偏见", "lgbt": "LGBTQ歧视",
    }

    def normalize_record(rec: dict) -> dict:
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
    parser = argparse.ArgumentParser(description="vLLM加速NPU推理脚本")
    parser.add_argument("--data_path", type=str,
                        default="/home/ma-user/work/test/split_data/test.parquet",
                        help="输入文件路径 (json/jsonl/parquet)")
    parser.add_argument("--model_path", type=str,
                        default="/home/ma-user/work/test/models_base/Qwen/Qwen2.5-3B-Instruct",
                        help="模型路径")
    parser.add_argument("--output_dir", type=str,
                        default="/home/ma-user/work/test/eval_results",
                        help="评测结果输出目录")
    parser.add_argument("--num_npus", type=int, default=1,
                        help="使用的NPU数量（tensor_parallel_size）")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="vLLM最大模型上下文长度")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="显存利用率 (0~1)")
    parser.add_argument("--tag", type=str, default="",
                        help="结果文件标签 (如 base, lora)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="vLLM最大并发序列数（max_num_seqs），相当于batch size，越大吞吐越高，显存占用越多")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="LoRA adapter目录路径（可选）。提供时--model_path应为基座模型路径")
    parser.add_argument("--max_lora_rank", type=int, default=64,
                        help="LoRA最大rank，需与训练时一致（默认64）")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="禁用CUDA Graph（NPU兼容性需要时启用）")

    args = parser.parse_args()

    print("=" * 80)
    print("ToxiCN有毒语言检测 vLLM加速NPU批量推理")
    print("=" * 80)
    print(f"模型: {args.model_path}")
    print(f"数据: {args.data_path}")
    print(f"NPU数 (tensor_parallel): {args.num_npus}")
    print(f"最大上下文长度: {args.max_model_len}")
    print(f"显存利用率: {args.gpu_memory_utilization}")
    print(f"Batch size (max_num_seqs): {args.batch_size}")
    if args.lora_path:
        print(f"LoRA adapter: {args.lora_path}")
        print(f"LoRA max rank: {args.max_lora_rank}")
    print(f"enforce_eager: {args.enforce_eager}")

    # 自动检测：若model_path是LoRA目录（含adapter_config.json）但未指定--lora_path，报错提示
    _adapter_cfg = Path(args.model_path) / "adapter_config.json"
    _model_cfg = Path(args.model_path) / "config.json"
    if _adapter_cfg.exists() and not _model_cfg.exists():
        if not args.lora_path:
            raise ValueError(
                f"检测到 '{args.model_path}' 是LoRA adapter目录（含adapter_config.json）。\n"
                "请使用 --lora_path 指定adapter路径，并用 --model_path 指定基座模型路径。\n"
                "示例：\n"
                f"  --model_path /path/to/base_model --lora_path {args.model_path}"
            )
    print()

    # ========== 1. 加载数据 ==========
    print("加载数据...")
    data = load_eval_data(args.data_path)
    total_data = len(data)
    print(f"✓ 总数据量: {total_data}\n")

    # ========== 2. 构造消息列表 ==========
    print("构造推理prompt...")
    all_messages = [build_messages(item["文本"]) for item in data]
    print(f"✓ 构造完成: {len(all_messages)} 条\n")

    # ========== 3. 初始化vLLM引擎 ==========
    print("初始化vLLM引擎...")
    use_lora = args.lora_path is not None
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.num_npus,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        dtype="bfloat16",
        max_num_seqs=args.batch_size,
        enable_lora=use_lora,
        max_lora_rank=args.max_lora_rank if use_lora else 16,
    )
    print("✓ vLLM引擎初始化完成\n")

    # 采样参数：temperature=0 表示贪心解码（等价于原脚本的do_sample=False）
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
    )

    # 构造LoRA请求（若使用LoRA）
    lora_request = LoRARequest("eval_lora", 1, args.lora_path) if use_lora else None

    # ========== 4. 批量推理 ==========
    print("开始vLLM批量推理...")
    outputs = llm.chat(
        messages=all_messages,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=lora_request,
    )
    print(f"✓ 推理完成: {len(outputs)} 条\n")

    # ========== 5. 解析结果 ==========
    results_all = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text

        pred_full = extract_prediction_full(response)
        predicted_label = pred_full["category"]
        predicted_binary = pred_full["binary"]

        results_all.append({
            "index": i,
            "文本": data[i]["文本"],
            "标签": data[i]["标签"],
            "predict_label": predicted_label,
            "predict_binary": predicted_binary,
            "predict_toxic_type": pred_full.get("toxic_type"),
            "predict_expression": pred_full.get("expression"),
            "response": response
        })

    # ========== 6. 计算指标 ==========
    metrics = calculate_metrics(results_all)

    # ========== 7. 保存结果 ==========
    os.makedirs(args.output_dir, exist_ok=True)
    # 若使用LoRA，结果文件名使用lora目录名，否则用base模型名
    model_name = Path(args.lora_path).name if use_lora else Path(args.model_path).name
    tag = f"_{args.tag}" if args.tag else ""
    output_file = os.path.join(args.output_dir, f"eval_vllm_npu_{model_name}{tag}.json")

    print(f"\n保存结果到: {output_file}")
    output_data = {
        "metrics": metrics,
        "results": results_all
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✓ 完成！")


if __name__ == "__main__":
    main()
