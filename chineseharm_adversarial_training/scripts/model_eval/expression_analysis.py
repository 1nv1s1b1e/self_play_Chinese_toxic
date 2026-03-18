#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于expression类型的难度分层评估工具
====================================

ToxiCN原始标注体系 (Monitor Toxic Frame):
  Level I   : Whether Toxic (toxic=0/1)
  Level II  : Toxic Type (non-toxic=0, offensive=1, hate_speech=2)
  Level III : Targeted Group (LGBTQ/Region/Sexism/Racism/Others/non-hate)
  Level IV  : Expression Category (non-hate=0, explicit=1, implicit=2, reporting=3)

本脚本利用ToxiCN_1.0.csv的完整标注，按expression类型分析模型表现。
implicit(隐式) 和 reporting(转述) 是最具挑战性的样本。

输入:
  --eval_results: 评测结果JSON (含predict_label和原文)
  --toxicn_csv:   ToxiCN_1.0.csv原始数据 (含expression字段)

输出:
  - 按expression分层的准确率/F1
  - 难度梯度分析
  - 最适合论文使用的对比表格
"""

import json
import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd

# ToxiCN原始标注映射
EXPRESSION_MAP = {0: "non-hate", 1: "explicit", 2: "implicit", 3: "reporting"}
TARGET_INDEX = {0: "LGBTQ歧视", 1: "地域偏见", 2: "性别歧视", 3: "种族歧视", 4: "其他仇恨", 5: "无毒"}
TOXIC_TYPE_MAP = {0: "non-toxic", 1: "offensive_language", 2: "hate_speech"}

ALL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]


def load_toxicn_csv(csv_path: str) -> pd.DataFrame:
    """
    加载ToxiCN_1.0.csv，解析多标签target + expression字段
    CSV列: post, toxic, toxic_type, expression, target
    target是多标签列表, 如 [0, 0, 1, 0, 0, 0] 表示Sexism
    """
    df = pd.read_csv(csv_path)
    
    # 解析target列(字符串形式的列表 → 实际列表)
    import ast
    df['target_list'] = df['target'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # 转换为主类别标签(与项目6分类对齐)
    def get_primary_category(row):
        if row['toxic'] == 0:
            return "无毒"
        target = row['target_list']
        # 按优先级: Sexism(2) > Racism(3) > Region(1) > LGBTQ(0) > Others(4)
        priority = [2, 3, 1, 0, 4]
        for idx in priority:
            if idx < len(target) and target[idx] == 1:
                return TARGET_INDEX[idx]
        return "其他仇恨"  # fallback
    
    df['category'] = df.apply(get_primary_category, axis=1)
    df['expression_name'] = df['expression'].map(EXPRESSION_MAP)
    df['toxic_type_name'] = df['toxic_type'].map(TOXIC_TYPE_MAP)
    
    return df


def build_text_to_metadata(toxicn_df: pd.DataFrame) -> dict:
    """
    构建 text → metadata 映射
    用于将评测结果与原始标注关联
    """
    lookup = {}
    for _, row in toxicn_df.iterrows():
        text = str(row['post']).strip()
        lookup[text] = {
            'expression': row['expression_name'],
            'toxic_type': row['toxic_type_name'],
            'category': row['category'],
            'is_toxic': int(row['toxic']),
            'is_multi_target': sum(row['target_list'][:5]) > 1 if row['toxic'] == 1 else False,
        }
    return lookup


def compute_metrics_for_group(predictions: list) -> dict:
    """计算一组预测的P/R/F1"""
    label_stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    correct = 0
    total = len(predictions)
    
    for item in predictions:
        true_label = item['true_label']
        pred_label = item['pred_label']
        
        if pred_label is None:
            label_stats[true_label]["FN"] += 1
        elif pred_label == true_label:
            correct += 1
            label_stats[true_label]["TP"] += 1
        else:
            label_stats[pred_label]["FP"] += 1
            label_stats[true_label]["FN"] += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    
    # Macro F1
    f1_scores = []
    for cat in ALL_CATEGORIES:
        s = label_stats[cat]
        p = s["TP"] / (s["TP"] + s["FP"]) if (s["TP"] + s["FP"]) > 0 else 0
        r = s["TP"] / (s["TP"] + s["FN"]) if (s["TP"] + s["FN"]) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_scores.append(f1)
    
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    # Weighted F1
    total_w = sum(label_stats[c]["TP"] + label_stats[c]["FN"] for c in ALL_CATEGORIES)
    weighted_f1 = sum(
        f1_scores[i] * (label_stats[ALL_CATEGORIES[i]]["TP"] + label_stats[ALL_CATEGORIES[i]]["FN"])
        for i in range(len(ALL_CATEGORIES))
    ) / total_w if total_w > 0 else 0
    
    return {
        "accuracy": round(accuracy, 2),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "total": total,
        "correct": correct,
    }


def analyze_by_expression(eval_results: list, text_lookup: dict) -> dict:
    """按expression类型分组分析"""
    groups = defaultdict(list)
    unmatched = 0
    
    for item in eval_results:
        text = item.get('text', item.get('文本', '')).strip()
        pred = item.get('predict_label', item.get('predicted_label', None))
        true_label = item.get('标签', item.get('true_label', ''))
        
        meta = text_lookup.get(text, None)
        
        pred_item = {'true_label': true_label, 'pred_label': pred, 'text': text}
        
        if meta:
            expr = meta['expression']
            groups[expr].append(pred_item)
            groups[f"type_{meta['toxic_type']}"].append(pred_item)
            if meta['is_multi_target']:
                groups['multi_target'].append(pred_item)
        else:
            unmatched += 1
            groups['unmatched'].append(pred_item)
        
        groups['overall'].append(pred_item)
    
    # 计算每组指标
    results = {}
    for group_name, preds in groups.items():
        if preds:
            results[group_name] = compute_metrics_for_group(preds)
    
    results['_unmatched_count'] = unmatched
    return results


def print_expression_analysis(analysis: dict):
    """打印expression分层分析 (论文表格格式)"""
    print(f"\n{'='*80}")
    print(f"ToxiCN Expression 分层评测 (难度梯度)")
    print(f"{'='*80}")
    
    # 按难度排序: explicit → reporting → implicit
    expr_order = ['explicit', 'implicit', 'reporting', 'non-hate']
    difficulty_label = {
        'explicit': '显式(Easy)',
        'implicit': '隐式(Hard)', 
        'reporting': '转述(Hard)',
        'non-hate': '非仇恨(Baseline)',
    }
    
    print(f"\n{'表达类型':<20} {'样本数':<8} {'Accuracy':<12} {'Macro-F1':<12} {'Weighted-F1':<12}")
    print(f"{'-'*65}")
    
    for expr in expr_order:
        if expr in analysis:
            m = analysis[expr]
            label = difficulty_label.get(expr, expr)
            print(f"{label:<20} {m['total']:<8} {m['accuracy']:<12.2f} {m['macro_f1']:<12.4f} {m['weighted_f1']:<12.4f}")
    
    if 'overall' in analysis:
        m = analysis['overall']
        print(f"{'-'*65}")
        print(f"{'总体':<20} {m['total']:<8} {m['accuracy']:<12.2f} {m['macro_f1']:<12.4f} {m['weighted_f1']:<12.4f}")
    
    # Toxic Type分析
    print(f"\n{'='*80}")
    print(f"ToxiCN Toxic Type 分层评测")
    print(f"{'='*80}")
    type_order = ['type_offensive_language', 'type_hate_speech', 'type_non-toxic']
    type_label = {
        'type_offensive_language': '攻击语言(Offensive)',
        'type_hate_speech': '仇恨言论(Hate Speech)',
        'type_non-toxic': '非有毒(Non-toxic)',
    }
    
    print(f"\n{'有毒类型':<25} {'样本数':<8} {'Accuracy':<12} {'Macro-F1':<12} {'Weighted-F1':<12}")
    print(f"{'-'*70}")
    
    for ttype in type_order:
        if ttype in analysis:
            m = analysis[ttype]
            label = type_label.get(ttype, ttype)
            print(f"{label:<25} {m['total']:<8} {m['accuracy']:<12.2f} {m['macro_f1']:<12.4f} {m['weighted_f1']:<12.4f}")
    
    # 多标签样本
    if 'multi_target' in analysis:
        m = analysis['multi_target']
        print(f"\n多标签样本(Multi-target): {m['total']}条 Acc={m['accuracy']:.1f}% F1={m['macro_f1']:.4f}")
    
    unmatched = analysis.get('_unmatched_count', 0)
    if unmatched > 0:
        print(f"\n⚠️  未匹配到expression标注的样本: {unmatched}")


def generate_latex_table(analysis: dict) -> str:
    """生成LaTeX格式表格 (适合论文)"""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Performance breakdown by expression type}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Expression Type & Samples & Accuracy & Weighted-F1 \\")
    lines.append(r"\midrule")
    
    for expr, label in [('explicit', 'Explicit'), ('implicit', 'Implicit'), 
                        ('reporting', 'Reporting'), ('non-hate', 'Non-hate')]:
        if expr in analysis:
            m = analysis[expr]
            lines.append(f"{label} & {m['total']} & {m['accuracy']:.1f} & {m['weighted_f1']:.4f} \\\\")
    
    if 'overall' in analysis:
        m = analysis['overall']
        lines.append(r"\midrule")
        lines.append(f"Overall & {m['total']} & {m['accuracy']:.1f} & {m['weighted_f1']:.4f} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="ToxiCN Expression分层难度评估",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--eval_results", type=str, required=True,
                       help="评测结果JSON文件路径")
    parser.add_argument("--toxicn_csv", type=str, 
                       default=None,
                       help="ToxiCN_1.0.csv路径 (含expression标注)")
    parser.add_argument("--split_data_dir", type=str,
                       default=None,
                       help="split_data目录 (若无CSV则尝试从已有数据匹配)")
    parser.add_argument("--output", type=str, default=None,
                       help="输出JSON路径")
    parser.add_argument("--latex", action="store_true",
                       help="输出LaTeX表格")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ToxiCN Expression 分层评估")
    print("=" * 80)
    
    # 加载评测结果
    print("\n加载评测结果...")
    with open(args.eval_results, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    # 评测结果可能在 debug_samples 或直接是列表
    if isinstance(eval_data, dict):
        results_list = eval_data.get('debug_samples', [])
        if not results_list:
            print("⚠️  eval_results JSON中未找到debug_samples，尝试其他字段...")
            # 需要带predict_label的完整结果文件
            results_list = eval_data.get('results', [])
    else:
        results_list = eval_data
    
    print(f"  评测样本: {len(results_list)}")
    
    # 加载ToxiCN CSV标注 
    if args.toxicn_csv and os.path.exists(args.toxicn_csv):
        print(f"\n加载ToxiCN标注: {args.toxicn_csv}")
        toxicn_df = load_toxicn_csv(args.toxicn_csv)
        text_lookup = build_text_to_metadata(toxicn_df)
        print(f"  标注条目: {len(text_lookup)}")
        
        # 统计expression分布
        expr_dist = toxicn_df['expression_name'].value_counts()
        print(f"\n  Expression分布:")
        for name, count in expr_dist.items():
            print(f"    {name}: {count}")
    else:
        print("\n⚠️  未提供ToxiCN_1.0.csv，无法进行expression分层分析")
        print("   请在 https://github.com/DUT-lujunyu/ToxiCN 下载 ToxiCN_1.0.csv")
        print("   并使用 --toxicn_csv 参数指定路径")
        text_lookup = {}
    
    # 分析
    analysis = analyze_by_expression(results_list, text_lookup)
    
    # 打印
    print_expression_analysis(analysis)
    
    # LaTeX
    if args.latex:
        latex = generate_latex_table(analysis)
        print(f"\n{'='*80}")
        print("LaTeX表格:")
        print(latex)
    
    # 保存
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 分析结果已保存: {args.output}")


if __name__ == "__main__":
    main()
