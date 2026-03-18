#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评测结果汇总脚本
读取所有eval_results中的JSON并生成对比表格
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_results(eval_dir: str):
    """加载所有评测结果"""
    results = {}
    eval_path = Path(eval_dir)
    
    for f in sorted(eval_path.glob("*.json")):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            
            # 兼容两种格式 (vllm直接返回 vs npu包一层metrics)
            if "metrics" in data:
                metrics = data["metrics"]
            else:
                metrics = data
            
            results[f.stem] = metrics
        except Exception as e:
            print(f"⚠️  读取失败 {f.name}: {e}")
    
    return results


def print_summary(results: dict):
    """打印汇总表格"""
    if not results:
        print("没有找到评测结果。")
        return
    
    # 所有类别
    all_categories = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]
    
    # 表头
    print("\n" + "=" * 100)
    print("评测结果汇总")
    print("=" * 100)
    
    # 准确率对比
    print(f"\n{'模型':<40} {'准确率':>8} {'Macro-F1':>10} {'Weighted-F1':>12} {'二分类F1':>10}")
    print("-" * 85)
    
    for name, metrics in results.items():
        acc = metrics.get("overall_accuracy", 0)
        macro = metrics.get("macro_f1", metrics.get("average_f1", 0))
        weighted = metrics.get("weighted_f1", 0)
        binary = metrics.get("binary_metrics", {})
        binary_f1 = binary.get("f1_score", 0) if isinstance(binary, dict) else 0
        
        w_str = f"{weighted:>11.4f}" if weighted else f"{'N/A':>12}"
        b_str = f"{binary_f1:>9.4f}" if binary_f1 else f"{'N/A':>10}"
        
        if macro == 0 and "category_accuracy" in metrics:
            print(f"{name:<40} {acc:>7.2f}%  {'N/A':>10} {w_str} {b_str}")
        else:
            print(f"{name:<40} {acc:>7.2f}%  {macro:>9.4f} {w_str} {b_str}")
    
    # 各类别详情
    print(f"\n{'='*100}")
    print("各类别准确率/F1对比")
    print(f"{'='*100}")
    
    header = f"{'模型':<30}"
    for cat in all_categories:
        header += f" {cat:>8}"
    print(header)
    print("-" * 100)
    
    for name, metrics in results.items():
        row = f"{name:<30}"
        
        # 兼容两种指标格式
        cat_data = metrics.get("category_metrics", metrics.get("category_accuracy", {}))
        
        for cat in all_categories:
            if cat in cat_data:
                info = cat_data[cat]
                if "f1_score" in info:
                    val = info["f1_score"] * 100
                elif "accuracy" in info:
                    val = info["accuracy"]
                else:
                    val = 0
                row += f" {val:>7.1f}%"
            else:
                row += f" {'N/A':>8}"
        
        print(row)
    
    print(f"{'='*100}")


def export_csv(results: dict, output_path: str):
    """导出CSV格式"""
    all_categories = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]
    
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        # 表头
        header = ["模型", "总体准确率", "Macro-F1", "Weighted-F1", "Weighted-P", "Weighted-R", "二分类F1"]
        for cat in all_categories:
            header.extend([f"{cat}_P", f"{cat}_R", f"{cat}_F1"])
        f.write(",".join(header) + "\n")
        
        for name, metrics in results.items():
            acc = metrics.get("overall_accuracy", 0)
            macro = metrics.get("macro_f1", metrics.get("average_f1", 0))
            weighted = metrics.get("weighted_f1", 0)
            w_p = metrics.get("weighted_precision", 0)
            w_r = metrics.get("weighted_recall", 0)
            binary = metrics.get("binary_metrics", {})
            binary_f1 = binary.get("f1_score", 0) if isinstance(binary, dict) else 0
            row = [name, f"{acc:.2f}", f"{macro:.4f}", f"{weighted:.4f}", f"{w_p:.4f}", f"{w_r:.4f}", f"{binary_f1:.4f}"]
            
            cat_data = metrics.get("category_metrics", {})
            for cat in all_categories:
                if cat in cat_data:
                    info = cat_data[cat]
                    row.append(f"{info.get('precision', 0):.4f}")
                    row.append(f"{info.get('recall', 0):.4f}")
                    row.append(f"{info.get('f1_score', 0):.4f}")
                else:
                    row.extend(["", "", ""])
            
            f.write(",".join(row) + "\n")
    
    print(f"\n✓ CSV已导出: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="评测结果汇总")
    parser.add_argument("--eval_dir", type=str, 
                       default="/home/ma-user/work/test/eval_results",
                       help="评测结果目录")
    parser.add_argument("--results_dir", type=str, default=None,
                       help="评测结果目录(别名，优先级高于val_dir)")
    parser.add_argument("--export_csv", type=str, default=None,
                       help="导出CSV路径 (可选)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="输出JSON文件路径 (可选)")
    
    args = parser.parse_args()
    
    # 兼容两种参数名
    eval_dir = args.results_dir if args.results_dir else args.eval_dir
    
    results = load_results(eval_dir)
    print_summary(results)
    
    # 保存JSON汇总(如果指定)
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ JSON已导出: {args.output_file}")
    
    if args.export_csv:
        export_csv(results, args.export_csv)
    else:
        # 默认导出到eval_dir
        csv_path = os.path.join(eval_dir, "summary.csv")
        export_csv(results, csv_path)


if __name__ == "__main__":
    main()
