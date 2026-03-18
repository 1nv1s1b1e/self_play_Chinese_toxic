#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验结果汇总分析与可视化
==============================

读取各消融实验的JSON结果，生成:
1. 汇总对比表 (终端打印 + JSON)
2. 可视化图表 (matplotlib)
   - Prompt消融: 不同prompt变体的F1对比柱状图
   - Epoch消融: 准确率/F1随epoch变化的折线图
   - Base vs SFT: 多维度雷达图 + 分类别增益对比
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # 无GUI后端
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    # 中文字体设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import numpy as np

ALL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]


def load_json(path: Path) -> Optional[Dict]:
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ============================================================
# 1. Prompt消融分析
# ============================================================
def analyze_prompt_ablation(results_dir: Path, output_dir: Path):
    """分析prompt消融结果"""
    summary_files = list(results_dir.glob("prompt_ablation_summary_*.json"))
    if not summary_files:
        print("  未找到prompt消融结果")
        return

    all_summaries = []
    for f in summary_files:
        data = load_json(f)
        if data:
            all_summaries.append(data)

    print(f"\n{'=' * 70}")
    print("Prompt消融实验分析")
    print(f"{'=' * 70}")

    for summary in all_summaries:
        model = Path(summary.get("model", "")).name
        tag = summary.get("tag", "")
        results = summary.get("results", {})

        label = f"{model}" + (f" ({tag})" if tag else "")
        print(f"\n模型: {label}")
        print(f"{'Prompt变体':<16} {'Accuracy':>10} {'Macro-F1':>10} {'Failed':>8}")
        print("-" * 48)
        for variant in ["full_rules", "brief_rules", "no_rules", "zero_shot"]:
            if variant in results:
                m = results[variant]
                print(f"{variant:<16} {m['accuracy']:>10.2%} {m['macro_f1']:>10.4f} {m['failed_extractions']:>8}")

        # RULES贡献度
        if "full_rules" in results and "no_rules" in results:
            fr = results["full_rules"]
            nr = results["no_rules"]
            print(f"\n  RULES贡献: Acc {fr['accuracy']-nr['accuracy']:+.2%}, F1 {fr['macro_f1']-nr['macro_f1']:+.4f}")

    # 可视化
    if HAS_MPL and all_summaries:
        _plot_prompt_ablation(all_summaries, output_dir)


def _plot_prompt_ablation(summaries: List[Dict], output_dir: Path):
    """Prompt消融柱状图"""
    variant_order = ["full_rules", "brief_rules", "no_rules", "zero_shot"]
    variant_labels = ["完整规则", "精简规则", "无规则", "零样本"]

    for summary in summaries:
        model = Path(summary.get("model", "unknown")).name
        tag = summary.get("tag", "")
        results = summary.get("results", {})

        accs = [results.get(v, {}).get("accuracy", 0) for v in variant_order]
        f1s = [results.get(v, {}).get("macro_f1", 0) for v in variant_order]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        x = np.arange(len(variant_labels))
        width = 0.5

        bars1 = ax1.bar(x, [a * 100 for a in accs], width, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'])
        ax1.set_ylabel('准确率 (%)')
        ax1.set_title(f'Prompt消融 - 准确率\n{model} {tag}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(variant_labels, rotation=15)
        ax1.set_ylim(0, 105)
        for bar, acc in zip(bars1, accs):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{acc:.1%}', ha='center', va='bottom', fontsize=9)

        bars2 = ax2.bar(x, f1s, width, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'])
        ax2.set_ylabel('Macro-F1')
        ax2.set_title(f'Prompt消融 - Macro-F1\n{model} {tag}')
        ax2.set_xticks(x)
        ax2.set_xticklabels(variant_labels, rotation=15)
        ax2.set_ylim(0, 1.1)
        for bar, f1 in zip(bars2, f1s):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        out_path = output_dir / f"prompt_ablation_{model}_{tag}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  图表已保存: {out_path}")


# ============================================================
# 2. Epoch消融分析
# ============================================================
def analyze_epoch_ablation(results_dir: Path, output_dir: Path):
    """分析epoch消融结果"""
    epoch_files = list(results_dir.glob("epoch_ablation_*.json"))
    if not epoch_files:
        print("  未找到epoch消融结果")
        return

    print(f"\n{'=' * 70}")
    print("Epoch消融实验分析")
    print(f"{'=' * 70}")

    for f in epoch_files:
        data = load_json(f)
        if not data:
            continue

        model = Path(data.get("model", "")).name
        results = data.get("results", {})

        print(f"\n模型: {model}")
        print(f"{'Epoch':<10} {'Accuracy':>10} {'Macro-F1':>10}")
        print("-" * 34)

        epoch_keys = sorted(results.keys(), key=lambda x: int(x.split("_")[1]))
        epochs_nums = []
        accs = []
        f1s = []

        for k in epoch_keys:
            m = results[k]
            ep = int(k.split("_")[1])
            epochs_nums.append(ep)
            accs.append(m['accuracy'])
            f1s.append(m['macro_f1'])
            print(f"{k:<10} {m['accuracy']:>10.2%} {m['macro_f1']:>10.4f}")

        # 饱和分析
        if len(accs) >= 3:
            gain_01 = f1s[1] - f1s[0] if len(f1s) > 1 else 0
            gain_12 = f1s[2] - f1s[1] if len(f1s) > 2 else 0
            total_gain = f1s[-1] - f1s[0]
            print(f"\n  epoch 0→1 增益: F1 {gain_01:+.4f}")
            print(f"  epoch 1→2 增益: F1 {gain_12:+.4f}")
            if total_gain > 0:
                print(f"  epoch 1 占总增益: {gain_01/total_gain*100:.1f}%")

        # 可视化
        if HAS_MPL:
            _plot_epoch_ablation(model, epochs_nums, accs, f1s, output_dir)


def _plot_epoch_ablation(model: str, epochs: list, accs: list, f1s: list, output_dir: Path):
    """Epoch消融折线图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, [a*100 for a in accs], 'o-', color='#2196F3', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('准确率 (%)')
    ax1.set_title(f'Epoch消融 - 准确率\n{model}')
    ax1.set_xticks(epochs)
    ax1.grid(True, alpha=0.3)
    for ep, acc in zip(epochs, accs):
        ax1.annotate(f'{acc:.1%}', (ep, acc*100), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    ax2.plot(epochs, f1s, 's-', color='#F44336', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro-F1')
    ax2.set_title(f'Epoch消融 - Macro-F1\n{model}')
    ax2.set_xticks(epochs)
    ax2.grid(True, alpha=0.3)
    for ep, f1 in zip(epochs, f1s):
        ax2.annotate(f'{f1:.4f}', (ep, f1), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    out_path = output_dir / f"epoch_ablation_{model}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {out_path}")


# ============================================================
# 3. Base vs SFT 分析
# ============================================================
def analyze_base_vs_sft(results_dir: Path, output_dir: Path):
    """分析Base vs SFT结果"""
    comparison_files = list(results_dir.glob("base_vs_sft_*.json"))
    if not comparison_files:
        print("  未找到Base vs SFT结果")
        return

    print(f"\n{'=' * 70}")
    print("Base vs SFT 对比分析")
    print(f"{'=' * 70}")

    for f in comparison_files:
        data = load_json(f)
        if not data:
            continue

        ba = data.get("base_analysis", {})
        sa = data.get("sft_analysis", {})

        print(f"\nBase: {data.get('base_model', 'N/A')}")
        print(f"SFT:  {data.get('sft_model', 'N/A')}")

        print(f"\n{'指标':<28} {'Base':>12} {'SFT':>12} {'差值':>12}")
        print("-" * 66)

        rows = [
            ("格式对齐率", ba.get('format_alignment_rate',0), sa.get('format_alignment_rate',0)),
            ("准确率(全部)", ba.get('accuracy_all',0), sa.get('accuracy_all',0)),
            ("准确率(可解析)", ba.get('accuracy_parseable',0), sa.get('accuracy_parseable',0)),
            ("Macro-F1", ba.get('macro_f1',0), sa.get('macro_f1',0)),
            ("平均输出长度(字符)", ba.get('avg_response_length_chars',0), sa.get('avg_response_length_chars',0)),
        ]
        for name, b, s in rows:
            diff = s - b
            if isinstance(b, float) and b <= 1.0:
                print(f"{name:<28} {b:>12.2%} {s:>12.2%} {diff:>+12.2%}")
            else:
                print(f"{name:<28} {b:>12.1f} {s:>12.1f} {diff:>+12.1f}")

        # 分类别F1增益
        print(f"\n  分类别F1增益:")
        for cat in ALL_CATEGORIES:
            bf1 = ba.get('category_metrics',{}).get(cat,{}).get('f1',0)
            sf1 = sa.get('category_metrics',{}).get(cat,{}).get('f1',0)
            bar = "█" * int((sf1-bf1)*50) if sf1 > bf1 else "▒" * int((bf1-sf1)*50)
            print(f"    {cat:<8} {bf1:.3f} → {sf1:.3f} ({sf1-bf1:+.3f}) {bar}")

        # 可视化
        if HAS_MPL:
            _plot_base_vs_sft(data, output_dir)


def _plot_base_vs_sft(data: Dict, output_dir: Path):
    """Base vs SFT 分类别对比柱状图"""
    ba = data.get("base_analysis", {})
    sa = data.get("sft_analysis", {})

    cats = ALL_CATEGORIES
    base_f1 = [ba.get('category_metrics',{}).get(c,{}).get('f1',0) for c in cats]
    sft_f1 = [sa.get('category_metrics',{}).get(c,{}).get('f1',0) for c in cats]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(cats))
    width = 0.35

    bars1 = ax.bar(x - width/2, base_f1, width, label='Base', color='#90CAF9', edgecolor='#1976D2')
    bars2 = ax.bar(x + width/2, sft_f1, width, label='SFT', color='#EF9A9A', edgecolor='#D32F2F')

    ax.set_ylabel('F1-Score')
    ax.set_title('Base vs SFT — 分类别F1对比')
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    base_name = Path(data.get("base_model","")).name
    sft_name = Path(data.get("sft_model","")).name
    out_path = output_dir / f"base_vs_sft_{base_name}_vs_{sft_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图表已保存: {out_path}")


# ============================================================
# 4. 综合报告
# ============================================================
def generate_comprehensive_report(base_dir: Path, output_dir: Path):
    """生成综合消融实验报告"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "█" * 70)
    print("  消融实验综合分析报告")
    print("█" * 70)

    # Prompt消融
    prompt_dir = base_dir / "prompt_ablation"
    if prompt_dir.exists():
        analyze_prompt_ablation(prompt_dir, output_dir)

    # Epoch消融
    epoch_dir = base_dir / "epoch_ablation"
    if epoch_dir.exists():
        analyze_epoch_ablation(epoch_dir, output_dir)

    # Base vs SFT
    bvs_dir = base_dir / "base_vs_sft"
    if bvs_dir.exists():
        analyze_base_vs_sft(bvs_dir, output_dir)

    # 综合结论
    print(f"\n\n{'=' * 70}")
    print("综合结论")
    print(f"{'=' * 70}")
    print("""
消融实验旨在回答以下核心问题:

Q1: RULES在prompt中的贡献有多大?
    → 对比 full_rules vs no_rules/zero_shot 的F1差异。
    如果差异很大，说明模型高度依赖prompt中的关键词规则进行分类，
    而非真正学到了语义判断能力。

Q2: SFT的核心价值是什么 — 格式对齐还是分类能力?
    → 对比 Base和SFT 在 "格式对齐率" 和 "可解析准确率" 上的差异。
    如果格式对齐率差异大但可解析准确率差异小，说明SFT主要在做格式对齐。

Q3: SFT训练是否过拟合?
    → 对比 epoch 0/1/2/3 的F1。如果 epoch 1 已获得 >90% 的总增益，
    后续epoch主要在过拟合。

这些结论直接论证了对抗RL训练的必要性:
  - 如果RULES贡献大 → 模型对隐蔽有毒文本(不含明显关键词)的检测能力不足
  - 如果SFT只做格式对齐 → 需要RL来真正提升分类能力
  - 如果SFT已过拟合 → 需要对抗样本来打破过拟合
""")

    print(f"\n所有图表已保存到: {output_dir}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="消融实验结果汇总分析")
    parser.add_argument("--results_dir", type=str,
                       default="/home/ma-user/work/test/ablation_results",
                       help="消融结果根目录")
    parser.add_argument("--output_dir", type=str,
                       default="/home/ma-user/work/test/ablation_results/analysis",
                       help="分析输出目录")

    args = parser.parse_args()
    generate_comprehensive_report(Path(args.results_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
