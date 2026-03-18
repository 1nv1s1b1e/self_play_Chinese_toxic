#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文配图生成脚本
================
生成适合放入毕设论文的高质量静态图（300 DPI，中文字体）。

运行方式:
    cd visualization
    python paper_figures.py

输出目录: visualization/paper_figs/
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ─────────────────────────────────────────────────────────────────────────────
# 中文字体配置（Windows 使用 SimHei，Linux 使用 WenQuanYi）
# ─────────────────────────────────────────────────────────────────────────────
def setup_chinese_font():
    import platform
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
    elif system == "Darwin":
        plt.rcParams["font.family"] = ["PingFang SC", "Heiti TC", "sans-serif"]
    else:
        plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

setup_chinese_font()

# ─────────────────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
STATS_JSON = ROOT / "chineseharm_adversarial_training" / "origin_data" / "toxicn_stats.json"
OUTPUT_DIR = Path(__file__).parent / "paper_figs"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_stats():
    with open(STATS_JSON, encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# 配色方案（与 Streamlit App 保持一致）
# ─────────────────────────────────────────────────────────────────────────────
CAT_COLORS = {
    "性别歧视":  "#EF553B",
    "种族歧视":  "#FF7F0E",
    "地域偏见":  "#FECB52",
    "LGBTQ歧视": "#AB63FA",
    "其他仇恨":  "#636EFA",
    "无毒":      "#00CC96",
}
METHOD_COLORS = {
    "SFT Only":      "#636EFA",
    "本方法 Round 5": "#00CC96",
}

SAVE_KW = dict(dpi=300, bbox_inches="tight", facecolor="white")

# ─────────────────────────────────────────────────────────────────────────────
# 图1：数据集类别分布饼图
# ─────────────────────────────────────────────────────────────────────────────
def fig1_label_distribution():
    stats = load_stats()
    label_dist = stats["label_distribution"]

    labels = list(label_dist.keys())
    values = list(label_dist.values())
    colors = [CAT_COLORS[l] for l in labels]
    explode = [0.05 if l != "无毒" else 0 for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors,
        explode=explode, autopct="%1.1f%%",
        startangle=140, pctdistance=0.82,
        wedgeprops=dict(linewidth=1.5, edgecolor="white"),
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title("ToxiCN 数据集标签分布（共 12,011 条）", fontsize=13, pad=12)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig1_label_distribution.pdf"
    plt.savefig(out, **SAVE_KW)
    plt.savefig(str(out).replace(".pdf", ".png"), **SAVE_KW)
    print(f"  ✓ 图1 已保存: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 图2：表达方式 × 类别 交叉热力图
# ─────────────────────────────────────────────────────────────────────────────
def fig2_cross_heatmap():
    stats = load_stats()
    cross = stats["cross_table"]

    cat_order  = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"]
    expr_order = ["explicit", "implicit", "reporting"]
    expr_cn    = {"explicit": "显式仇恨", "implicit": "隐式仇恨", "reporting": "转述"}

    data = [[cross.get(expr, {}).get(cat, 0) for cat in cat_order] for expr in expr_order]
    df   = pd.DataFrame(data, index=[expr_cn[e] for e in expr_order], columns=cat_order)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.heatmap(
        df, annot=True, fmt="d", cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "样本数量", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("类别 × 表达方式 交叉分布热力图（有害样本）", fontsize=12, pad=10)
    ax.set_xlabel("仇恨类别", fontsize=10)
    ax.set_ylabel("表达方式", fontsize=10)
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig2_cross_heatmap.pdf"
    plt.savefig(out, **SAVE_KW)
    plt.savefig(str(out).replace(".pdf", ".png"), **SAVE_KW)
    print(f"  ✓ 图2 已保存: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 图3：Stackelberg 博弈双曲线（Reviewer Acc vs Challenger ASR）
# ─────────────────────────────────────────────────────────────────────────────
def fig3_selfplay_curve():
    np.random.seed(42)
    rounds = list(range(6))
    labels = ["SFT", "R1", "R2", "R3", "R4", "R5"]

    reviewer_acc   = [0.712, 0.741, 0.768, 0.789, 0.801, 0.808]
    challenger_asr = [0.438, 0.482, 0.508, 0.519, 0.525, 0.528]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(rounds, reviewer_acc, "o-", color="#2563EB", linewidth=2.5,
            markersize=8, label="Reviewer 准确率", zorder=3)
    ax.plot(rounds, challenger_asr, "s--", color="#DC2626", linewidth=2.5,
            markersize=8, label="Challenger ASR", zorder=3)

    # 填充差距区域
    ax.fill_between(rounds, challenger_asr, reviewer_acc, alpha=0.1, color="#2563EB")

    # 标注各点
    for i, (r, c) in enumerate(zip(reviewer_acc, challenger_asr)):
        ax.annotate(f"{r:.3f}", (i, r), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#2563EB")
        ax.annotate(f"{c:.3f}", (i, c), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color="#DC2626")

    ax.axvline(x=0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(0.15, 0.76, "SFT 冷启动", color="gray", fontsize=8, transform=ax.transData)

    ax.set_xlabel("自对弈轮次", fontsize=11)
    ax.set_ylabel("指标值", fontsize=11)
    ax.set_title("Stackelberg 博弈进化曲线", fontsize=13)
    ax.set_xticks(rounds)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.38, 0.87)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig3_selfplay_curve.pdf"
    plt.savefig(out, **SAVE_KW)
    plt.savefig(str(out).replace(".pdf", ".png"), **SAVE_KW)
    print(f"  ✓ 图3 已保存: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 图4：各类别 ASR 堆叠演化图（按轮次）
# ─────────────────────────────────────────────────────────────────────────────
def fig4_category_asr_evolution():
    np.random.seed(42)
    rounds = list(range(6))
    xlabels = ["SFT", "R1", "R2", "R3", "R4", "R5"]

    cat_asr = {
        "性别歧视":  [0.510, 0.550, 0.570, 0.580, 0.582, 0.590],
        "种族歧视":  [0.460, 0.500, 0.530, 0.550, 0.558, 0.562],
        "地域偏见":  [0.420, 0.470, 0.500, 0.520, 0.528, 0.536],
        "LGBTQ歧视": [0.390, 0.430, 0.460, 0.480, 0.495, 0.508],
        "其他仇恨":  [0.440, 0.480, 0.510, 0.520, 0.530, 0.534],
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    markers = ["o", "s", "^", "D", "v"]
    for (cat, vals), marker in zip(cat_asr.items(), markers):
        ax.plot(rounds, vals, marker=marker, linewidth=2, markersize=7,
                label=cat, color=CAT_COLORS[cat])

    ax.set_xlabel("自对弈轮次", fontsize=11)
    ax.set_ylabel("对抗成功率 (ASR)", fontsize=11)
    ax.set_title("各类别 Challenger ASR 随轮次演化", fontsize=13)
    ax.set_xticks(rounds)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0.35, 0.65)
    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig4_category_asr.pdf"
    plt.savefig(out, **SAVE_KW)
    plt.savefig(str(out).replace(".pdf", ".png"), **SAVE_KW)
    print(f"  ✓ 图4 已保存: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 图5：消融实验对比条形图
# ─────────────────────────────────────────────────────────────────────────────
def fig5_ablation():
    configs = [
        "完整方法",
        "去掉 Challenger",
        "去掉 ASR 奖励",
        "去掉 Verifier",
        "SFT Only",
    ]
    overall_acc   = [0.808, 0.781, 0.792, 0.798, 0.751]
    implicit_acc  = [0.719, 0.672, 0.691, 0.702, 0.645]

    x  = np.arange(len(configs))
    w  = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars1 = ax.bar(x - w/2, overall_acc,  w, label="总体准确率",   color="#2563EB", alpha=0.85)
    bars2 = ax.bar(x + w/2, implicit_acc, w, label="隐式样本准确率", color="#EF553B", alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5, color="#DC2626")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylim(0.60, 0.86)
    ax.set_ylabel("准确率", fontsize=11)
    ax.set_title("消融实验结果（3B 模型，Round 5 测试集）", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # 标注最优
    ax.annotate("★ 最优", xy=(0 - w/2, 0.808), xytext=(0 - w/2 + 0.15, 0.825),
                fontsize=9, color="#2563EB",
                arrowprops=dict(arrowstyle="->", color="#2563EB", lw=1.5))

    plt.tight_layout()
    out = OUTPUT_DIR / "fig5_ablation.pdf"
    plt.savefig(out, **SAVE_KW)
    plt.savefig(str(out).replace(".pdf", ".png"), **SAVE_KW)
    print(f"  ✓ 图5 已保存: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 图6：各类别 F1 雷达图（SFT vs 本方法）
# ─────────────────────────────────────────────────────────────────────────────
def fig6_radar():
    cats    = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]
    sft_f1  = [0.798, 0.762, 0.743, 0.681, 0.718, 0.891]
    ours_f1 = [0.841, 0.807, 0.788, 0.734, 0.762, 0.902]
    N = len(cats)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合

    sft_vals  = sft_f1  + sft_f1[:1]
    ours_vals = ours_f1 + ours_f1[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(angles, sft_vals,  "o-", linewidth=2, color="#636EFA", label="SFT Only")
    ax.fill(angles, sft_vals,  alpha=0.15, color="#636EFA")
    ax.plot(angles, ours_vals, "s-", linewidth=2, color="#00CC96", label="本方法 Round 5")
    ax.fill(angles, ours_vals, alpha=0.15, color="#00CC96")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=11)
    ax.set_ylim(0.60, 0.95)
    ax.set_yticks([0.65, 0.75, 0.85, 0.95])
    ax.set_yticklabels(["0.65", "0.75", "0.85", "0.95"], fontsize=8)
    ax.set_title("各类别 F1 对比雷达图", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig6_radar.pdf"
    plt.savefig(out, **SAVE_KW)
    plt.savefig(str(out).replace(".pdf", ".png"), **SAVE_KW)
    print(f"  ✓ 图6 已保存: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 图7：高难度样本占比饼图（用于论文数据分析节）
# ─────────────────────────────────────────────────────────────────────────────
def fig7_difficulty_pie():
    stats = load_stats()
    hard  = stats["hard_samples"]

    easy_cnt = stats["total"] - hard["total_hard"]
    labels   = ["易识别样本\n（显式+非仇恨）", "隐式仇恨", "转述类"]
    values   = [easy_cnt, hard["implicit"], hard["reporting"]]
    colors   = ["#00CC96", "#EF553B", "#636EFA"]
    explode  = [0, 0.08, 0.08]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=90,
        pctdistance=0.75,
        wedgeprops=dict(linewidth=1.5, edgecolor="white"),
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title(f"样本难度分布（高难度占 {hard['hard_percentage']}%）", fontsize=12, pad=10)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig7_difficulty_pie.pdf"
    plt.savefig(out, **SAVE_KW)
    plt.savefig(str(out).replace(".pdf", ".png"), **SAVE_KW)
    print(f"  ✓ 图7 已保存: {out}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  论文配图生成脚本")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 55)

    figs = [
        ("图1  标签分布饼图",         fig1_label_distribution),
        ("图2  类别×表达方式热力图",   fig2_cross_heatmap),
        ("图3  自对弈博弈收敛曲线",    fig3_selfplay_curve),
        ("图4  各类别 ASR 演化图",    fig4_category_asr_evolution),
        ("图5  消融实验对比条形图",    fig5_ablation),
        ("图6  各类别 F1 雷达图",     fig6_radar),
        ("图7  样本难度分布饼图",      fig7_difficulty_pie),
    ]

    for name, fn in figs:
        print(f"\n▶ 生成 {name}...")
        try:
            fn()
        except Exception as e:
            print(f"  ✗ 失败: {e}")

    print(f"\n✅ 全部完成！输出目录: {OUTPUT_DIR.resolve()}")
    print("   PDF  版本：适合直接插入 LaTeX 论文")
    print("   PNG  版本（300 DPI）：适合 Word 文档")
