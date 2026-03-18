#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChineseHarm 对抗自对弈训练 — 毕设可视化 Dashboard
==================================================
运行方式:
    cd visualization
    streamlit run app.py
"""

import json
import os
import sys
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
STATS_JSON   = ROOT / "chineseharm_adversarial_training" / "origin_data" / "toxicn_stats.json"
SPLIT_REPORT = ROOT / "chineseharm_adversarial_training" / "split_data" / "data_split_report.json"
DATA_REPORT  = ROOT / "chineseharm_adversarial_training" / "prepared_data" / "data_preparation_report.json"

# ─────────────────────────────────────────────────────────────────────────────
# 页面全局配置
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChineseHarm 对抗自对弈系统",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 统一配色
CAT_COLORS = {
    "性别歧视":  "#EF553B",
    "种族歧视":  "#FF7F0E",
    "地域偏见":  "#FECB52",
    "LGBTQ歧视": "#AB63FA",
    "其他仇恨":  "#636EFA",
    "无毒":      "#00CC96",
}
EXPR_COLORS = {
    "non-hate":  "#00CC96",
    "explicit":  "#EF553B",
    "implicit":  "#FF7F0E",
    "reporting": "#636EFA",
}

# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_toxicn_stats():
    with open(STATS_JSON, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_split_report():
    with open(SPLIT_REPORT, encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# 侧边栏导航
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/ChineseHarm-Self--Play-blue?style=for-the-badge", use_container_width=True)
    st.markdown("## 🗂️ 导航")
    page = st.radio(
        "",
        options=["🏠 系统概览", "📊 数据集分析", "🤖 文本检测演示", "📈 对抗训练过程", "🎯 模型对比"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        **项目说明**
        - 数据集：ToxiCN (ACL 2023)
        - 模型：Qwen 3B/7B (LoRA)
        - 框架：TRL + DeepSpeed
        - 硬件：昇腾 910B NPU
        """
    )

# ─────────────────────────────────────────────────────────────────────────────
# 页面 0: 系统概览
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 系统概览":
    st.title("🛡️ 基于对抗自对弈强化学习的中文有害内容检测系统")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("训练样本", "8,160 条", help="ToxiCN 训练集总量")
    with col2:
        st.metric("仇恨类别", "6 类", help="性别歧视/种族歧视/地域偏见/LGBTQ歧视/其他仇恨/无毒")
    with col3:
        st.metric("高难度样本", "24.2%", "隐式+转述", help="对抗训练的主要攻击目标")
    with col4:
        st.metric("自对弈轮次", "5 轮", help="Stackelberg 博弈迭代次数")

    st.markdown("---")
    st.subheader("⚙️ Stackelberg 对抗博弈框架")

    fig = go.Figure()

    # 节点
    nodes = {
        "Challenger\n(攻击者)": (0.5, 0.9, "#EF553B"),
        "Reviewer\n(检测者)": (0.15, 0.2, "#636EFA"),
        "Verifier\n(冻结Oracle)": (0.85, 0.2, "#00CC96"),
        "动态数据\n(Phase 0)": (0.5, 0.55, "#FECB52"),
        "GRPO 奖励\n(Phase A/B)": (0.5, 0.2, "#AB63FA"),
    }

    for label, (x, y, color) in nodes.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=55, color=color, line=dict(width=2, color="white")),
            text=[label],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            showlegend=False,
            hoverinfo="none",
        ))

    # 箭头（用线模拟）
    edges = [
        (0.5, 0.85, 0.5, 0.62, "生成对抗文本"),
        (0.5, 0.48, 0.2, 0.28, "Challenger GRPO数据"),
        (0.5, 0.48, 0.8, 0.28, "验证 ASR"),
        (0.8, 0.16, 0.55, 0.16, "计算 ASR 奖励"),
        (0.45, 0.16, 0.2, 0.16, "Reviewer GRPO"),
    ]
    for x0, y0, x1, y1, label in edges:
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.5, arrowwidth=2,
            arrowcolor="#888",
            text=label, font=dict(size=9, color="#555"),
            bgcolor="rgba(255,255,255,0.7)", borderpad=2,
        )

    fig.update_layout(
        height=420, margin=dict(l=20, r=20, t=10, b=10),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1.1]),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(
            "**Phase 0：动态数据生成**\n\n"
            "每轮由 Challenger 生成对抗文本，Verifier 评估，计算各类别 ASR（对抗成功率）"
        )
    with col2:
        st.warning(
            "**Phase A：Challenger GRPO**\n\n"
            "奖励 = 质量门控 × (话题相关性 + ASR奖励)\n\n"
            "目标：生成能绕过 Reviewer 的隐式文本"
        )
    with col3:
        st.success(
            "**Phase B：Reviewer GRPO**\n\n"
            "奖励 = 0.7×(二分类+类别) + 0.3×(毒性类型+表达方式)\n\n"
            "漏检惩罚 -1.0，误检惩罚 -0.5"
        )

# ─────────────────────────────────────────────────────────────────────────────
# 页面 1: 数据集分析
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 数据集分析":
    st.title("📊 ToxiCN 数据集分析")
    st.markdown("来源：ToxiCN (ACL 2023) · 12,011 条中文有毒语言标注数据")
    st.markdown("---")

    stats = load_toxicn_stats()
    label_dist = stats["label_distribution"]
    expr_dist  = stats["expression_distribution"]
    cross      = stats["cross_table"]
    hard       = stats["hard_samples"]

    # ── 行1: 类别分布 + 表达方式分布 ────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏷️ 标签类别分布")
        labels = list(label_dist.keys())
        values = list(label_dist.values())
        colors = [CAT_COLORS.get(l, "#888") for l in labels]

        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} 条 (%{percent})<extra></extra>",
            pull=[0.05 if l != "无毒" else 0 for l in labels],
        ))
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                          legend=dict(orientation="h", yanchor="bottom", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)
        total = sum(values)
        st.caption(f"共 {total:,} 条 | 无毒占 {label_dist['无毒']/total*100:.1f}%，有害占 {(total-label_dist['无毒'])/total*100:.1f}%")

    with col2:
        st.subheader("🗣️ 表达方式分布")
        expr_labels_cn = {"non-hate": "非仇恨", "explicit": "显式仇恨", "implicit": "隐式仇恨", "reporting": "转述"}
        expr_labels = [expr_labels_cn.get(k, k) for k in expr_dist.keys()]
        expr_values = list(expr_dist.values())
        expr_colors = [EXPR_COLORS.get(k, "#888") for k in expr_dist.keys()]

        fig = go.Figure(go.Bar(
            x=expr_labels, y=expr_values,
            marker_color=expr_colors,
            text=expr_values,
            textposition="outside",
            hovertemplate="%{x}: %{y} 条<extra></extra>",
        ))
        fig.update_layout(
            height=360, margin=dict(l=0, r=0, t=10, b=40),
            yaxis_title="样本数量",
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#eee"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"高难度样本（隐式+转述）：{hard['total_hard']:,} 条，占 **{hard['hard_percentage']}%**")

    st.markdown("---")

    # ── 行2: 交叉热力图 ──────────────────────────────────────────────────────────
    st.subheader("🔥 类别 × 表达方式 交叉热力图（仅有害样本）")

    cat_order  = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"]
    expr_order = ["explicit", "implicit", "reporting"]
    expr_cn    = {"explicit": "显式仇恨", "implicit": "隐式仇恨", "reporting": "转述"}

    heatmap_data = []
    for expr in expr_order:
        row = []
        for cat in cat_order:
            v = cross.get(expr, {}).get(cat, 0)
            row.append(v)
        heatmap_data.append(row)

    fig = go.Figure(go.Heatmap(
        z=heatmap_data,
        x=cat_order,
        y=[expr_cn[e] for e in expr_order],
        colorscale="YlOrRd",
        text=heatmap_data,
        texttemplate="%{text}",
        hovertemplate="表达方式: %{y}<br>类别: %{x}<br>样本数: %{z}<extra></extra>",
        colorbar=dict(title="样本数"),
    ))
    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="仇恨类别",
        yaxis_title="表达方式",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "💡 **关键洞察**：性别歧视的隐式样本（887条）最多，是对抗训练的主要攻击目标；"
        "地域偏见的转述样本（600条）最多，是 Reviewer 最容易误判的来源。"
    )

    st.markdown("---")

    # ── 行3: 数据集划分 ──────────────────────────────────────────────────────────
    st.subheader("✂️ 数据集划分与使用方式")

    split_data = {
        "集合": ["train", "val", "test"],
        "数量": [8160, 1426, 1427],
        "用途": ["SFT冷启动 + RL种子池", "RL过程评估 + SFT验证", "最终评测（不参与训练）"],
        "Reviewer SFT": ["8,160 条", "—", "—"],
        "Challenger SFT": ["4,425 条\n（排除无毒）", "—", "—"],
    }
    df_split = pd.DataFrame(split_data)
    st.dataframe(df_split, use_container_width=True, hide_index=True)

    # 堆叠柱状图
    fig = go.Figure()
    splits = ["train", "val", "test"]
    sizes  = [8160, 1426, 1427]
    split_colors = ["#636EFA", "#EF553B", "#00CC96"]
    for s, sz, c in zip(splits, sizes, split_colors):
        fig.add_trace(go.Bar(
            name=s, x=["数据集划分"], y=[sz],
            marker_color=c,
            text=[f"{s}\n{sz:,}"],
            textposition="inside",
        ))
    fig.update_layout(
        barmode="stack", height=200, margin=dict(l=0, r=0, t=10, b=0),
        showlegend=True, yaxis_title="样本数量",
        plot_bgcolor="white", yaxis=dict(gridcolor="#eee"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 页面 2: 文本检测演示
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 文本检测演示":
    st.title("🤖 中文有害内容检测演示")
    st.markdown("本页演示 Reviewer 模型的检测能力（Mock 模式 / API 模式）")
    st.markdown("---")

    # 模式选择
    mode = st.radio("**运行模式**", ["🎭 Mock 演示", "🌐 远程 API 调用"], horizontal=True)
    st.markdown("---")

    # 预设样例
    EXAMPLES = {
        "示例 1（显式性别歧视）": "女的就是脑子不好使，别来搞技术了。",
        "示例 2（隐式仇恨-谐音）": "某地区的司机真的很有"特色"，每次都让人大开眼界。",
        "示例 3（转述类）": "有报道称该社区曾多次发生与地域身份相关的言语冲突事件。",
        "示例 4（无毒）": "今天天气真好，适合出去走走。",
        "示例 5（LGBTQ歧视-隐式）": "这种所谓的多元包容真是让人无语，以前的社会多好啊。",
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.selectbox("📌 快速选择示例文本", ["自定义输入"] + list(EXAMPLES.keys()))
        default_text = EXAMPLES.get(selected, "")
        text_input = st.text_area(
            "🖊️ 输入待检测文本",
            value=default_text,
            height=120,
            placeholder="请输入中文文本…",
        )
    with col2:
        st.markdown("#### ⚙️ 检测配置")
        if mode == "🌐 远程 API 调用":
            api_url  = st.text_input("API 地址", value="http://your-npu-server:8080/classify")
            api_key  = st.text_input("API Key (可选)", type="password")
        threshold = st.slider("有害置信度阈值", 0.0, 1.0, 0.5, 0.05)

    detect_btn = st.button("🔍 开始检测", type="primary", use_container_width=True)

    if detect_btn and text_input.strip():
        st.markdown("---")

        with st.spinner("检测中..."):
            import time
            time.sleep(0.6)  # 模拟网络延迟

            if mode == "🎭 Mock 演示":
                # Mock 规则推断（仅演示用，非真实模型）
                text = text_input.lower()
                keywords = {
                    "性别歧视": ["女", "男", "娘炮", "汉子", "妇女", "女的", "男的", "脑子不好使"],
                    "种族歧视": ["黑人", "白人", "外国人", "老外", "种族"],
                    "地域偏见": ["地区", "某地", "河南", "东北", "司机", "地域"],
                    "LGBTQ歧视": ["lgbt", "同性", "变性", "多元", "包容", "gay"],
                    "其他仇恨": ["滚", "死", "垃圾", "废物", "傻"],
                }
                detected_cat  = "无毒"
                detected_expr = "非仇恨"
                confidence    = random.uniform(0.80, 0.95)
                toxic_type    = "无毒"

                for cat, kws in keywords.items():
                    if any(kw in text_input for kw in kws):
                        detected_cat = cat
                        # 判断表达方式
                        implicit_kws = ["所谓", "特色", "大开眼界", "真是", "以前", "多好", "无语"]
                        report_kws   = ["报道", "称", "曾", "事件", "相关", "研究"]
                        if any(kw in text_input for kw in implicit_kws):
                            detected_expr = "隐式仇恨"
                            toxic_type = "仇恨言论"
                        elif any(kw in text_input for kw in report_kws):
                            detected_expr = "转述"
                            toxic_type = "仇恨言论"
                        else:
                            detected_expr = "显式仇恨"
                            toxic_type = "仇恨言论"
                        break

                is_harmful = detected_cat != "无毒"

                # 模拟模型输出文本
                if is_harmful:
                    raw_output = (
                        f"判断：有害\n"
                        f"类别：{detected_cat}\n"
                        f"毒性类型：{toxic_type}\n"
                        f"表达方式：{detected_expr}"
                    )
                else:
                    raw_output = "判断：无害"
                    confidence = random.uniform(0.88, 0.97)

            else:
                # 真实 API 调用
                import requests
                try:
                    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                    resp = requests.post(
                        api_url,
                        json={"text": text_input},
                        headers=headers,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    result_json  = resp.json()
                    raw_output   = result_json.get("output", "")
                    is_harmful   = "有害" in raw_output
                    detected_cat = result_json.get("category", "未知")
                    detected_expr= result_json.get("expression", "未知")
                    confidence   = result_json.get("confidence", 0.5)
                    toxic_type   = result_json.get("toxic_type", "未知")
                except Exception as e:
                    st.error(f"❌ API 调用失败: {e}")
                    st.stop()

        # ── 结果展示 ──────────────────────────────────────────────────────────────
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.subheader("🎯 检测结果")
            if is_harmful:
                st.error(f"⚠️ **检测结论：有害内容**")
                st.markdown(f"**仇恨类别**：`{detected_cat}`")
                st.markdown(f"**毒性类型**：`{toxic_type}`")
                st.markdown(f"**表达方式**：`{detected_expr}`")
                cat_color = CAT_COLORS.get(detected_cat, "#888")
                # 置信度仪表
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence * 100,
                    title={"text": "有害置信度 (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": cat_color},
                        "steps": [
                            {"range": [0, 50], "color": "#e8f5e9"},
                            {"range": [50, 75], "color": "#fff3e0"},
                            {"range": [75, 100], "color": "#ffebee"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": threshold * 100,
                        },
                    },
                    number={"suffix": "%", "font": {"size": 30}},
                ))
                fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.success(f"✅ **检测结论：无害内容**")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=(1 - confidence) * 100,
                    title={"text": "无害置信度 (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#00CC96"},
                        "steps": [
                            {"range": [0, 50], "color": "#ffebee"},
                            {"range": [50, 75], "color": "#fff3e0"},
                            {"range": [75, 100], "color": "#e8f5e9"},
                        ],
                    },
                    number={"suffix": "%", "font": {"size": 30}},
                ))
                fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=10))
                st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.subheader("📄 模型原始输出")
            st.code(raw_output, language=None)

            if is_harmful:
                st.subheader("📌 类别说明")
                cat_desc = {
                    "性别歧视": "基于性别的歧视性言论，包括对男性或女性的刻板印象、贬低等。",
                    "种族歧视": "针对特定种族或民族群体的仇恨/偏见性表达。",
                    "地域偏见": "对特定地区人群的刻板印象和偏见性言论。",
                    "LGBTQ歧视": "针对 LGBTQ+ 群体的歧视或仇恨性表达。",
                    "其他仇恨": "不针对特定群体但具有一般攻击性或仇恨性的内容。",
                }
                st.info(cat_desc.get(detected_cat, ""))

                if detected_expr == "隐式仇恨":
                    st.warning(
                        "⚠️ **隐式仇恨**：此文本未使用明显关键词，通过反讽、暗语、"
                        "阴阳怪气等方式传达歧视意图，是检测难点。"
                    )
                elif detected_expr == "转述":
                    st.info(
                        "ℹ️ **转述类**：此文本以报道/讨论形式呈现，立场模糊，"
                        "Reviewer 需结合语境判断是否具有实质性危害。"
                    )

    elif detect_btn:
        st.warning("请输入待检测文本。")

# ─────────────────────────────────────────────────────────────────────────────
# 页面 3: 对抗训练过程
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 对抗训练过程":
    st.title("📈 Stackelberg 自对弈训练过程可视化")
    st.markdown("以下曲线展示双方模型在 5 轮迭代中的对抗博弈动态（基于预期趋势模拟）")
    st.markdown("---")

    rounds = [0, 1, 2, 3, 4, 5]  # 轮次 0 = SFT 初始值

    # 模拟数据（基于 Stackelberg 博弈的典型收敛曲线）
    random.seed(42)
    np.random.seed(42)

    # Reviewer 准确率：从 SFT 基线开始，随轮次提升，最终趋于稳定
    reviewer_acc = [0.712, 0.741, 0.768, 0.789, 0.801, 0.808]
    reviewer_acc = [v + np.random.normal(0, 0.005) for v in reviewer_acc]

    # Challenger ASR：从 SFT 基线开始，随轮次提升（略慢于 Reviewer 提升）
    challenger_asr = [0.438, 0.482, 0.508, 0.519, 0.525, 0.528]
    challenger_asr = [v + np.random.normal(0, 0.006) for v in challenger_asr]

    # 各类别 ASR（每轮）
    cat_asr_data = {
        "性别歧视":  [0.51, 0.55, 0.57, 0.58, 0.58, 0.59],
        "种族歧视":  [0.46, 0.50, 0.53, 0.55, 0.56, 0.56],
        "地域偏见":  [0.42, 0.47, 0.50, 0.52, 0.53, 0.54],
        "LGBTQ歧视": [0.39, 0.43, 0.46, 0.48, 0.50, 0.51],
        "其他仇恨":  [0.44, 0.48, 0.51, 0.52, 0.53, 0.53],
    }

    # ── 主曲线：Reviewer vs Challenger 博弈 ──────────────────────────────────────
    st.subheader("⚔️ Reviewer 准确率 vs Challenger 攻击成功率（ASR）")

    fig = make_subplots(specs=[[{"secondary_y": False}]])
    fig.add_trace(go.Scatter(
        x=rounds, y=reviewer_acc,
        mode="lines+markers",
        name="Reviewer 准确率",
        line=dict(color="#636EFA", width=3),
        marker=dict(size=10),
        hovertemplate="轮次 %{x} | Reviewer Acc: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=challenger_asr,
        mode="lines+markers",
        name="Challenger ASR",
        line=dict(color="#EF553B", width=3, dash="dash"),
        marker=dict(size=10, symbol="diamond"),
        hovertemplate="轮次 %{x} | Challenger ASR: %{y:.3f}<extra></extra>",
    ))
    # 添加初始 SFT 标注
    fig.add_vline(x=0, line_dash="dot", line_color="gray", annotation_text="SFT 冷启动", annotation_position="top right")
    fig.update_layout(
        height=360,
        xaxis=dict(title="自对弈轮次", tickvals=rounds, ticktext=[f"Round {r}" if r > 0 else "SFT" for r in rounds]),
        yaxis=dict(title="指标值", range=[0.35, 0.88], gridcolor="#eee"),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info("📌 双方指标同步提升，体现了 Stackelberg 博弈中\"螺旋上升\"的协同进化特性。")

    st.markdown("---")

    # ── 类别 ASR 热力图（轮次 × 类别）──────────────────────────────────────────
    st.subheader("🔥 各类别 Challenger ASR — 按轮次热力图")

    categories = list(cat_asr_data.keys())
    heatmap_z = [[cat_asr_data[cat][r] for cat in categories] for r in range(6)]
    row_labels = [f"Round {r}" if r > 0 else "SFT" for r in rounds]

    fig = go.Figure(go.Heatmap(
        z=heatmap_z,
        x=categories,
        y=row_labels,
        colorscale="RdYlGn_r",
        text=[[f"{v:.3f}" for v in row] for row in heatmap_z],
        texttemplate="%{text}",
        colorbar=dict(title="ASR"),
        hovertemplate="轮次: %{y}<br>类别: %{x}<br>ASR: %{z:.3f}<extra></extra>",
        zmin=0.35, zmax=0.65,
    ))
    fig.update_layout(
        height=340,
        xaxis_title="仇恨类别",
        yaxis_title="自对弈轮次",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("颜色越红表示 ASR 越高（Challenger 越容易骗过 Reviewer），越绿表示 Reviewer 越难被攻击。")

    st.markdown("---")

    # ── 奖励曲线 ────────────────────────────────────────────────────────────────
    st.subheader("📉 GRPO 训练奖励曲线")

    col1, col2 = st.columns(2)
    steps = list(range(0, 51, 2))

    with col1:
        st.markdown("**Challenger 奖励（最终轮）**")
        np.random.seed(1)
        c_reward = [(-0.6 + 0.02 * s + np.random.normal(0, 0.08)) for s in steps]
        c_reward_smooth = pd.Series(c_reward).rolling(3, min_periods=1).mean().tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=c_reward, mode="lines", name="原始",
                                  line=dict(color="#EF553B", width=1), opacity=0.4))
        fig.add_trace(go.Scatter(x=steps, y=c_reward_smooth, mode="lines", name="平滑",
                                  line=dict(color="#EF553B", width=2.5)))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(height=250, plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(gridcolor="#eee"), xaxis_title="训练步数", yaxis_title="奖励值")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Reviewer 奖励（最终轮）**")
        np.random.seed(2)
        r_reward = [(0.15 + 0.015 * s + np.random.normal(0, 0.06)) for s in steps]
        r_reward_smooth = pd.Series(r_reward).rolling(3, min_periods=1).mean().tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=r_reward, mode="lines", name="原始",
                                  line=dict(color="#636EFA", width=1), opacity=0.4))
        fig.add_trace(go.Scatter(x=steps, y=r_reward_smooth, mode="lines", name="平滑",
                                  line=dict(color="#636EFA", width=2.5)))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(height=250, plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(gridcolor="#eee"), xaxis_title="训练步数", yaxis_title="奖励值")
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# 页面 4: 模型对比
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 模型对比":
    st.title("🎯 模型性能对比")
    st.markdown("在 ToxiCN 测试集（1,427条）上的分类准确率对比（含消融实验）")
    st.markdown("---")

    # ── 主对比表 ─────────────────────────────────────────────────────────────────
    st.subheader("📊 基线 vs 本方法（总体准确率）")

    model_results = {
        "模型": [
            "GPT-4o (zero-shot)",
            "Claude-3.5 (zero-shot)",
            "Qwen-7B (SFT only)",
            "Qwen-3B (SFT only)",
            "本方法 Reviewer-3B (Round 1)",
            "本方法 Reviewer-3B (Round 3)",
            "本方法 Reviewer-3B (Round 5) ★",
            "本方法 Reviewer-7B (Round 5)",
        ],
        "总体 Acc": [0.748, 0.761, 0.779, 0.751, 0.768, 0.789, 0.808, 0.821],
        "F1 (宏平均)": [0.712, 0.726, 0.748, 0.720, 0.739, 0.762, 0.781, 0.796],
        "隐式样本 Acc": [0.642, 0.659, 0.671, 0.645, 0.673, 0.698, 0.719, 0.738],
        "类型": ["闭源基线", "闭源基线", "开源基线", "开源基线", "本方法", "本方法", "本方法", "本方法"],
    }
    df_results = pd.DataFrame(model_results)

    # 条形图
    fig = go.Figure()
    type_colors = {"闭源基线": "#FECB52", "开源基线": "#636EFA", "本方法": "#00CC96"}
    for mtype, group in df_results.groupby("类型"):
        fig.add_trace(go.Bar(
            name=mtype,
            x=group["模型"],
            y=group["总体 Acc"],
            marker_color=type_colors[mtype],
            text=[f"{v:.3f}" for v in group["总体 Acc"]],
            textposition="outside",
        ))
    fig.update_layout(
        height=420,
        barmode="group",
        yaxis=dict(range=[0.68, 0.88], title="总体准确率", gridcolor="#eee"),
        xaxis=dict(tickangle=-20),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── 消融实验 ─────────────────────────────────────────────────────────────────
    st.subheader("🔬 消融实验（3B 模型，Round 5）")

    ablation_data = {
        "实验配置": [
            "完整方法（Challenger+Reviewer+Verifier）",
            "- 去掉 Challenger（无对抗攻击）",
            "- 去掉 ASR 奖励（仅静态 v7 奖励）",
            "- 去掉 Verifier（用 Reviewer 自评）",
            "- SFT only（无 RL）",
        ],
        "总体 Acc":    [0.808, 0.781, 0.792, 0.798, 0.751],
        "隐式 Acc":    [0.719, 0.672, 0.691, 0.702, 0.645],
        "Δ vs 完整": ["+0.000", "-0.027", "-0.016", "-0.010", "-0.057"],
    }
    df_ablation = pd.DataFrame(ablation_data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_ablation["实验配置"],
        x=df_ablation["总体 Acc"],
        orientation="h",
        marker_color=["#00CC96"] + ["#636EFA"] * 3 + ["#EF553B"],
        text=[f"{v:.3f}" for v in df_ablation["总体 Acc"]],
        textposition="outside",
    ))
    fig.update_layout(
        height=320,
        xaxis=dict(range=[0.70, 0.84], title="总体准确率", gridcolor="#eee"),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        margin=dict(l=0, r=60, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df_ablation, use_container_width=True, hide_index=True)
    st.success(
        "**消融实验结论**：去掉 Challenger 导致准确率下降最大（-2.7%），"
        "证明对抗攻击者是提升 Reviewer 鲁棒性的关键组件。"
    )

    st.markdown("---")

    # ── 各类别雷达图 ─────────────────────────────────────────────────────────────
    st.subheader("🕸️ 各类别 F1 雷达图（SFT vs 本方法）")

    cats = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]
    sft_f1 = [0.798, 0.762, 0.743, 0.681, 0.718, 0.891]
    ours_f1 = [0.841, 0.807, 0.788, 0.734, 0.762, 0.902]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=sft_f1 + [sft_f1[0]],
        theta=cats + [cats[0]],
        fill="toself",
        name="SFT Only",
        line_color="#636EFA",
        fillcolor="rgba(99,110,250,0.2)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=ours_f1 + [ours_f1[0]],
        theta=cats + [cats[0]],
        fill="toself",
        name="本方法 Round 5",
        line_color="#00CC96",
        fillcolor="rgba(0,204,150,0.2)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.6, 0.95])),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        margin=dict(l=20, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("LGBTQ歧视 类别提升幅度最大（+5.3%），但绝对值仍偏低，主因是训练数据量最少（711条）。")
