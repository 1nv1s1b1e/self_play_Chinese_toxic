#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练轻量级类别验证器 (Category Verifier)
用于 Challenger 奖励函数中的类别忠实度信号

══════════════════════════════════════════════════════════════════
 文献依据
══════════════════════════════════════════════════════════════════

 1. RLHF (Ouyang et al., 2022):
    使用独立训练的奖励模型 (Reward Model) 提供奖励信号。
    我们的 category verifier 本质上就是一个轻量级 reward model。

 2. Perez et al. (2022) "Red Teaming Language Models with Language Models":
    使用 RoBERTa hate speech classifier 作为 R(y)。
    我们训练一个 TF-IDF + LogReg 分类器, 作为 R(x) 的核心。

 3. CRT (Hong et al., ICLR 2024) "Curiosity-driven Red-teaming":
    使用 RoBERTa toxicity classifier 判定有毒性概率。

 4. RLAIF (Lee et al., 2023):
    使用 AI feedback 替代 human feedback 提供奖励信号。

 5. SSP (arXiv:2510.18821):
    R(τ) = 1 - solver_success_rate。
    classifier_confidence 近似了 "solver 能否正确分类" 的概率。

══════════════════════════════════════════════════════════════════
 方法: TF-IDF (字符 n-gram) + Logistic Regression
══════════════════════════════════════════════════════════════════

 选择理由:
   1. 轻量: 模型 < 50MB, 推理 < 1ms/样本 → 适合在 reward function 中调用
   2. 概率校准: LogReg 天然输出校准概率 → 直接作为奖励信号
   3. 字符 n-gram: 适合中文 (无需分词), 捕获语言模式而非精确匹配
   4. 可复现: sklearn 是标准库, 无额外依赖

 为什么不用 BM25/检索:
   BM25 检索返回最相似的训练样本 → 鼓励模型复制训练样本 (v5.1 已验证)
   分类器学习决策边界 → 泛化到新文本, 不鼓励复制

使用方式:
    python train_category_verifier.py --train_data /path/to/train.json --output /path/to/verifier.pkl
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# 确保能导入项目模块
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))


def load_data(data_path: str):
    """加载训练数据, 支持 json/parquet 格式"""
    data_path = Path(data_path)

    if data_path.suffix == '.json':
        with open(data_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        texts, labels = [], []
        for r in records:
            text = r.get('文本', r.get('original_text', r.get('content', '')))
            label = r.get('标签', r.get('category', r.get('topic', '')))
            if text and label:
                texts.append(text)
                labels.append(label)
    elif data_path.suffix == '.parquet':
        import pandas as pd
        df = pd.read_parquet(data_path)
        text_col = '文本' if '文本' in df.columns else 'original_text' if 'original_text' in df.columns else 'content'
        label_col = '标签' if '标签' in df.columns else 'category' if 'category' in df.columns else 'topic'
        texts = df[text_col].tolist()
        labels = df[label_col].tolist()
    else:
        raise ValueError(f"Unsupported format: {data_path.suffix}")

    print(f"  加载 {len(texts)} 条数据, 来源: {data_path.name}")
    return texts, labels


def train_verifier(train_data_path: str, save_path: str, val_data_path: str = None):
    """
    训练类别验证器并保存

    Pipeline:
      TfidfVectorizer (char n-gram 2~5) → LogisticRegression (多类, 概率输出)

    Args:
        train_data_path: 训练数据路径
        save_path: 模型保存路径
        val_data_path: 可选, 验证数据路径

    Returns:
        dict: 训练报告 (accuracy, per-class metrics)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, accuracy_score
    import joblib

    print("\n" + "=" * 60)
    print("训练类别验证器 (Category Verifier)")
    print("=" * 60)

    # ── 加载数据 ──
    train_texts, train_labels = load_data(train_data_path)

    # ── 构建 Pipeline ──
    #
    # TfidfVectorizer:
    #   analyzer='char': 字符级别, 适合中文 (无需分词)
    #   ngram_range=(2, 5): 捕获 bi-gram 到 5-gram 的字符模式
    #   max_features=50000: 限制特征空间, 防止过拟合
    #   sublinear_tf=True: 使用 1 + log(tf), 抑制高频词的主导
    #
    # LogisticRegression:
    #   C=1.0: 正则化强度, 默认值
    #   max_iter=1000: 确保收敛
    #   multi_class='multinomial': 多类分类, softmax 概率
    #   solver='lbfgs': 适合小数据集
    #
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 5),
            max_features=50000,
            sublinear_tf=True,
            min_df=2,
        )),
        ('clf', LogisticRegression(
            C=1.0,
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced',  # 处理类别不平衡
        )),
    ])

    print(f"\n  训练数据: {len(train_texts)} 条")
    print(f"  类别: {sorted(set(train_labels))}")
    print(f"  类别分布:")
    from collections import Counter
    for cat, cnt in sorted(Counter(train_labels).items()):
        print(f"    {cat}: {cnt} ({cnt/len(train_labels):.1%})")

    # ── 训练 ──
    print("\n  训练中...")
    pipeline.fit(train_texts, train_labels)
    print("  ✓ 训练完成")

    # ── 训练集评估 ──
    train_pred = pipeline.predict(train_texts)
    train_acc = accuracy_score(train_labels, train_pred)
    print(f"\n  训练集准确率: {train_acc:.4f}")

    # ── 验证集评估 ──
    report = {"train_accuracy": train_acc, "train_size": len(train_texts)}

    if val_data_path and Path(val_data_path).exists():
        val_texts, val_labels = load_data(val_data_path)
        val_pred = pipeline.predict(val_texts)
        val_acc = accuracy_score(val_labels, val_pred)
        val_report = classification_report(val_labels, val_pred, output_dict=True, zero_division=0)

        print(f"\n  验证集准确率: {val_acc:.4f}")
        print(f"  验证集分类报告:")
        print(classification_report(val_labels, val_pred, zero_division=0))

        report["val_accuracy"] = val_acc
        report["val_size"] = len(val_texts)
        report["val_classification_report"] = val_report

    # ── 保存模型 ──
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)

    model_size = save_path.stat().st_size / 1024 / 1024
    print(f"\n  ✓ 模型已保存: {save_path} ({model_size:.1f} MB)")
    report["model_path"] = str(save_path)
    report["model_size_mb"] = round(model_size, 2)

    # ── 保存训练报告 ──
    report_path = save_path.parent / "verifier_training_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 训练报告: {report_path}")

    # ── 推理速度测试 ──
    import time
    test_texts = train_texts[:100]
    t0 = time.time()
    for _ in range(10):
        pipeline.predict_proba(test_texts)
    elapsed = (time.time() - t0) / 10
    print(f"\n  推理速度: {elapsed*1000:.1f}ms / 100条 ({elapsed/len(test_texts)*1000*1000:.0f}μs/条)")

    return report


def main():
    parser = argparse.ArgumentParser(description="训练类别验证器")
    parser.add_argument("--train_data", type=str, required=True,
                       help="训练数据路径 (json/parquet)")
    parser.add_argument("--val_data", type=str, default=None,
                       help="验证数据路径 (json/parquet)")
    parser.add_argument("--output", type=str, required=True,
                       help="模型保存路径 (.pkl)")
    args = parser.parse_args()

    train_verifier(args.train_data, args.output, args.val_data)


if __name__ == "__main__":
    main()
