#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版中文有害文本数据集下载与整合
=====================================

为双智能体对抗博弈引入更难、更多样的数据源，使任务复杂度显著提升。

数据集:
  1. ToxiCN (ACL 2023): 12K知乎/贴吧评论, 含隐式仇恨言论 → 模型不能只靠关键词
  2. COLD (EMNLP 2022): 37K中文冒犯语言, 含anti-bias干扰 → 反偏见言论易误判
  3. Safety-Prompts (清华CoAI): 100K安全场景, 含指令攻击 → 对抗性极强
  4. CDial-Bias (清华CoAI): 对话偏见数据 → 隐含偏见检测
  5. SWSR: 中文性别歧视 → 隐式有害

统一输出格式:
  {"text": "...", "label": "...", "source": "...", "difficulty": "easy/medium/hard"}

难度分级:
  - easy:   显式有害，含明显关键词 (ChineseHarm原始)
  - medium: 有害但关键词不明显 (ToxiCN explicit, COLD attacking)
  - hard:   隐式有害/反偏见/指令攻击 (ToxiCN implicit, COLD anti-bias, Safety-Prompts attack)
"""

import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

import pandas as pd
import numpy as np


# ============================================================
# 统一类别映射体系 (扩展ChineseHarm的6类为更细粒度)
# ============================================================

# 扩展类别体系：在原有6类基础上增加难度维度
UNIFIED_CATEGORIES = [
    "博彩",        # 保持原有
    "低俗色情",    # 保持原有
    "谩骂引战",    # 保持原有 + COLD攻击个人/群体 + ToxiCN仇恨言论
    "欺诈",        # 保持原有
    "黑产广告",    # 保持原有
    "歧视偏见",    # 新增: ToxiCN性别/种族/地域歧视 + CDial-Bias
    "违法犯罪",    # 新增: Safety-Prompts违法犯罪类
    "隐私侵犯",    # 新增: Safety-Prompts隐私类
    "不违规",      # 保持原有 + COLD anti-bias (表面冒犯实则反偏见)
]

# 备选方案：如果希望保持原有6类体系，使用此映射
CATEGORY_MAP_TO_6CLASS = {
    # ToxiCN 映射
    "toxicn_sexism": "谩骂引战",
    "toxicn_racism": "谩骂引战",
    "toxicn_regional_bias": "谩骂引战",
    "toxicn_anti_lgbtq": "谩骂引战",
    "toxicn_others": "谩骂引战",
    "toxicn_non_toxic": "不违规",
    
    # COLD 映射
    "cold_attack_individual": "谩骂引战",
    "cold_attack_group": "谩骂引战",
    "cold_anti_bias": "不违规",       # 关键! 反偏见归为不违规但含敏感词
    "cold_non_offensive": "不违规",
    
    # Safety-Prompts 映射
    "safety_insult": "谩骂引战",
    "safety_discrimination": "谩骂引战",
    "safety_crimes": "欺诈",           # 违法犯罪 → 欺诈
    "safety_physical_harm": "谩骂引战",
    "safety_mental_health": "不违规",   # 心理健康多为正常讨论
    "safety_privacy": "黑产广告",       # 隐私侵犯 → 黑产
    "safety_ethics": "不违规",          # 道德讨论多为正常
    "safety_goal_hijack": "欺诈",
    "safety_prompt_leak": "欺诈",
    "safety_role_play": "谩骂引战",
    "safety_unsafe_topic": "谩骂引战",
    "safety_unsafe_opinion": "谩骂引战",
    "safety_reverse_exposure": "低俗色情",
}


# ============================================================
# 1. ToxiCN 下载与处理
# ============================================================

def download_and_process_toxicn(output_dir: Path) -> Optional[pd.DataFrame]:
    """
    下载并处理 ToxiCN 数据集
    核心价值: 隐式仇恨言论 (implicit hate speech) 让分类任务变难
    """
    print("=" * 60)
    print("[1/5] ToxiCN (ACL 2023) - 含隐式仇恨言论")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        ds = load_dataset("JunyuLu/ToxiCN", split="train")
        df = ds.to_pandas()
        print(f"  原始样本数: {len(df)}")
        print(f"  列名: {list(df.columns)}")
        
        # ToxiCN 标注:
        # toxic: 0(non-toxic) / 1(toxic)
        # toxic_type: 0(non-toxic) / 1(general offensive) / 2(hate speech)
        # expression: 0(non-hate) / 1(explicit) / 2(implicit) / 3(reporting)
        # target: [LGBTQ, Region, Sexism, Racism, others, non-hate]
        
        records = []
        for _, row in df.iterrows():
            text = str(row.get('text', '')).strip()
            if not text or len(text) < 5:
                continue
            
            toxic = int(row.get('toxic', 0))
            toxic_type = int(row.get('toxic_type', 0))
            expression = int(row.get('expression', 0))
            
            # 解析目标群体
            target = row.get('target', [0, 0, 0, 0, 0, 1])
            if isinstance(target, str):
                target = json.loads(target)
            
            target_names = ["LGBTQ", "Region", "Sexism", "Racism", "Others", "Non-hate"]
            active_targets = [target_names[i] for i, v in enumerate(target) if v == 1]
            
            # 确定统一标签
            if toxic == 0:
                label = "不违规"
                source_label = "toxicn_non_toxic"
            else:
                # 根据 target 映射
                if target[2] == 1:  # Sexism
                    source_label = "toxicn_sexism"
                elif target[3] == 1:  # Racism
                    source_label = "toxicn_racism"
                elif target[1] == 1:  # Region
                    source_label = "toxicn_regional_bias"
                elif target[0] == 1:  # LGBTQ
                    source_label = "toxicn_anti_lgbtq"
                else:
                    source_label = "toxicn_others"
                label = CATEGORY_MAP_TO_6CLASS[source_label]
            
            # 难度分级
            if toxic == 0:
                difficulty = "easy"
            elif expression == 1:  # explicit
                difficulty = "medium"
            elif expression == 2:  # implicit → 最难!
                difficulty = "hard"
            elif expression == 3:  # reporting
                difficulty = "hard"  # 报道类容易误判
            else:
                difficulty = "medium"
            
            records.append({
                "text": text,
                "label": label,
                "source": "toxicn",
                "source_label": source_label,
                "difficulty": difficulty,
                "targets": ",".join(active_targets),
                "expression_type": ["non-hate", "explicit", "implicit", "reporting"][expression],
            })
        
        result_df = pd.DataFrame(records)
        
        # 统计
        print(f"\n  处理后样本数: {len(result_df)}")
        print(f"  标签分布:")
        for label, count in result_df['label'].value_counts().items():
            print(f"    {label}: {count}")
        print(f"  难度分布:")
        for diff, count in result_df['difficulty'].value_counts().items():
            print(f"    {diff}: {count}")
        print(f"  隐式仇恨言论: {len(result_df[result_df['expression_type'] == 'implicit'])} 条 (最有价值!)")
        
        out_path = output_dir / "toxicn_processed.parquet"
        result_df.to_parquet(out_path, index=False)
        print(f"  ✓ 保存到: {out_path}")
        
        return result_df
        
    except Exception as e:
        print(f"  ⚠️ 处理失败: {e}")
        print("  手动下载: pip install datasets && python -c \"from datasets import load_dataset; load_dataset('JunyuLu/ToxiCN')\"")
        return None


# ============================================================
# 2. COLD 下载与处理
# ============================================================

def download_and_process_cold(output_dir: Path) -> Optional[pd.DataFrame]:
    """
    下载并处理 COLD 数据集
    核心价值: anti-bias 样本 (表面像冒犯但实际反对歧视) 极易误判
    """
    print("\n" + "=" * 60)
    print("[2/5] COLD (EMNLP 2022) - 含anti-bias干扰样本")
    print("=" * 60)
    
    try:
        # COLD 可从 GitHub 下载 CSV 文件
        # 尝试通过 requests 直接下载
        import requests
        
        base_url = "https://raw.githubusercontent.com/thu-coai/COLDataset/main/COLDataset"
        
        all_records = []
        
        for split_name, has_fine_label in [("train", False), ("dev", False), ("test", True)]:
            url = f"{base_url}/{split_name}.csv"
            print(f"  下载 {split_name}...")
            
            resp = requests.get(url, timeout=30)
            resp.encoding = 'utf-8'
            
            if resp.status_code != 200:
                print(f"    ⚠️ 下载失败: HTTP {resp.status_code}")
                continue
            
            # 解析CSV
            lines = resp.text.strip().split('\n')
            reader = csv.DictReader(lines)
            
            for row in reader:
                text = row.get('TEXT', '').strip()
                if not text or len(text) < 5:
                    continue
                
                label_val = int(row.get('label', 0))
                
                if has_fine_label:
                    fine_label = int(row.get('fine-grained-label', 0))
                    # 0: safe (other-Non-offen)
                    # 1: attack individual
                    # 2: attack group
                    # 3: safe (anti-bias) ← 最有价值!
                    if fine_label == 0:
                        source_label = "cold_non_offensive"
                        difficulty = "easy"
                    elif fine_label == 1:
                        source_label = "cold_attack_individual"
                        difficulty = "medium"
                    elif fine_label == 2:
                        source_label = "cold_attack_group"
                        difficulty = "medium"
                    elif fine_label == 3:
                        source_label = "cold_anti_bias"
                        difficulty = "hard"  # anti-bias是最难的!
                    else:
                        source_label = "cold_non_offensive"
                        difficulty = "easy"
                else:
                    if label_val == 0:
                        source_label = "cold_non_offensive"
                        difficulty = "easy"
                    else:
                        source_label = "cold_attack_individual"  # 粗粒度标注
                        difficulty = "medium"
                
                unified_label = CATEGORY_MAP_TO_6CLASS[source_label]
                
                all_records.append({
                    "text": text,
                    "label": unified_label,
                    "source": "cold",
                    "source_label": source_label,
                    "difficulty": difficulty,
                    "split": split_name,
                })
        
        if not all_records:
            print("  ⚠️ 未获取到数据")
            return None
        
        result_df = pd.DataFrame(all_records)
        
        # 统计
        print(f"\n  处理后样本数: {len(result_df)}")
        print(f"  标签分布:")
        for label, count in result_df['label'].value_counts().items():
            print(f"    {label}: {count}")
        print(f"  难度分布:")
        for diff, count in result_df['difficulty'].value_counts().items():
            print(f"    {diff}: {count}")
        anti_bias_count = len(result_df[result_df['source_label'] == 'cold_anti_bias'])
        print(f"  anti-bias样本: {anti_bias_count} 条 (标注为不违规但含敏感词，最易误判!)")
        
        out_path = output_dir / "cold_processed.parquet"
        result_df.to_parquet(out_path, index=False)
        print(f"  ✓ 保存到: {out_path}")
        
        return result_df
        
    except Exception as e:
        print(f"  ⚠️ 处理失败: {e}")
        print("  手动下载: git clone https://github.com/thu-coai/COLDataset.git")
        return None


# ============================================================
# 3. Safety-Prompts 处理 (聚焦指令攻击场景)
# ============================================================

def download_and_process_safety_prompts(output_dir: Path) -> Optional[pd.DataFrame]:
    """
    下载并处理 Safety-Prompts
    核心价值: 指令攻击场景 (目标劫持/角色扮演/反面诱导) 天然对抗性最强
    """
    print("\n" + "=" * 60)
    print("[3/5] Safety-Prompts (清华CoAI) - 指令攻击场景")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        
        all_records = []
        
        # 典型安全场景
        typical_map = {
            "Insult": ("safety_insult", "medium"),
            "Unfairness and Discrimination": ("safety_discrimination", "medium"),
            "Crimes and Illegal Activities": ("safety_crimes", "medium"),
            "Physical Harm": ("safety_physical_harm", "medium"),
            "Mental Health": ("safety_mental_health", "hard"),  # 边界模糊
            "Privacy and Property": ("safety_privacy", "medium"),
            "Ethics and Morality": ("safety_ethics", "hard"),   # 边界模糊
        }
        
        for cat, (source_label, difficulty) in typical_map.items():
            try:
                ds = load_dataset(
                    "thu-coai/Safety-Prompts",
                    data_files='typical_safety_scenarios.json',
                    field=cat,
                    split='train'
                )
                for item in ds:
                    prompt = item.get('prompt', '').strip()
                    if prompt and len(prompt) >= 10:
                        all_records.append({
                            "text": prompt,
                            "label": CATEGORY_MAP_TO_6CLASS[source_label],
                            "source": "safety_prompts",
                            "source_label": source_label,
                            "difficulty": difficulty,
                            "scenario": "typical",
                        })
                print(f"  typical/{cat}: {len(ds)} 条")
            except Exception as e:
                print(f"  ⚠️ {cat}: {e}")
        
        # 指令攻击场景 (最有价值!)
        attack_map = {
            "Goal Hijacking": ("safety_goal_hijack", "hard"),
            "Prompt Leaking": ("safety_prompt_leak", "hard"),
            "Role Play Instruction": ("safety_role_play", "hard"),
            "Unsafe Instruction Topic": ("safety_unsafe_topic", "hard"),
            "Inquiry with Unsafe Opinion": ("safety_unsafe_opinion", "hard"),
            "Reverse Exposure": ("safety_reverse_exposure", "hard"),
        }
        
        for cat, (source_label, difficulty) in attack_map.items():
            try:
                ds = load_dataset(
                    "thu-coai/Safety-Prompts",
                    data_files='instruction_attack_scenarios.json',
                    field=cat,
                    split='train'
                )
                for item in ds:
                    prompt = item.get('prompt', '').strip()
                    if prompt and len(prompt) >= 10:
                        all_records.append({
                            "text": prompt,
                            "label": CATEGORY_MAP_TO_6CLASS[source_label],
                            "source": "safety_prompts",
                            "source_label": source_label,
                            "difficulty": difficulty,
                            "scenario": "attack",
                        })
                print(f"  attack/{cat}: {len(ds)} 条")
            except Exception as e:
                print(f"  ⚠️ {cat}: {e}")
        
        if not all_records:
            return None
        
        result_df = pd.DataFrame(all_records)
        
        print(f"\n  处理后样本数: {len(result_df)}")
        print(f"  标签分布:")
        for label, count in result_df['label'].value_counts().items():
            print(f"    {label}: {count}")
        attack_count = len(result_df[result_df['scenario'] == 'attack'])
        print(f"  指令攻击样本: {attack_count} 条 (对抗性最强!)")
        
        out_path = output_dir / "safety_prompts_processed.parquet"
        result_df.to_parquet(out_path, index=False)
        print(f"  ✓ 保存到: {out_path}")
        
        return result_df
        
    except Exception as e:
        print(f"  ⚠️ 处理失败: {e}")
        return None


# ============================================================
# 4. CDial-Bias (可选)
# ============================================================

def download_and_process_cdial_bias(output_dir: Path) -> Optional[pd.DataFrame]:
    """
    下载 CDial-Bias 对话偏见数据
    核心价值: 对话场景中的隐含偏见
    """
    print("\n" + "=" * 60)
    print("[4/5] CDial-Bias (清华CoAI) - 对话偏见")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        ds = load_dataset("thu-coai/CDial-Bias-Dataset", split="train")
        df = ds.to_pandas()
        
        print(f"  原始样本数: {len(df)}")
        print(f"  列名: {list(df.columns)}")
        
        records = []
        for _, row in df.iterrows():
            text = str(row.get('text', row.get('context', ''))).strip()
            if not text or len(text) < 10:
                continue
            
            # CDial-Bias 通常有 bias 标签
            bias = row.get('bias', row.get('label', 0))
            
            if bias:
                label = "谩骂引战"
                difficulty = "hard"  # 对话偏见通常很隐含
            else:
                label = "不违规"
                difficulty = "easy"
            
            records.append({
                "text": text,
                "label": label,
                "source": "cdial_bias",
                "source_label": f"cdial_{'bias' if bias else 'safe'}",
                "difficulty": difficulty,
            })
        
        if not records:
            return None
            
        result_df = pd.DataFrame(records)
        print(f"  处理后样本数: {len(result_df)}")
        
        out_path = output_dir / "cdial_bias_processed.parquet"
        result_df.to_parquet(out_path, index=False)
        print(f"  ✓ 保存到: {out_path}")
        
        return result_df
        
    except Exception as e:
        print(f"  ⚠️ CDial-Bias 下载失败: {e}")
        print("  此数据集为可选，跳过不影响主流程")
        return None


# ============================================================
# 5. ChineseHarm 原始数据重新标注难度
# ============================================================

def process_chineseharm_with_difficulty(
    bench_path: Path, 
    output_dir: Path
) -> Optional[pd.DataFrame]:
    """
    为 ChineseHarm 原始数据标注难度等级
    根据关键词覆盖率评估每条样本的难度
    """
    print("\n" + "=" * 60)
    print("[5/5] ChineseHarm原始数据 - 难度标注")
    print("=" * 60)
    
    # 每个类别的强信号关键词
    STRONG_KEYWORDS = {
        "博彩": ["赌", "彩", "盘", "注", "赔率", "下注", "开奖", "中奖", "时时彩",
                 "棋牌", "扑克", "老虎机", "红包", "接龙", "扫雷", "投注"],
        "低俗色情": ["约", "夜", "性", "色", "裸", "胸", "服务", "按摩", "上门",
                    "私密", "情趣", "内衣", "福利", "91", "吃瓜"],
        "谩骂引战": ["狗", "蠢", "废物", "垃圾", "死", "滚", "贱", "丑",
                    "歧视", "恶心", "傻"],
        "欺诈": ["兼职", "佣金", "日赚", "免费", "中奖", "领取", "回馈",
                "投资", "收益", "贷款", "清退", "兑付", "客服", "退款"],
        "黑产广告": ["日结", "点赞", "关注", "刷单", "账号", "实名", "解封",
                    "代注册", "信用", "征信", "套现", "接码", "引流", "粉丝"],
    }
    
    with open(bench_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for item in data:
        text = item.get('文本', '').strip()
        label = item.get('标签', '').strip()
        if not text or not label:
            continue
        
        # 评估难度: 统计关键词命中数
        if label == "不违规":
            difficulty = "easy"
        else:
            keywords = STRONG_KEYWORDS.get(label, [])
            hit_count = sum(1 for kw in keywords if kw in text)
            if hit_count >= 3:
                difficulty = "easy"     # 关键词明显，太简单
            elif hit_count >= 1:
                difficulty = "medium"   # 有关键词但不多
            else:
                difficulty = "hard"     # 无明显关键词，隐式有害
        
        records.append({
            "text": text,
            "label": label,
            "source": "chineseharm",
            "source_label": f"chineseharm_{label}",
            "difficulty": difficulty,
        })
    
    result_df = pd.DataFrame(records)
    
    print(f"  总样本数: {len(result_df)}")
    print(f"  难度分布:")
    for diff, count in result_df['difficulty'].value_counts().items():
        print(f"    {diff}: {count}")
    hard_pct = len(result_df[result_df['difficulty'] == 'hard']) / len(result_df) * 100
    print(f"  hard样本占比: {hard_pct:.1f}% (越高说明原数据集本身越有挑战)")
    
    out_path = output_dir / "chineseharm_with_difficulty.parquet"
    result_df.to_parquet(out_path, index=False)
    print(f"  ✓ 保存到: {out_path}")
    
    return result_df


# ============================================================
# 数据整合策略
# ============================================================

def merge_datasets(
    output_dir: Path,
    chineseharm_df: Optional[pd.DataFrame],
    toxicn_df: Optional[pd.DataFrame],
    cold_df: Optional[pd.DataFrame],
    safety_df: Optional[pd.DataFrame],
    cdial_df: Optional[pd.DataFrame],
    strategy: str = "balanced_hard",
) -> pd.DataFrame:
    """
    整合多数据集，支持多种混合策略
    
    策略:
      - "all": 合并全部数据
      - "balanced_hard": 平衡采样，偏重hard样本 (推荐!)
      - "hard_only": 只保留medium和hard样本
      - "adversarial_focus": 针对对抗训练优化的采样策略
    """
    print("\n" + "=" * 60)
    print(f"数据整合 (策略: {strategy})")
    print("=" * 60)
    
    dfs = []
    for name, df in [
        ("ChineseHarm", chineseharm_df),
        ("ToxiCN", toxicn_df),
        ("COLD", cold_df),
        ("Safety-Prompts", safety_df),
        ("CDial-Bias", cdial_df),
    ]:
        if df is not None and len(df) > 0:
            # 确保统一列
            keep_cols = ["text", "label", "source", "difficulty"]
            existing_cols = [c for c in keep_cols if c in df.columns]
            dfs.append(df[existing_cols].copy())
            print(f"  {name}: {len(df)} 条")
    
    if not dfs:
        print("  ⚠️ 没有可用数据")
        return pd.DataFrame()
    
    merged = pd.concat(dfs, ignore_index=True)
    
    # 去重 (基于text)
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=['text'], keep='first')
    print(f"  去重: {before_dedup} → {len(merged)}")
    
    if strategy == "all":
        final_df = merged
        
    elif strategy == "balanced_hard":
        # 每个类别尽量平衡，hard样本权重更高
        final_records = []
        
        for label in merged['label'].unique():
            label_df = merged[merged['label'] == label]
            
            # hard样本全部保留
            hard = label_df[label_df['difficulty'] == 'hard']
            medium = label_df[label_df['difficulty'] == 'medium']
            easy = label_df[label_df['difficulty'] == 'easy']
            
            # 每个类别目标: hard全保留, medium保留50%, easy保留20%
            final_records.append(hard)
            if len(medium) > 0:
                n_medium = max(int(len(medium) * 0.5), min(len(medium), 500))
                final_records.append(medium.sample(n=min(n_medium, len(medium)), random_state=42))
            if len(easy) > 0:
                n_easy = max(int(len(easy) * 0.2), min(len(easy), 300))
                final_records.append(easy.sample(n=min(n_easy, len(easy)), random_state=42))
        
        final_df = pd.concat(final_records, ignore_index=True)
        
    elif strategy == "hard_only":
        final_df = merged[merged['difficulty'].isin(['medium', 'hard'])]
        
    elif strategy == "adversarial_focus":
        # 对抗训练专用: 最大化Reviewer的困难度
        # 1. 全部hard样本
        # 2. anti-bias样本 (容易误判)
        # 3. implicit toxic样本
        # 4. 少量easy样本防止过拟合
        hard = merged[merged['difficulty'] == 'hard']
        medium = merged[merged['difficulty'] == 'medium']
        easy = merged[merged['difficulty'] == 'easy']
        
        final_records = [hard]
        if len(medium) > 0:
            final_records.append(medium.sample(n=min(len(medium), 3000), random_state=42))
        if len(easy) > 0:
            final_records.append(easy.sample(n=min(len(easy), 1000), random_state=42))
        
        final_df = pd.concat(final_records, ignore_index=True)
    else:
        final_df = merged
    
    # 打乱
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n  最终数据集大小: {len(final_df)}")
    print(f"  标签分布:")
    for label, count in final_df['label'].value_counts().items():
        print(f"    {label}: {count}")
    print(f"  难度分布:")
    for diff, count in final_df['difficulty'].value_counts().items():
        print(f"    {diff}: {count}")
    print(f"  来源分布:")
    for src, count in final_df['source'].value_counts().items():
        print(f"    {src}: {count}")
    
    # 保存
    out_path = output_dir / f"merged_{strategy}.parquet"
    final_df.to_parquet(out_path, index=False)
    print(f"\n  ✓ 保存到: {out_path}")
    
    # 同时保存 JSON 格式 (兼容现有pipeline)
    json_records = []
    for _, row in final_df.iterrows():
        json_records.append({
            "文本": row['text'],
            "标签": row['label'],
            "来源": row.get('source', 'unknown'),
            "难度": row.get('difficulty', 'unknown'),
        })
    
    json_path = output_dir / f"merged_{strategy}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_records, f, ensure_ascii=False, indent=2)
    print(f"  ✓ JSON格式: {json_path}")
    
    # 生成报告
    report = {
        "strategy": strategy,
        "total_samples": len(final_df),
        "label_distribution": final_df['label'].value_counts().to_dict(),
        "difficulty_distribution": final_df['difficulty'].value_counts().to_dict(),
        "source_distribution": final_df['source'].value_counts().to_dict(),
        "cross_table": final_df.groupby(['source', 'difficulty']).size().to_dict(),
    }
    # 将元组键转为字符串
    report["cross_table"] = {str(k): v for k, v in report["cross_table"].items()}
    
    report_path = output_dir / f"merge_report_{strategy}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 报告: {report_path}")
    
    return final_df


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="增强版中文有害文本数据集下载与整合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载全部数据集并以balanced_hard策略整合
  python download_enhanced_datasets.py --output_dir ./enhanced_data --strategy balanced_hard
  
  # 只使用ToxiCN和COLD (最推荐的组合)
  python download_enhanced_datasets.py --output_dir ./enhanced_data --datasets toxicn,cold --strategy adversarial_focus
  
  # 下载全部但只保留hard样本
  python download_enhanced_datasets.py --output_dir ./enhanced_data --strategy hard_only

推荐策略 (按论文需求):
  balanced_hard     → SFT冷启动 (平衡各类别，偏重困难样本)
  adversarial_focus → RL对抗训练 (最大化Reviewer难度)
  hard_only         → 消融实验 (只用困难样本验证方法有效性)
        """
    )
    parser.add_argument("--output_dir", type=str, 
                       default="./enhanced_data",
                       help="输出目录")
    parser.add_argument("--bench_path", type=str,
                       default=None,
                       help="ChineseHarm bench.json路径 (自动检测)")
    parser.add_argument("--datasets", type=str, default="all",
                       help="要下载的数据集, 逗号分隔. 可选: toxicn,cold,safety,cdial,chineseharm. 默认: all")
    parser.add_argument("--strategy", type=str, default="balanced_hard",
                       choices=["all", "balanced_hard", "hard_only", "adversarial_focus"],
                       help="数据整合策略")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 自动检测 bench.json 路径
    if args.bench_path:
        bench_path = Path(args.bench_path)
    else:
        # 尝试常见路径
        candidates = [
            Path(__file__).parent.parent.parent / "origin_data" / "bench.json",
            Path(__file__).parent.parent.parent.parent / "ChineseHarm-bench" / "benchmark" / "bench.json",
        ]
        bench_path = None
        for p in candidates:
            if p.exists():
                bench_path = p
                break
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   增强版中文有害文本数据集下载与整合                          ║")
    print("║   目标: 让双智能体对抗博弈更有挑战性                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  输出目录:  {output_dir}")
    print(f"  整合策略:  {args.strategy}")
    print(f"  数据集:    {args.datasets}")
    print()
    
    datasets_to_download = args.datasets.split(',') if args.datasets != "all" else \
        ["toxicn", "cold", "safety", "cdial", "chineseharm"]
    
    # 下载各数据集
    toxicn_df = None
    cold_df = None
    safety_df = None
    cdial_df = None
    chineseharm_df = None
    
    if "toxicn" in datasets_to_download:
        toxicn_df = download_and_process_toxicn(output_dir)
    
    if "cold" in datasets_to_download:
        cold_df = download_and_process_cold(output_dir)
    
    if "safety" in datasets_to_download:
        safety_df = download_and_process_safety_prompts(output_dir)
    
    if "cdial" in datasets_to_download:
        cdial_df = download_and_process_cdial_bias(output_dir)
    
    if "chineseharm" in datasets_to_download and bench_path and bench_path.exists():
        chineseharm_df = process_chineseharm_with_difficulty(bench_path, output_dir)
    
    # 整合
    merged_df = merge_datasets(
        output_dir=output_dir,
        chineseharm_df=chineseharm_df,
        toxicn_df=toxicn_df,
        cold_df=cold_df,
        safety_df=safety_df,
        cdial_df=cdial_df,
        strategy=args.strategy,
    )
    
    print("\n" + "=" * 60)
    print("完成! 后续步骤:")
    print("=" * 60)
    print(f"  1. 检查输出: {output_dir}")
    print(f"  2. 用 merged_{args.strategy}.json 替换原始 bench.json")
    print(f"  3. 重新运行 split_dataset.py 划分训练/验证/测试集")
    print(f"  4. 重新运行 prepare_all_data.py 准备SFT和RL数据")
    print(f"  5. 重新训练 LoRA, 观察效果是否下降到合理水平")
    print()
    print("  提示: 如果效果仍然太好，尝试 --strategy hard_only")
    print("  提示: 如果效果下降太多，尝试 --strategy all")


if __name__ == "__main__":
    main()
