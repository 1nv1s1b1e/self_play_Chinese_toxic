#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
课程学习调度器 (Curriculum Learning Scheduler)
================================================
对应 PDF 图5「动态策略熵正则化」:

  通过训练过程中的问题进行难度分类:
    - 困难问题: 对优化采用熵正则化以加强探索能力，避免迭代答案生成时过度自信
    - 简单问题: 不采用熵正则化，提高置信度

核心思路:
  1. 根据 Verifier 置信度 / 迭代收敛速度给样本打难度分
  2. 训练早期先用简单样本 (warm-up)
  3. 逐步引入困难样本 (curriculum)
  4. 动态调整 KL 惩罚系数 (难 → 大 KL 鼓励探索, 简单 → 小 KL 鼓励确定性)

难度信号来源:
  - API Verifier 的 confidence (多源不一致 → 难)
  - 迭代 Reviewer 的 convergence_rate (不收敛 → 难)
  - Reviewer vs Verifier 分歧 (分歧 → 难)
  - 历史轮次该样本的类别 ASR (ASR 高 → 该类别整体难)
"""

import math
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SampleDifficulty:
    """单条样本的难度信息"""
    text: str
    category: str
    difficulty_score: float          # 综合难度 [0, 1], 1=最难
    verifier_confidence: float = 1.0 # Verifier 置信度
    convergence_rate: float = 1.0    # 迭代收敛率
    reviewer_verifier_agree: bool = True  # R/V 是否一致
    category_asr: float = 0.0       # 该类别历史 ASR
    
    # 课程学习参数
    kl_weight: float = 1.0          # KL 惩罚权重 (难题更大)
    reward_weight: float = 1.0      # 奖励加权 (课程加权)
    include_in_training: bool = True # 是否纳入当前轮训练


@dataclass
class CurriculumConfig:
    """课程学习配置"""
    # 难度计算权重
    w_verifier_confidence: float = 0.3   # Verifier 置信度权重
    w_convergence: float = 0.3           # 迭代收敛权重
    w_rv_agreement: float = 0.2          # R/V 一致性权重
    w_category_asr: float = 0.2          # 类别 ASR 权重

    # 课程进度参数
    warmup_rounds: int = 1               # 前 N 轮只用简单样本
    difficulty_threshold_start: float = 0.3  # 初始难度阈值 (只训练 ≤ 此阈值的样本)
    difficulty_threshold_end: float = 1.0    # 最终难度阈值 (训练所有样本)
    schedule_type: str = "linear"        # 进度调度: "linear" / "cosine" / "step"
    
    # KL 动态调整
    kl_base: float = 0.05               # 基础 KL 系数
    kl_hard_multiplier: float = 3.0     # 困难样本 KL 放大倍数
    kl_easy_multiplier: float = 0.5     # 简单样本 KL 缩小倍数
    difficulty_split: float = 0.5        # 难/易分界线

    # 熵正则化
    entropy_bonus_hard: float = 0.01     # 困难样本的熵奖励
    entropy_bonus_easy: float = 0.0      # 简单样本的熵奖励 (= 0, 不正则化)


class CurriculumScheduler:
    """
    课程学习调度器

    用法:
        scheduler = CurriculumScheduler(total_rounds=5)
        
        # 每轮开始时
        difficulties = scheduler.compute_difficulties(samples, round_idx)
        filtered = scheduler.filter_by_curriculum(difficulties, round_idx)
        kl_weights = scheduler.get_kl_weights(difficulties)
    """

    def __init__(
        self,
        total_rounds: int = 5,
        config: CurriculumConfig = None,
    ):
        self.total_rounds = total_rounds
        self.config = config or CurriculumConfig()
        self.history: Dict[int, Dict[str, float]] = {}  # round → category_asr

    def compute_difficulty(
        self,
        verifier_confidence: float = 1.0,
        convergence_rate: float = 1.0,
        reviewer_verifier_agree: bool = True,
        category_asr: float = 0.0,
    ) -> float:
        """
        计算单条样本的综合难度分数

        difficulty ∈ [0, 1], 1 = 最难

        信号加权:
          difficulty = w1 × (1 - confidence)      # 低置信度 → 难
                     + w2 × (1 - convergence)     # 低收敛率 → 难
                     + w3 × (0 if agree else 1)   # R/V 不一致 → 难
                     + w4 × category_asr           # 高 ASR → 该类别整体难
        """
        cfg = self.config
        
        d = (cfg.w_verifier_confidence * (1.0 - verifier_confidence)
             + cfg.w_convergence * (1.0 - convergence_rate)
             + cfg.w_rv_agreement * (0.0 if reviewer_verifier_agree else 1.0)
             + cfg.w_category_asr * category_asr)

        return max(0.0, min(1.0, d))

    def compute_difficulties(
        self,
        samples: List[dict],
        round_idx: int = 0,
    ) -> List[SampleDifficulty]:
        """
        批量计算样本难度

        Args:
            samples: List of dicts, 每个包含:
                text, category, verifier_confidence, convergence_rate,
                reviewer_verifier_agree, ...
            round_idx: 当前轮次
        """
        # 获取当前轮的类别 ASR
        cat_asr = self.history.get(round_idx - 1, {})

        difficulties = []
        for sample in samples:
            category = sample.get("category", "无毒")
            v_conf = sample.get("verifier_confidence", 1.0)
            conv_rate = sample.get("convergence_rate", 1.0)
            rv_agree = sample.get("reviewer_verifier_agree", True)
            c_asr = cat_asr.get(category, 0.0)

            score = self.compute_difficulty(
                verifier_confidence=v_conf,
                convergence_rate=conv_rate,
                reviewer_verifier_agree=rv_agree,
                category_asr=c_asr,
            )

            sd = SampleDifficulty(
                text=sample.get("text", ""),
                category=category,
                difficulty_score=score,
                verifier_confidence=v_conf,
                convergence_rate=conv_rate,
                reviewer_verifier_agree=rv_agree,
                category_asr=c_asr,
            )

            difficulties.append(sd)

        return difficulties

    def get_difficulty_threshold(self, round_idx: int) -> float:
        """
        根据当前轮次获取难度阈值

        课程进度: round_idx → threshold ∈ [start, end]
        """
        cfg = self.config

        if round_idx < cfg.warmup_rounds:
            return cfg.difficulty_threshold_start

        # 进度 ∈ [0, 1]
        progress = (round_idx - cfg.warmup_rounds) / max(1, self.total_rounds - cfg.warmup_rounds)
        progress = min(1.0, progress)

        if cfg.schedule_type == "linear":
            threshold = (cfg.difficulty_threshold_start
                         + progress * (cfg.difficulty_threshold_end - cfg.difficulty_threshold_start))
        elif cfg.schedule_type == "cosine":
            # cosine schedule: 前期慢，后期快
            cosine_progress = 0.5 * (1 - math.cos(math.pi * progress))
            threshold = (cfg.difficulty_threshold_start
                         + cosine_progress * (cfg.difficulty_threshold_end - cfg.difficulty_threshold_start))
        elif cfg.schedule_type == "step":
            # 阶梯式
            steps = 3
            step_idx = min(int(progress * steps), steps - 1)
            step_thresholds = np.linspace(
                cfg.difficulty_threshold_start,
                cfg.difficulty_threshold_end,
                steps
            )
            threshold = step_thresholds[step_idx]
        else:
            threshold = cfg.difficulty_threshold_end

        return float(threshold)

    def filter_by_curriculum(
        self,
        difficulties: List[SampleDifficulty],
        round_idx: int,
    ) -> List[SampleDifficulty]:
        """
        根据课程进度过滤样本

        Args:
            difficulties: 样本难度列表
            round_idx: 当前轮次

        Returns:
            过滤后的样本难度列表 (include_in_training=True 的子集)
        """
        threshold = self.get_difficulty_threshold(round_idx)

        included = 0
        for sd in difficulties:
            if sd.difficulty_score <= threshold:
                sd.include_in_training = True
                included += 1
            else:
                sd.include_in_training = False

        logger.info(f"课程过滤 (Round {round_idx}): "
                     f"阈值={threshold:.2f}, "
                     f"纳入={included}/{len(difficulties)} "
                     f"({included/len(difficulties)*100:.1f}%)")

        return [sd for sd in difficulties if sd.include_in_training]

    def get_kl_weights(self, difficulties: List[SampleDifficulty]) -> List[float]:
        """
        根据难度动态计算 KL 惩罚权重

        PDF 设计:
          困难样本 → 大 KL → 鼓励探索 (熵正则化)
          简单样本 → 小 KL → 鼓励确定性
        """
        cfg = self.config
        kl_weights = []

        for sd in difficulties:
            if sd.difficulty_score >= cfg.difficulty_split:
                # 困难
                kl = cfg.kl_base * cfg.kl_hard_multiplier
            else:
                # 简单
                kl = cfg.kl_base * cfg.kl_easy_multiplier

            sd.kl_weight = kl
            kl_weights.append(kl)

        return kl_weights

    def get_reward_weights(self, difficulties: List[SampleDifficulty]) -> List[float]:
        """
        获取奖励加权系数

        困难样本给更高权重 (让模型更关注边界区域)
        简单样本给较低权重 (已经学会的不需要反复强调)
        """
        weights = []
        for sd in difficulties:
            # 线性加权: 难度 0 → 权重 0.5, 难度 1 → 权重 1.5
            w = 0.5 + sd.difficulty_score
            sd.reward_weight = w
            weights.append(w)
        return weights

    def get_entropy_bonuses(self, difficulties: List[SampleDifficulty]) -> List[float]:
        """
        获取每条样本的熵正则化奖励

        PDF 图5:
          困难样本 → 正熵奖励 → 鼓励探索
          简单样本 → 无熵奖励 → 收敛到确定答案
        """
        cfg = self.config
        bonuses = []
        for sd in difficulties:
            if sd.difficulty_score >= cfg.difficulty_split:
                bonuses.append(cfg.entropy_bonus_hard)
            else:
                bonuses.append(cfg.entropy_bonus_easy)
        return bonuses

    def update_history(self, round_idx: int, category_asr: Dict[str, float]):
        """记录本轮各类别 ASR，供下一轮难度计算使用"""
        self.history[round_idx] = category_asr
        logger.info(f"更新历史 ASR (Round {round_idx}): {category_asr}")

    def get_stats(self, difficulties: List[SampleDifficulty]) -> Dict[str, Any]:
        """获取当前批次的难度统计"""
        scores = [sd.difficulty_score for sd in difficulties]
        
        stats = {
            "total_samples": len(difficulties),
            "difficulty_mean": sum(scores) / len(scores) if scores else 0,
            "difficulty_std": float(np.std(scores)) if scores else 0,
            "difficulty_min": min(scores) if scores else 0,
            "difficulty_max": max(scores) if scores else 0,
            "easy_count": sum(1 for s in scores if s < 0.3),
            "medium_count": sum(1 for s in scores if 0.3 <= s < 0.7),
            "hard_count": sum(1 for s in scores if s >= 0.7),
            "included_count": sum(1 for sd in difficulties if sd.include_in_training),
        }

        # 按类别统计
        from collections import defaultdict
        cat_stats = defaultdict(list)
        for sd in difficulties:
            cat_stats[sd.category].append(sd.difficulty_score)

        stats["per_category"] = {
            cat: {
                "mean_difficulty": sum(scores) / len(scores),
                "count": len(scores),
            }
            for cat, scores in cat_stats.items()
        }

        return stats


# ── CLI 工具 (测试用) ────────────────────────────────────────────────────────────

def main():
    """演示课程学习调度器"""
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 模拟样本
    samples = [
        {"text": "简单样本", "category": "无毒", "verifier_confidence": 0.95, "convergence_rate": 1.0, "reviewer_verifier_agree": True},
        {"text": "中等样本", "category": "性别歧视", "verifier_confidence": 0.7, "convergence_rate": 0.6, "reviewer_verifier_agree": True},
        {"text": "困难样本", "category": "其他仇恨", "verifier_confidence": 0.5, "convergence_rate": 0.3, "reviewer_verifier_agree": False},
        {"text": "极难样本", "category": "地域偏见", "verifier_confidence": 0.3, "convergence_rate": 0.2, "reviewer_verifier_agree": False},
    ]

    scheduler = CurriculumScheduler(total_rounds=5)

    for round_idx in range(5):
        print(f"\n=== Round {round_idx} ===")
        difficulties = scheduler.compute_difficulties(samples, round_idx)
        filtered = scheduler.filter_by_curriculum(difficulties, round_idx)
        kl_weights = scheduler.get_kl_weights(difficulties)
        reward_weights = scheduler.get_reward_weights(difficulties)
        entropy_bonuses = scheduler.get_entropy_bonuses(difficulties)
        
        stats = scheduler.get_stats(difficulties)
        print(f"难度统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")

        for sd, kl, rw, eb in zip(difficulties, kl_weights, reward_weights, entropy_bonuses):
            print(f"  [{sd.category}] difficulty={sd.difficulty_score:.2f} "
                  f"kl={kl:.4f} reward_w={rw:.2f} entropy={eb:.4f} "
                  f"include={sd.include_in_training}")

        # 模拟 ASR 更新
        scheduler.update_history(round_idx, {
            "性别歧视": 0.3 + round_idx * 0.05,
            "其他仇恨": 0.5 + round_idx * 0.03,
            "地域偏见": 0.4 + round_idx * 0.04,
        })


if __name__ == "__main__":
    main()
