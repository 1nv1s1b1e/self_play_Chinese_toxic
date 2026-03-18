#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reviewer Agent 奖励函数 (v2: binary + category + toxic_type + expression)

Reviewer 任务:
  1. 二分类: 判断是否有害 ("有害" / "无害")
  2. 若有害，判断目标群体类别

奖励设计:
  基础分 (binary + category):        0.7 权重
  附加分 (toxic_type + expression):  0.3 权重

  漏检: -1.0 (最严重)
  误检: -0.5
  无法解析: -0.8
"""

ALL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]
HARMFUL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"]
TOXIC_TYPES = ["无毒", "一般攻击性", "仇恨言论"]
EXPRESSIONS = ["非仇恨", "显式仇恨", "隐式仇恨", "转述"]

def extract_prediction_full(output_text: str) -> dict:
    """
    从模型输出中提取完整多维预测:
      binary:     "有害" / "无害" / None
      category:   类别字符串 / None
      toxic_type: "一般攻击性" / "仇恨言论" / None
      expression: "显式仇恨" / "隐式仇恨" / "转述" / None
    """
    result = {"binary": None, "category": None, "toxic_type": None, "expression": None}
    if not output_text:
        return result

    text = output_text.strip()

    # ---- 新格式解析 ----
    has_harmful = ("判断：有害" in text or "判断:有害" in text
                   or "判断: 有害" in text or "判断 ：有害" in text)
    has_safe = ("判断：无害" in text or "判断:无害" in text
                or "判断: 无害" in text or "判断 ：无害" in text)

    if has_harmful:
        result["binary"] = "有害"
        # 提取类别
        for cat in HARMFUL_CATEGORIES:
            if cat in text:
                result["category"] = cat
                break
        if result["category"] is None:
            result["category"] = "其他仇恨"
        # 提取毒性类型
        for tt in ["一般攻击性", "仇恨言论"]:
            if tt in text:
                result["toxic_type"] = tt
                break
        # 提取表达方式
        for expr in ["隐式仇恨", "显式仇恨", "转述"]:
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

    # ---- 兼容旧格式: 直接匹配类别关键词 ----
    for cat in ALL_CATEGORIES:
        if cat in text[:30]:
            result["category"] = cat
            result["binary"] = "有害" if cat in HARMFUL_CATEGORIES else "无害"
            return result
    for cat in ALL_CATEGORIES:
        if cat in text:
            result["category"] = cat
            result["binary"] = "有害" if cat in HARMFUL_CATEGORIES else "无害"
            return result

    return result


def extract_prediction(output_text: str) -> str:
    """从模型输出中提取预测类别 (向后兼容接口)"""
    return extract_prediction_full(output_text)["category"]



try:
    from reward_functions.reward_logger import RewardLogger
except ImportError:
    from reward_logger import RewardLogger

# 模块级单例 Logger
_logger = RewardLogger("reviewer")


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Reviewer 奖励函数 (v2: binary + category + toxic_type + expression)
    # 日志控制: REWARD_DEBUG=0(默认) / 1(逐样本详情) / 2(+JSONL)
    if data_source != "toxicn_reviewer":
        return 0.0

    pred = extract_prediction_full(solution_str)

    import json as _json
    gt_toxic_type = None
    gt_expression = None

    if isinstance(ground_truth, dict):
        true_label    = ground_truth.get("category", ground_truth.get("ground_truth", ""))
        gt_toxic_type = ground_truth.get("toxic_type")
        gt_expression = ground_truth.get("expression")
    elif isinstance(ground_truth, str):
        try:
            gt_dict = _json.loads(ground_truth)
            if isinstance(gt_dict, dict):
                true_label    = gt_dict.get("category", "")
                gt_toxic_type = gt_dict.get("toxic_type")
                gt_expression = gt_dict.get("expression")
            else:
                true_label = ground_truth
        except (ValueError, _json.JSONDecodeError):
            true_label = ground_truth
    else:
        true_label = str(ground_truth) if ground_truth else ""

    true_is_harmful = true_label in HARMFUL_CATEGORIES
    true_binary     = "有害" if true_is_harmful else "无害"

    true_toxic_type = gt_toxic_type
    true_expression = gt_expression
    if extra_info and isinstance(extra_info, dict):
        if not true_toxic_type:
            true_toxic_type = extra_info.get("toxic_type")
        if not true_expression:
            true_expression = extra_info.get("expression")

    # 获取文本预览（用于日志）
    text_preview = ""
    if extra_info and isinstance(extra_info, dict):
        text_preview = extra_info.get("original_text", "")[:60]

    # ---- 基础分: binary + category (权重 0.7) ----
    if pred["binary"] is None:
        _logger.log_reviewer_sample(
            text_preview=text_preview, pred=pred,
            true_cat=true_label, true_is_harmful=true_is_harmful,
            base_score=-0.8, bonus=0.0, reward=-0.8,
        )
        _logger.log_batch_summary()
        return -0.8

    if pred["binary"] != true_binary:
        if true_is_harmful and pred["binary"] == "无害":
            reward = -1.0   # 漏检
        else:
            reward = -0.5   # 误检
        _logger.log_reviewer_sample(
            text_preview=text_preview, pred=pred,
            true_cat=true_label, true_is_harmful=true_is_harmful,
            base_score=reward, bonus=0.0, reward=reward,
        )
        _logger.log_batch_summary()
        return reward

    if pred["category"] == true_label:
        base_score = 1.0
    elif true_is_harmful:
        base_score = -0.1
    else:
        base_score = 1.0

    # ---- 附加分: toxic_type + expression (权重 0.3) ----
    bonus = 0.0
    if true_is_harmful and true_toxic_type and true_expression:
        tt_correct = (pred.get("toxic_type") == true_toxic_type)
        ex_correct = (pred.get("expression") == true_expression)
        if tt_correct and ex_correct:
            bonus = 1.0
        elif tt_correct or ex_correct:
            bonus = 0.5
        else:
            bonus = -0.3

    final_score = max(-1.0, min(1.0, base_score * 0.7 + bonus * 0.3))

    _logger.log_reviewer_sample(
        text_preview=text_preview, pred=pred,
        true_cat=true_label, true_is_harmful=true_is_harmful,
        base_score=base_score, bonus=bonus, reward=final_score,
    )
    _logger.log_batch_summary()
    return final_score
