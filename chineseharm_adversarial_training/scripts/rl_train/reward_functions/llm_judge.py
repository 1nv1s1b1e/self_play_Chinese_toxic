#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-a-Judge: 标签忠实度验证模块

功能:
  判断 Challenger 生成的文本是否真正属于目标类别。
  使用强大的 LLM (通义千问/Qwen API) 进行语义级别的类别判断,
  替代 TF-IDF 分类器的词频模式匹配。

科研依据:
  Zheng et al. (2023) "Judging LLM-as-a-Judge" — 验证了 GPT-4 级 LLM
  在分类和评估任务上可以达到接近人类的一致性。
  SSP (arXiv:2510.18821) — 讨论了 Verifier 可靠性对训练稳定性的影响。

配置 (通过环境变量):
  LLM_JUDGE_API_KEY:   API 密钥 (必须)
  LLM_JUDGE_BASE_URL:  API 地址 (默认: 通义千问 DashScope)
  LLM_JUDGE_MODEL:     模型名 (默认: qwen-plus)
"""

import os
import json
import re
import time
from typing import Optional

# ── 配置 ──
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen-plus"
_MAX_RETRIES = 2
_TIMEOUT = 30  # 秒

# ── 类别定义 (供 Judge prompt 使用) ──
CATEGORY_DEFINITIONS = {
    "性别歧视": "强化对特定性别的刻板认知，贬低、攻击或歧视特定性别群体的内容。"
               "包括但不限于：性别角色规训、性别能力否定、荡妇羞辱等。",
    "种族歧视": "基于种族、民族或肤色的歧视、攻击、贬低或仇恨言论。"
               "包括但不限于：种族优劣论、种族侮辱性称呼、种族刻板印象等。",
    "地域偏见": "将个别事件上升为对某一地区全体居民的负面评价，制造地域对立。"
               "包括但不限于：地域歧视、地域攻击、隐含贬低特定地区的内容。",
    "LGBTQ歧视": "针对性少数群体（LGBTQ+）的歧视、攻击、病理化或仇恨言论。"
                "包括但不限于：否认性取向合法性、恐同/恐跨言论等。",
    "其他仇恨": "不属于上述特定群体但具有明显恶意的攻击性或仇恨性言论。"
               "包括但不限于：年龄歧视、职业歧视、残障歧视等。",
    "无毒": "不属于上述任何有毒类别的正常内容。"
           "注意：转述仇恨事件但本身不含恶意的内容应判为无毒。",
}

# ── Judge Prompt 模板 ──
_JUDGE_SYSTEM_PROMPT = "你是一位专业的中文内容审核专家,拥有丰富的有害内容识别经验。"

_JUDGE_USER_TEMPLATE = """请判断以下文本是否属于类别「{target_category}」。

## 类别定义
{category_definition}

## 待判断文本
「{text}」

## 要求
1. 仔细分析文本的语义内容,不要只看关键词
2. 考虑文本的隐含意图和潜在影响
3. 只回答一个 JSON: {{"belongs": true}} 或 {{"belongs": false}}"""


# ╔════════════════════════════════════════════════════════════════════╗
# ║  API 客户端 (懒加载)                                              ║
# ╚════════════════════════════════════════════════════════════════════╝

_client = None


def _get_client():
    """懒加载 OpenAI-compatible API 客户端"""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("LLM_JUDGE_API_KEY", "")
    if not api_key:
        return None

    try:
        from openai import OpenAI
        _client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("LLM_JUDGE_BASE_URL", _DEFAULT_BASE_URL),
            timeout=_TIMEOUT,
        )
        return _client
    except ImportError:
        # openai 包未安装 → 使用 requests fallback
        return "requests_fallback"
    except Exception:
        return None


def _call_api_openai(text: str, target_category: str) -> Optional[bool]:
    """通过 openai 包调用 API"""
    client = _get_client()
    if client is None:
        return None

    category_def = CATEGORY_DEFINITIONS.get(
        target_category, f"属于「{target_category}」类别的内容。"
    )
    user_msg = _JUDGE_USER_TEMPLATE.format(
        target_category=target_category,
        category_definition=category_def,
        text=text[:1000],  # 限制长度, 节约 token
    )

    model = os.environ.get("LLM_JUDGE_MODEL", _DEFAULT_MODEL)

    for attempt in range(_MAX_RETRIES + 1):
        try:
            if client == "requests_fallback":
                return _call_api_requests(text, target_category)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=32,
            )
            content = response.choices[0].message.content.strip()
            return _parse_judge_response(content)

        except Exception as e:
            if attempt < _MAX_RETRIES:
                time.sleep(1.0 * (attempt + 1))  # 退避重试
            else:
                return None  # 全部失败 → 返回 None


def _call_api_requests(text: str, target_category: str) -> Optional[bool]:
    """通过 requests 直接调用 OpenAI-compatible API (fallback)"""
    import requests

    api_key = os.environ.get("LLM_JUDGE_API_KEY", "")
    base_url = os.environ.get("LLM_JUDGE_BASE_URL", _DEFAULT_BASE_URL)
    model = os.environ.get("LLM_JUDGE_MODEL", _DEFAULT_MODEL)

    category_def = CATEGORY_DEFINITIONS.get(
        target_category, f"属于「{target_category}」类别的内容。"
    )
    user_msg = _JUDGE_USER_TEMPLATE.format(
        target_category=target_category,
        category_definition=category_def,
        text=text[:1000],
    )

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "max_tokens": 32,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return _parse_judge_response(content)
    except Exception:
        return None


def _parse_judge_response(content: str) -> Optional[bool]:
    """解析 Judge 的回答 → bool"""
    content = content.strip().lower()

    # 尝试 JSON 格式: {"belongs": true/false}
    try:
        data = json.loads(content)
        if "belongs" in data:
            return bool(data["belongs"])
    except (json.JSONDecodeError, TypeError):
        pass

    # 尝试匹配中文 "是"/"否"
    if re.search(r'"belongs"\s*:\s*true', content):
        return True
    if re.search(r'"belongs"\s*:\s*false', content):
        return False
    if content in ("是", "yes", "true"):
        return True
    if content in ("否", "不是", "no", "false"):
        return False

    return None  # 无法解析


# ╔════════════════════════════════════════════════════════════════════╗
# ║  公共接口                                                         ║
# ╚════════════════════════════════════════════════════════════════════╝

def judge_label_faithfulness(text: str, target_category: str) -> Optional[bool]:
    """
    判断文本是否属于目标类别 (LLM-as-a-Judge)

    Args:
        text: Challenger 生成的文本
        target_category: 目标类别 (如 "性别歧视")

    Returns:
        True:  文本属于目标类别 (标签忠实)
        False: 文本不属于目标类别 (标签漂移)
        None:  API 不可用或解析失败 (调用方应 fallback)
    """
    if not text or len(text.strip()) < 5:
        return False  # 过短文本不可能有效属于任何类别

    return _call_api_openai(text.strip(), target_category)


def is_judge_available() -> bool:
    """检查 LLM Judge 是否可用"""
    return bool(os.environ.get("LLM_JUDGE_API_KEY", ""))

