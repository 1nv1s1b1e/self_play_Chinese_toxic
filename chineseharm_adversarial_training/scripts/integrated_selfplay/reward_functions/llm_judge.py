#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-as-a-Judge: 标签忠实度验证模块 (集成版)

功能:
  判断 Challenger 生成的文本是否真正属于目标类别。
  使用强大的 LLM (通义千问/Qwen API) 进行语义级别的类别判断,
  替代 TF-IDF 分类器的词频模式匹配。

改进 (集成版):
  - 类别定义统一从 constants.py 导入

配置 (通过环境变量):
  LLM_JUDGE_API_KEY:   API 密钥 (必须)
  LLM_JUDGE_BASE_URL:  API 地址 (默认: 通义千问 DashScope)
  LLM_JUDGE_MODEL:     模型名 (默认: qwen-plus)
"""

import os
import sys
import json
import re
import time
from typing import Optional

# ── 导入统一常量 ──
_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SELF_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from constants import HARMFUL_CATEGORIES, ALL_CATEGORIES, RULES, get_category_rules

# ── 配置 ──
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen-plus"
_MAX_RETRIES = 2
_TIMEOUT = 30

# ── 类别定义 (从 constants.py 的 RULES 构建，供 Judge prompt 使用) ──
CATEGORY_DEFINITIONS = {}
for cat in ALL_CATEGORIES:
    rules_text = get_category_rules(cat)
    if rules_text:
        CATEGORY_DEFINITIONS[cat] = rules_text
    elif cat == "无毒":
        CATEGORY_DEFINITIONS[cat] = (
            "不属于性别歧视、种族歧视、地域偏见、LGBTQ歧视、其他仇恨中任何类别的正常内容。"
            "注意：转述仇恨事件但本身不含恶意的内容应判为无毒。"
        )

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
        text=text[:1000],
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

        except Exception:
            if attempt < _MAX_RETRIES:
                time.sleep(1.0 * (attempt + 1))
            else:
                return None


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

    try:
        data = json.loads(content)
        if "belongs" in data:
            return bool(data["belongs"])
    except (json.JSONDecodeError, TypeError):
        pass

    if re.search(r'"belongs"\s*:\s*true', content):
        return True
    if re.search(r'"belongs"\s*:\s*false', content):
        return False
    if content in ("是", "yes", "true"):
        return True
    if content in ("否", "不是", "no", "false"):
        return False

    return None


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
        True:  标签忠实
        False: 标签漂移
        None:  API 不可用或解析失败
    """
    if not text or len(text.strip()) < 5:
        return False
    return _call_api_openai(text.strip(), target_category)


def is_judge_available() -> bool:
    """检查 LLM Judge 是否可用"""
    return bool(os.environ.get("LLM_JUDGE_API_KEY", ""))
