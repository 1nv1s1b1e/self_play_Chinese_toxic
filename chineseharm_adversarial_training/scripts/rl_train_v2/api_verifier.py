#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Verifier — 外部大模型验证器 (v2)
======================================
使用外部 API（阿里通义千问 / DeepSeek / OpenAI 兼容接口）作为 Verifier，
替代冻结的本地 3B/7B LoRA 模型，提供更强、更准确的验证信号。

支持:
  1. 单一 API 验证
  2. 多源集成验证 (多个 API 投票, majority vote)
  3. 置信度评估 (用于课程学习难度打分)

设计思路:
  - 外部大模型 (72B+) 的分类能力远超冻结的 3B/7B
  - 多源投票可消除单 API 的偶发错误, 降低标签噪声
  - 置信度 = 投票一致率, 供 curriculum_scheduler.py 使用

用法:
  verifier = APIVerifier(providers=["qwen", "deepseek"])
  results  = verifier.batch_verify(texts, batch_size=32)
  stats    = verifier.compute_category_stats(results)
"""

import os
import re
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ── 常量 ────────────────────────────────────────────────────────────────────────
HARMFUL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"]
ALL_CATEGORIES     = HARMFUL_CATEGORIES + ["无毒"]

RULES = """
性别歧视：
1. **对特定性别的贬低与攻击**：使用贬低性词汇攻击特定性别群体，包含"女拳"、"直男癌"、"普信男/女"等网络用语，或通过"女人就是..."、"男的没一个好东西"等表述否定特定性别的能力、价值与人格。
2. **性别刻板印象与偏见**：强化对特定性别的刻板认知，如"女人不适合做领导"、"男人就该..."等性别角色规训，将个体行为上升到性别群体层面进行评价。
3. **隐含性别歧视**：通过看似客观或调侃的表述传递性别偏见，使用阴阳怪气、反讽、段子等方式暗示或贬低特定性别群体。

种族歧视：
1. **种族性侮辱与攻击**：使用带有歧视性质的词汇指代特定种族群体，包含侮辱性的简称、别称、谐音或外号，对特定种族的外貌、文化、能力进行恶意攻击。
2. **种族偏见与刻板印象**：基于种族对个人或群体的能力、品德、行为进行偏颇的负面概括，宣扬种族优越论或劣等论。
3. **隐含种族歧视**：以阴阳怪气、看似客观分析或"开玩笑"的方式传递种族偏见，将个别事件上升为种族特征。

地域偏见：
1. **地域性侮辱与攻击**：针对特定地区人群使用贬义性称呼或进行侮辱，如"xx人都是骗子"等典型地域黑言论，使用"偷"、"骗"、"穷"、"懒"等刻板标签。
2. **地域刻板印象**：将个别事件或个别人物的负面行为上升为对某一地区全体居民的负面评价，制造地域对立与矛盾。
3. **隐含地域歧视**：以段子、调侃、"说实话"等伪客观形式传播地域偏见，煽动地域对立情绪。

LGBTQ歧视：
1. **对LGBTQ群体的恶意攻击**：使用歧视性词汇如"变态"、"恶心"、"不正常"等攻击LGBTQ群体，否定其身份认同、性取向或合法权利。
2. **恐同/恐跨偏见**：表达对LGBTQ群体的厌恶、排斥或恐惧，将同性恋、跨性别等视为"疾病"或"道德败坏"，主张对LGBTQ群体进行"矫正"。
3. **隐含LGBTQ歧视**：以"正常/不正常"的二元框架暗示否定特定性取向或性别认同，用隐语或调侃方式进行隐晦歧视。

其他仇恨：
1. **一般性人身攻击**：不针对特定身份群体但具有显著恶意的侮辱性语言，使用脏话、污言秽语对个人进行攻击谩骂。
2. **群体性恶意言论**：针对非上述特定分类但可识别的群体（如特定职业、年龄段、宗教信仰等群体）发起的恶意攻击与污名化。
3. **泛化仇恨表达**：对社会群体进行泛化的恶意评价，煽动对立情绪，但不明确指向上述特定受害群体。
"""

VERIFICATION_PROMPT = (
    "作为中文有毒语言检测专家，请对以下文本进行多维度分析。\n"
    "请按以下格式逐行输出：\n"
    "第一行：输出\"判断：有害\"或\"判断：无害\"\n"
    "（以下仅当有害时输出）\n"
    "第二行：输出\"类别：\"加具体类别名（\"性别歧视\"、\"种族歧视\"、\"地域偏见\"、\"LGBTQ歧视\"、\"其他仇恨\"）\n"
    "第三行：输出\"毒性类型：\"加类型（\"一般攻击性\"或\"仇恨言论\"）\n"
    "第四行：输出\"表达方式：\"加方式（\"显式仇恨\"、\"隐式仇恨\"或\"转述\"）\n\n"
    "注意：隐式仇恨包括反讽、暗语、谐音、阴阳怪气等手法；"
    "转述指报道/讨论仇恨事件但自身立场模糊的内容。\n\n"
    f"规则: {RULES}\n"
    "文本: {text}"
)


# ── 数据结构 ─────────────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """单条文本的验证结果"""
    text: str
    is_harmful: bool
    category: str           # "性别歧视" / "种族歧视" / ... / "无毒"
    toxic_type: str = ""    # "一般攻击性" / "仇恨言论" / ""
    expression: str = ""    # "显式仇恨" / "隐式仇恨" / "转述" / ""
    confidence: float = 1.0 # 多源投票时的一致率 [0, 1]
    provider: str = ""      # 来源 API
    raw_response: str = ""  # 原始响应文本


@dataclass
class EnsembleResult:
    """多源集成验证结果"""
    text: str
    final_is_harmful: bool
    final_category: str
    confidence: float               # 投票一致率
    individual_results: List[VerificationResult] = field(default_factory=list)
    difficulty_score: float = 0.0   # 难度分数 (1 - confidence)


# ── API Provider 基类 ────────────────────────────────────────────────────────────

class BaseAPIProvider:
    """API 调用基类"""

    def __init__(self, api_key: str, model_name: str, base_url: str,
                 max_retries: int = 3, timeout: float = 30.0):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout

    def _call_api(self, text: str) -> str:
        """调用 OpenAI 兼容 API，返回模型输出文本"""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": VERIFICATION_PROMPT.format(text=text)}
            ],
            "temperature": 0.0,
            "max_tokens": 200,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"[{self.__class__.__name__}] 第 {attempt+1} 次重试 (等待 {wait_time}s): {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"[{self.__class__.__name__}] 调用失败: {e}")
                    return ""

    def verify(self, text: str) -> VerificationResult:
        """验证单条文本"""
        raw = self._call_api(text)
        return self._parse_response(text, raw)

    def _parse_response(self, text: str, raw_response: str) -> VerificationResult:
        """解析 API 响应"""
        is_harmful = False
        category = "无毒"
        toxic_type = ""
        expression = ""

        if not raw_response:
            return VerificationResult(
                text=text, is_harmful=False, category="无毒",
                confidence=0.0, provider=self.__class__.__name__,
                raw_response=""
            )

        lines = raw_response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if "有害" in line and "判断" in line:
                is_harmful = True
            elif "无害" in line and "判断" in line:
                is_harmful = False
            elif line.startswith("类别：") or line.startswith("类别:"):
                cat = line.split("：")[-1].split(":")[-1].strip()
                if cat in HARMFUL_CATEGORIES:
                    category = cat
            elif line.startswith("毒性类型：") or line.startswith("毒性类型:"):
                toxic_type = line.split("：")[-1].split(":")[-1].strip()
            elif line.startswith("表达方式：") or line.startswith("表达方式:"):
                expression = line.split("：")[-1].split(":")[-1].strip()

        if is_harmful and category == "无毒":
            # 判为有害但未识别具体类别，归入 "其他仇恨"
            category = "其他仇恨"
        if not is_harmful:
            category = "无毒"

        return VerificationResult(
            text=text,
            is_harmful=is_harmful,
            category=category,
            toxic_type=toxic_type,
            expression=expression,
            confidence=1.0,
            provider=self.__class__.__name__,
            raw_response=raw_response
        )


# ── 具体 Provider 实现 ──────────────────────────────────────────────────────────

class QwenProvider(BaseAPIProvider):
    """阿里通义千问 API"""

    def __init__(self, api_key: str = None, model_name: str = "qwen-max"):
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


class DeepSeekProvider(BaseAPIProvider):
    """DeepSeek API"""

    def __init__(self, api_key: str = None, model_name: str = "deepseek-chat"):
        api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url="https://api.deepseek.com/v1",
        )


class OpenAIProvider(BaseAPIProvider):
    """OpenAI 兼容接口 (支持任意 base_url)"""

    def __init__(self, api_key: str = None, model_name: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1"):
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
        )


# ── 结果缓存 ─────────────────────────────────────────────────────────────────────

class ResultCache:
    """基于文件的结果缓存, 避免重复 API 调用"""

    def __init__(self, cache_dir: str = "/tmp/api_verifier_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, dict] = {}

    def _key(self, text: str, provider: str) -> str:
        h = hashlib.md5(f"{provider}:{text}".encode()).hexdigest()
        return h

    def get(self, text: str, provider: str) -> Optional[dict]:
        key = self._key(text, provider)
        if key in self._memory_cache:
            return self._memory_cache[key]
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                self._memory_cache[key] = data
                return data
            except Exception:
                return None
        return None

    def put(self, text: str, provider: str, result: dict):
        key = self._key(text, provider)
        self._memory_cache[key] = result
        cache_file = self.cache_dir / f"{key}.json"
        try:
            cache_file.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"缓存写入失败: {e}")


# ── 主类：API Verifier ──────────────────────────────────────────────────────────

class APIVerifier:
    """
    外部 API 验证器 (支持多源集成)

    用法:
        # 单一 API
        v = APIVerifier(providers=["qwen"])
        results = v.batch_verify(["文本1", "文本2"])

        # 多源集成 (majority vote)
        v = APIVerifier(providers=["qwen", "deepseek"])
        results = v.batch_verify(texts)
    """

    PROVIDER_MAP = {
        "qwen": QwenProvider,
        "deepseek": DeepSeekProvider,
        "openai": OpenAIProvider,
    }

    def __init__(
        self,
        providers: List[str] = None,
        cache_dir: str = "/tmp/api_verifier_cache",
        max_workers: int = 8,
        provider_configs: Dict[str, dict] = None,
    ):
        """
        Args:
            providers: API 名称列表, e.g. ["qwen", "deepseek"]
            cache_dir: 缓存目录
            max_workers: 并行线程数
            provider_configs: 各 provider 的额外配置, e.g. {"qwen": {"model_name": "qwen-plus"}}
        """
        providers = providers or ["qwen"]
        provider_configs = provider_configs or {}

        self.providers: List[BaseAPIProvider] = []
        for name in providers:
            cls = self.PROVIDER_MAP.get(name)
            if cls is None:
                logger.warning(f"未知 provider: {name}, 跳过")
                continue
            kwargs = provider_configs.get(name, {})
            self.providers.append(cls(**kwargs))

        if not self.providers:
            raise ValueError("至少需要一个有效的 provider")

        self.cache = ResultCache(cache_dir)
        self.max_workers = max_workers
        self.use_ensemble = len(self.providers) > 1

        logger.info(f"APIVerifier 初始化: {len(self.providers)} 个 provider, "
                     f"ensemble={'ON' if self.use_ensemble else 'OFF'}")

    def verify_single(self, text: str) -> EnsembleResult:
        """验证单条文本 (含多源集成)"""
        individual_results = []

        for provider in self.providers:
            provider_name = provider.__class__.__name__

            # 检查缓存
            cached = self.cache.get(text, provider_name)
            if cached is not None:
                result = VerificationResult(**cached)
            else:
                result = provider.verify(text)
                # 写入缓存
                self.cache.put(text, provider_name, {
                    "text": result.text,
                    "is_harmful": result.is_harmful,
                    "category": result.category,
                    "toxic_type": result.toxic_type,
                    "expression": result.expression,
                    "confidence": result.confidence,
                    "provider": result.provider,
                    "raw_response": result.raw_response,
                })

            individual_results.append(result)

        # 集成投票
        if self.use_ensemble:
            return self._ensemble_vote(text, individual_results)
        else:
            r = individual_results[0]
            return EnsembleResult(
                text=text,
                final_is_harmful=r.is_harmful,
                final_category=r.category,
                confidence=r.confidence,
                individual_results=individual_results,
                difficulty_score=0.0,
            )

    def _ensemble_vote(self, text: str, results: List[VerificationResult]) -> EnsembleResult:
        """多源投票集成"""
        # 1. 二分类投票 (有害/无害)
        harmful_votes = sum(1 for r in results if r.is_harmful)
        total = len(results)
        is_harmful = harmful_votes > total / 2

        # 2. 类别投票 (仅在判为有害时)
        if is_harmful:
            cat_votes = [r.category for r in results if r.is_harmful and r.category != "无毒"]
            if cat_votes:
                cat_counter = Counter(cat_votes)
                final_category = cat_counter.most_common(1)[0][0]
            else:
                final_category = "其他仇恨"
        else:
            final_category = "无毒"

        # 3. 计算一致率作为置信度
        if is_harmful:
            agreement = harmful_votes / total
        else:
            agreement = (total - harmful_votes) / total

        # 4. 难度分数 = 1 - 置信度 (disagreement 越大越难)
        difficulty = 1.0 - agreement

        return EnsembleResult(
            text=text,
            final_is_harmful=is_harmful,
            final_category=final_category,
            confidence=agreement,
            individual_results=results,
            difficulty_score=difficulty,
        )

    def batch_verify(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[EnsembleResult]:
        """
        批量验证文本

        Args:
            texts: 待验证文本列表
            batch_size: 并行 batch 大小 (控制并发 API 调用数)
            show_progress: 是否显示进度条

        Returns:
            List[EnsembleResult]
        """
        results = []
        total = len(texts)

        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, total, batch_size), desc="API验证")
        else:
            iterator = range(0, total, batch_size)

        for start in iterator:
            batch = texts[start:start + batch_size]
            batch_results = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.verify_single, text): text for text in batch}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        text = futures[future]
                        logger.error(f"验证失败: {e}, text={text[:50]}...")
                        batch_results.append(EnsembleResult(
                            text=text, final_is_harmful=False, final_category="无毒",
                            confidence=0.0, difficulty_score=1.0,
                        ))

            results.extend(batch_results)

        # 按原始顺序排序 (ThreadPoolExecutor 可能乱序)
        text_to_result = {r.text: r for r in results}
        ordered_results = [text_to_result.get(t, EnsembleResult(
            text=t, final_is_harmful=False, final_category="无毒",
            confidence=0.0, difficulty_score=1.0,
        )) for t in texts]

        return ordered_results

    def compute_category_stats(
        self,
        results: List[EnsembleResult],
        true_categories: List[str] = None,
    ) -> Dict[str, Any]:
        """
        计算各类别统计指标

        Returns:
            {
                "cat_adversarial_success_rate": {cat: float},  # 各类别 ASR
                "cat_label_verified_rate": {cat: float},       # 各类别标签验证通过率
                "overall_accuracy": float,                      # 总体准确率 (若有 true_categories)
                "avg_confidence": float,                        # 平均置信度
                "avg_difficulty": float,                        # 平均难度分数
            }
        """
        stats = {
            "cat_adversarial_success_rate": {},
            "cat_label_verified_rate": {},
            "overall_accuracy": 0.0,
            "avg_confidence": 0.0,
            "avg_difficulty": 0.0,
        }

        confidences = [r.confidence for r in results]
        difficulties = [r.difficulty_score for r in results]
        stats["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
        stats["avg_difficulty"] = sum(difficulties) / len(difficulties) if difficulties else 0.0

        if true_categories:
            correct = 0
            cat_total = defaultdict(int)
            cat_correct = defaultdict(int)
            cat_verified = defaultdict(int)

            for result, true_cat in zip(results, true_categories):
                cat_total[true_cat] += 1

                if result.final_category == true_cat:
                    correct += 1
                    cat_correct[true_cat] += 1

                # 标签验证: API 判定与 true_category 一致
                if result.final_is_harmful == (true_cat in HARMFUL_CATEGORIES):
                    cat_verified[true_cat] += 1

            stats["overall_accuracy"] = correct / len(results) if results else 0.0

            for cat in ALL_CATEGORIES:
                total = cat_total.get(cat, 0)
                if total > 0:
                    stats["cat_label_verified_rate"][cat] = cat_verified.get(cat, 0) / total
                    # ASR: Challenger 生成的有害文本中，Reviewer 漏判 = 被 fool 的比例
                    # 这里简化为 1 - 准确率
                    stats["cat_adversarial_success_rate"][cat] = 1 - cat_correct.get(cat, 0) / total

        return stats


# ── 兼容旧版 Verifier 接口 ──────────────────────────────────────────────────────

class APIVerifierCompat:
    """
    兼容旧版 Verifier 的接口封装，使 generate_dynamic_data_v2.py 可以无缝切换

    用法 (替代 verifier.py 中的 Verifier 类):
        verifier = APIVerifierCompat(providers=["qwen"])
        classification = verifier.classify_text(text)
        batch_results  = verifier.batch_classify(texts)
    """

    def __init__(self, providers: List[str] = None, **kwargs):
        self.api_verifier = APIVerifier(providers=providers, **kwargs)

    def classify_text(self, text: str) -> Dict[str, str]:
        """分类单条文本，返回与旧版 Verifier 相同的格式"""
        result = self.api_verifier.verify_single(text)
        return {
            "is_harmful": result.final_is_harmful,
            "category": result.final_category,
            "confidence": result.confidence,
            "difficulty_score": result.difficulty_score,
        }

    def batch_classify(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """批量分类，返回列表"""
        results = self.api_verifier.batch_verify(texts, batch_size=batch_size)
        return [
            {
                "text": r.text,
                "is_harmful": r.final_is_harmful,
                "category": r.final_category,
                "confidence": r.confidence,
                "difficulty_score": r.difficulty_score,
            }
            for r in results
        ]

    def compute_rewards(
        self,
        generated_texts: List[str],
        reviewer_outputs: List[str],
        true_categories: List[str],
    ) -> Dict[str, Any]:
        """
        计算对抗奖励信号 (与旧版 Verifier.compute_rewards 兼容)

        Returns:
            {
                "verifier_results":   List[EnsembleResult],  # API 验证结果
                "reviewer_was_fooled": List[bool],             # Reviewer 是否被骗
                "reviewer_is_correct": List[bool],             # Reviewer 是否正确
                "cat_stats":          Dict,                    # 类别级统计
            }
        """
        api_results = self.api_verifier.batch_verify(generated_texts)

        reviewer_was_fooled = []
        reviewer_is_correct = []

        for api_result, reviewer_out in zip(api_results, reviewer_outputs):
            # 解析 Reviewer 输出
            reviewer_says_harmful = "有害" in reviewer_out and "无害" not in reviewer_out

            if api_result.final_is_harmful and not reviewer_says_harmful:
                reviewer_was_fooled.append(True)
                reviewer_is_correct.append(False)
            elif not api_result.final_is_harmful and reviewer_says_harmful:
                reviewer_was_fooled.append(False)
                reviewer_is_correct.append(False)
            else:
                reviewer_was_fooled.append(False)
                reviewer_is_correct.append(True)

        cat_stats = self.api_verifier.compute_category_stats(api_results, true_categories)

        return {
            "verifier_results": api_results,
            "reviewer_was_fooled": reviewer_was_fooled,
            "reviewer_is_correct": reviewer_is_correct,
            "cat_stats": cat_stats,
        }


# ── CLI 工具 (独立使用) ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="API Verifier — 外部大模型验证工具")
    parser.add_argument("--providers", nargs="+", default=["qwen"],
                        choices=["qwen", "deepseek", "openai"],
                        help="使用的 API providers")
    parser.add_argument("--input", type=str, required=True,
                        help="输入文件 (每行一条文本, 或 JSON/JSONL)")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 JSON 文件")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cache_dir", type=str, default="/tmp/api_verifier_cache")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # 加载输入
    texts = []
    input_path = Path(args.input)
    if input_path.suffix == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj.get("text", obj.get("content", "")))
    elif input_path.suffix == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                texts = [d.get("text", d.get("content", "")) if isinstance(d, dict) else str(d) for d in data]
    else:
        with open(input_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    logger.info(f"加载 {len(texts)} 条文本")

    # 验证
    verifier = APIVerifier(
        providers=args.providers,
        cache_dir=args.cache_dir,
    )
    results = verifier.batch_verify(texts, batch_size=args.batch_size)

    # 输出
    output_data = []
    for r in results:
        output_data.append({
            "text": r.text,
            "is_harmful": r.final_is_harmful,
            "category": r.final_category,
            "confidence": r.confidence,
            "difficulty_score": r.difficulty_score,
            "providers": [ir.provider for ir in r.individual_results],
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已保存到 {args.output}")

    # 打印统计
    harmful_count = sum(1 for r in results if r.final_is_harmful)
    logger.info(f"有害: {harmful_count}/{len(results)} ({harmful_count/len(results)*100:.1f}%)")
    logger.info(f"平均置信度: {sum(r.confidence for r in results)/len(results):.3f}")


if __name__ == "__main__":
    main()
