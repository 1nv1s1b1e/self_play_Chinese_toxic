#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迭代答案收敛机制 (Iterative Gold Answer Generation)
====================================================
对应 PDF 图3「自动化QA对生成机制」中的右侧流程：
  Reviewer 对同一文本反复生成答案，直到连续 K 次输出一致 (a_n = a_{n-1} = a_{n-2})
  最终稳定答案作为 Gold Answer，供后续 RL 训练使用。

核心思想:
  - 单次推理可能不稳定（尤其边界样本），多次推理取收敛结果更可靠
  - 收敛速度本身是难度信号：快收敛 = 简单，慢收敛 = 困难
  - 不收敛的样本单独标记，可用于课程学习的"困难池"

用法:
  reviewer = IterativeReviewer(model_path="/path/to/reviewer")
  gold_answers = reviewer.generate_gold_answers(texts, max_iterations=5, k=3)
"""

import re
import gc
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter

import torch

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

REVIEWER_SYSTEM_PROMPT = "You are a helpful assistant."

REVIEWER_USER_TEMPLATE = (
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
class ParsedAnswer:
    """解析后的单次推理答案"""
    is_harmful: bool = False
    category: str = "无毒"
    toxic_type: str = ""
    expression: str = ""
    raw_text: str = ""

    def signature(self) -> str:
        """答案签名，用于判断收敛"""
        return f"{self.is_harmful}|{self.category}|{self.toxic_type}|{self.expression}"


@dataclass
class GoldAnswer:
    """Gold Answer（经迭代收敛后的稳定答案）"""
    text: str                       # 输入文本
    answer: ParsedAnswer            # 最终收敛答案
    converged: bool                 # 是否收敛
    iterations_needed: int          # 收敛所需迭代次数
    total_iterations: int           # 总迭代次数
    convergence_rate: float         # 最终答案在所有迭代中的出现率
    difficulty_score: float         # 难度分数 (基于收敛速度)
    all_answers: List[ParsedAnswer] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "is_harmful": self.answer.is_harmful,
            "category": self.answer.category,
            "toxic_type": self.answer.toxic_type,
            "expression": self.answer.expression,
            "converged": self.converged,
            "iterations_needed": self.iterations_needed,
            "total_iterations": self.total_iterations,
            "convergence_rate": self.convergence_rate,
            "difficulty_score": self.difficulty_score,
        }


# ── 解析工具 ─────────────────────────────────────────────────────────────────────

def parse_reviewer_output(raw_text: str) -> ParsedAnswer:
    """解析 Reviewer 输出为结构化 ParsedAnswer"""
    answer = ParsedAnswer(raw_text=raw_text)

    if not raw_text or not raw_text.strip():
        return answer

    lines = raw_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if "有害" in line and "判断" in line and "无害" not in line:
            answer.is_harmful = True
        elif "无害" in line and "判断" in line:
            answer.is_harmful = False
        elif line.startswith("类别：") or line.startswith("类别:"):
            cat = line.split("：")[-1].split(":")[-1].strip()
            if cat in HARMFUL_CATEGORIES:
                answer.category = cat
        elif line.startswith("毒性类型：") or line.startswith("毒性类型:"):
            answer.toxic_type = line.split("：")[-1].split(":")[-1].strip()
        elif line.startswith("表达方式：") or line.startswith("表达方式:"):
            answer.expression = line.split("：")[-1].split(":")[-1].strip()

    if answer.is_harmful and answer.category == "无毒":
        answer.category = "其他仇恨"
    if not answer.is_harmful:
        answer.category = "无毒"
        answer.toxic_type = ""
        answer.expression = ""

    return answer


def check_convergence(answers: List[ParsedAnswer], k: int = 3) -> Tuple[bool, Optional[ParsedAnswer]]:
    """
    检查最近 K 次答案是否收敛

    收敛条件: 最近连续 K 次答案的 signature 相同
    """
    if len(answers) < k:
        return False, None

    recent = answers[-k:]
    signatures = [a.signature() for a in recent]

    if len(set(signatures)) == 1:
        return True, recent[-1]
    return False, None


# ── 本地模型推理器 ───────────────────────────────────────────────────────────────

class LocalReviewerInference:
    """本地 Reviewer 模型推理 (支持 NPU/GPU)"""

    def __init__(self, model_path: str, device: str = "npu:0", max_new_tokens: int = 150):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = device
        self.max_new_tokens = max_new_tokens

        logger.info(f"加载 Reviewer 模型: {model_path} → {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if 'npu' in device else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True,
            device_map=device,
        )
        self.model.eval()
        logger.info("Reviewer 模型加载完成")

    def generate_single(self, text: str, temperature: float = 0.3) -> str:
        """生成单条文本的审核结果"""
        messages = [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": REVIEWER_USER_TEMPLATE.format(text=text)},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                top_p=0.9 if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_batch(self, texts: List[str], temperature: float = 0.3, batch_size: int = 8) -> List[str]:
        """批量生成"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_messages = []
            for text in batch:
                messages = [
                    {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                    {"role": "user", "content": REVIEWER_USER_TEMPLATE.format(text=text)},
                ]
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                batch_messages.append(input_text)

            inputs = self.tokenizer(
                batch_messages, return_tensors="pt", padding=True, truncation=True,
                max_length=2048,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    do_sample=(temperature > 0),
                    top_p=0.9 if temperature > 0 else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            for j, output in enumerate(outputs):
                generated = output[inputs["input_ids"].shape[1]:]
                text_out = self.tokenizer.decode(generated, skip_special_tokens=True)
                results.append(text_out)

        return results

    def cleanup(self):
        """释放模型显存"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── 迭代 Reviewer 主类 ──────────────────────────────────────────────────────────

class IterativeReviewer:
    """
    迭代答案收敛 Reviewer

    对每条文本反复生成答案，直到连续 K 次输出一致或达到最大迭代次数。

    用法:
        reviewer = IterativeReviewer(model_path="/path/to/reviewer")
        gold_answers = reviewer.generate_gold_answers(
            texts=["文本1", "文本2"],
            max_iterations=5,
            k=3,
        )
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = "npu:0",
        api_verifier=None,
        use_api: bool = False,
    ):
        """
        Args:
            model_path: 本地模型路径 (use_api=False 时使用)
            device: 设备
            api_verifier: APIVerifier 实例 (use_api=True 时使用)
            use_api: 是否使用 API 进行迭代推理
        """
        self.use_api = use_api

        if use_api:
            if api_verifier is None:
                raise ValueError("use_api=True 时必须提供 api_verifier")
            self.api_verifier = api_verifier
            self.local_model = None
        else:
            if model_path is None:
                raise ValueError("use_api=False 时必须提供 model_path")
            self.local_model = LocalReviewerInference(model_path, device)
            self.api_verifier = None

    def _generate_once(self, text: str, temperature: float = 0.3) -> ParsedAnswer:
        """单次推理"""
        if self.use_api:
            result = self.api_verifier.verify_single(text)
            return ParsedAnswer(
                is_harmful=result.final_is_harmful,
                category=result.final_category,
                raw_text=str(result),
            )
        else:
            raw = self.local_model.generate_single(text, temperature=temperature)
            return parse_reviewer_output(raw)

    def _generate_batch_once(self, texts: List[str], temperature: float = 0.3) -> List[ParsedAnswer]:
        """单轮批量推理"""
        if self.use_api:
            results = self.api_verifier.batch_verify(texts, show_progress=False)
            return [
                ParsedAnswer(
                    is_harmful=r.final_is_harmful,
                    category=r.final_category,
                    raw_text=str(r),
                )
                for r in results
            ]
        else:
            raw_outputs = self.local_model.generate_batch(texts, temperature=temperature)
            return [parse_reviewer_output(raw) for raw in raw_outputs]

    def generate_gold_answer(
        self,
        text: str,
        max_iterations: int = 5,
        k: int = 3,
        temperature: float = 0.3,
    ) -> GoldAnswer:
        """
        对单条文本进行迭代推理直到收敛

        Args:
            text: 待分析文本
            max_iterations: 最大迭代次数
            k: 连续 K 次一致视为收敛
            temperature: 推理温度 (>0 引入随机性以检测稳定性)

        Returns:
            GoldAnswer
        """
        all_answers = []

        for iteration in range(1, max_iterations + 1):
            answer = self._generate_once(text, temperature=temperature)
            all_answers.append(answer)

            converged, stable_answer = check_convergence(all_answers, k=k)
            if converged:
                # 计算难度分数: 收敛越快越简单
                difficulty = (iteration - k) / (max_iterations - k) if max_iterations > k else 0.0

                # 计算该答案在所有迭代中的出现率
                final_sig = stable_answer.signature()
                convergence_rate = sum(1 for a in all_answers if a.signature() == final_sig) / len(all_answers)

                return GoldAnswer(
                    text=text,
                    answer=stable_answer,
                    converged=True,
                    iterations_needed=iteration,
                    total_iterations=iteration,
                    convergence_rate=convergence_rate,
                    difficulty_score=difficulty,
                    all_answers=all_answers,
                )

        # 未收敛: 取众数作为最终答案
        signatures = [a.signature() for a in all_answers]
        sig_counter = Counter(signatures)
        most_common_sig = sig_counter.most_common(1)[0][0]
        majority_answer = next(a for a in all_answers if a.signature() == most_common_sig)

        convergence_rate = sig_counter[most_common_sig] / len(all_answers)
        difficulty = 1.0  # 未收敛 = 最难

        return GoldAnswer(
            text=text,
            answer=majority_answer,
            converged=False,
            iterations_needed=max_iterations,
            total_iterations=max_iterations,
            convergence_rate=convergence_rate,
            difficulty_score=difficulty,
            all_answers=all_answers,
        )

    def generate_gold_answers(
        self,
        texts: List[str],
        max_iterations: int = 5,
        k: int = 3,
        temperature: float = 0.3,
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[GoldAnswer]:
        """
        批量迭代推理

        优化策略：
          - 每轮对所有未收敛文本进行批量推理
          - 收敛的文本立即退出，减少计算量
        """
        n = len(texts)
        all_answers: List[List[ParsedAnswer]] = [[] for _ in range(n)]
        gold_answers: List[Optional[GoldAnswer]] = [None] * n
        active_indices = list(range(n))  # 未收敛的样本索引

        if show_progress:
            from tqdm import tqdm

        for iteration in range(1, max_iterations + 1):
            if not active_indices:
                break

            if show_progress:
                logger.info(f"迭代 {iteration}/{max_iterations}: {len(active_indices)} 条待收敛")

            # 批量推理当前活跃样本
            active_texts = [texts[i] for i in active_indices]
            batch_answers = self._generate_batch_once(active_texts, temperature=temperature)

            # 更新并检查收敛
            new_active = []
            for idx, answer in zip(active_indices, batch_answers):
                all_answers[idx].append(answer)

                converged, stable_answer = check_convergence(all_answers[idx], k=k)
                if converged:
                    difficulty = (iteration - k) / (max_iterations - k) if max_iterations > k else 0.0
                    final_sig = stable_answer.signature()
                    convergence_rate = sum(
                        1 for a in all_answers[idx] if a.signature() == final_sig
                    ) / len(all_answers[idx])

                    gold_answers[idx] = GoldAnswer(
                        text=texts[idx],
                        answer=stable_answer,
                        converged=True,
                        iterations_needed=iteration,
                        total_iterations=iteration,
                        convergence_rate=convergence_rate,
                        difficulty_score=difficulty,
                        all_answers=all_answers[idx],
                    )
                else:
                    new_active.append(idx)

            active_indices = new_active

        # 处理未收敛的样本
        for idx in active_indices:
            answers = all_answers[idx]
            signatures = [a.signature() for a in answers]
            sig_counter = Counter(signatures)
            most_common_sig = sig_counter.most_common(1)[0][0]
            majority_answer = next(a for a in answers if a.signature() == most_common_sig)

            convergence_rate = sig_counter[most_common_sig] / len(answers)

            gold_answers[idx] = GoldAnswer(
                text=texts[idx],
                answer=majority_answer,
                converged=False,
                iterations_needed=max_iterations,
                total_iterations=max_iterations,
                convergence_rate=convergence_rate,
                difficulty_score=1.0,
                all_answers=answers,
            )

        # 统计
        converged_count = sum(1 for g in gold_answers if g.converged)
        avg_iterations = sum(g.iterations_needed for g in gold_answers) / len(gold_answers)
        avg_difficulty = sum(g.difficulty_score for g in gold_answers) / len(gold_answers)

        logger.info(f"迭代收敛完成: 收敛率={converged_count}/{n} ({converged_count/n*100:.1f}%), "
                     f"平均迭代={avg_iterations:.1f}, 平均难度={avg_difficulty:.2f}")

        return gold_answers

    def cleanup(self):
        """释放资源"""
        if self.local_model:
            self.local_model.cleanup()


# ── CLI ──────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="迭代答案收敛 Gold Answer 生成")
    parser.add_argument("--model_path", type=str, required=True, help="Reviewer 模型路径")
    parser.add_argument("--input", type=str, required=True, help="输入文件 (JSONL，含 text 字段)")
    parser.add_argument("--output", type=str, required=True, help="输出 JSONL 文件")
    parser.add_argument("--max_iterations", type=int, default=5, help="最大迭代次数")
    parser.add_argument("--k", type=int, default=3, help="连续 K 次一致视为收敛")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="npu:0")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 加载输入
    texts = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])

    logger.info(f"加载 {len(texts)} 条文本")

    # 迭代收敛
    reviewer = IterativeReviewer(model_path=args.model_path, device=args.device)
    gold_answers = reviewer.generate_gold_answers(
        texts,
        max_iterations=args.max_iterations,
        k=args.k,
        temperature=args.temperature,
    )

    # 输出
    with open(args.output, "w", encoding="utf-8") as f:
        for ga in gold_answers:
            f.write(json.dumps(ga.to_dict(), ensure_ascii=False) + "\n")

    logger.info(f"Gold Answers 已保存到 {args.output}")

    reviewer.cleanup()


if __name__ == "__main__":
    main()
