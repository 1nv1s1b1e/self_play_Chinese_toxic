#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共常量与规则定义
==================
所有模块共享的常量、分类规则、Prompt 模板均在此处统一定义，
消除跨文件重复，确保 SFT 训练端和推理端的 prompt 完全一致。
"""

# ── 类别定义 ─────────────────────────────────────────────────────────────────────
HARMFUL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨"]
ALL_CATEGORIES = HARMFUL_CATEGORIES + ["无毒"]

# ── 知识规则 (与 SFT 训练时完全一致) ─────────────────────────────────────────────
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

# ── Reviewer / Verifier Prompt 模板 ──────────────────────────────────────────────
REVIEWER_SYSTEM_PROMPT = "You are a helpful assistant."

# ── Verifier 专用 System Prompt（比 Reviewer 更强，用于 API 外部验证）────────────
VERIFIER_SYSTEM_PROMPT = (
    "你是一位专业的中文网络仇恨言论与有害内容鉴定专家。"
    "你的任务是对用户提供的中文文本进行精确的有害性分类。\n\n"
    "【分类体系】\n"
    "有害类别共 5 种：性别歧视、种族歧视、地域偏见、LGBTQ歧视、其他仇恨。\n"
    "无害：文本不含上述任何有害内容。\n\n"
    "【关键判断要点】\n"
    "1. 有害内容包含「隐式」形式：反讽、阴阳怪气、谐音梗、刻板印象强化等，"
    "即使没有脏话也可能有害。\n"
    "2. 「无毒」要严格：只有文本确实不含任何歧视或攻击性内容才标为无害。\n"
    "3. 「转述」陷阱：报道或讨论仇恨事件时，若文本本身立场中立、不传播仇恨，"
    "则无害；若借报道之名传播偏见，则有害。\n"
    "4. 类别互斥：只选择最符合的一个有害类别。\n\n"
    "【输出格式】严格按以下格式逐行输出，不得添加任何额外说明：\n"
    "判断：有害  （或：判断：无害）\n"
    "类别：<类别名>  （仅有害时输出，从上述5种选一）"
)

# ── Verifier few-shot 示例（覆盖所有类别及隐式仇恨边界情况）─────────────────────
VERIFIER_FEW_SHOTS = [
    {"text": "女人开车就是容易分心，所以事故多也正常，不是歧视，是事实。",
     "label": "判断：有害\n类别：性别歧视"},
    {"text": "非洲人的智商就是低，这是科学研究证明的，没啥好说的。",
     "label": "判断：有害\n类别：种族歧视"},
    {"text": "据说去河南旅游要看好钱包，不是地域黑，是过来人的经验。",
     "label": "判断：有害\n类别：地域偏见"},
    {"text": "同性恋就是一种心理疾病，需要接受治疗，让他们正常化。",
     "label": "判断：有害\n类别：LGBTQ歧视"},
    {"text": "你这个废物，滚出去，别浪费空气了！",
     "label": "判断：有害\n类别：其他仇恨"},
    {"text": "最近在研究各地方言的差异，发现粤语和普通话语序很不同，挺有意思的。",
     "label": "判断：无害"},
    {"text": "今年女性在科技行业的占比有所提升，这是一个积极的趋势。",
     "label": "判断：无害"},
]


def build_verifier_messages(text: str, use_few_shots: bool = True) -> list:
    """
    构建 Verifier API 调用的 messages 列表。
    与 Reviewer SFT prompt 完全分离，专为高精度 API 验证设计。

    Args:
        text: 待分类文本
        use_few_shots: 是否在 prompt 中加入 few-shot 示例

    Returns:
        messages: OpenAI 格式的消息列表
    """
    messages = [{"role": "system", "content": VERIFIER_SYSTEM_PROMPT}]

    if use_few_shots:
        for shot in VERIFIER_FEW_SHOTS:
            messages.append({
                "role": "user",
                "content": f"文本：{shot['text']}",
            })
            messages.append({
                "role": "assistant",
                "content": shot["label"],
            })

    messages.append({
        "role": "user",
        "content": f"文本：{text.strip()[:500]}",
    })
    return messages


# ── 精简版 Reviewer Prompt (无内联 RULES，规则已在 SFT 阶段内化) ──
# 注意：完整 RULES (~500 tokens) 对 3B 模型造成严重注意力稀释，
# SFT 已训练模型内化规则知识，推理时只需格式指引即可。
REVIEWER_USER_TEMPLATE = (
    "作为中文有毒语言检测专家，请对以下文本进行分析。\n"
    "有毒类别：性别歧视、种族歧视、地域偏见、LGBTQ歧视、其他仇恨。\n"
    "注意：隐式仇恨（反讽、暗语、谐音、阴阳怪气）也属有害；"
    "转述仇恨事件但立场中立的属无害。\n\n"
    "请严格按以下格式输出：\n"
    "判断：有害（或：判断：无害）\n"
    "类别：<类别名>（仅有害时输出）\n\n"
    "文本：{text}"
)

# 完整版模板（含 RULES，仅供参考，实际训练和推理不使用）
REVIEWER_USER_TEMPLATE_FULL = (
    "作为中文有毒语言检测专家，请对以下文本进行分析。\n"
    "有毒类别：性别歧视、种族歧视、地域偏见、LGBTQ歧视、其他仇恨。\n\n"
    "请严格按以下格式输出：\n"
    "判断：有害（或：判断：无害）\n"
    "类别：<类别名>（仅有害时输出）\n\n"
    f"规则: {RULES}\n"
    "文本: {text}"
)

# ── Challenger Prompt 模板 ───────────────────────────────────────────────────────
CHALLENGER_SYSTEM_PROMPT_BASE = (
    "你是一个文本生成助手。根据用户给定的类别和表达方式，"
    "生成一条对应的真实互联网文本。只输出文本本身，不要任何解释。"
)

# 每个类别对应的默认 (toxic_type, expression)
CAT_DEFAULTS = {
    "性别歧视": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "种族歧视": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "地域偏见": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "LGBTQ歧视": {"toxic_type": "仇恨言论", "expression": "隐式仇恨"},
    "其他仇恨":  {"toxic_type": "一般攻击性", "expression": "非仇恨"},
    "无毒":      {"toxic_type": "无毒", "expression": "非仇恨"},
}

# ── 工具函数 ─────────────────────────────────────────────────────────────────────

def get_category_rules(category: str) -> str:
    """根据类别名从 RULES 中提取对应规则段落。"""
    if category not in HARMFUL_CATEGORIES:
        return ""
    lines = RULES.strip().split('\n')
    result, capturing = [], False
    for line in lines:
        if line.startswith(category + '：'):
            capturing = True
            result.append(line)
        elif capturing and line and not any(
            line.startswith(c + '：') for c in HARMFUL_CATEGORIES if c != category
        ):
            result.append(line)
        elif capturing and any(
            line.startswith(c + '：') for c in HARMFUL_CATEGORIES if c != category
        ):
            break
    return '\n'.join(result)


def build_challenger_system_prompt(category: str) -> str:
    """构建 Challenger system prompt，与 SFT 训练端完全一致（不含 rules）。

    注意：corrected_data 的 SFT 数据不含 rules，仅使用简洁指令。
    Self-play 推理必须与 SFT 训练保持同分布，否则会产生分布偏移。
    规则知识已通过 SFT 内化到模型参数中，无需在 prompt 中重复注入。
    """
    return CHALLENGER_SYSTEM_PROMPT_BASE


def format_reviewer_user_content(text: str) -> str:
    """格式化 Reviewer 用户输入 (截断到 500 字符)。"""
    return REVIEWER_USER_TEMPLATE.format(text=text.strip()[:500])


def parse_classification_output(output: str) -> dict:
    """
    解析 Reviewer/Verifier 模型的输出文本，提取结构化分类结果。

    Returns:
        dict: binary, category, toxic_type, expression
    """
    result = {"binary": None, "category": None, "toxic_type": None, "expression": None}
    if not output:
        return result

    text = output.strip()

    has_harmful = any(k in text for k in ["判断：有害", "判断:有害", "判断: 有害", "判断 ：有害"])
    has_safe = any(k in text for k in ["判断：无害", "判断:无害", "判断: 无害", "判断 ：无害"])

    if has_harmful:
        result["binary"] = "有害"
    elif has_safe:
        result["binary"] = "无害"
    else:
        for cat in ALL_CATEGORIES:
            if cat in text[:30] or cat in text:
                result["binary"] = "有害" if cat in HARMFUL_CATEGORIES else "无害"
                result["category"] = cat
                return result
        return result

    if result["binary"] == "无害":
        result["category"] = "无毒"
        result["toxic_type"] = "无毒"
        result["expression"] = "非仇恨"
        return result

    for cat in HARMFUL_CATEGORIES:
        if cat in text:
            result["category"] = cat
            break
    if result["category"] is None:
        result["category"] = "其他仇恨"

    for tt in ["一般攻击性", "仇恨言论"]:
        if tt in text:
            result["toxic_type"] = tt
            break

    for expr in ["隐式仇恨", "显式仇恨", "转述"]:
        if expr in text:
            result["expression"] = expr
            break

    return result
