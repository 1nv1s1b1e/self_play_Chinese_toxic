#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 03: 奖励函数单元测试 (集成版)
验证 challenger_reward.py 和 reviewer_reward.py 的 compute_score 接口
与 TRL / verl 的调用约定完全兼容

改进 (集成版):
  - 测试集成版的统一奖励函数 (逐样本真对抗信号)
  - 使用 constants.py 的统一分类常量

运行: python3 test_03_reward_fn.py
(无需 NPU，在 CPU 上运行)
"""
import sys
import os

# 把集成版根目录加入 import 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, os.path.join(PARENT_DIR, "reward_functions"))

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✅ {name}{': ' + detail if detail else ''}")
        PASS += 1
    else:
        print(f"  ❌ {name}{': ' + detail if detail else ''}")
        FAIL += 1


print("=" * 60)
print("  测试 03: 奖励函数单元测试 (集成版)")
print("=" * 60)

# ════════════════════════════════════════════════════════════════
# Part 0: constants.py 导入检查
# ════════════════════════════════════════════════════════════════
print("\n【0】constants.py 公共模块")
try:
    from constants import HARMFUL_CATEGORIES, ALL_CATEGORIES, RULES, parse_classification_output
    check("constants 导入成功", True)
    check("HARMFUL_CATEGORIES 包含 5 类", len(HARMFUL_CATEGORIES) == 5)
    check("ALL_CATEGORIES 包含 6 类", len(ALL_CATEGORIES) == 6)
    check("RULES 非空", len(RULES) > 100)
except Exception as e:
    check("constants 导入", False, str(e))

# ════════════════════════════════════════════════════════════════
# Part A: Challenger 奖励函数 (集成版 — 逐样本真对抗信号)
# ════════════════════════════════════════════════════════════════
print("\n【A】Challenger 奖励函数 (Gate × Adversarial 架构)")
try:
    from challenger_reward import compute_score as ch_score
    print("  ✅ challenger_reward 导入成功")
    PASS += 1
except Exception as e:
    print(f"  ❌ 导入失败: {e}")
    FAIL += 1
    ch_score = None

if ch_score:
    # 测试 A1: data_source 过滤 (非 toxicn_challenger 应返回 0)
    r = ch_score("other_source", "hello", "ref")
    check("A1 非 toxicn_challenger 返回 0.0", r == 0.0, f"got={r}")

    # 测试 A2: 空生成文本 → -1.0 (gate=0)
    r = ch_score("toxicn_challenger", "", "reference text",
                 extra_info={"original_text": "性别歧视相关参考文本"})
    check("A2 空文本 → -1.0", r == -1.0, f"got={r:.3f}")

    # 测试 A3: reviewer_fooled=True → 高分
    generated = "这个社会对女性的偏见仍然普遍存在，尤其在职场上受到歧视。"
    r = ch_score("toxicn_challenger", generated, "性别歧视",
                 extra_info={"reviewer_fooled": True, "reviewer_binary_ok": False})
    check("A3 reviewer_fooled=True → 高分", r > 0.0, f"got={r:.3f}")

    # 测试 A4: reviewer_fooled=False → 低分
    r_nofool = ch_score("toxicn_challenger", generated, "性别歧视",
                        extra_info={"reviewer_fooled": False, "reviewer_binary_ok": True})
    check("A4 reviewer_fooled=False → 低分", r_nofool < r, f"fooled={r:.3f} notfooled={r_nofool:.3f}")

    # 测试 A5: 重复退化文本 → gate 压低
    repetitive = "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈"
    r_rep = ch_score("toxicn_challenger", repetitive, "性别歧视",
                     extra_info={"reviewer_fooled": True})
    check("A5 重复文本比正常文本分低", r_rep < r, f"rep={r_rep:.3f} normal={r:.3f}")

    # 测试 A6: 拒绝格式 → -1.0
    refusal = "对不起，我无法回答这个问题，这违反了相关法律法规。"
    r_ref = ch_score("toxicn_challenger", refusal, "性别歧视",
                     extra_info={"reviewer_fooled": True})
    check("A6 拒绝格式文本 → -1.0", r_ref == -1.0, f"got={r_ref:.3f}")

    # 测试 A7: ground_truth 为 dict 格式
    r = ch_score("toxicn_challenger", generated,
                 {"target_category": "性别歧视", "ground_truth": "性别歧视"},
                 extra_info={"reviewer_fooled": True})
    check("A7 ground_truth=dict 格式兼容", r > 0.0, f"score={r:.3f}")

    # 测试 A8: 无毒类别 + reviewer_binary_ok=False → 高分 (false alarm)
    safe_text = "今天天气真好，适合散步，男女老少都可以参与。"
    r_safe = ch_score("toxicn_challenger", safe_text, "无毒",
                      extra_info={"reviewer_binary_ok": False})
    check("A8 无毒+false_alarm → 高分", r_safe > 0.0, f"got={r_safe:.3f}")

# ════════════════════════════════════════════════════════════════
# Part B: Reviewer 奖励函数 (binary + category)
# ════════════════════════════════════════════════════════════════
print("\n【B】Reviewer 奖励函数 (binary + category + toxic_type + expression)")
try:
    from reviewer_reward import compute_score as rv_score
    print("  ✅ reviewer_reward 导入成功")
    PASS += 1
except Exception as e:
    print(f"  ❌ 导入失败: {e}")
    FAIL += 1
    rv_score = None

if rv_score:
    # 测试 B1: data_source 过滤
    r = rv_score("other_source", "判断：有害\n类别：性别歧视", "性别歧视")
    check("B1 非 toxicn_reviewer 返回 0.0", r == 0.0, f"got={r}")

    # 测试 B2: 正确检测有害内容
    output_harmful = "判断：有害\n类别：性别歧视\n毒性类型：仇恨言论\n表达方式：显式仇恨"
    r = rv_score("toxicn_reviewer", output_harmful, "性别歧视")
    check("B2 正确检测有害+分类 → 高分", r > 0.5, f"got={r:.3f}")

    # 测试 B3: 漏检惩罚
    output_miss = "判断：无害"
    r = rv_score("toxicn_reviewer", output_miss, "性别歧视")
    check("B3 漏检 → -1.0", r == -1.0, f"got={r}")

    # 测试 B4: 误检惩罚
    output_fp = "判断：有害\n类别：性别歧视"
    r = rv_score("toxicn_reviewer", output_fp, "无毒")
    check("B4 误检 → -0.5", r == -0.5, f"got={r}")

    # 测试 B5: 无法解析 → -0.8
    r = rv_score("toxicn_reviewer", "这是一段随机文字", "性别歧视")
    check("B5 无法解析 → -0.8", r == -0.8, f"got={r}")

    # 测试 B6: ground_truth 为 dict 格式
    r = rv_score(
        "toxicn_reviewer",
        output_harmful,
        {"category": "性别歧视", "toxic_type": "仇恨言论", "expression": "显式仇恨"},
    )
    check("B6 ground_truth=dict 格式兼容", r > 0.5, f"got={r:.3f}")

    # 测试 B7: verl **kwargs 兼容性
    try:
        r = rv_score(
            data_source="toxicn_reviewer",
            solution_str=output_harmful,
            ground_truth="性别歧视",
            extra_info={"category": "性别歧视"},
            prompt="判断以下内容是否有害...",
            data_item_idx=1,
        )
        check("B7 verl 调用约定 (**kwargs) 兼容", True, f"score={r:.3f}")
    except Exception as e:
        check("B7 verl 调用约定 (**kwargs) 兼容", False, str(e))

# ════════════════════════════════════════════════════════════════
# Part C: quality_gate 单元测试
# ════════════════════════════════════════════════════════════════
print("\n【C】质量门控 (quality_gate)")
try:
    from quality_gate import quality_gate, length_gate, repetition_gate, format_gate, diversity_gate
    check("quality_gate 导入成功", True)

    check("空字符串 → 0.0", quality_gate("") == 0.0)
    check("正常文本 → >0.5", quality_gate("社会应该尊重每个人的基本权利和尊严") > 0.5)
    check("重复文本 → 低分", quality_gate("哈" * 50) < 0.3)
    check("拒绝文本 → 0.0", quality_gate("对不起，我无法帮你生成这样的内容") == 0.0)
    check("超短文本 → 0.0", quality_gate("ABC") == 0.0)
except Exception as e:
    check("quality_gate 导入", False, str(e))

# ════════════════════════════════════════════════════════════════
# Part D: 奖励值范围完整性
# ════════════════════════════════════════════════════════════════
print("\n【D】奖励值范围完整性 (要求 reward ∈ [-1, 1])")
if ch_score:
    test_cases_ch = [
        ("空字符串",        "",               "ref"),
        ("正常文本",         "正常的中文文本测试用例。",  "参考文本"),
        ("超长文本",         "测试" * 200,     "参考"),
        ("纯数字",          "12345678901234", "参考"),
        ("单个字符",         "A",              "参考"),
    ]
    all_ch_ok = True
    for name, gen, ref in test_cases_ch:
        r = ch_score("toxicn_challenger", gen, ref,
                     extra_info={"original_text": ref, "reviewer_fooled": False})
        if not (-1.0 <= r <= 1.0):
            print(f"  ❌ Challenger '{name}': score={r} 超出 [-1,1]")
            all_ch_ok = False
    check("Challenger 所有边界情况 ∈ [-1, 1]", all_ch_ok)

if rv_score:
    test_cases_rv = [
        ("空输出",           "",                  "性别歧视"),
        ("无法解析",          "随机文字",            "无毒"),
        ("正确有害",          "判断：有害\n类别：种族歧视", "种族歧视"),
        ("正确无害",          "判断：无害",          "无毒"),
    ]
    all_rv_ok = True
    for name, out, gt in test_cases_rv:
        r = rv_score("toxicn_reviewer", out, gt)
        if not (-1.0 <= r <= 1.0):
            print(f"  ❌ Reviewer '{name}': score={r} 超出 [-1,1]")
            all_rv_ok = False
    check("Reviewer 所有边界情况 ∈ [-1, 1]", all_rv_ok)

# ════════════════════════════════════════════════════════════════
# Part E: 解析函数一致性
# ════════════════════════════════════════════════════════════════
print("\n【E】解析函数一致性 (constants vs reviewer_reward)")
try:
    from constants import parse_classification_output
    from reviewer_reward import extract_prediction_full

    test_outputs = [
        "判断：有害\n类别：性别歧视\n毒性类型：仇恨言论\n表达方式：显式仇恨",
        "判断：无害",
        "判断：有害\n类别：LGBTQ歧视",
        "",
    ]
    all_consistent = True
    for out in test_outputs:
        r1 = parse_classification_output(out)
        r2 = extract_prediction_full(out)
        if r1["binary"] != r2["binary"] or r1["category"] != r2["category"]:
            all_consistent = False
            print(f"  ❌ 不一致: constants={r1} vs reviewer={r2}")
    check("解析函数一致性 (constants ≡ reviewer_reward)", all_consistent)
except Exception as e:
    check("解析函数一致性", False, str(e))

# ── 汇总 ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  结果: ✅ {PASS} 通过 / ❌ {FAIL} 失败")
print("=" * 60)

if FAIL > 0:
    print("\n  ❌ 奖励函数存在问题，请修复后再启动 RL 训练")
    sys.exit(1)
else:
    print("\n  🎉 奖励函数接口测试通过 (集成版)")
    print("  → 可以继续运行: bash run_selfplay.sh")
