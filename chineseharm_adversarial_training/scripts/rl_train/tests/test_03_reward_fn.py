#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 03: 奖励函数单元测试
验证 challenger_reward_v7.py 和 reviewer_reward.py 的 compute_score 接口
与 verl 的调用约定完全兼容

运行: python3 test_03_reward_fn.py
(无需 NPU，在 CPU 上运行)
"""
import sys
import os

# 把 reward_functions 加入 import 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "reward_functions"))

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
print("  测试 03: 奖励函数单元测试")
print("=" * 60)

# ════════════════════════════════════════════════════════════════
# Part A: Challenger 奖励函数 (challenger_reward_v7.py)
# ════════════════════════════════════════════════════════════════
print("\n【A】Challenger 奖励函数 (Gate × Signal 架构)")
try:
    from challenger_reward_v7 import compute_score as ch_score
    print("  ✅ challenger_reward_v7 导入成功")
    PASS += 1
except Exception as e:
    print(f"  ❌ 导入失败: {e}")
    FAIL += 1
    ch_score = None

if ch_score:
    # 测试 A1: data_source 过滤 (非 toxicn 应返回 0)
    r = ch_score("other_source", "hello", "ref")
    check("A1 非 toxicn_challenger 返回 0.0", r == 0.0, f"got={r}")

    # 测试 A2: 空生成文本 → 低分 (gate=0)
    r = ch_score("toxicn_challenger", "", "reference text", extra_info={"original_text": "性别歧视相关参考文本"})
    check("A2 空文本 → 低分", r <= -0.5, f"got={r:.3f}")

    # 测试 A3: 正常生成的中文文本 → 分数在合法范围内
    generated = "这个社会对女性的偏见仍然普遍存在，尤其在职场上受到歧视。"
    reference  = "社会存在性别歧视，女性在职场遭受不平等对待。"
    r = ch_score("toxicn_challenger", generated, reference,
                 extra_info={"original_text": reference, "category": "性别歧视"})
    check("A3 正常文本分数在 [-1, 1]", -1.0 <= r <= 1.0, f"got={r:.3f}")
    check("A3 正常文本分数 > 空文本", r > -0.5, f"got={r:.3f}")

    # 测试 A4: 重复退化文本 → gate 压低分数
    repetitive = "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈"
    r_rep = ch_score("toxicn_challenger", repetitive, reference,
                     extra_info={"original_text": reference})
    check("A4 重复退化文本比正常文本分低", r_rep < r, f"rep={r_rep:.3f} normal={r:.3f}")

    # 测试 A5: 包含拒绝格式的文本 → format_gate=0
    refusal = "对不起，我无法回答这个问题，这违反了相关法律法规。"
    r_ref = ch_score("toxicn_challenger", refusal, reference,
                     extra_info={"original_text": reference})
    check("A5 拒绝格式文本分数极低", r_ref < -0.5, f"got={r_ref:.3f}")

    # 测试 A6: verl 调用约定兼容性 (用 **kwargs 传入额外参数)
    try:
        r = ch_score(
            data_source="toxicn_challenger",
            solution_str=generated,
            ground_truth=reference,
            extra_info={"original_text": reference},
            # verl 可能传入额外字段
            prompt="写一个关于...",
            data_item_idx=0,
        )
        check("A6 verl 调用约定 (**kwargs) 兼容", True, f"score={r:.3f}")
    except Exception as e:
        check("A6 verl 调用约定 (**kwargs) 兼容", False, str(e))

# ════════════════════════════════════════════════════════════════
# Part B: Reviewer 奖励函数 (reviewer_reward.py)
# ════════════════════════════════════════════════════════════════
print("\n【B】Reviewer 奖励函数 (binary + category)")
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

    # 测试 B2: 正确检测有害内容 (有害 + 类别正确)
    output_harmful = "判断：有害\n类别：性别歧视\n毒性类型：仇恨言论\n表达方式：显式仇恨"
    r = rv_score("toxicn_reviewer", output_harmful, "性别歧视")
    check("B2 正确检测有害+分类 → 高分", r > 0.5, f"got={r:.3f}")

    # 测试 B3: 漏检惩罚 (有害文本判为无害 → -1.0)
    output_miss = "判断：无害"
    r = rv_score("toxicn_reviewer", output_miss, "性别歧视")
    check("B3 漏检 → -1.0", r == -1.0, f"got={r}")

    # 测试 B4: 误检惩罚 (无害文本判为有害 → -0.5)
    output_fp = "判断：有害\n类别：性别歧视"
    r = rv_score("toxicn_reviewer", output_fp, "无毒")
    check("B4 误检 → -0.5", r == -0.5, f"got={r}")

    # 测试 B5: 无法解析的输出 → -0.8
    r = rv_score("toxicn_reviewer", "这是一段无法解析的随机文字", "性别歧视")
    check("B5 无法解析 → -0.8", r == -0.8, f"got={r}")

    # 测试 B6: ground_truth 为 dict 格式 (verl extra_info 场景)
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
# Part C: 奖励值范围完整性检验
# ════════════════════════════════════════════════════════════════
print("\n【C】奖励值范围完整性 (verl 要求 reward ∈ [-1, 1])")
if ch_score:
    test_cases_ch = [
        ("空字符串",        "",               "ref"),
        ("正常文本",         "正常的中文文本测试用例",  "参考文本"),
        ("超长文本",         "测试" * 200,     "参考"),
        ("纯数字",          "12345678901234", "参考"),
        ("单个字符",         "A",              "参考"),
    ]
    all_ch_ok = True
    for name, gen, ref in test_cases_ch:
        r = ch_score("toxicn_challenger", gen, ref, extra_info={"original_text": ref})
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

print("\n" + "=" * 60)
print(f"  结果: ✅ {PASS} 通过 / ❌ {FAIL} 失败")
print("=" * 60)

if FAIL > 0:
    print("\n  ❌ 奖励函数存在问题，请修复后再启动 RL 训练")
    sys.exit(1)
else:
    print("\n  🎉 奖励函数接口测试通过，与 verl 调用约定兼容")
    print("  → 可以继续运行 test_04_verl_grpo_smoke.sh")
