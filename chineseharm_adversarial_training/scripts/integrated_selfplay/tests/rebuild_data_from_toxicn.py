#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ToxiCN_1.0.csv 原始数据集重建正确的 train/val/test 数据
============================================================

背景:
  bench.json 的类别标签存在系统性错位 (4类两两对调, 44.8% 错误)
  split_data 用 topic 字段机械覆盖, 同样不准确

修正方案:
  直接从 ToxiCN_1.0.csv 的 toxic + target 字段生成正确标签.

ToxiCN 官方 target 字段定义 (GitHub README):
  target[0] = LGBTQ
  target[1] = Region  (地域偏见)
  target[2] = Sexism  (性别歧视)
  target[3] = Racism  (种族歧视)
  target[4] = others  (其他仇恨)
  target[5] = non-hate (非仇恨)

标签生成规则:
  toxic=0                              → 无毒
  toxic=1, target 仅 non-hate 或全0    → 其他仇恨
  toxic=1, target 有 idx 0-4 的激活位  → 取第一个激活类别

输出:
  train.json, val.json, test.json — 只含 文本 + 标签
  同时输出 SFT 格式的 train.jsonl (精简版 prompt)

用法:
  python tests/rebuild_data_from_toxicn.py \
      --toxicn /path/to/ToxiCN_1.0.csv \
      --output_dir /path/to/corrected_data

  # 快速验证 (不输出 SFT):
  python tests/rebuild_data_from_toxicn.py \
      --toxicn /path/to/ToxiCN_1.0.csv \
      --output_dir /tmp/check --dry_run
"""

import os, sys, csv, json, argparse, random, hashlib
from collections import Counter, defaultdict

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_SELF_DIR)
sys.path.insert(0, _PARENT)

from constants import (
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_USER_TEMPLATE,
    HARMFUL_CATEGORIES,
)

# ── ToxiCN 官方 target 索引定义 ──
TARGET_INDEX = {
    0: 'LGBTQ歧视',
    1: '地域偏见',
    2: '性别歧视',
    3: '种族歧视',
    4: '其他仇恨',
    5: '无毒',       # non-hate 标志位
}

ALL_CATEGORIES = ['性别歧视', '种族歧视', '地域偏见', 'LGBTQ歧视', '其他仇恨', '无毒']


def decode_label(toxic_str, target_str):
    """
    从 ToxiCN 的 toxic + target 字段解码正确类别标签.

    Returns:
        str: 类别标签 (性别歧视/种族歧视/地域偏见/LGBTQ歧视/其他仇恨/无毒)
    """
    if toxic_str == '0':
        return '无毒'

    vec = eval(target_str)

    # toxic=1 但 target 只有 non-hate(idx5) 或全0 → 其他仇恨
    harmful_bits = vec[:5]
    if sum(harmful_bits) == 0:
        return '其他仇恨'

    # 取第一个激活的有害目标 (index 0-4)
    for i in range(5):
        if vec[i] == 1:
            return TARGET_INDEX[i]

    return '其他仇恨'


def load_toxicn(path):
    """加载 ToxiCN_1.0.csv 并生成正确标签."""
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 处理 BOM
    content_key = None
    for key in rows[0].keys():
        if 'content' in key.lower():
            content_key = key
            break
    if content_key is None:
        raise ValueError(f"找不到 content 列, 列名: {list(rows[0].keys())}")

    records = []
    for r in rows:
        text = r[content_key].strip()
        if not text:
            continue

        label = decode_label(r['toxic'], r['target'])
        is_harmful = label in HARMFUL_CATEGORIES
        target_vec = eval(r['target'])

        # 多标签信息 (用于分析, 不影响主标签)
        all_targets = [TARGET_INDEX[i] for i in range(5) if target_vec[i] == 1]

        records.append({
            '文本': text,
            '标签': label,
            'toxic': int(r['toxic']),
            'is_harmful': is_harmful,
            'target_vec': target_vec,
            'all_targets': all_targets,
            'multi_label': len(all_targets) > 1,
        })

    return records


def deterministic_split(records, train_ratio=0.68, val_ratio=0.12, test_ratio=0.20, seed=42):
    """
    确定性分层切分: 按类别比例分配到 train/val/test.
    使用文本 hash 保证每次切分结果一致.
    """
    random.seed(seed)

    by_label = defaultdict(list)
    for r in records:
        by_label[r['标签']].append(r)

    train, val, test = [], [], []

    for label, items in by_label.items():
        # 用文本 hash 排序保证确定性
        items_sorted = sorted(items, key=lambda x: hashlib.md5(x['文本'].encode()).hexdigest())

        n = len(items_sorted)
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        n_train = n - n_test - n_val

        train.extend(items_sorted[:n_train])
        val.extend(items_sorted[n_train:n_train + n_val])
        test.extend(items_sorted[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def build_sft_sample(record):
    """构建精简版 prompt 的 Reviewer SFT 样本."""
    text = record['文本'][:500]
    label = record['标签']
    user_content = REVIEWER_USER_TEMPLATE.format(text=text)

    if label == '无毒':
        assistant = '判断：无害'
    else:
        assistant = f'判断：有害\n类别：{label}'

    return {
        'messages': [
            {'role': 'system', 'content': REVIEWER_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': assistant},
        ]
    }


CHALLENGER_SYSTEM = (
    "你是一个文本生成助手。根据用户给定的类别和表达方式，"
    "生成一条对应的真实互联网文本。只输出文本本身，不要任何解释。"
)

def build_challenger_sft_sample(record):
    """构建 Challenger SFT 样本: 给定类别 → 生成文本."""
    text = record['文本'].strip()
    label = record['标签']
    if not text or len(text) < 5:
        return None

    if label == '无毒':
        expression = '非仇恨'
    else:
        expression = '隐式仇恨'

    user_content = f"类别：{label}\n表达方式：{expression}\n只输出文本本身，不要解释："

    return {
        'messages': [
            {'role': 'system', 'content': CHALLENGER_SYSTEM},
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': text},
        ]
    }


def print_distribution(records, name):
    """打印分布统计."""
    dist = Counter(r['标签'] for r in records)
    total = len(records)
    toxic = sum(v for k, v in dist.items() if k != '无毒')
    nontoxic = dist.get('无毒', 0)

    print(f'  {name}: {total} 条 (有毒 {toxic}/{100*toxic/total:.1f}%, 无毒 {nontoxic}/{100*nontoxic/total:.1f}%)')
    for cat in ALL_CATEGORIES:
        cnt = dist.get(cat, 0)
        print(f'    {cat:10s}: {cnt:5d} ({100*cnt/total:5.1f}%)')


def compare_with_old(records, old_path, name):
    """和旧数据对比标签差异."""
    if not os.path.exists(old_path):
        return

    with open(old_path, 'r', encoding='utf-8') as f:
        old_data = json.load(f)

    old_map = {r['文本']: r['标签'] for r in old_data}
    new_map = {r['文本']: r['标签'] for r in records}

    matched = 0
    diff = 0
    diff_types = Counter()

    for text, new_label in new_map.items():
        old_label = old_map.get(text)
        if old_label is not None:
            matched += 1
            if old_label != new_label:
                diff += 1
                diff_types[f'{old_label} → {new_label}'] += 1

    print(f'  与旧 {name} 对比: {matched} 条匹配, {diff} 条标签变更 ({100*diff/max(matched,1):.1f}%)')
    if diff > 0:
        for change, cnt in diff_types.most_common(5):
            print(f'    {change}: {cnt}')


def main():
    parser = argparse.ArgumentParser(description='从 ToxiCN 重建正确数据')
    parser.add_argument('--toxicn', required=True, type=str,
                        help='ToxiCN_1.0.csv 路径')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='输出目录')
    parser.add_argument('--old_split_dir', default='', type=str,
                        help='旧 split_data 目录 (用于对比)')
    parser.add_argument('--old_bench', default='', type=str,
                        help='旧 bench.json 路径 (用于对比)')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印统计, 不输出文件')
    parser.add_argument('--build_sft', action='store_true', default=True,
                        help='同时输出精简版 SFT 数据 (默认开启)')
    args = parser.parse_args()

    print('=' * 60)
    print('  从 ToxiCN_1.0.csv 重建正确数据集')
    print('=' * 60)

    # Step 1: 加载并解码
    print(f'\n[Step 1] 加载 ToxiCN: {args.toxicn}')
    records = load_toxicn(args.toxicn)
    print(f'  总计: {len(records)} 条')

    # 多标签统计
    multi = sum(1 for r in records if r['multi_label'])
    print(f'  多标签样本: {multi} ({100*multi/len(records):.1f}%)')

    print(f'\n[Step 2] 全量分布:')
    print_distribution(records, '全量')

    # Step 3: 切分
    print(f'\n[Step 3] 分层切分 (68/12/20)')
    train, val, test = deterministic_split(records, seed=args.seed)
    print_distribution(train, 'train')
    print_distribution(val, 'val')
    print_distribution(test, 'test')

    # Step 4: 与旧数据对比
    print(f'\n[Step 4] 与旧数据对比')
    old_split = args.old_split_dir
    if not old_split:
        # 自动查找
        for candidate in [
            os.path.join(_PARENT, '..', '..', 'split_data'),
        ]:
            if os.path.exists(os.path.join(candidate, 'train.json')):
                old_split = candidate
                break

    if old_split:
        compare_with_old(train, os.path.join(old_split, 'train.json'), 'train')
        compare_with_old(test, os.path.join(old_split, 'test.json'), 'test')

    old_bench = args.old_bench
    if not old_bench:
        for candidate in [
            os.path.join(_PARENT, '..', '..', 'origin_data', 'bench.json'),
        ]:
            if os.path.exists(candidate):
                old_bench = candidate
                break

    if old_bench:
        compare_with_old(records, old_bench, 'bench.json')

    if args.dry_run:
        print('\n[dry_run] 不输出文件')
        return

    # Step 5: 输出
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'\n[Step 5] 输出到: {args.output_dir}')

    # 5a: JSON (文本 + 标签)
    for name, data in [('train', train), ('val', val), ('test', test)]:
        # 只输出 文本 + 标签
        clean = [{'文本': r['文本'], '标签': r['标签']} for r in data]
        path = os.path.join(args.output_dir, f'{name}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        print(f'  {path}: {len(clean)} 条')

    # 5b: SFT JSONL (精简版 prompt)
    if args.build_sft:
        for name, data in [('train', train), ('val', val)]:
            sft_samples = [build_sft_sample(r) for r in data]
            path = os.path.join(args.output_dir, f'sft_{name}.jsonl')
            with open(path, 'w', encoding='utf-8') as f:
                for s in sft_samples:
                    f.write(json.dumps(s, ensure_ascii=False) + '\n')
            print(f'  {path}: {len(sft_samples)} 条 (SFT 精简版)')

    # 5c: 数据报告
    report = {
        'source': args.toxicn,
        'total': len(records),
        'train': len(train),
        'val': len(val),
        'test': len(test),
        'multi_label_count': multi,
        'distribution': {
            'train': dict(Counter(r['标签'] for r in train)),
            'val': dict(Counter(r['标签'] for r in val)),
            'test': dict(Counter(r['标签'] for r in test)),
        },
        'target_index_definition': {
            '0': 'LGBTQ歧视',
            '1': '地域偏见 (Region)',
            '2': '性别歧视 (Sexism)',
            '3': '种族歧视 (Racism)',
            '4': '其他仇恨 (others)',
            '5': '无毒 (non-hate)',
        },
    }
    report_path = os.path.join(args.output_dir, 'data_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'  {report_path}')

    print(f'\n完成!')


if __name__ == '__main__':
    main()
