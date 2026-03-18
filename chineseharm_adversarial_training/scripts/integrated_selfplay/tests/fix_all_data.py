#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量数据修正 + 多标签支持
========================
1. 从 ToxiCN_1.0.csv 构建正确标签映射 (text -> label, text -> all_labels)
2. 修正 split_data/, prepared_data/, origin_data/ 中所有含标签的文件
3. 重建 SFT 数据 (精简版 prompt)
4. 输出多标签映射表供 reward 函数使用

用法:
  python tests/fix_all_data.py --toxicn /path/to/ToxiCN_1.0.csv
"""

import os, sys, csv, json, argparse, hashlib
from collections import Counter, defaultdict
from pathlib import Path

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_SELF_DIR)
sys.path.insert(0, _PARENT)

from constants import (
    REVIEWER_SYSTEM_PROMPT, REVIEWER_USER_TEMPLATE,
    HARMFUL_CATEGORIES, ALL_CATEGORIES,
)

TARGET_MAP = {0: 'LGBTQ歧视', 1: '地域偏见', 2: '性别歧视', 3: '种族歧视', 4: '其他仇恨', 5: '无毒'}


def build_label_maps(toxicn_path):
    """从 ToxiCN 构建 text->label 和 text->all_labels 映射."""
    with open(toxicn_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    content_key = [k for k in rows[0].keys() if 'content' in k.lower()][0]

    primary_map = {}   # text -> primary label
    multi_map = {}     # text -> list of all labels
    toxic_map = {}     # text -> toxic (0/1)

    for r in rows:
        text = r[content_key].strip()
        if not text:
            continue

        toxic = r['toxic']
        toxic_map[text] = int(toxic)

        if toxic == '0':
            primary_map[text] = '无毒'
            multi_map[text] = ['无毒']
            continue

        vec = eval(r['target'])
        labels = [TARGET_MAP[i] for i in range(5) if vec[i] == 1]

        if not labels:
            primary_map[text] = '其他仇恨'
            multi_map[text] = ['其他仇恨']
        else:
            primary_map[text] = labels[0]
            multi_map[text] = labels

    return primary_map, multi_map, toxic_map


def fix_json_file(path, primary_map, label_col='标签', text_col='文本'):
    """修正 JSON 文件中的标签."""
    if not os.path.exists(path):
        return None

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        return None

    # 检测列名
    sample = data[0]
    if text_col not in sample:
        for alt in ['文本', 'original_text', 'content', 'text']:
            if alt in sample:
                text_col = alt
                break
    if label_col not in sample:
        for alt in ['标签', 'category', 'label']:
            if alt in sample:
                label_col = alt
                break

    if text_col not in sample or label_col not in sample:
        return None

    fixed = 0
    for r in data:
        text = r.get(text_col, '').strip()
        correct = primary_map.get(text)
        if correct and r.get(label_col) != correct:
            r[label_col] = correct
            fixed += 1

    # 备份并写回
    bak = path + '.bak_wrong_labels'
    if not os.path.exists(bak):
        os.rename(path, bak)
    else:
        pass  # 已有备份

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {'path': path, 'total': len(data), 'fixed': fixed}


def fix_jsonl_file(path, primary_map):
    """修正 JSONL 文件 (SFT messages 格式) 中 assistant 回复的标签."""
    if not os.path.exists(path):
        return None

    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if not lines:
        return None

    fixed = 0
    new_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue

        # 提取文本
        text = ''
        if 'messages' in obj:
            for m in obj['messages']:
                if m.get('role') == 'user':
                    content = m['content']
                    # 从 prompt 中提取 "文本：xxx" 或 "文本: xxx"
                    for marker in ['文本：', '文本:', '文本: ']:
                        idx = content.find(marker)
                        if idx >= 0:
                            text = content[idx + len(marker):].strip()
                            break
                    break

        correct = primary_map.get(text)
        if correct and 'messages' in obj:
            # 修正 assistant 回复
            for m in obj['messages']:
                if m.get('role') == 'assistant':
                    old_content = m['content']
                    if correct == '无毒':
                        new_content = '判断：无害'
                    else:
                        new_content = f'判断：有害\n类别：{correct}'

                    if old_content != new_content:
                        m['content'] = new_content
                        fixed += 1
                    break

            # 修正 reward_model 等元数据
            if 'reward_model' in obj and isinstance(obj['reward_model'], dict):
                rm = obj['reward_model']
                for key in ['category', 'ground_truth']:
                    if key in rm:
                        rm[key] = correct

            if 'extra_info' in obj and isinstance(obj['extra_info'], dict):
                ei = obj['extra_info']
                if 'category' in ei:
                    ei['category'] = correct

        new_lines.append(json.dumps(obj, ensure_ascii=False))

    # 备份并写回
    bak = path + '.bak_wrong_labels'
    if not os.path.exists(bak):
        os.rename(path, bak)

    with open(path, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line + '\n')

    return {'path': path, 'total': len(new_lines), 'fixed': fixed}


def build_sft_data(primary_map, multi_map, toxic_map, train_texts, val_texts, output_dir):
    """用正确标签重建 SFT 数据."""
    os.makedirs(output_dir, exist_ok=True)

    for name, texts in [('train', train_texts), ('val', val_texts)]:
        samples = []
        for text in texts:
            label = primary_map.get(text)
            if not label or not text.strip():
                continue

            user_content = REVIEWER_USER_TEMPLATE.format(text=text.strip()[:500])
            if label == '无毒':
                assistant = '判断：无害'
            else:
                assistant = f'判断：有害\n类别：{label}'

            samples.append({
                'messages': [
                    {'role': 'system', 'content': REVIEWER_SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_content},
                    {'role': 'assistant', 'content': assistant},
                ]
            })

        path = os.path.join(output_dir, f'sft_{name}.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print(f'  SFT {name}: {path} ({len(samples)} 条)')


def build_multi_label_map(multi_map, output_path):
    """输出多标签映射表, 供 reward 函数运行时加载.

    key 策略: 同时存完整文本和截断到200字符的版本,
    因为 extra_info['original_text'] 在 build_parquet 中被截断到 200 chars.
    """
    multi_only = {}
    for text, labels in multi_map.items():
        if len(labels) <= 1:
            continue
        multi_only[text] = labels
        # 同时存截断版, 使 reward 函数通过 original_text[:200] 也能查到
        truncated = text[:200]
        if truncated != text and truncated not in multi_only:
            multi_only[truncated] = labels

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(multi_only, f, ensure_ascii=False, indent=2)

    real_count = sum(1 for t, l in multi_map.items() if len(l) > 1)
    print(f'  多标签映射: {output_path} ({real_count} 条多标签样本, {len(multi_only)} 个 key 含截断版)')
    return multi_only


def main():
    parser = argparse.ArgumentParser(description='全量数据修正')
    parser.add_argument('--toxicn', required=True, type=str)
    parser.add_argument('--project_dir', default='', type=str,
                        help='项目根目录 (默认自动检测)')
    args = parser.parse_args()

    # 自动检测项目目录
    project_dir = args.project_dir
    if not project_dir:
        project_dir = os.path.abspath(os.path.join(_PARENT, '..', '..'))
    print(f'项目目录: {project_dir}')

    # Step 1: 构建正确标签映射
    print('\n[Step 1] 从 ToxiCN 构建正确标签映射')
    primary_map, multi_map, toxic_map = build_label_maps(args.toxicn)
    print(f'  总计: {len(primary_map)} 条')
    print(f'  多标签: {sum(1 for v in multi_map.values() if len(v) > 1)} 条')
    dist = Counter(primary_map.values())
    for k, v in dist.most_common():
        print(f'    {k}: {v}')

    # Step 2: 修正 split_data JSON
    print('\n[Step 2] 修正 split_data/')
    for name in ['train.json', 'val.json', 'test.json']:
        path = os.path.join(project_dir, 'split_data', name)
        result = fix_json_file(path, primary_map)
        if result:
            print(f'  {name}: {result["fixed"]}/{result["total"]} 修正')

    # Step 3: 修正 origin_data/bench.json
    print('\n[Step 3] 修正 origin_data/bench.json')
    result = fix_json_file(os.path.join(project_dir, 'origin_data', 'bench.json'), primary_map)
    if result:
        print(f'  bench.json: {result["fixed"]}/{result["total"]} 修正')

    # Step 4: 修正 prepared_data/rl/
    print('\n[Step 4] 修正 prepared_data/rl/')
    for name in ['train_seed.json', 'test_eval.json', 'val_eval.json']:
        path = os.path.join(project_dir, 'prepared_data', 'rl', name)
        result = fix_json_file(path, primary_map)
        if result:
            print(f'  {name}: {result["fixed"]}/{result["total"]} 修正')

    # Step 5: 修正 prepared_data/reviewer_sft/ JSONL
    print('\n[Step 5] 修正 prepared_data/reviewer_sft/')
    for name in ['train.jsonl', 'val.jsonl']:
        path = os.path.join(project_dir, 'prepared_data', 'reviewer_sft', name)
        result = fix_jsonl_file(path, primary_map)
        if result:
            print(f'  {name}: {result["fixed"]}/{result["total"]} 修正')

    # Step 6: 修正 prepared_data_v2 (如果存在)
    print('\n[Step 6] 修正 prepared_data_v2/')
    for subdir in ['rl', 'reviewer_sft', 'challenger_sft']:
        for name in ['train_seed.json', 'test_eval.json', 'val_eval.json',
                      'train.jsonl', 'val.jsonl']:
            path = os.path.join(project_dir, 'prepared_data_v2', subdir, name)
            if not os.path.exists(path):
                continue
            if name.endswith('.jsonl'):
                result = fix_jsonl_file(path, primary_map)
            else:
                result = fix_json_file(path, primary_map)
            if result and result['fixed'] > 0:
                print(f'  {subdir}/{name}: {result["fixed"]}/{result["total"]} 修正')

    # Step 7: 修正根目录 train.jsonl
    print('\n[Step 7] 修正根目录 train.jsonl')
    path = os.path.join(project_dir, 'train.jsonl')
    result = fix_jsonl_file(path, primary_map)
    if result:
        print(f'  train.jsonl: {result["fixed"]}/{result["total"]} 修正')

    # Step 8: 重建 corrected_data (全量正确数据 + SFT)
    print('\n[Step 8] 重建 corrected_data/')
    corrected_dir = os.path.join(project_dir, 'corrected_data')
    os.makedirs(corrected_dir, exist_ok=True)

    # 从 ToxiCN 重新切分
    all_texts = list(primary_map.keys())
    import random
    random.seed(42)

    by_label = defaultdict(list)
    for text in all_texts:
        by_label[primary_map[text]].append(text)

    train_texts, val_texts, test_texts = [], [], []
    for label, texts in by_label.items():
        texts_sorted = sorted(texts, key=lambda x: hashlib.md5(x.encode()).hexdigest())
        n = len(texts_sorted)
        n_test = max(1, round(n * 0.20))
        n_val = max(1, round(n * 0.12))
        n_train = n - n_test - n_val
        train_texts.extend(texts_sorted[:n_train])
        val_texts.extend(texts_sorted[n_train:n_train + n_val])
        test_texts.extend(texts_sorted[n_train + n_val:])

    random.shuffle(train_texts)
    random.shuffle(val_texts)
    random.shuffle(test_texts)

    # 输出 train/val/test.json (包含多标签信息)
    for name, texts in [('train', train_texts), ('val', val_texts), ('test', test_texts)]:
        records = []
        for text in texts:
            records.append({
                '文本': text,
                '标签': primary_map[text],
                'all_labels': multi_map.get(text, [primary_map[text]]),
            })
        path = os.path.join(corrected_dir, f'{name}.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        dist = Counter(r['标签'] for r in records)
        print(f'  {name}.json: {len(records)} 条')
        for cat in ALL_CATEGORIES:
            print(f'    {cat}: {dist.get(cat, 0)}')

    # Reviewer SFT 数据
    build_sft_data(primary_map, multi_map, toxic_map, train_texts, val_texts, corrected_dir)

    # Challenger SFT 数据
    print('\n  生成 Challenger SFT 数据...')
    CHALLENGER_SYS = (
        "你是一个文本生成助手。根据用户给定的类别和表达方式，"
        "生成一条对应的真实互联网文本。只输出文本本身，不要任何解释。"
    )
    for sft_name, texts in [('train', train_texts), ('val', val_texts)]:
        samples = []
        for text in texts:
            label = primary_map.get(text)
            if not label or len(text.strip()) < 5:
                continue
            expr = '非仇恨' if label == '无毒' else '隐式仇恨'
            samples.append({
                'messages': [
                    {'role': 'system', 'content': CHALLENGER_SYS},
                    {'role': 'user', 'content': f"类别：{label}\n表达方式：{expr}\n只输出文本本身，不要解释："},
                    {'role': 'assistant', 'content': text.strip()},
                ]
            })
        path = os.path.join(corrected_dir, f'challenger_sft_{sft_name}.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print(f'  Challenger SFT {sft_name}: {path} ({len(samples)} 条)')

    # Step 9: 输出多标签映射表
    print('\n[Step 9] 输出多标签映射表')
    multi_path = os.path.join(corrected_dir, 'multi_label_map.json')
    build_multi_label_map(multi_map, multi_path)

    # 同时复制到 integrated_selfplay 目录供运行时使用
    runtime_path = os.path.join(_PARENT, 'multi_label_map.json')
    with open(multi_path, 'r') as f:
        content = f.read()
    with open(runtime_path, 'w') as f:
        f.write(content)
    print(f'  运行时副本: {runtime_path}')

    # 也替换 split_data 为最终版
    for name in ['train.json', 'val.json', 'test.json']:
        src = os.path.join(corrected_dir, name)
        dst = os.path.join(project_dir, 'split_data', name)
        with open(src, 'r') as f:
            content = f.read()
        with open(dst, 'w') as f:
            f.write(content)
    print(f'  split_data/ 已同步替换')

    # Step 10: 验证
    print('\n[Step 10] 最终验证')
    for name in ['train.json', 'val.json', 'test.json']:
        data = json.load(open(os.path.join(corrected_dir, name)))
        wrong = sum(1 for r in data if primary_map.get(r['文本']) != r['标签'])
        multi_cnt = sum(1 for r in data if len(r.get('all_labels', [])) > 1)
        print(f'  {name}: {len(data)} 条, 标签错误={wrong}, 多标签={multi_cnt}')

    print('\n全部修正完成!')


if __name__ == '__main__':
    main()
