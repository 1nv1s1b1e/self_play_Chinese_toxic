#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复JSONL文件中的JSON解析错误
问题：pyarrow JSON parser 对非法转义字符 (如 \x, \a 等) 会直接报错
方案：逐行读取并用 json.loads 重新序列化，修复转义问题

用法:
  python fix_jsonl.py                             # 修复所有默认目录下的 .jsonl
  python fix_jsonl.py --data_dir /path/to/data    # 指定目录
  python fix_jsonl.py --file /path/to/file.jsonl  # 指定单个文件
"""

import os
import re
import json
import argparse
import shutil
from pathlib import Path


def fix_invalid_escapes(text: str) -> str:
    """
    修复字符串中的非法JSON转义字符
    JSON只允许: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
    其他如 \x, \a, \p 等都是非法的
    """
    # 匹配反斜杠后面跟着非法转义字符的模式
    # 合法的: " \ / b f n r t u
    def replace_invalid(match):
        char = match.group(1)
        if char in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't'):
            return match.group(0)  # 合法转义，保留
        if char == 'u':
            return match.group(0)  # \uXXXX，保留
        # 非法转义 → 移除反斜杠，只保留字符
        return char
    
    return re.sub(r'\\(.)', replace_invalid, text)


def fix_jsonl_file(file_path: str, backup: bool = True) -> dict:
    """
    修复单个JSONL文件
    
    Returns:
        统计信息 dict
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"  文件不存在: {file_path}")
        return {"status": "not_found"}
    
    print(f"\n处理: {file_path}")
    
    # 先尝试用二进制模式读取，检测编码问题
    with open(file_path, 'rb') as f:
        raw_bytes = f.read()
    
    # 尝试UTF-8解码，有错就替换
    try:
        content = raw_bytes.decode('utf-8')
        had_encoding_error = False
    except UnicodeDecodeError:
        content = raw_bytes.decode('utf-8', errors='replace')
        had_encoding_error = True
        print(f"  ⚠ 发现编码错误，已用替换字符修复")
    
    lines = content.splitlines()
    
    fixed_records = []
    stats = {
        "total_lines": len(lines),
        "valid": 0,
        "fixed": 0,
        "skipped": 0,
        "encoding_error": had_encoding_error,
        "errors": []
    }
    
    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        # 第一次尝试：直接解析
        try:
            obj = json.loads(line)
            fixed_records.append(obj)
            stats["valid"] += 1
            continue
        except json.JSONDecodeError:
            pass
        
        # 第二次尝试：修复非法转义后解析
        try:
            fixed_line = fix_invalid_escapes(line)
            obj = json.loads(fixed_line)
            fixed_records.append(obj)
            stats["fixed"] += 1
            if stats["fixed"] <= 10:
                print(f"  🔧 第{line_no}行: 修复了非法转义字符")
            continue
        except json.JSONDecodeError as e:
            stats["skipped"] += 1
            stats["errors"].append({"line": line_no, "error": str(e)})
            if stats["skipped"] <= 5:
                print(f"  ✗ 第{line_no}行: 无法修复，跳过 ({e})")
    
    # 备份原文件
    if backup:
        backup_path = str(file_path) + '.bak'
        shutil.copy2(file_path, backup_path)
        print(f"  备份: {backup_path}")
    
    # 写入修复后的文件
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in fixed_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"  结果: {stats['valid']}条正常 + {stats['fixed']}条已修复 + {stats['skipped']}条丢弃")
    print(f"  输出: {len(fixed_records)}条 → {file_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="修复JSONL文件中的JSON解析错误")
    parser.add_argument("--data_dir", type=str, 
                       default="/home/ma-user/work/test/prepared_data",
                       help="数据目录（递归搜索所有.jsonl文件）")
    parser.add_argument("--file", type=str, default=None,
                       help="指定单个JSONL文件")
    parser.add_argument("--no-backup", action="store_true",
                       help="不创建备份文件")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("JSONL 数据修复工具")
    print("=" * 60)
    
    jsonl_files = []
    
    if args.file:
        jsonl_files = [args.file]
    else:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"目录不存在: {data_dir}")
            return
        jsonl_files = sorted(data_dir.rglob("*.jsonl"))
        print(f"在 {data_dir} 找到 {len(jsonl_files)} 个 .jsonl 文件")
    
    all_stats = {}
    for f in jsonl_files:
        stats = fix_jsonl_file(str(f), backup=not args.no_backup)
        all_stats[str(f)] = stats
    
    # 汇总
    print("\n" + "=" * 60)
    print("修复汇总")
    print("=" * 60)
    total_fixed = sum(s.get("fixed", 0) for s in all_stats.values())
    total_skipped = sum(s.get("skipped", 0) for s in all_stats.values())
    
    for path, stats in all_stats.items():
        status = "✓" if stats.get("skipped", 0) == 0 else "⚠"
        name = Path(path).name
        print(f"  {status} {name}: {stats.get('valid',0)}正常 + {stats.get('fixed',0)}修复 + {stats.get('skipped',0)}丢弃")
    
    if total_fixed > 0 or total_skipped > 0:
        print(f"\n共修复 {total_fixed} 条，丢弃 {total_skipped} 条")
    else:
        print("\n所有文件均无需修复")


if __name__ == "__main__":
    main()
