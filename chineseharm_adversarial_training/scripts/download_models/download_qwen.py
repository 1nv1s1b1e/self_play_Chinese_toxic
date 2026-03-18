#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载Qwen2.5 Instruct系列模型
支持: 0.5B, 1.5B, 3B (7B可选)
使用modelscope或huggingface下载
"""

import os
import argparse
from pathlib import Path


# 模型ID映射
MODELS = {
    "0.5B": {
        "hf": "Qwen/Qwen2.5-0.5B-Instruct",
        "ms": "Qwen/Qwen2.5-0.5B-Instruct",
    },
    "1.5B": {
        "hf": "Qwen/Qwen2.5-1.5B-Instruct",
        "ms": "Qwen/Qwen2.5-1.5B-Instruct",
    },
    "3B": {
        "hf": "Qwen/Qwen2.5-3B-Instruct",
        "ms": "Qwen/Qwen2.5-3B-Instruct",
    },
    "7B": {
        "hf": "Qwen/Qwen2.5-7B-Instruct",
        "ms": "Qwen/Qwen2.5-7B-Instruct",
    },
}


def download_from_modelscope(model_id: str, output_dir: str):
    """使用modelscope下载(国内推荐)"""
    from modelscope import snapshot_download
    print(f"  [modelscope] 下载 {model_id} → {output_dir}")
    snapshot_download(model_id, cache_dir=output_dir)
    print(f"  ✓ 完成")


def download_from_huggingface(model_id: str, output_dir: str):
    """使用huggingface_hub下载"""
    from huggingface_hub import snapshot_download
    print(f"  [huggingface] 下载 {model_id} → {output_dir}")
    snapshot_download(repo_id=model_id, local_dir=os.path.join(output_dir, model_id))
    print(f"  ✓ 完成")


def main():
    parser = argparse.ArgumentParser(description="下载Qwen2.5模型")
    parser.add_argument(
        "--models", type=str, nargs="+",
        default=["0.5B", "1.5B", "3B"],
        choices=list(MODELS.keys()),
        help="要下载的模型尺寸列表"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/home/ma-user/work/test/models_base",
        help="模型保存根目录"
    )
    parser.add_argument(
        "--source", type=str, default="modelscope",
        choices=["modelscope", "huggingface"],
        help="下载源"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("下载Qwen2.5 Instruct模型")
    print("=" * 60)
    print(f"模型列表: {args.models}")
    print(f"输出目录: {args.output_dir}")
    print(f"下载源:   {args.source}")
    print()

    for size in args.models:
        model_info = MODELS[size]
        model_id = model_info["ms"] if args.source == "modelscope" else model_info["hf"]
        print(f"\n[{size}] 下载 {model_id}...")

        try:
            if args.source == "modelscope":
                download_from_modelscope(model_id, str(output_dir))
            else:
                download_from_huggingface(model_id, str(output_dir))
        except Exception as e:
            print(f"  ✗ 下载失败: {e}")
            continue

    # 列出已下载的模型
    print("\n" + "=" * 60)
    print("已下载模型:")
    print("=" * 60)
    for item in sorted(output_dir.rglob("config.json")):
        model_dir = item.parent
        print(f"  {model_dir}")

    print("\n✓ 下载完成!")


if __name__ == "__main__":
    main()
