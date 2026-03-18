#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验 2: Epoch 消融
========================

在不同训练epoch (1,2,3) 下保存Reviewer LoRA checkpoint，然后分别评测。
验证 SFT 是否在 1 epoch 就已饱和 / 过拟合。

功能:
  1. 训练模式: 训练 Reviewer LoRA，每个 epoch 保存 checkpoint
  2. 评测模式: 对各 epoch checkpoint 分别评测，输出对比结果
"""

import os, sys, types
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

# ------ 防御性 NPU 初始化: 必须在其他任何库之前完成 ------
import torch
if 'torch_npu' not in sys.modules:
    try:
        import torch_npu  # noqa: F401
    except (ImportError, RuntimeError):
        sys.modules['torch_npu'] = types.ModuleType('torch_npu')
# ------------------------------------------------------------------

import json
import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
import pandas as pd


# ============================================================
# 训练部分
# ============================================================

class SaveEachEpochCallback(TrainerCallback):
    """每个epoch结束时保存完整的LoRA checkpoint"""

    def __init__(self, output_base_dir: str, tokenizer):
        self.output_base_dir = Path(output_base_dir)
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch)
        save_dir = self.output_base_dir / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  >> 保存 epoch {epoch} checkpoint → {save_dir}")
        model.save_pretrained(str(save_dir))
        self.tokenizer.save_pretrained(str(save_dir))


def load_reviewer_sft_dataset(data_path: str, tokenizer, max_length: int = 2048):
    """加载并处理Reviewer SFT数据集（与 train_reviewer_lora.py 一致）"""
    if data_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=data_path, split='train')
    elif data_path.endswith('.parquet'):
        dataset = load_dataset('parquet', data_files=data_path, split='train')
    else:
        raise ValueError("不支持的数据格式")

    print(f"  数据集大小: {len(dataset)}")

    def process_func(examples):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for messages in examples['messages']:
            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_messages = messages[:-1]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )

            full_tok = tokenizer(
                full_text, truncation=True, max_length=max_length,
                padding=False, return_tensors=None
            )
            prompt_tok = tokenizer(
                prompt_text, truncation=True, max_length=max_length,
                padding=False, return_tensors=None
            )

            prompt_length = len(prompt_tok['input_ids'])
            labels = [-100] * prompt_length + full_tok['input_ids'][prompt_length:]
            if len(labels) != len(full_tok['input_ids']):
                labels = labels[:len(full_tok['input_ids'])]

            model_inputs['input_ids'].append(full_tok['input_ids'])
            model_inputs['attention_mask'].append(full_tok['attention_mask'])
            model_inputs['labels'].append(labels)

        return model_inputs

    tokenized = dataset.map(
        process_func, batched=True,
        remove_columns=dataset.column_names, desc="Tokenizing"
    )
    return tokenized


def train_with_epoch_checkpoints(args):
    """训练并每epoch保存"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Epoch消融: 训练Reviewer LoRA (每epoch保存)")
    print("=" * 60)
    print(f"基座模型:   {args.model_path}")
    print(f"数据集:     {args.data_path}")
    print(f"最大epoch:  {args.max_epochs}")
    print(f"输出目录:   {output_dir}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, padding_side='right'
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if 'npu' in args.device or 'cuda' in args.device else torch.float32,
        device_map=args.device, trust_remote_code=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        inference_mode=False
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_reviewer_sft_dataset(args.data_path, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_output"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",  # 不使用默认保存，由callback保存
        fp16=True if 'cuda' in args.device or 'npu' in args.device else False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    epoch_callback = SaveEachEpochCallback(output_dir, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt"),
        callbacks=[epoch_callback],
    )

    print("\n开始训练...")
    trainer.train()
    print(f"\n训练完成! 各epoch checkpoint保存在: {output_dir}/epoch_*/")


# ============================================================
# 评测部分
# ============================================================

ALL_CATEGORIES = ["性别歧视", "种族歧视", "地域偏见", "LGBTQ歧视", "其他仇恨", "无毒"]
SYSTEM_PROMPT = "You are a helpful assistant."

FULL_RULES = """
性别歧视：
1. **对特定性别的贬低与攻击**：使用贬低性词汇攻击特定性别群体。
2. **性别刻板印象与偏见**：强化对特定性别的刻板认知。
3. **隐含性别歧视**：通过调侃、反讽等方式暗示贬低特定性别群体。

种族歧视：
1. **种族性侮辱与攻击**：使用歧视性词汇指代特定种族群体。
2. **种族偏见与刻板印象**：基于种族进行偏颇的负面概括。
3. **隐含种族歧视**：以看似客观或"开玩笑"的方式传递种族偏见。

地域偏见：
1. **地域性侮辱与攻击**：针对特定地区人群使用贬义性称呼或进行侮辱。
2. **地域刻板印象**：将个别负面行为上升为对地区全体居民的评价。
3. **隐含地域歧视**：以段子、调侃等形式传播地域偏见。

LGBTQ歧视：
1. **对LGBTQ群体的恶意攻击**：使用"变态"、"不正常"等词汇攻击LGBTQ群体。
2. **恐同/恐跨偏见**：将同性恋、跨性别视为"疾病"或"道德败坏"。
3. **隐含LGBTQ歧视**：以二元框架暗示否定特定性取向或性别认同。

其他仇恨：
1. **一般性人身攻击**：具有显著恶意的侮辱性语言。
2. **群体性恶意言论**：针对特定职业、年龄等群体的恶意攻击与污名化。
3. **泛化仇恨表达**：对社会群体泛化恶意评价，煽动对立情绪。
"""


def create_prompt(text: str) -> str:
    """创建评测prompt（与训练一致）"""
    return (
        '作为有毒语言检测专家，请依据有毒语言检测标准，判断以下文本是否包含有毒内容。\n'
        '如果文本有毒，请指出有毒类别，类别范围限定为："性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"。如果文本无毒，请回答"无毒"。\n'
        '请直接输出一个类别：包括"性别歧视"、"种族歧视"、"地域偏见"、"LGBTQ歧视"、"其他仇恨"或"无毒"。\n\n'
        f'规则: {FULL_RULES}\n'
        f'文本: {text}'
    )


def extract_prediction(output_text: str) -> Optional[str]:
    output_clean = output_text.strip()
    for cat in ALL_CATEGORIES:
        if cat in output_clean[:30]:
            return cat
    for cat in ALL_CATEGORIES:
        if cat in output_clean:
            return cat
    return None


def calculate_metrics(results: List[Dict]) -> Dict:
    label_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    correct = total = failed = 0
    for item in results:
        true_label, pred_label = item["true_label"], item["pred_label"]
        total += 1
        if pred_label is None:
            failed += 1; label_stats[true_label]['FN'] += 1
        elif pred_label == true_label:
            correct += 1; label_stats[true_label]['TP'] += 1
        else:
            label_stats[pred_label]['FP'] += 1; label_stats[true_label]['FN'] += 1
    accuracy = correct / total if total > 0 else 0
    f1_scores = []
    cat_metrics = {}
    for cat in ALL_CATEGORIES:
        s = label_stats[cat]
        p = s['TP'] / (s['TP'] + s['FP']) if (s['TP'] + s['FP']) > 0 else 0
        r = s['TP'] / (s['TP'] + s['FN']) if (s['TP'] + s['FN']) > 0 else 0
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        f1_scores.append(f1)
        cat_metrics[cat] = {"precision": round(p,4), "recall": round(r,4), "f1": round(f1,4), "support": s['TP']+s['FN']}
    macro_f1 = sum(f1_scores) / len(f1_scores)
    return {"accuracy": round(accuracy,4), "macro_f1": round(macro_f1,4), "correct": correct,
            "total": total, "failed_extractions": failed, "category_metrics": cat_metrics}


def load_eval_data(path: str) -> List[Dict]:
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
        return [{"文本": row.get("original_text", row.get("文本","")),
                 "标签": row.get("category", row.get("标签",""))} for _, row in df.iterrows()]
    elif path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(l.strip()) for l in f]
    else:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)


def eval_merged_model(model_path: str, data: List[Dict], device: str, batch_size: int) -> List[Dict]:
    """评测一个已合并(或base)的模型"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()

    results = []
    for start in range(0, len(data), batch_size):
        batch = data[start:start+batch_size]
        formatted = []
        for item in batch:
            messages = [{"role":"system","content":SYSTEM_PROMPT},
                        {"role":"user","content":create_prompt(item['文本'])}]
            formatted.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        for j, o in enumerate(out):
            inp_len = inputs.input_ids[j].shape[0]
            resp = tokenizer.decode(o[inp_len:], skip_special_tokens=True).strip()
            results.append({"true_label": data[start+j]['标签'], "pred_label": extract_prediction(resp), "response": resp})

    del model
    import gc; gc.collect()
    try:
        if hasattr(torch, 'npu') and torch.npu.is_available(): torch.npu.empty_cache()
        elif torch.cuda.is_available(): torch.cuda.empty_cache()
    except: pass
    return results


def eval_lora_checkpoint(base_model_path: str, lora_path: str, data: List[Dict],
                         device: str, batch_size: int) -> List[Dict]:
    """直接加载base+LoRA adapter评测（不需要先merge）"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()
    model.eval()

    results = []
    for start in range(0, len(data), batch_size):
        batch = data[start:start+batch_size]
        formatted = []
        for item in batch:
            messages = [{"role":"system","content":SYSTEM_PROMPT},
                        {"role":"user","content":create_prompt(item['文本'])}]
            formatted.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        for j, o in enumerate(out):
            inp_len = inputs.input_ids[j].shape[0]
            resp = tokenizer.decode(o[inp_len:], skip_special_tokens=True).strip()
            results.append({"true_label": data[start+j]['标签'], "pred_label": extract_prediction(resp), "response": resp})

    del model, base_model
    import gc; gc.collect()
    try:
        if hasattr(torch, 'npu') and torch.npu.is_available(): torch.npu.empty_cache()
        elif torch.cuda.is_available(): torch.cuda.empty_cache()
    except: pass
    return results


def evaluate_all_epochs(args):
    """评测各epoch checkpoint"""
    output_dir = Path(args.output_dir)
    ckpt_dir = Path(args.ckpt_dir)

    print("=" * 60)
    print("Epoch消融: 评测各epoch checkpoint")
    print("=" * 60)
    print(f"基座模型:     {args.model_path}")
    print(f"Checkpoint目录: {ckpt_dir}")
    print()

    data = load_eval_data(args.data_path)
    if args.num_samples:
        data = data[:args.num_samples]
    print(f"测试数据: {len(data)} 条\n")

    all_metrics = {}

    # 1. 评测 base (epoch=0)
    print(f"\n{'─' * 50}")
    print("[epoch_0] 基座模型 (无SFT)")
    print(f"{'─' * 50}")
    t0 = time.time()
    raw = eval_merged_model(args.model_path, data, args.device, args.batch_size)
    m = calculate_metrics(raw)
    m['elapsed_sec'] = round(time.time() - t0, 1)
    all_metrics["epoch_0"] = m
    print(f"  Acc={m['accuracy']:.2%}  F1={m['macro_f1']:.4f}  Time={m['elapsed_sec']:.0f}s")

    # 2. 评测各epoch
    epoch_dirs = sorted(ckpt_dir.glob("epoch_*"), key=lambda x: int(x.name.split("_")[1]))
    for ep_dir in epoch_dirs:
        epoch_name = ep_dir.name
        epoch_num = int(epoch_name.split("_")[1])

        print(f"\n{'─' * 50}")
        print(f"[{epoch_name}] LoRA checkpoint")
        print(f"{'─' * 50}")

        t0 = time.time()
        raw = eval_lora_checkpoint(args.model_path, str(ep_dir), data, args.device, args.batch_size)
        m = calculate_metrics(raw)
        m['elapsed_sec'] = round(time.time() - t0, 1)
        all_metrics[epoch_name] = m
        print(f"  Acc={m['accuracy']:.2%}  F1={m['macro_f1']:.4f}  Time={m['elapsed_sec']:.0f}s")

    # 3. 汇总
    print(f"\n\n{'=' * 60}")
    print("Epoch消融实验汇总")
    print(f"{'=' * 60}")
    print(f"\n{'Epoch':<12} {'Accuracy':>10} {'Macro-F1':>10} {'Failed':>8}")
    print("-" * 44)
    for name in sorted(all_metrics, key=lambda x: int(x.split("_")[1])):
        m = all_metrics[name]
        print(f"{name:<12} {m['accuracy']:>10.2%} {m['macro_f1']:>10.4f} {m['failed_extractions']:>8}")

    # SFT增益分析
    if 'epoch_0' in all_metrics and 'epoch_1' in all_metrics:
        base_f1 = all_metrics['epoch_0']['macro_f1']
        ep1_f1 = all_metrics['epoch_1']['macro_f1']
        last_key = sorted(all_metrics, key=lambda x: int(x.split("_")[1]))[-1]
        last_f1 = all_metrics[last_key]['macro_f1']

        total_gain = last_f1 - base_f1
        ep1_gain = ep1_f1 - base_f1

        print(f"\nSFT增益分析:")
        print(f"  epoch 1 增益:  F1 {ep1_gain:+.4f} (占总增益 {ep1_gain/total_gain*100:.1f}%)" if total_gain > 0 else f"  epoch 1 增益: F1 {ep1_gain:+.4f}")
        print(f"  总增益:        F1 {total_gain:+.4f}")
        if total_gain > 0 and ep1_gain / total_gain > 0.9:
            print(f"  ⚠️ epoch 1 已获得 >{ep1_gain/total_gain*100:.0f}% 的总增益，后续epoch可能在过拟合")

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.model_path).name
    summary_file = output_dir / f"epoch_ablation_{model_name}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({"model": args.model_path, "results": all_metrics}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {summary_file}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Epoch消融实验")
    sub = parser.add_subparsers(dest="action", help="操作模式")

    # 训练子命令
    p_train = sub.add_parser("train", help="训练并保存每epoch checkpoint")
    p_train.add_argument("--model_path", type=str, default="/home/ma-user/work/test/models_base/Qwen/Qwen2.5-3B-Instruct")
    p_train.add_argument("--data_path", type=str, default="/home/ma-user/work/test/prepared_data/reviewer_sft/train.jsonl")
    p_train.add_argument("--output_dir", type=str, default="/home/ma-user/work/test/ablation_results/epoch_ablation/checkpoints")
    p_train.add_argument("--max_epochs", type=int, default=3)
    p_train.add_argument("--lora_rank", type=int, default=32)
    p_train.add_argument("--lora_alpha", type=int, default=64)
    p_train.add_argument("--batch_size", type=int, default=4)
    p_train.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p_train.add_argument("--learning_rate", type=float, default=2e-4)
    p_train.add_argument("--max_length", type=int, default=2048)
    p_train.add_argument("--device", type=str, default="npu:0")

    # 评测子命令
    p_eval = sub.add_parser("eval", help="评测各epoch checkpoint")
    p_eval.add_argument("--model_path", type=str, default="/home/ma-user/work/test/models_base/Qwen/Qwen2.5-3B-Instruct")
    p_eval.add_argument("--ckpt_dir", type=str, default="/home/ma-user/work/test/ablation_results/epoch_ablation/checkpoints")
    p_eval.add_argument("--data_path", type=str, default="/home/ma-user/work/test/split_data/test.parquet")
    p_eval.add_argument("--output_dir", type=str, default="/home/ma-user/work/test/ablation_results/epoch_ablation")
    p_eval.add_argument("--batch_size", type=int, default=8)
    p_eval.add_argument("--device", type=str, default="npu:0")
    p_eval.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()

    if args.action == "train":
        train_with_epoch_checkpoints(args)
    elif args.action == "eval":
        evaluate_all_epochs(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
