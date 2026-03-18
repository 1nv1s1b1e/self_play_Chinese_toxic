#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Challenger LoRA训练脚本
使用PEFT和transformers对Qwen模型进行LoRA微调
"""

import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'

import sys
import types
import torch

# 昇腾 NPU 初始化
if 'torch_npu' not in sys.modules:
    try:
        import torch_npu
    except (ImportError, RuntimeError):
        sys.modules['torch_npu'] = types.ModuleType('torch_npu')

import argparse
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import set_seed
import json

def load_sft_dataset(data_path: str, tokenizer, max_length: int = 512):
    """
    加载并处理Challenger SFT数据集
    使用 apply_chat_template 与 Qwen2.5-Instruct 的预训练格式对齐
    支持两种格式:
    - messages格式: {"messages": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]}
    - flat格式: {"instruction":"...", "input":"...", "output":"..."}
    只对assistant部分计算loss
    """
    print(f"加载数据集: {data_path}")
    
    # 加载数据 — JSONL 用逐行 json.loads() 避免 pyarrow 对转义字符的严格报错
    dataset = None
    data_format = None  # 'messages' or 'flat'

    if data_path.endswith('.jsonl'):
        records = []
        skipped = 0
        with open(data_path, 'r', encoding='utf-8', errors='replace') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    skipped += 1
                    if skipped <= 5:
                        print(f"  ⚠ 跳过第{line_no}行 (JSON解析错误): {e}")
        if skipped > 0:
            print(f"  共跳过 {skipped} 条有问题的数据")
        if not records:
            raise ValueError(f"数据文件为空或所有行均解析失败: {data_path}")
        from datasets import Dataset
        columns = {}
        for key in records[0].keys():
            columns[key] = [r.get(key) for r in records]
        dataset = Dataset.from_dict(columns)
    elif data_path.endswith('.parquet'):
        dataset = load_dataset('parquet', data_files=data_path, split='train')
    else:
        raise ValueError("不支持的数据格式，请使用 .jsonl 或 .parquet")
    
    print(f"数据集大小: {len(dataset)}")

    # ---- 自动检测数据格式 ----
    sample = dataset[0]
    col_names = list(sample.keys())
    print(f"样本字段: {col_names}")

    if 'messages' in col_names and isinstance(sample['messages'], list):
        data_format = 'messages'
        print(f"  检测到 messages 格式")
        for msg in sample['messages']:
            print(f"    {msg['role']}: {str(msg['content'])[:80]}...")
    elif 'instruction' in col_names and 'output' in col_names:
        data_format = 'flat'
        print(f"  检测到 flat 格式 (instruction/output)，将自动构建messages")
        print(f"    instruction: {str(sample['instruction'])[:80]}...")
        print(f"    output: {str(sample['output'])[:80]}...")
    else:
        raise ValueError(
            f"数据格式不支持，需要 'messages' 或 'instruction'+'output' 字段\n"
            f"实际字段: {col_names}"
        )
    
    # Challenger 系统提示 — 让模型理解自己的角色
    CHALLENGER_SYSTEM_PROMPT = (
        "你是一个文本生成助手。根据用户给定的类别和表达方式，"
        "生成一条对应的真实互联网文本。只输出文本本身，不要任何解释。"
    )

    def process_func(examples):
        """
        处理每个样本: 使用 chat template 编码, 只对 assistant 部分计算 loss
        """
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        if data_format == 'messages':
            items = examples['messages']
        else:
            # flat → messages: 构建对话
            items = []
            for i in range(len(examples['instruction'])):
                instr = examples['instruction'][i]
                inp = examples['input'][i] if 'input' in examples and examples['input'][i] else ""
                out = examples['output'][i]
                user_content = f"{instr}\n{inp}" if inp else str(instr)
                msgs = [
                    {"role": "system", "content": CHALLENGER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": str(out)}
                ]
                items.append(msgs)
        
        for messages in items:
            # 完整对话（含assistant回复）
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # prompt部分（不含assistant回复）
            prompt_messages = messages[:-1]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize完整文本
            full_tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None
            )
            
            # Tokenize prompt部分
            prompt_tokenized = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None
            )
            
            prompt_length = len(prompt_tokenized['input_ids'])
            
            # 创建labels: prompt部分为-100, 只对assistant部分计算loss
            labels = [-100] * prompt_length + full_tokenized['input_ids'][prompt_length:]
            
            # 确保 labels 长度与 input_ids 一致
            if len(labels) != len(full_tokenized['input_ids']):
                labels = labels[:len(full_tokenized['input_ids'])]
            
            model_inputs['input_ids'].append(full_tokenized['input_ids'])
            model_inputs['attention_mask'].append(full_tokenized['attention_mask'])
            model_inputs['labels'].append(labels)
        
        return model_inputs
    
    # 处理数据集
    tokenized_dataset = dataset.map(
        process_func,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing challenger dataset (chat template)"
    )

    # 统计token长度
    lengths = [len(ids) for ids in tokenized_dataset['input_ids']]
    avg_len = sum(lengths) / len(lengths)
    max_len = max(lengths)
    print(f"Token长度统计: 平均={avg_len:.0f}, 最大={max_len}, 截断阈值={max_length}")
    
    return tokenized_dataset

def main():
    parser = argparse.ArgumentParser(description="Challenger LoRA训练")
    parser.add_argument("--model_path", type=str, 
                       default="/home/ma-user/work/test/models_base/Qwen/Qwen2.5-3B-Instruct",
                       help="基础模型路径")
    parser.add_argument("--data_path", type=str,
                       default="/home/ma-user/work/test/prepared_data/challenger_sft/train.jsonl",
                       help="SFT数据路径")
    parser.add_argument("--val_data_path", type=str,
                       default=None,
                       help="验证集数据路径(可选，会显示val loss)")
    parser.add_argument("--output_dir", type=str,
                       default="/home/ma-user/work/test/lora_models_toxicn/challenger",
                       help="LoRA输出目录")
    parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--batch_size", type=int, default=4, help="训练batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--max_length", type=int, default=1024, help="最大序列长度（few-shot 多轮模式）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="npu:0", help="设备 (单卡时指定，如 npu:0)")
    parser.add_argument("--n_devices", type=int, default=1, help="使用的NPU/GPU数量 (≥2时自动多卡并行)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Challenger LoRA训练")
    print("=" * 60)
    print(f"基础模型: {args.model_path}")
    print(f"数据集: {args.data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"LoRA配置: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"NPU数量: {args.n_devices}")
    print()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side='right'  # 重要：设置padding方向
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 检测可用设备
    def detect_device(requested_device):
        """自动检测可用设备"""
        if 'cuda' in requested_device and torch.cuda.is_available():
            return requested_device
        if 'npu' in requested_device:
            try:
                if hasattr(torch, 'npu') and torch.npu.is_available():
                    return requested_device
            except:
                pass
        if torch.cuda.is_available():
            return requested_device.replace('npu', 'cuda')
        return 'cpu'
    
    # 加载模型
    print("加载基础模型...")
    use_multi = args.n_devices > 1
    
    if use_multi:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # 自动检测 NPU/CUDA
        if hasattr(torch, 'npu') and torch.npu.is_available():
            device_str = f"npu:{local_rank}"
            torch.npu.set_device(local_rank)
        elif torch.cuda.is_available():
            device_str = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)
        else:
            device_str = "cpu"
        device_map = {"": local_rank}
        print(f"  多卡模式: LOCAL_RANK={local_rank}, 设备={device_str}")
    else:
        device_str = detect_device(args.device)
        device_map = {"":  device_str}
        print(f"  单卡模式: 设备={device_str}")
    
    # NPU优先用bf16，CUDA用fp16
    if 'npu' in device_str:
        model_dtype = torch.bfloat16
    elif 'cuda' in device_str:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    
    # 配置LoRA
    print("配置LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        inference_mode=False
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 启用gradient checkpointing (3B模型节省显存)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # 加载数据集
    train_dataset = load_sft_dataset(args.data_path, tokenizer, max_length=args.max_length)
    
    # 加载验证集(可选)
    val_dataset = None
    if args.val_data_path and os.path.exists(args.val_data_path):
        print(f"加载验证集: {args.val_data_path}")
        val_dataset = load_sft_dataset(args.val_data_path, tokenizer)
    
    # 配置训练参数
    ddp_kwargs = {}
    if use_multi:
        ddp_kwargs["ddp_find_unused_parameters"] = False
        # 升腾NPU使用HCCL后端
        if 'npu' in device_str:
            ddp_kwargs["ddp_backend"] = "hccl"
    
    eval_strategy = "steps" if val_dataset is not None else "no"
    eval_steps = 50 if val_dataset is not None else None
    
    # 根据设备选择精度
    use_bf16 = 'npu' in device_str
    use_fp16 = 'cuda' in device_str and not use_bf16
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        logging_dir=os.path.join(args.output_dir, "logs"),
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=False,  # DDP多卡训练结束时显存已满，reload best ckpt会OOM；最佳ckpt已由save_steps保存到磁盘
        metric_for_best_model="eval_loss" if val_dataset is not None else None,
        greater_is_better=False if val_dataset is not None else None,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,  # NPU环境下多 worker易出错
        **ddp_kwargs,
    )
    
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("\n开始训练...")
    train_result = trainer.train()
    
    # 保存模型（DDP 模式下只在 rank 0 保存）
    if not use_multi or int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("\n保存LoRA模型...")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # 保存训练指标
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        print(f"\n✓ 训练完成！LoRA模型已保存到: {args.output_dir}")
        print(f"  Train Loss: {metrics.get('train_loss', 'N/A')}")
    else:
        print(f"\n✓ Rank {os.environ.get('LOCAL_RANK')} 训练完成 (由 rank 0 保存模型)")

if __name__ == "__main__":
    main()
