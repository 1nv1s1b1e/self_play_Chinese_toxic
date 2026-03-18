import os
import sys
import torch
import torch_npu
from datasets import load_dataset
# 引入 AutoModelForCausalLM 准备手动加载模型
from transformers import AutoTokenizer, AutoModelForCausalLM 
from trl import GRPOConfig, GRPOTrainer

print("2. [初始化] 正在挂载奖励函数...")
# 请确保当前运行路径下有 reward_functions 文件夹，或者改为绝对路径！
sys.path.insert(0, "/home/ma-user/work/test/RL_SCRIPTS/reward_functions") 
try:
    from challenger_reward_v7 import compute_score
    print("   ✅ 奖励函数加载成功！")
except Exception as e:
    print(f"   ❌ 奖励函数加载失败: {e}")
    sys.exit(1)

def custom_reward_func(prompts, completions, **kwargs):
    """
    修复版奖励函数：
    TRL 会将 dataset 中的额外列作为 list 传入 kwargs。
    例如 kwargs['reward_model'] 会是一个字典列表，包含了当前 batch 的所有 reward_model 数据。
    """
    scores = []
    
    # 安全提取数据集中的自定义列
    reward_models = kwargs.get("reward_model", [])
    extra_infos = kwargs.get("extra_info", [])
    
    for i in range(len(prompts)):
        # 提取生成的文本内容
        solution_str = completions[i][0]["content"] 
        
        # 从 reward_model 字典中安全提取 ground_truth
        gt = ""
        if i < len(reward_models) and isinstance(reward_models[i], dict):
            gt = reward_models[i].get("ground_truth", "")
            
        # 提取 extra_info
        extra = extra_infos[i] if i < len(extra_infos) else {}
        
        # 调用你的底层评分逻辑
        score = compute_score(
            data_source="toxicn_challenger", 
            solution_str=solution_str,
            ground_truth=gt,
            extra_info=extra
        )
        scores.append(score)
        
    return scores

def main():
    print("3. [配置] 解析模型和数据路径...")
    model_id = os.environ.get("MODEL_PATH", "/home/ma-user/work/test/merged_models_toxicn/challenger_0.5B")
    dataset_path = os.environ.get("DATA_PATH", "/tmp/verl_smoke_test/data/smoke_train.parquet")
    output_dir = "./grpo_challenger_output"

    if not os.path.exists(dataset_path):
        print(f"   ❌ 找不到数据集: {dataset_path}")
        sys.exit(1)

    print("4. [加载] 正在加载 Tokenizer, 模型与数据集...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========================================================
    # 核心修复：手动加载模型对象，明确把分配权彻底交给 DeepSpeed
    # ========================================================
    print("   -> 正在从本地磁盘读取模型权重 (无 device_map)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=None  # <=== 关键！坚决不使用 "auto"
    )

    dataset = load_dataset("parquet", data_files={"train": dataset_path})["train"]
    print(f"   ✅ 数据集加载完成，共 {len(dataset)} 条。")

    print("5. [配置] 设定 TRL GRPO 参数 (集成 DeepSpeed)...")
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-6,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=2,
        num_generations=4, 
        max_completion_length=128,
        bf16=True,
        logging_steps=1,
        max_steps=50, 
        save_steps=50,
        gradient_checkpointing=True,
        use_vllm=False, 
        deepspeed="ds_zero2.json", 
    )

    print("6. [启动] 正在初始化 GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,  # <=== 这里传入加载好的模型对象，而不是字符串
        reward_funcs=[custom_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    print("7. 🚀 [训练] 开始 PPO 更新！")
    trainer.train()
    trainer.save_model(output_dir)
    print("🎉 训练大功告成！")

if __name__ == "__main__":
    main()