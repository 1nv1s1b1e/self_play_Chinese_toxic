# Plan 4: API 外部验证器 (Verifier API)

## 核心问题

v1 的 Verifier 是冻结的 7B LoRA Reviewer 模型:
1. **显存/时间开销大**: Phase 0 每轮需加载 7B 模型到 NPU，占用大量显存和时间
2. **标签质量有限**: 7B 模型对隐式仇恨等边界情况判断不准，导致 ground truth 有噪声
3. **与 Reviewer 同架构**: Verifier 和 Reviewer 都是 Qwen2.5 + LoRA，容易共享偏见

## 改进方案

使用外部 API (如 Qwen-72B / DeepSeek API) 作为 Verifier:
- **更准确的标签**: 72B 模型远超 7B 在分类任务上的准确率
- **更快的 Phase 0**: 无需在 NPU 上加载 7B，API 调用是异步 IO 而非 GPU 计算
- **架构去偏**: 不同模型架构减少共享偏见

## 实现细节

| 特性 | 说明 |
|------|------|
| API 兼容 | OpenAI 格式 (兼容 Qwen/DeepSeek/OpenAI) |
| 批量并发 | asyncio + semaphore 控制并发数 |
| 超时重试 | 3 次重试 + 指数退避 |
| 降级机制 | API 失败时 fallback 到本地 7B Verifier |
| 成本控制 | 只在 Phase 0 使用，每轮 ~1500 次 API 调用 |

## 修改的文件

| 文件 | 作用 |
|------|------|
| `api_verifier.py` | API 验证器核心实现 |
| `run_selfplay_plan4.sh` | 修改版主循环: 使用 API Verifier |
| `README.md` | 本文件 |

## 如何使用

```bash
# 设置 API 密钥
export VERIFIER_API_KEY="your-api-key"
export VERIFIER_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export VERIFIER_API_MODEL="qwen-plus"

bash scripts/plan_verifier_api/run_selfplay_plan4.sh
```

## 成本估算

- 每轮 Phase 0: ~1500 条文本 × ~200 token/条 = ~300K tokens
- Qwen-plus 价格: ~0.004 元/千 token → 每轮约 1.2 元
- 5 轮总计: ~6 元

## 可与方案 1/2/3 组合

方案 4 替换的是 Verifier 来源，与 reward/优化器/数据管线完全正交。
