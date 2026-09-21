# AgentX 模拟精度提升 TODO

> 上次更新：2026-09-21
> 分支：`agentx-support`（最新提交 `91322dc`）
> 对照基准：InferenceX 开源仓库 `github.com/SemiAnalysisAI/InferenceX` + SemiAnalysis 博客

## 0. 当前状态摘要

### 已完成

- EP8/per_dp 并行修复（`2a40d9e`）
- AgentX replay、sampling、stats、scheduler 对齐（`91322dc`）
- 经验幂律外推（decode attention α ≈ 0.24–0.45 替代 roofline α = 1.0）
- 四配置完整实验（B200/B300 × HBM/DRAM，1800s profile）

### 当前差距（B200 DRAM，empirical extrapolation）

| 指标 | 模拟值 | 博客参考值 | 差距 |
|------|--------|-----------|------|
| 输出 tok/s/GPU | 7.88 | 449.91 | 57x |
| P90 TPOT | 0.341s | 0.050s | 6.8x |
| P90 TTFT | 95.7s | 16.2s | 5.9x |
| Mean E2E | 138.3s | 43.4s | 3.2x |
| HBM hit rate | 57.8% | 73% | -15pp |

### 差距根因分解

通过分析 InferenceX 开源配置（`configs/nvidia-master.yaml`）和 vLLM 代码，确认博客中 DeepSeek-V4 在 B200/B300 的真实配置：

```
B200 高并发: tp=8, ep=8, dp-attn=true, dcp-size=1, kv-offloading=dram
             framework=vllm, spec-decoding=dspark(K=6), concurrency=192
B300 高并发: tp=8, ep=8, dp-attn=true, dcp-size=1, kv-offloading=dram
             framework=sglang, spec-decoding=dspark(K=6), concurrency=384-576
```

关键发现：**DSV4 不使用 DCP**（`dcp-size` 默认为 1）。DCP 仅用于 Kimi-K3（4 KV heads）。DSV4 的 MLA（`num_kv_heads=1`）下 decode attention 仍是单 rank 扫描完整 KV cache。

---

## 1. [高影响 — P0] FP4 权重 + FP8 KV Cache 精度对齐

**误差性质**：硬性的精度/算力差，不是外推能补救的。不修则所有结论要标注"未考虑量化"，误差可能 20%+。

### 1.1 当前状态

TokenSim 的 parquet 数据（来自 AIConfigurator 同源）**已包含所有精度**：

| 算子 | 已有精度 | 查表路径 | 状态 |
|------|---------|---------|------|
| GEMM | bf16, fp8, fp8_block, nvfp4 | `"dtype": model.dtype` → nvfp4 | ✅ 已正确 |
| MoE | bf16, fp8, fp8_block, nvfp4, mxfp4 等 | `"dtype": model.dtype` → nvfp4 | ✅ 已正确 |
| Attention | bf16 KV, fp8 KV | `"kv_cache_dtype": model.kv_cache_dtype` → bf16 | ❌ 应为 fp8 |

真实系统用 `--kv-cache-dtype fp8`（见 `dsv4_fp4_b200_vllm_mtp.sh:275`），模拟器 YAML 写的是 `bf16`。

### 1.2 实测精度差异

**GEMM NVFP4 vs BF16**（已在生效，供参考）：

| m | NVFP4 (us) | BF16 (us) | 加速 |
|---|-----------|----------|------|
| 1 | — | — | 1.43x |
| 64 | — | — | 1.72x |
| 1024 | — | — | 2.32x |
| 8192 | — | — | 2.68x |

**Attention FP8 KV vs BF16 KV**（MLA shape: num_heads=16, num_kv_heads=1, head_dim=64）：

| batch | context | bf16 KV (us) | fp8 KV (us) | 加速 |
|-------|---------|-------------|------------|------|
| 8 | 65K | 33.21 | 23.17 | 1.43x |
| 8 | 131K | 52.11 | 36.38 | 1.43x |
| 32 | 65K | 88.56 | 54.07 | 1.64x |

短 context（<8K）差异可忽略（~1.0x）。

### 1.3 FP8 KV 对 HBM 容量的间接影响

FP8 KV 使每 token 的 KV cache 字节数减半：

| kv_cache_dtype | bytes/token/layer | 8-rank HBM 容量 (tokens) |
|---------------|-------------------|------------------------|
| bf16 | 256 | ~19.3M |
| fp8 | 128 | ~38.5M |

当前 cluster config `kv_cache_capacity_tokens_total: 21943624`（接近 bf16 理论值）。改成 fp8 后容量翻倍，HBM hit rate 会从 57.8% 大幅提升，可能接近博客的 73%。

### 1.4 具体修改

1. **`data/models/deepseek-v4-proxy.yaml`**：`kv_cache_dtype: bf16` → `kv_cache_dtype: fp8`（一行）
2. **`data/clusters/64_b200/tp8_dp8_ep.json`**：按 fp8 重算 `kv_cache_capacity_tokens_total`（约 38.5M 或按 vLLM 脚本的 `DEP_KV_CACHE_BYTES=37580963840` 精确计算）
3. **验证**：跑 B200 DRAM 配置，对比 attention 延迟和 HBM hit rate 变化

### 1.5 Quant/Dequant 算子

真实系统在 NVFP4 GEMM 前后有 quant/dequant 开销。目前 TokenSim 的 elementwise 表缺失（100% analytical），这部分开销可能被低估或漏掉。短期可忽略（占比 <1%），但在端到端验证中需要关注。

---

## 2. [高影响 — P0] 超长上下文 Decode Attention 外推

**误差性质**：46.5% 的 decode attention 查询超出测量边界（131K），当前用固定幂律指数（α ≈ 0.24–0.45）外推。长 context 下高估 ~2.8x，通过闭环放大级联为更大的吞吐误差。

### 2.1 当前方法的问题

经验幂律 `latency(ctx) = latency(ctx_max) × (ctx / ctx_max)^α` 是对整体 attention 的盲外推，没有利用算子的物理结构。

### 2.2 改进方向：按算子分解建模

Decode attention 的物理行为可分解为：

- **KV cache 读取**：线性于 context_len（memory-bound，带宽受限）
- **QK^T 计算**：线性于 context_len × num_heads（compute，但 MLA 下 num_kv_heads=1 使其 memory-bound）
- **Softmax**：超线性于 context_len（需要两遍扫描：max-reduce + normalize）
- **SV 乘法**：线性于 context_len

策略：

1. 在 2–3 个代表性 context 长度（如 8K、32K、131K）上采样实测点
2. 按理论复杂度函数拟合各分量系数（而非盲外推整体延迟）
3. 按分解模型插值/外推到 500K+

### 2.3 按 batch size 分别拟合

当前 α 是上四分位平均值，但实测差异大：

| batch_size | α |
|-----------|---|
| 1 | 0.236 |
| 8 | 0.270 |
| 16 | 0.451 |

应按 batch_size 分别拟合参数，而非统一 α。

### 2.4 涉及文件

- `TokenSim/operator_data/lookup.py`：`_fit_scaling_exponent()` 改为 per-batch-size 拟合 + 分解模型
- `data/operator_data/`：补充长 context 测量数据（如果能获得 >131K 的 MLA 实测）

---

## 3. [中等影响 — P1] MLA 解析近似

**误差性质**：完全模拟代价大，但可以用解析近似把误差从"不支持"降到"有限近似"。

### 3.1 MLA 的本质

MLA 把 KV cache 从 per-head 压成共享低秩 latent：

- **KV 读取量**：压缩比 = `kv_lora_rank / (num_heads × head_dim)`。DSV4: `kv_lora_rank=512, num_heads=128, head_dim=128` → 压缩比 1/32
- **显存占用**：按压缩比折算进现有 GQA 模型（已部分体现在 `num_kv_heads=1, head_dim=64` 配置中）
- **额外算力**：decode 时多两次小的 up-projection GEMM（latent → K, latent → V），维度 `[batch, kv_lora_rank] → [batch, num_heads, head_dim]`

### 3.2 近似方案

TokenSim 现有的 `num_kv_heads=1, head_dim=64` 配置已经隐式捕获了 MLA 的 KV 压缩效果（查表时 KV 读取量自动按这个 shape 计算）。需要补充的是：

1. **Up-projection GEMM 开销**：每个 decode step 额外 2 次 GEMM，shape `[batch, 512] × [512, 128×128]`。可从现有 GEMM 表查询或 analytical 估算
2. **Prefill 路径**：prefill 时的 absorbed attention 可以跳过 up-projection（直接在 latent space 计算），但需要不同的 attention kernel 延迟

### 3.3 写报告时的标注

标注"MLA 用 GQA 等效 shape 近似，up-projection 开销未计入"。误差预估 5–15%，远小于完全不支持。

### 3.4 涉及文件

- `TokenSim/latency/operator_table.py`：在 `_attention_decode()` 后追加 up-projection GEMM 查询
- `data/models/deepseek-v4-proxy.yaml`：增加 `kv_lora_rank` 字段

---

## 4. [中等影响 — P1] DSpark 投机解码

**误差性质**：如果目标场景不用投机解码，直接声明 scope 排除是合理的。如果要覆盖，用接受率参数化的解析模型即可。

### 4.1 博客配置

InferenceX 使用 DSpark（模型自身 MTP head 做 draft），参数来自 `dsv4_fp4_b200_vllm_mtp.sh`：

```json
{
  "method": "dspark",
  "num_speculative_tokens": 6,
  "draft_sample_method": "probabilistic",
  "rejection_sample_method": "synthetic",
  "synthetic_acceptance_length": 3.77
}
```

每个 decode step 产出 ~3.77 token（而非 1 个），等效 TPOT = step_latency / 3.77。

### 4.2 解析模型（推荐）

不需要完整模拟 draft-verify 循环。关键变量：

- **acceptance_length**（AL）：配置为 3.77（synthetic），或从 golden AL distribution 采样
- **draft 开销**：MTP head 的 K 次 forward pass ≈ K 次小 GEMM（hidden_size → vocab_size），远小于完整 decode step
- **verify 开销**：1 次 prefill-like forward pass（K+1 tokens）

简化公式：
```
effective_tpot = (decode_step_latency + K × draft_latency + verify_latency) / AL
```

其中 `draft_latency` 和 `verify_latency` 可从现有 GEMM/attention 表估算。

### 4.3 简化方案（最小改动）

在 scheduler 中加入 `speculative_acceptance_length` 参数，每个 decode step 产出 AL 个 token：

- 不改延迟模型，只改每步 token 产出数
- 等效于假设 draft + verify 开销 ≈ 0（乐观估计）
- 预估改善：throughput 提升 ~3.77x

### 4.4 涉及文件

- `TokenSim/llm/llm_scheduler.py`：decode step token 产出逻辑
- `TokenSim/latency/operator_table.py`：draft/verify 延迟计算（解析模型方案）
- `data/models/deepseek-v4-proxy.yaml`：增加 `speculative_decoding` 配置段

---

## 5. [低影响 — P2] FlashInfer MLA Sparse Attention

与 dense FlashAttention 差距不大，尤其 sparsity 不高时。**标为已知近似即可。**

当前 decode attention 用通用 `generation_attention` 表查询。博客用 `FLASHINFER_MLA_SPARSE_DSV4` + FP4 indexer，但 MLA 的 KV head 数极少（1 head, dim=64），sparse 的收益主要在极长 context（>256K）下才显著。

如果端到端验证发现 attention 延迟仍然是主要误差源，再考虑收集 MLA sparse kernel 的 profile 数据。

---

## 6. [P1] Prefill-Decode 调度交错

### 6.1 真实系统配置

- vLLM: `prefill-schedule-interval=16`，`long-prefill-token-threshold=512`
- SGLang: `prefill-decode-interval=20–24`

博客指出引入后 output throughput +141%，P99 ITL -97.3%。

### 6.2 当前状态

模拟器的 chunked prefill 调度是每步都可以混合 prefill 和 decode，没有显式 interval 控制。

### 6.3 需要做的

1. 实现 `prefill_schedule_interval` 参数：N 步 decode-only 后才允许一次 prefill
2. 实现 `long_prefill_token_threshold`：超过阈值的 prefill 单独调度

---

## 7. [P1] HBM/DRAM Cache Hit Rate 对齐

### 7.1 当前差距

| 配置 | 模拟 HBM | 博客 HBM | 模拟 DRAM | 博客 DRAM |
|------|---------|---------|----------|---------|
| B200 DRAM | 57.8% | 73% | 36.3% | 20% |
| B300 DRAM | 49.6% | 91% | 44.1% | 1.36% |

注意：修复 FP8 KV 后 HBM 容量翻倍，hit rate 会自动提升（见 §1.3）。建议先完成 §1 再评估剩余差距。

### 7.2 真实系统的 cache 策略

- vLLM: `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768`
- SGLang: `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`
- DEP8 下 session 通过 router hash 固定到 DP rank，影响 cache 复用

### 7.3 需要做的

1. 对齐 prefix cache retention 策略
2. 验证 session → DP rank 亲和性路由
3. 修复 FP8 KV 后重新评估差距

---

## 8. [P1] DEP 路由和 Cache 亲和性

TokenSim 已支持 `data_parallel_size=8` + `expert_parallel_scope=per_dp`，但 DEP8 的精细行为未完全对齐：

- **Session 亲和性路由**：vllm-router（consistent_hash）或 sglang-router（cache_aware）
- **Per-DP KV cache 独立性**：每个 DP rank 容量 = 总 HBM / DP_size
- **Load balance**：vLLM `balance-abs-threshold=32`，SGLang `total_requests`

---

## 9. [P2] 其他

### 9.1 Elementwise 算子数据

`elementwise` 表完全缺失（100% analytical）：rmsnorm、residual_add、rope、swiglu。总延迟占比约 0.9%，短期可忽略。

### 9.2 博客参考点详细核对

- 博客数据对应哪个 concurrency 和 framework 组合
- Profile duration / warmup / drain 规则
- 统计口径（root + subagent? incomplete requests?）

---

## 10. 执行计划

### Phase 1 — 消除硬性精度差 + 外推失败（预估 2-4x 改善）

```
[P0] 1. FP8 KV cache 对齐        ← 一行 YAML + capacity 重算，立竿见影
[P0] 2. 超长 context 外推重构     ← 按算子分解建模替代盲幂律
```

### Phase 2 — 端到端验证

**在 Phase 1 完成后**，用 2-3 个真实 workload 做端到端验证：

| 实验 | Context | 量化 | 关注指标 |
|------|---------|------|---------|
| A | 8K | fp8 kv | TPOT, throughput baseline |
| B | 32K | fp8 kv | attention 占比, hit rate |
| C | 128K | fp8 kv | 外推残余误差, E2E |

对比结果决定 §3/§4 的投入是否值得。

### Phase 3 — 按验证结果补齐

```
[P1] 3. MLA 解析近似              ← 如果 attention 延迟仍是主要误差源
[P1] 4. DSpark 投机解码           ← 如果目标场景需要覆盖（否则声明 scope 排除）
[P1] 5. Prefill-decode 调度交错
[P1] 6. HBM/DRAM cache hit rate
[P1] 7. DEP 路由和 cache 亲和性
```

### Phase 4 — 精打细磨

```
[P2] 8. FlashInfer MLA sparse     ← 标为已知近似
[P2] 9. Elementwise 算子数据
[P2] 10. 博客参考点核对
```

Phase 1 完成后预计：attention 延迟降低 1.4-1.6x（FP8 KV）+ 外推误差大幅收窄 + HBM 容量翻倍带来的 cache hit rate 提升。加上闭环正反馈，总差距有望从 57x 显著缩小。端到端验证后才能给出精确的剩余差距估算。

---

## 11. 关键参考资料

| 资源 | 路径/URL |
|------|---------|
| InferenceX 仓库 | `github.com/SemiAnalysisAI/InferenceX` |
| B200 vLLM 启动脚本 | `benchmarks/single_node/agentic/dsv4_fp4_b200_vllm_mtp.sh` |
| B200 SGLang 启动脚本 | `benchmarks/single_node/agentic/dsv4_fp4_b200_sglang_mtp.sh` |
| 硬件配置矩阵 | `configs/nvidia-master.yaml` |
| vLLM DCP 实现 | `/home/zhuohang/Serving/vllm-note/vllm/v1/attention/ops/dcp.py` |
| vLLM MLA attention | `/home/zhuohang/Serving/vllm-note/vllm/models/deepseek_v32/attention.py` |
| AIConfigurator 数据 | `github.com/ai-dynamo/aiconfigurator` — 同源于 TokenSim parquet |
| 经验外推实验结果 | `results/agentx-empirical-extrap/` |
| 博客 | `newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat` |
