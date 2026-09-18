## Context

用户确认沿用其它算子的「拆分 → 离线 Roofline → 运行时查表 → AF 调度」，缺少实测稍后补充。旧实现仅放开 MLA 枚举，仍套用 QKV/GQA 公式，不能表达 V3。

## Goals / Non-Goals

**Goals:**

- 扩展模型元数据以表达 MLA 维度。
- 离线生成 MLA analytical operator-data，运行时精确查表并区分来源。
- 保持 MoE A/F planner 与既有模型兼容。
- 为 DeepSeek‑V3 建立可运行的模型配置与校验。

**Non-Goals:**

- 本变更不实现 vLLM 权重加载或真实 DeepSeek 推理。
- 没有设备实测数据时不宣称精确性能预测。
- 不改变普通 MHA/GQA/MQA 的公式和数据兼容性。
- 不实现 V3 noaux_tc 路由、MTP、完整 prefill 调度；prefill 先验证等长完整单层 Attention。

## Decisions

1. **统一维度。** `MLADimensions` 表达 q_lora、kv_lora、rope、nope、v。Q/K head=192、cache width=576；不使用 hidden/heads=56。
2. **算子拆分。** decode：Q/KV 投影、norm、RoPE → grouped Q absorption → compressed core → grouped value expansion → output。prefill：KV expansion → expanded causal core。前置步骤按顺序建模；cached/chunked/ragged prefill 明确拒绝。
3. **专用公式。** decode matmul FLOPs=`2*H*sum(S)*(2*kv_lora_rank+rope_dim)`；softmax 近似 `5*H*sum(S)`；cache 读取=`sum(S)*(kv_lora_rank+rope_dim)*dtype_bytes`，另计本步写入。Q latent rank 只用于投影。prefill 按有效 causal pairs 计算，展开 K/V 是临时张量。
4. **可选表。** `mla_attention` 携带完整核心 shape，精确匹配；`batched_gemm` 表达 H 组独立逐头乘法，一次启动、沿 m 插值。普通表不变；cache 更新在 core 测量边界内。
5. **离线理论路径。** 未测 shape 离线生成 Roofline 数据；runtime 缺表显式失败，不回退 GQA 或现场 analytical。合格实测在生成时优先，固定开销不重复叠加。
6. **TP 与账本。** A TP 分片 query heads、up/output 权重，复制 down 投影和 latent cache。V3 cache 每 rank 每层每 token 为 576 元素，embedding/lm_head 未绑定。
7. **示例。** 单层虚拟 1A1F、完整 61 层虚拟 1A32F 均使用 AFExecutor。保留前 3 层 Dense、后 58 层 MoE 和 shared expert。硬件与固定功耗假设独立于模型文件。

## Risks / Trade-offs

- [Risk] 公式假设 fused softmax、cache 一次逻辑读取，未刻画重复加载、padding 和 occupancy；启动、网络和功耗尚未实测 → Mitigation：来源明确标记 analytical，误差范围未知，不能因 exact shape 命中声称实测。
- [Risk] 新 key 字段导致旧 operator-data 无法命中 → Mitigation：普通 Attention 走旧 key；MLA 不允许静默回退到 GQA。
- [Risk] 大模型配置数值错误 → Mitigation：维度校验、配置快照和最小 decode smoke test。
- [Risk] BF16 模拟不同于官方 FP8 checkpoint，synthetic uniform routing 不等于 noaux_tc；33 卡为虚拟资源 → Mitigation：结果披露边界，保留显存检查，不声称四卡可装下完整 BF16 V3。

## Migration Plan

旧数据包无需迁移，MLA 必须生成可选表。`requested-shapes.json`、`measurement-backlog.json` 和 `docs/deepseek-v3-mla.md` 给出补测形状及边界。用户明确同意实测稍后补充，不阻塞本轮理论模拟。
