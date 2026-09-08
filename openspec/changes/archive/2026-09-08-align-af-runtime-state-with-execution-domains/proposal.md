## Why

当前 AF planner 已按 execution domain 展开 TP/DP/EP 计算，但 request-owning worker 的模型内存与 KV Cache 仍按另一套全局 `ParallelConfig` 计算，且非 leader A TP rank 没有 rank-local KV 状态。这会使“多 rank 计算”与“单 rank 内存记账”同时存在，导致 KV 容量、模型是否可装载、最大并发请求数和 OOM 判断不可信。

## What Changes

- 在 AF 启动时将 execution domains 编译为唯一的 runtime plan，为每个 A/F logical rank 绑定并行坐标、physical worker、驻留模型组件与内存容量。
- 将“请求调度者”与“加速卡 rank 状态”分离：A TP leader 继续运行 scheduler，但 A TP group 的每个 rank 都拥有自己的模型分片、KV 分片和容量记账。
- 为每个 A DP replica 建立组级 KV Cache；一个逻辑 KV block 必须在该 replica 的全部必要 TP ranks 上原子分配或原子回滚，replica 可用 block 容量由最小 rank-local 容量决定，不对 ranks 求和。
- 按 A/F 分工重新计算 rank-local 模型驻留内存：A 保存 Attention/KV、首尾组件，并由 A leader 保存 Router；F ranks 按 expert placement 保存 Expert/FFN 组件；不再让 A leader 代表整个分离模型的全量显存。
- 将 MoE Router/Top-K 放在 A TP compute/collective 之后、A→F 之前的 A leader 上；Router 只将原始 token 展开为 routed token-expert assignments 一次，A→F 本身承担跨域 EP dispatch，F→A 本身承担 EP combine，不再串行模拟第二次相同 payload 搬运。
- 使 F rank 的 Expert/Dense FFN 算子形状真实按 TP 切分，并让每个本地 Expert 的 routed assignment 数进入 latency lookup；`top_k` 不得在 Expert FLOPs 中再次相乘。
- 对外部 Expert histogram 校验 `sum(histogram) == token_count * top_k`，在 workload 边界拒绝含义不一致的输入。
- 将 KV head 的 TP 分片或复制规则与 A domain TP 绑定，对无法整除或无法无歧义复制的拓扑在模拟前明确失败。
- 增加 AF 并行语义校验，确保 scheduler owner、planner TP/EP group、rank runtime state、KV 分片与 expert 驻留使用同一拓扑。
- 保持 `af.enabled=false` 的现有全局 `ParallelConfig` 和 KV Cache 路径不变。本 change 不实现算子级 TP DAG、跨 DP 的 vLLM-style global EP、`PP>1`、EP rank-to-rank traffic matrix 或 speculative decode。

## Capabilities

### New Capabilities

- `af-runtime-state`: 定义 AF execution domain 到 rank-local 模型/KV 状态、A replica 组级 KV 事务、容量计算与语义校验的行为契约。

### Modified Capabilities

- `af-execution`: 将现有“KV 状态仍归 A 侧所有”的要求收紧为“调度由 A replica leader 负责，KV 分片在该 replica 的必要 A TP ranks 上保持一致分配”。

## Impact

- 主要影响 `TokenSim/llm/llm_engine.py`、`TokenSim/config/cache_config.py`、`TokenSim/config/cluster_config.py`、`TokenSim/af/planning.py` 及 block manager/scheduler 的 KV 容量边界。
- 新增 AF runtime plan、rank state、replica KV Cache 和 semantic validator 边界，并将 AF DAG 收敛为 A→Router→A→F/EP Dispatch→Expert→F→A/EP Combine，使 Router 归属、payload、计算量与 A 侧路由决策一致。
- AF 多 rank 场景的 KV 容量、可调度请求数和显存失败结果可能与当前输出不同；这是修正错误语义后的预期变化。
- 需要增加 TP KV 分片、DP 独立容量、异构 rank 最小容量、原子回滚、F expert 内存驻留和 AF-disabled 回归测试。
