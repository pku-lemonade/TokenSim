## Why

当前 AF Executor 每层只调度 A-to-F 与 F-to-A 两个 `af_transfer`，虽然 transfer provider 能估计 MoE dispatch、combine 和 AllReduce，但这些内部通信没有成为资源事件，因而不会进入排队、trace 或 makespan。同时，集群把 Prefill、A、F 的全部 worker 展开为一套 global ranks，并强制总数等于 `TP * PP * DP`，无法表达 PD+AF 异构部署中各执行域独立的并行拓扑。

## What Changes

- **BREAKING**：将 AF worker group 的逻辑 rank 与物理 worker 分别建模，并要求每个 execution domain 内采用严格一对一 placement；不再对 Prefill、A、F worker 总和执行单一 world-size 校验。
- 为完整 Prefill、Decode A 和 Decode F 分别定义并校验独立的并行执行域；每个域可声明自己的 TP、PP、DP、EP 关系及 rank 到物理设备的映射。
- A/F 各自要求每个 logical rank 唯一映射到一个 physical worker，且同一 domain 的 physical worker 不得承载多个 ranks；A 与 F 的 worker 数可以彼此不同，但每侧必须与自身 logical rank placement 完整匹配。
- 将每层 AF 执行从四段链扩展为显式执行计划：A compute、域内 collective、A-to-F、Router/Gating compute、MoE dispatch、expert compute、域内 collective、MoE combine、F-to-A；仅在当前模型与并行计划要求时创建对应 collective。
- 将原有合并的 F compute latency composition 拆为互不重叠的 Router/Gating 与 Expert 两个 service time，确保 Router 在 dispatch 前完成且任何算子恰好计时一次。
- 让 dispatch、combine 和 AllReduce/collective 使用独立的 typed transfer query、payload、参与 rank、通信资源和 trace 事件，不再仅停留在 provider 的孤立查询能力。
- 保持 request、scheduler 与 KV Cache 归属在请求拥有侧；F logical instance 不取得 request 或 KV Cache 所有权。
- 增加端到端测试，证明内部通信进入 service、queue、resource contention、trace 与 makespan，并证明不同 P/A/F 设备数和并行拓扑不再被错误地合并校验。
- 保持 `af.enabled=false` 和不显式配置新拓扑的既有单卡配置兼容；对无法无歧义迁移的显式多 rank AF 配置给出可操作错误，而不猜测 rank mapping。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `af-separation-config`：拆分 physical device/rank、AF logical instance 与 execution domain，并为 Prefill、A、F 分别定义并行拓扑和配置校验。
- `af-execution`：把 dispatch、combine 和执行计划要求的 collective 纳入逐层事件链、共享资源竞争、trace 和 makespan。
- `operator-data-latency-provider`：拆分 Router/Expert compute composition，并要求运行时为内部通信事件构造完整 typed query，保留 operation、参与 ranks、payload、来源和匹配元数据。

## Impact

- 主要影响 `TokenSim/config/cluster_config.py`、`TokenSim/config/parallel_config.py`、`TokenSim/parallel.py`、`TokenSim/af/resources.py`、`TokenSim/af/executor.py`、`TokenSim/af/latency.py`、`TokenSim/operator_data/providers.py` 和 `TokenSim/llm/llm_engine.py`。
- AF 集群配置将新增 execution domain、logical instance 和 rank/device mapping 字段；现有示例需要迁移。
- AF trace 将新增 `moe_dispatch`、`moe_combine` 和必要 collective 事件及其参与者元数据。
- 不新增 Ada/Groq 性能数据，不改变“机制与模型自洽通过，物理准确性待实测”的结论边界。
