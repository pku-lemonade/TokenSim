## Why

当前 Decode AF 已能展开 DP/TP/EP rank 与资源竞争，但逐层 Dense/MoE 选择、逐 token 路由、Shared Expert、F-to-A 合并、EP rank 复用以及 dtype/算子公式仍与真实执行语义不一致。这些偏差会同时改变计算量、通信量、设备数和 makespan，因此必须先完成正确性闭环，后续才能可信地研究通信能力、micro-batch 流水与吞吐量的关系。

## What Changes

- Decode AF planner 按模型的逐层类型生成不同 DAG：Dense 层执行 A Attention、F Dense FFN 与返回；MoE 层才执行 A Router、routed/shared Expert、weighted combine 与 residual。
- 用逐 token routing assignments 保存 token owner、目标 Expert 和 combine 所需的 weight 语义；histogram 只从 assignments 派生并用于负载统计。
- F-to-A 以一个 logical stage 下的 endpoint legs，按 token/tensor ownership 将 partials 直接发送到对应 A TP ranks；A/F TP 不同时通过 global hidden interval 交集映射 shard，每条 leg 只等待自己的 producer，各 A rank只等待并合并属于自己的 partials，下一层需要不同 layout 时再执行明确的 A TP collective。
- Shared Expert 在 F 侧对每个原始 token 恰好执行一次，与 routed Expert 分支并行，并进入同一个 A 侧合并点。
- **BREAKING**：F 域只支持 `EP=1`（Expert TP 沿用 F TP，按 F DP replica 复制）或 vLLM-style rank-reuse `EP=TP*DP`（Expert TP=1，Expert 跨现有 F world 放置）；两种模式都只使用已有 `TP * DP` physical ranks，不支持隐式 `TP * EP * DP` 拓扑。
- 对所有支持的 TP 分片实施严格 shape 校验；除已明确定义的 KV-head 复制外，非法非整除配置在仿真前失败，不再使用 `ceil` 扩大模型维度。
- 拆分权重、activation、KV Cache 和两个 AF 传输方向的 dtype，并统一 dtype-to-byte 解析；传输 dtype 与 activation dtype 不同时显式计入量化/反量化；兼容入口仅在未提供新字段时将旧 `af_dtype` 映射到各字段并输出弃用信息。
- 修正用于离线 operator-data generation 的 SwiGLU、GQA/MQA 和 sliding-window analytical 公式；运行时仍只允许 package exact match 或受限插值，缺失时明确失败。本变更覆盖标准 MHA/GQA/MQA 与 SwiGLU，不扩展 AF Prefill 的变长请求模型，也不新增 DeepSeek MLA/MFA 专用公式。
- 让 collective 的 operation、replica-local participants、显式 resource lane 和独立 communication fabric registry 真正决定 DAG、时延查询与资源竞争；planner 传递 logical bytes，由 provider 恰好一次推导 wire volume，缺少必要配置时显式失败，不再静默借用 AF 跨域链路。
- 本变更的目标配置与验收使用 `af.phases=["decode"]`，此时 Prefill 保持既有完整模型 aggregate 路径；既有 AF-Prefill capability 不做准确性重构，但增加适配公共 rank/dtype/query 契约的 compatibility adapter，避免因本次接口迁移而失效。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `af-execution`：补全 Decode AF 的逐层 Dense/MoE DAG、逐 token 路由、Shared Expert、A 侧 combine/residual、ownership return 与真实 collective 语义。
- `af-separation-config`：定义 rank-reuse EP、分离 dtype、严格 TP shape 和显式 collective fabric 的配置与校验规则。
- `af-runtime-state`：使组件内存、Expert residency、A TP activation ownership 和 dtype 记账与新执行语义一致。
- `operator-data-latency-provider`：补全 Dense/MoE/Shared/Combine/Residual 的互斥算子组成，并修正 dtype、SwiGLU、GQA/MQA 与 Decode window 公式。

## Impact

- 主要影响 `TokenSim/af/` 的 block 模型、runtime plan、planner、executor、trace 与通信资源解析。
- 影响 `TokenSim/moe/` 的 routing 输出和 Expert placement，以及 `TokenSim/operator_data/` 的 manifest、查询键、离线 analytical generator 和生成数据。
- 影响 AF 集群 JSON、benchmark dtype 参数、operator-data schema/package、runtime summary、示例与确定性 oracle；现有显式二维 F `TP * EP * DP` 配置必须迁移为 `EP=1` 或 rank-reuse `EP=TP*DP`。
- Decode-only AF 下 Prefill 的调度与 aggregate 计算路径保持不变；既有 AF-Prefill 和 AF-disabled 行为必须保持回归兼容。
