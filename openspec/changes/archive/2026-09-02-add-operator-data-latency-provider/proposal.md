## Why

当前 AF executor 已能正确模拟计算、通信资源竞争与流水并行，但正式运行仍只能依赖调用方注入的 stub latency provider，无法使用 Ada/Groq 的可追溯算子数据评估物理时延。现在需要建立“实测 kernel 优先、校准公式补齐”的离线数据生成与确定性运行时查询链路，使 AF 分离实验既能端到端运行，也能说明每个 service time 的来源、匹配方式和组合构成。

## What Changes

- 新增版本化 operator-data package，定义硬件、算子性能、分析明细、shape manifest 与 generation metadata 的 schema、单位和一致性校验。
- 新增离线数据生成流程：优先采用可追溯实测样本，以 FLOPs、内存搬运、硬件参数和校准系数补齐覆盖范围，并严格隔离 calibration 与 holdout 数据。
- 新增 typed runtime lookup：支持 exact match 和按算子类别约束的单维插值；第一版禁止外推，缺失、越界、重复键或非法数据均明确失败。
- 新增 A/F operator composition：分别组合 Attention 与 FFN/MoE 的必要算子，并通过 breakdown 规则防止 QKV、输出投影、Router、dispatch/combine、softmax 和 SwiGLU 重复计时。
- 新增通信 service provider：按可追溯的固定延迟、有效带宽和 payload bytes 估算 A-to-F/F-to-A service time；链路竞争等待仍完全由 AF executor 产生。
- 扩充 provider 与 trace 的可追溯元数据，保留 dataset、source、match type 与 composition breakdown，但这些元数据不得参与资源选择或调度。
- 新增以真实生成数据包驱动的端到端验证和按算子类别输出的 coverage、误差分位数、bias、bottleneck consistency 与 worst-shape 报告；无目标硬件 ground truth 时只能声明模型自洽。
- 借鉴 AIConfigurator 的分表查询、来源解析、受控插值和算子组合，以及 InferSim 的公式拆解、通信量与 benchmark 校准；TokenSim 运行时不依赖两个参考仓库，也不沿用其设备专属经验常数作为 Groq 真值。

## Capabilities

### New Capabilities

- `operator-data-latency-provider`: 定义 operator-data package、离线生成与校准、运行时匹配和插值、A/F 算子组合、通信 service 估算、来源追踪及精度验证行为。

### Modified Capabilities

- `af-execution`: 扩充正式 provider 结果与 AF trace 的 provenance、match 和 composition 元数据契约，同时保持 service time 与 queue time 的职责边界。

## Impact

- 影响 `TokenSim/af/latency.py` 的 query/result 接口及其正式 compute/transfer provider 实现、AF trace 序列化和配置装配入口。
- 新增离线 operator-data 生成、校准、验证工具及版本化数据包；运行时需要明确选定并预检 Parquet reader 依赖，不能静默假设 pandas/pyarrow 已存在。
- 新增 Ada/Groq 硬件与算子数据接入点，但不会生成或伪造目标硬件 ground truth；公开或估算数据必须标注证据等级和来源。
- AF 关闭时的既有 Prefill/Decode 路径、已有 Roofline/LLMCompass backend，以及使用 7/10 ms stub 的 executor 基线保持不变。
