## Why

TokenSim 当前只能通过 `role` 描述 Prefill/Decode 部署，无法表达 Attention 与 FFN 分别由不同设备池承担，也没有 AF 流水阶段、粒度、链路端点和竞争资源的配置契约。需要先建立可校验且向后兼容的 AF 配置层，为后续 Prefill/Decode AF 调度、通信和异构设备建模提供稳定入口。

## What Changes

- 新增默认关闭的 `AFConfig`，配置 AF 作用阶段（Prefill、Decode 或两者）、AF block 大小、显式 A/F links 以及时间线开关。
- 为 worker group 新增稳定的 `name` 与 `capabilities`，用 `prefill`、`decode_full`、`af_attention`、`af_ffn` 明确完整阶段与 AF 计算职责，不再为 AF 扩张旧 `role` 枚举。
- 新增显式 AF link 配置，绑定 A/F worker group、网络性能项和物理 `resource_group`，并描述全/半双工及方向容量，使同一通信资源上的排队竞争无歧义。
- 在加载集群配置时校验 AF 字段、所选阶段所需的完整/拆分设备池、link 端点、网络引用和共享资源一致性；AF 关闭时保持现有执行语义。
- 保留旧 `role` 配置的兼容转换，但旧配置不得隐式启用 AF；继续将现有 `block_size` 解释为 KV Cache block，避免与 `af_block_size` 混用。
- 提供“Prefill+Decode 均 AF”和“仅 Decode AF”配置示例，以及针对默认值、阶段组合、link 竞争、非法拓扑和旧配置兼容性的测试。
- 本 change 仅建立配置与校验契约，不实现 AF block 执行器、调度状态机、通信流水、时延 provider，也不填入尚未提供的 Ada/Groq 性能数值。
- 后续 AF 时延模型采用“离线生成、运行时分表查询”：按 GEMM、Prefill Attention、Decode Attention、MoE 和逐元素算子生成带来源的 Parquet 数据，TokenSim 仅执行精确查询或受控一维插值，不在运行时静默执行公式 fallback 或外推。旧 Roofline/LLMCompass 只保留给非 AF 基线。本 change 只固定该扩展边界，不实现数据生成器或查询接口。
- 记录下游 AF 验证契约：以确定性端到端 oracle 验证流水机制，以可获得的算子/层级实测、公开模型基准、通信数据和真实系统 trace 分层校验量级、趋势与物理精度，并通过来源等级和敏感性分析限制结论；本 change 不实现测试 harness、数据采集或校准。

## Capabilities

### New Capabilities

- `af-separation-config`: 定义 Prefill/Decode AF 分离的阶段选择、worker capability 拓扑、显式通信资源、兼容行为和失败校验。

### Modified Capabilities

无。

## Impact

- `TokenSim/config/cluster_config.py`: 增加 AF 配置、worker capability 模型、兼容转换与跨字段校验。
- `TokenSim/config/config.py`: 导出新增配置类型和 capability 常量。
- `benchmark.py`: 接收并向模拟器配置层传递 AF 配置，同时保持既有 KV block 参数语义。
- `tests/`: 增加 AF 配置解析、验证和旧配置回归测试。
- `data/clusters/`: 增加不含虚构 Ada/Groq 性能参数的两阶段 AF 与 Decode-only AF 拓扑示例。
