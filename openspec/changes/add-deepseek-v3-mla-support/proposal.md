## Why

当前 TokenSim 仅接受 MHA/GQA/MQA，无法描述 DeepSeek-V3 的 MLA，导致 671B 级 MoE 架构无法建立可运行的 A/F decode 模拟。现有 DeepSeek-V3-like 元数据只覆盖专家部分，MLA 是继续做大模型架构推演的主要缺口。

## What Changes

- 增加 MLA 模型结构字段和校验路径。
- 增加 MLA decode/context Attention 的算子数据键与离线 analytical 生成器；运行时仅查表。
- 建模 compressed latent KV、rotary KV 和 query projection 的工作量及缓存字节数。
- 为 DeepSeek-V3 建立模型清单；硬件、固定功耗和虚拟卡数在实验配置中独立声明。
- 增加 MLA 单元测试、形状覆盖测试和端到端 decode smoke test。

## Capabilities

### New Capabilities
- `deepseek-mla-attention`: 描述并模拟 DeepSeek 风格 MLA 的结构、KV 缓存、Attention 延迟和 A/F 执行。

### Modified Capabilities
- `operator-data-latency-provider`: 扩展 Attention 查询和算子数据 schema 以覆盖 MLA。

## Impact

影响 `ModelShape`、operator-data schema/provider、analytical latency、AF planner 的 Attention 查询，以及模型配置和测试数据。普通 MHA/GQA/MQA 配置必须保持兼容；没有 MLA 专用测量数据时，系统应明确标记 analytical 估算而不是伪装为实测。
