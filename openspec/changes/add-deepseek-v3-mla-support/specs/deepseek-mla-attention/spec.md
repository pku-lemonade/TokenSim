## Purpose

为 DeepSeek‑V3 这类采用 Multi‑Head Latent Attention 的大规模 MoE 模型提供可配置、可验证且明确标注估算口径的 TokenSim decode 模拟能力。

## ADDED Requirements

### Requirement: MLA model metadata
系统 SHALL 接受 MLA 模型的 latent KV、rotary KV、query head 和压缩维度元数据，并校验维度为正且与 head 配置一致。

#### Scenario: valid DeepSeek metadata
- **WHEN** 用户提供 DeepSeek‑V3 的 MLA 元数据、256 experts 和 top-k 路由
- **THEN** 模型清单通过校验并可生成 decode execution plan

#### Scenario: invalid latent dimensions
- **WHEN** latent 或 rotary KV 维度缺失、为零或超过模型 hidden size
- **THEN** 系统 MUST 返回包含字段名的校验错误

### Requirement: MLA attention latency
系统 SHALL 为 context 和 generation Attention 生成包含 query projection、latent KV 读取和 Attention 核心工作量的 MLA 查询，并区分实测数据与 analytical 估算。

#### Scenario: offline analytical generation
- **WHEN** 尚无 MLA 实测数据且用户选择理论模拟
- **THEN** 系统 MUST 离线使用 MLA 专用公式生成 operator-data，并在 source metadata 中标记 analytical；运行时只查表

#### Scenario: exact operator data
- **WHEN** 存在匹配设备、dtype、batch、context 和 MLA 维度的 operator-data
- **THEN** 系统 MUST 优先使用 exact 数据且不得回退到普通 GQA 数据

### Requirement: MLA cache and A/F execution
系统 SHALL 将 MLA 压缩 KV cache 字节数用于 cache 显存估算，并允许 MLA Attention 与现有 MoE Router、Expert、Combine 的 A/F decode 流水组合。AF 通信继续传递 activation 和 expert partials，不将 latent cache 当作 AF payload。

#### Scenario: decode with DeepSeek MoE
- **WHEN** DeepSeek‑V3 MLA decode 使用 top-k experts 和 A/F 分离
- **THEN** execution trace MUST 包含 MLA Attention、dispatch、routed expert、combine 和 return stages

#### Scenario: unsupported exactness
- **WHEN** 用户请求精确 MLA 但没有 MLA operator-data
- **THEN** 系统 MUST 明确报告缺少数据，不得把普通 GQA 数据标为 MLA 实测

### Requirement: explicit execution boundary
系统 SHALL 区分 absorbed decode 与 expanded prefill，并声明支持范围与测量边界。

#### Scenario: unsupported prefix prefill
- **WHEN** 请求 cached、chunked 或 ragged MLA prefill
- **THEN** 系统 MUST 明确拒绝该 shape；完整 AF 调度验证限于 decode，prefill 验证等长完整单层 Attention

#### Scenario: replicated latent cache
- **WHEN** A 侧使用 TP=2、BF16、V3 latent=512 和 rope=64
- **THEN** 每个 A rank MUST 保留每层每 token 1152 字节的 cache，query heads 分片为 64
