## Purpose

为 DeepSeek-V3 671B 提供一个可被新配置目录发现、可由 operator-table 估算器定价、并能运行完整 AF decode 的稳定适配层，使 MLA、Dense/MoE 层和模型边界操作都能在同一条新运行时路径中被验证和 profiling。

## ADDED Requirements

### Requirement: DeepSeek-V3 ModelSpec is discoverable and complete

系统 SHALL 在标准模型目录中提供一个有效的 DeepSeek-V3 `ModelSpec` YAML。加载模型目录后，调用方 SHALL 能通过 `deepseek-v3-671b-mla` 及其声明的别名解析该模型；模型规格 MUST 包含 7168 hidden size、61 layers、128 attention heads、129280 vocabulary、BF16 compute/KV dtype、前三层 Dense、其余层 MoE、256 routed experts、top-8 routing、1 shared expert、18432 dense intermediate size 和 2048 routed intermediate size。

#### Scenario: Catalog loads DeepSeek-V3

- **WHEN** 系统从标准 `data/models` 目录创建模型 catalog
- **THEN** catalog 包含 `deepseek-v3-671b-mla`
- **AND** 通过 model id 或声明的 alias 得到同一个模型规格
- **AND** 规格的关键层数、宽度、MoE 和 dtype 字段与 DeepSeek-V3 配置一致

#### Scenario: Invalid V3 shape is rejected

- **WHEN** V3 YAML 缺少必需字段、包含非正维度，或 TP 所需的 attention/FFN 维度不满足整除约束
- **THEN** 模型加载在仿真前失败
- **AND** 错误包含模型 id、字段和违反的约束

### Requirement: MLA dimensions are represented explicitly

ModelSpec SHALL 显式表达 MLA attention 类型及 `kv_lora_rank`、`q_lora_rank`、`rope_head_dim`、`nope_head_dim` 和 `v_head_dim`。MLA 的 shape、KV cache ledger 和 attention operator query MUST 使用这些字段；系统 MUST NOT 将 MLA 当作普通 MHA 并用 `num_key_value_heads * head_dim` 推导 KV 宽度。

#### Scenario: MLA query uses latent dimensions

- **WHEN** DeepSeek-V3 decode 为一个 A rank 生成 attention query
- **THEN** query 保留 MLA attention type 和对应 latent/rope/nope/value dimensions
- **AND** KV cache bytes 与 attention shape 按 MLA latent KV 表示计算

#### Scenario: Unsupported MLA metadata fails explicitly

- **WHEN** MLA 模型缺少任一必需 low-rank 或 head-dimension字段
- **THEN** 模型解析或 runtime plan 创建失败
- **AND** 系统不静默退回普通 MHA shape

### Requirement: Full V3 AF decode uses the current operator-table runtime

系统 SHALL 提供完整 DeepSeek-V3 decode 入口，覆盖 embedding、每一层 attention、前三层 Dense FFN、其余层 Router/routed/shared MoE、residual/weighted combine、final norm、LM head 和 token selection。所有 compute 与 communication service time MUST 来自当前 operator-table/硬件上下文 provider，运行入口 MUST 不依赖已删除的旧 operator-data provider API。

#### Scenario: Full decode completes with new providers

- **WHEN** 用户以合法 AF cluster、DeepSeek-V3 ModelSpec、HardwareContext 和可用 operator package 运行一个或多个 decode step
- **THEN** 仿真完成并返回 makespan、阶段统计和 runtime plan
- **AND** trace 包含全部 61 个 transformer layers 及模型边界操作
- **AND** Dense 与 MoE layer 的执行类型与模型规格一致

#### Scenario: MLA-only smoke run uses the same model contract

- **WHEN** 用户运行 MLA 单层或单卡 smoke 入口
- **THEN** 入口解析同一个 DeepSeek-V3 ModelSpec 和 HardwareContext
- **AND** 其 shape 与完整 V3 decode 的 MLA shape 一致

### Requirement: Operator provenance and missing data are observable

V3 运行结果 SHALL 保留每个 compute/transfer estimate 的 source id、dataset version、match type 和 operator breakdown，并在 profiling 文件中记录 service duration 与 queue duration。若 operator-table 没有 exact 或允许的插值记录，系统 MUST 在对应 query 处明确失败并报告完整 key；不得调用旧 provider、静默使用完整 iteration latency 或隐藏缺失 shape。

#### Scenario: Profile reconstructs service time

- **WHEN** 完整 V3 decode 成功运行并启用 profiling
- **THEN** 输出包含 gzip/JSON trace 或等价 profiling 文件
- **AND** 每个 operator event 可追溯到来源、匹配类型、service duration 和 queue duration
- **AND** 阶段汇总可由 trace event 重建

#### Scenario: Missing operator row fails with context

- **WHEN** 某个 MLA、Dense、Router、Expert、boundary 或 communication query 没有合法 operator-table 记录
- **THEN** 仿真在产生不完整结果前失败
- **AND** 错误列出 model、layer、domain/rank、operation、shape 和 lookup key

### Requirement: Existing execution modes remain compatible

适配 SHALL 保持 AF-disabled 路径、非 V3 模型和 decode-only 约束的既有语义。V3 入口 MUST 明确声明当前支持的阶段；未实现的 checkpoint loading、prefill 逐 rank execution 或 MTP 不得被伪装成已支持。

#### Scenario: AF disabled remains unchanged

- **WHEN** 现有模型以 `af.enabled=false` 运行
- **THEN** 系统不创建 V3-specific AF plan 或 provider query
- **AND** 原有结果路径保持可用

#### Scenario: Unsupported phase is rejected clearly

- **WHEN** 用户要求当前 V3 入口尚未支持的 prefill/checkpoint/MTP 执行
- **THEN** 入口在开始仿真前明确报告不支持的阶段
- **AND** 不返回仅 decode 结果并标记为完整 serving 性能
