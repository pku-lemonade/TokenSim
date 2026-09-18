# af-separation-config Specification

## Purpose

为 TokenSim 的 Prefill/Decode Attention-FFN 分离提供稳定、可校验且向后兼容的配置契约，使后续调度、通信竞争与异构设备建模能够依据同一份阶段、拓扑和资源定义运行，同时不改变未启用 AF 的现有模拟行为。

## Requirements

### Requirement: AF configuration defaults to disabled

系统 SHALL 在集群配置未提供 `af` 节时创建等价的默认 AF 配置，其中 `enabled=false`、`phases=["prefill", "decode"]`、`af_block_size=1`、`links=[]` 且 `trace_enabled=true`。默认配置 MUST 不改变现有 Prefill、Decode、KV Cache 或 worker 选择行为。

#### Scenario: Existing configuration omits AF section

- **WHEN** 用户加载一份不含 `af` 节的现有集群配置
- **THEN** 系统接受该配置并将 AF 视为关闭
- **AND** 系统保留原有非 AF 配置语义

#### Scenario: User explicitly disables AF

- **WHEN** 用户设置 `af.enabled=false`
- **THEN** 系统接受合法的其余集群配置
- **AND** 下游组件读取到的 AF 模式为关闭

### Requirement: AF phases are explicitly selected

当 `af.enabled=true` 时，系统 MUST 要求 `phases` 是由 `prefill` 和 `decode` 组成的非空、无重复集合。系统 SHALL 支持仅 Prefill、仅 Decode 或 Prefill 与 Decode 两阶段均启用 AF；阶段列表的顺序 MUST 不改变配置语义。

#### Scenario: Both phases use AF

- **WHEN** 用户配置 `phases=["prefill", "decode"]`
- **THEN** 系统将 Prefill 和 Decode 都标记为 AF 阶段

#### Scenario: Decode-only experiment is accepted

- **WHEN** 用户配置 `phases=["decode"]`
- **THEN** 系统将 Decode 标记为 AF 阶段
- **AND** Prefill 继续要求完整 Prefill 执行能力

#### Scenario: Empty or unknown phase is rejected

- **WHEN** 已启用 AF 的配置提供空阶段列表、重复阶段或 `prefill`/`decode` 以外的阶段
- **THEN** 系统拒绝该配置
- **AND** 错误信息标明 `af.phases` 的允许值

### Requirement: AF block size has phase-independent request semantics

系统 MUST 要求 `af_block_size` 是严格正整数且不能是布尔值，并 SHALL 将其解释为一个 AF micro-batch 最多包含的 request 数。Prefill block 的实际 `token_count` SHALL 是其中 request 的 Prefill 计算 token 数之和；Decode block 中每个 request 每轮贡献一个 token，因此 `token_count` SHALL 等于 `request_count`。第一版 MUST 不为满足 `af_block_size` 而拆分单个 Prefill request。

#### Scenario: Prefill block contains variable token counts

- **WHEN** 两个 Prefill request 被分入同一个 AF block
- **THEN** block 的 `request_count` 为 2
- **AND** block 的 `token_count` 为两个 request 的 Prefill 计算 token 数之和

#### Scenario: Decode block count is unambiguous

- **WHEN** 一个 Decode AF block 包含 4 个 request
- **THEN** block 的 `request_count` 和 `token_count` 均为 4

#### Scenario: Invalid block size is rejected

- **WHEN** 用户将 `af_block_size` 设置为 0、负数或布尔值
- **THEN** 系统拒绝该配置
- **AND** 错误信息标明 `af_block_size` 必须为正整数

### Requirement: Worker groups declare explicit capabilities

系统 SHALL 支持 worker group 使用唯一 `name` 和非空 `capabilities` 声明职责。允许的 capability MUST 限于 `prefill`、`decode_full`、`af_attention` 和 `af_ffn`；同一 group 内重复或未知 capability MUST 被拒绝。`num_workers` MUST 表示该 group 中可独立竞争的物理计算 slot 数量，而不是 A/F 共用的 TP/PP/DP/EP 总数；其值 MUST 为正整数，group 引用的默认网络 MUST 存在于集群网络配置中。对每个启用的 AF execution domain，其全部 logical rank identities 与该 domain worker group 的 physical workers MUST 形成严格一对一映射。

#### Scenario: Physical worker count is independent of parallel ranks

- **WHEN** A group 与 F group 分别声明 2 个和 4 个 physical workers，且各自 execution domain 分别生成并一对一映射 2 个和 4 个 logical rank identities
- **THEN** 系统保留 2 个和 4 个可竞争计算 slot
- **AND** 系统不将两组 physical worker 数相加后与任一 domain 的 world size 比较

#### Scenario: Shared complete Prefill and AF Attention group is accepted

- **WHEN** 一个 worker group 同时声明 `prefill` 和 `af_attention`
- **THEN** 系统保留两个 capability
- **AND** 是否在 Prefill 阶段使用 AF 只由 `af.phases` 和 link 绑定决定

#### Scenario: Unknown capability is rejected

- **WHEN** 一个 worker group 声明不在允许集合中的 capability
- **THEN** 系统拒绝该配置
- **AND** 错误信息包含 group 名称和未知 capability

#### Scenario: Duplicate group names are rejected

- **WHEN** 两个 worker group 使用相同的 `name`
- **THEN** 系统拒绝该配置
- **AND** 错误信息标明重复的 group 名称

### Requirement: Each selected phase has one split topology

当 `af.enabled=true` 时，每个所选阶段 MUST 恰好由一条 AF link 覆盖。该 link 的 `attention_group` 与 `ffn_group` MUST 引用不同 worker group，前者 MUST 声明 `af_attention`，后者 MUST 声明 `af_ffn`。每个端点 MUST 引用与其职责相符的 execution domain；系统 MUST 不根据 Ada、Groq 或其他硬件名称推断职责或 rank topology。

#### Scenario: Decode AF uses separate A and F domains

- **WHEN** Decode link 将 A group 和 F group 分别绑定到合法的 `decode_attention` 与 `decode_ffn` execution domain
- **THEN** 系统接受两个 domain 各自的 topology 与 device mapping
- **AND** A 与 F 的 worker 数 MAY 彼此不同，但每个 domain 内 logical rank identity 数 MUST 等于其映射的 physical worker 数

#### Scenario: One link covers both phases

- **WHEN** 一条合法 link 的 `phases` 同时包含 `prefill` 和 `decode` 且端点 domain 覆盖相应职责
- **THEN** 系统接受同一组 A/F 设备承担两个阶段

#### Scenario: Separate links cover separate phases

- **WHEN** 一条合法 link 只覆盖 Prefill，另一条合法 link 只覆盖 Decode
- **THEN** 系统接受两个阶段使用不同 A/F 设备池和 execution domains

#### Scenario: Selected phase has no link

- **WHEN** `af.phases` 包含某阶段但没有 link 覆盖该阶段
- **THEN** 系统拒绝该配置
- **AND** 错误信息标明缺少 link 的阶段

#### Scenario: Selected phase has ambiguous links

- **WHEN** 两条 link 同时覆盖同一个 AF 阶段
- **THEN** 系统拒绝该配置
- **AND** 错误信息说明第一版每阶段只允许一条 link

#### Scenario: Link endpoints do not provide required capabilities

- **WHEN** link 的 A 或 F 端点未声明相应 `af_attention` 或 `af_ffn` capability，或未绑定相符 execution domain
- **THEN** 系统拒绝该配置
- **AND** 错误信息包含 link、端点 group 和 domain 名称

### Requirement: Execution domains separate topology from physical placement

系统 MUST 将 Prefill、Decode A 和 Decode F 建模为独立 execution domains，并分别保存 logical instances、domain-local rank topology 与 rank-to-device mapping。每个 domain MUST 独立校验其 TP、PP、DP 以及适用的 EP 关系和 mapping 完整性；系统 MUST NOT 将一套 global rank 顺序隐式复用于全部 domains。参与本变更 Decode AF 的 A 与 F domain 均使用 `PP=1`，因此 A 与 F 的基础 world size 分别为各自的 `TP * DP`。同一 domain 内每个基础 rank MUST 恰好映射一个 physical worker；F rank-reuse EP identity MUST 叠加在这些基础 ranks 上，不得再乘出额外 workers。

#### Scenario: P, A, and F use different topologies

- **WHEN** Prefill、Decode A 和 Decode F domains 分别声明不同的 topology 和合法 device mapping
- **THEN** 系统逐 domain 接受并校验这些配置
- **AND** 任一 domain 的 rank 编号都只在该 domain 内解释

#### Scenario: F rank reuse does not allocate extra workers

- **WHEN** F domain 配置 `TP=2`、`DP=2` 并使用 rank-reuse EP
- **THEN** F 基础 world 和 physical runtime state 数均为 4
- **AND** 派生 EP identity 不创建第五个 state，也不要求 `TP * EP * DP` 个 workers

#### Scenario: Derived EP identity resolves to one base rank

- **WHEN** 系统编译 rank-reuse F domain
- **THEN** 每个 EP identity 唯一解析到一个已有 TP/DP rank 和 physical worker
- **AND** 一个 physical worker 只记账一次容量与资源竞争

#### Scenario: One physical device backs multiple logical ranks

- **WHEN** 一个 mapping 将同一 execution domain 内的多个基础 rank identities 映射到同一 physical worker
- **THEN** 系统在模拟启动前拒绝配置
- **AND** 错误列出冲突的 domain、rank 与 physical worker

#### Scenario: Domain mapping is incomplete

- **WHEN** 一个显式多 rank domain 缺少 rank、device 或并行组 mapping
- **THEN** 系统在模拟前拒绝配置并标明 domain 与缺失 mapping
- **AND** 系统不得按 worker group 顺序猜测 global ranks

#### Scenario: Legacy single-device configuration remains valid

- **WHEN** AF 被关闭或合法单设备 AF 配置未显式声明新 domain topology
- **THEN** 系统生成无歧义的单实例 domain view
- **AND** 既有单卡执行行为保持不变

### Requirement: AF parallel topology must be executable

系统 MUST 在仿真开始前将每个参与 AF 的 execution domain 解析为完整的 DP replica、TP group、适用的 rank-reuse EP group、logical rank、physical worker、tensor layout 与必要 collective plan。每个并行参数 MUST 对应可执行的分片、复制、路由或通信语义。参与本变更 Decode AF 的 domain 的 `PP` MUST 等于 1。除明确支持的 KV-head 复制外，每个实际由 TP 分片的维度 MUST 满足对应算子的整除规则；系统 MUST NOT 使用 `ceil` 隐式增加 heads、hidden、query width 或 intermediate width。

#### Scenario: Unsupported AF pipeline parallelism is rejected

- **WHEN** 任一参与 Decode AF 的 A 或 F execution domain 配置 `PP > 1`
- **THEN** 系统在创建 execution events 前拒绝配置
- **AND** 错误说明 AF micro-batch 流水与 PP 是不同概念且本版本只支持 `PP=1`

#### Scenario: Resolvable TP DP and EP topology is accepted

- **WHEN** Decode AF domains 使用 `PP=1`，且每个基础 rank 的 DP、TP、适用 EP、layout 与 physical worker 均可唯一解析
- **THEN** 系统接受配置并保留所有 group membership
- **AND** TP、DP 或 EP 均不退化为只参与数值校验的参数

#### Scenario: Ambiguous rank responsibility is rejected

- **WHEN** 一个 AF rank 缺少 DP replica、TP group、适用 EP group、layout 或 physical worker，或者同一 membership 有多个解析结果
- **THEN** 系统在仿真开始前拒绝配置
- **AND** 错误包含 domain、rank 与冲突维度

#### Scenario: Non-divisible tensor shape is rejected

- **WHEN** Attention heads、query width、hidden shard 或 FFN intermediate 等实际分片维度不满足配置的 TP 规则
- **THEN** 系统在创建 runtime plan 前拒绝配置
- **AND** 错误列出 domain、operator、global dimension、TP size 与整除规则

#### Scenario: Supported KV head replication remains valid

- **WHEN** KV head 数小于 A TP size 且 TP size 可由 KV head 数整除
- **THEN** 系统按明确复制组接受配置
- **AND** 不将该例外推广到其他 tensor dimensions

### Requirement: Expert placement resolves routed work to F ranks

启用 MoE 的 F execution domain MUST 使用以下两种且仅两种模式之一。`EP=1` 时，Routed Experts MUST 在所选 F DP replica 的 TP group 上执行，Expert TP 等于 F TP，并在 F DP replicas 之间复制 Expert 集合。rank-reuse 模式下，`EP` MUST 等于 F 基础 world size `TP * DP`，每个已有 F rank 对应一个 EP identity，Expert TP 等于 1，Routed Expert placement 跨整个 F world；该模式不得增加 physical workers。`1 < EP < TP*DP` 和 `EP > TP*DP` MUST 在本版本中失败。Dense FFN 与 Shared Expert MUST 独立使用所选 F DP replica 的 TP group。每个 Routed Expert MUST 映射到合法执行 owner；每个原始 token MUST 只选择一个 Shared Expert TP group。

#### Scenario: EP one uses expert tensor parallelism

- **WHEN** F domain 配置 `TP=2`、`DP=2`、`EP=1`
- **THEN** 一个 block 的 Routed Experts 在选中的两 rank F TP group 上执行且 Expert TP 为 2
- **AND** 两个 F DP replicas 各自持有同一完整 Expert 集合的 TP shards

#### Scenario: EP equals the reused F world

- **WHEN** F domain 配置 `TP=2`、`DP=2`、`EP=4`
- **THEN** 四个现有 F ranks 形成 rank-reuse EP group且 Expert TP 为 1
- **AND** 系统仍只使用四个 physical workers

#### Scenario: Intermediate EP size is rejected

- **WHEN** F domain 配置 `TP=2`、`DP=2`、`EP=2`
- **THEN** 系统在模拟前拒绝配置
- **AND** 错误列出仅支持 `EP=1` 或 `EP=TP*DP=4`

#### Scenario: Complete expert placement is accepted

- **WHEN** 模型中的每个 Routed Expert 都映射到当前 EP 模式允许的 F ranks，且一次 assignment 可唯一确定执行 owner 或 TP group
- **THEN** 系统接受 placement
- **AND** placement 可解析出全部 compute participants 与 physical resources

#### Scenario: Routed expert has no legal rank

- **WHEN** placement 缺少模型中的某个 Routed Expert，或引用 F domain 外 rank
- **THEN** 系统在执行 AF iteration 前失败
- **AND** 错误列出 Expert id、F domain 与无效 placement

#### Scenario: Shared Expert placement is executable once per token

- **WHEN** 模型声明一个或多个 Shared Experts
- **THEN** 配置解析出一个所选 F DP replica 的 TP execution group，并将多个 Shared Experts 表示为该 group 上扩大的 shared intermediate
- **AND** 每个原始 token 逻辑上执行一次 shared branch，而不是在所有 EP identities 上重复完整工作

### Requirement: Non-AF phases retain complete execution topology

当只对部分阶段启用 AF 时，系统 MUST 要求未选择阶段仍有完整执行能力。`af.phases=["decode"]` 时 Prefill MUST 使用既有完整模型 aggregate 路径，不创建 Decode A/F blocks 或跨域 transfers；结果 MUST 将该精度等级标为 `prefill_aggregate`。本变更 MUST NOT 删除或改变既有 AF-Prefill capability，只是不重构或验收其逐 rank 执行语义。

#### Scenario: Decode-only AF retains complete Prefill

- **WHEN** `af.phases=["decode"]`
- **THEN** 系统要求至少一个 worker group 声明 `prefill` 并通过既有完整 Prefill 路径计算
- **AND** Prefill 不使用 Decode AF planner、ownership returns 或 Decode AF links

#### Scenario: Prefill approximation is observable

- **WHEN** Decode 使用 rank-local AF DAG 而 Prefill 使用既有完整路径
- **THEN** 结果区分 `decode_af_rank_local` 与 `prefill_aggregate`
- **AND** 系统不把 Prefill aggregate latency 宣称为逐 rank resource trace

#### Scenario: Prefill-only AF retains complete Decode

- **WHEN** `af.phases=["prefill"]`
- **THEN** 系统要求至少一个 worker group 声明 `decode_full`
- **AND** 既有 AF-Prefill capability 仍可按迁移后的合法配置被选择

### Requirement: AF links explicitly identify communication resources

每条 AF link MUST 提供唯一 `name`、非空 `phases`、`attention_group`、`ffn_group`、现有 `network` 引用、非空 `resource_group`、`duplex` 以及相应容量。`duplex` MUST 仅为 `full` 或 `half`；所有出现的容量 MUST 是严格正整数且不能是布尔值。

#### Scenario: Full-duplex link is accepted

- **WHEN** link 使用 `duplex="full"` 并提供 `a_to_f_capacity` 与 `f_to_a_capacity`
- **THEN** 系统保留两个方向各自的并发容量

#### Scenario: Half-duplex link is accepted

- **WHEN** link 使用 `duplex="half"` 并提供 `half_duplex_capacity`
- **THEN** 系统将该容量归一化为两个方向共享的并发容量

#### Scenario: Missing active capacity is rejected

- **WHEN** link 缺少当前双工模式所需的容量
- **THEN** 系统拒绝该配置
- **AND** 错误信息标明 link 名称和缺失字段

#### Scenario: Unknown network is rejected

- **WHEN** link 引用未在集群 `networks` 中定义的网络
- **THEN** 系统拒绝该配置
- **AND** 错误信息包含 link 名称和网络名称

### Requirement: Resource groups define communication competition

系统 SHALL 将相同 `resource_group` 的 links 归一化为同一物理通信资源。相同 resource group 的 links MUST 使用相同的 `network`、`duplex` 和对应容量，否则系统 MUST 拒绝配置。全双工 resource group SHALL 形成相互独立的 A→F 与 F→A 资源键；半双工 resource group SHALL 形成两个方向共享的单一资源键。不同 resource group SHALL 被视为独立资源。

#### Scenario: Full-duplex links share directional competition domains

- **WHEN** 多条全双工 link 使用相同 `resource_group`
- **THEN** 它们的所有 A→F 传输映射到同一个 A→F 资源键
- **AND** 它们的所有 F→A 传输映射到另一个 F→A 资源键

#### Scenario: Half-duplex links share one competition domain

- **WHEN** 多条半双工 link 使用相同 `resource_group`
- **THEN** 它们两个方向的传输均映射到同一个共享资源键

#### Scenario: Shared resource definitions conflict

- **WHEN** 相同 `resource_group` 的 links 声明不同网络、双工模式或容量
- **THEN** 系统拒绝该配置
- **AND** 错误信息标明冲突的 resource group 和 links

### Requirement: Communication timing separates queue and service semantics

配置产生的通信资源描述 MUST 允许后续执行器分别记录 `queue_duration` 与 `service_duration`。其中服务时间由 transfer provider 给出并表示固定时延与序列化时间，排队时间由申请共享资源到获得资源的间隔产生；端到端通信阶段时间 SHALL 为两者之和，链路利用率 MUST 只统计服务时间。

#### Scenario: Transfer waits for a busy resource

- **WHEN** 一个传输在 ready 后等待相同资源键上的前序传输释放资源
- **THEN** 其 `queue_duration` 为获得资源时间减去 ready 时间
- **AND** 其通信阶段总时间为 `queue_duration + service_duration`

#### Scenario: Transfer acquires an idle resource

- **WHEN** 一个传输在 ready 时立即获得资源
- **THEN** 其 `queue_duration` 为 0
- **AND** 其通信阶段总时间等于 `service_duration`

### Requirement: Legacy worker roles remain compatible

系统 SHALL 按当前代码的真实 role 将旧格式 worker group 归一化：`prefill` 映射为 `prefill`，`decode` 映射为 `decode_full`，`hybrid` 映射为 `prefill` 与 `decode_full`。未显式提供 `af.enabled=true` 的旧配置 MUST 保持 AF 关闭；若同一 group 同时提供旧 `role` 和新 `capabilities`，两者的完整阶段 capability MUST 一致，否则系统 MUST 拒绝配置。

#### Scenario: Existing prefill and decode groups load unchanged

- **WHEN** 用户加载仅使用 `prefill` 和 `decode` role 的现有配置
- **THEN** 系统接受该配置并生成对应 capability 视图
- **AND** AF 保持关闭

#### Scenario: Existing hybrid group loads unchanged

- **WHEN** 用户加载使用 `hybrid` role 的现有配置
- **THEN** 系统将其归一化为 `prefill` 与 `decode_full`

#### Scenario: Legacy and explicit declarations conflict

- **WHEN** 同一 worker group 的旧 role 与显式完整阶段 capabilities 表达不同职责
- **THEN** 系统拒绝该配置
- **AND** 错误信息标明冲突的 group

### Requirement: AF block size is distinct from KV block size

系统 MUST 将 `af_block_size` 解释为 AF micro-batch 的 request 上限，并 MUST 将现有 `block_size` 继续解释为 KV Cache token block 大小。配置解析和对外配置对象 MUST 保留两个独立值，不得互相覆盖或将 `af_block_size` 传递为 KV Cache block 大小。

#### Scenario: Both block sizes are configured

- **WHEN** 用户同时设置不同的 `af_block_size` 和 KV `block_size`
- **THEN** 系统分别保留两个值
- **AND** KV Cache 配置仍使用 KV `block_size`

#### Scenario: Only legacy KV block size is configured

- **WHEN** 用户只通过现有配置或 CLI 设置 `block_size`
- **THEN** 系统继续将其解释为 KV Cache block 大小
- **AND** `af_block_size` 保持自身默认值

### Requirement: Invalid or unsupported AF configuration fails before simulation

系统 MUST 在启动模拟事件循环前完成 AF 字段、阶段、拓扑、EP 模式、TP shape、dtype、Expert/component placement、tensor ownership、collective participants、network、resource group 和 resource lane 校验，并使用项目统一配置错误类型报告可操作上下文。合法且已启用的 AF 配置在 executor 与所需 providers 可用时 SHALL 进入 AF 路径；缺少必要依赖时 MUST 明确失败，不得静默退回完整 iteration latency。

#### Scenario: Configuration has multiple AF errors

- **WHEN** 配置同时包含非法容量和缺失阶段 link
- **THEN** 系统在模拟开始前拒绝配置
- **AND** 错误至少标明一个可操作字段或拓扑问题

#### Scenario: Valid AF configuration reaches a supported runtime

- **WHEN** 合法 Decode AF 配置被交给具有 executor、model metadata 和所需 providers 的入口
- **THEN** 系统创建对应 Decode AF 执行路径
- **AND** 系统不再以“AF executor 尚未实现”为由拒绝运行

#### Scenario: Valid AF configuration reaches a runtime without AF executor

- **WHEN** 合法且已启用的 AF 配置缺少 executor、provider、dtype、collective fabric 或必要模型元数据
- **THEN** 系统在产生 AF events 前拒绝运行并标明缺失依赖
- **AND** 系统不得使用完整 iteration backend 静默继续

#### Scenario: Orthogonal F TP EP DP mapping is rejected

- **WHEN** 配置或 mapping 试图按 `TP * EP * DP` 创建 F workers
- **THEN** 系统拒绝该配置并指出本版本的两种 EP 模式
- **AND** 不进入 runtime state 编译

### Requirement: AF dtypes describe distinct storage and transfer roles

系统 MUST 分别解析effective `weight_dtype`、`activation_dtype`、`kv_cache_dtype`、`a_to_f_dtype`与`f_to_a_dtype`。解析器 MUST 在填充defaults前记录raw field presence：显式legacy `af_dtype`与任一显式新role field混用 MUST 失败；只提供legacy字段时 MUST 将其映射到五个roles并产生可观测弃用信息；无legacy字段时，weight/activation MUST 缺省继承现有model dtype，KV MUST 缺省继承现有KV config或activation dtype，两个AF link directions MUST 缺省继承activation dtype。Operator manifest MUST 声明与最终effective compute/storage roles一致的数据覆盖，但不得覆盖runtime配置。所有roles MUST 通过同一个canonical dtype registry解析。

#### Scenario: Directional transfer dtypes differ

- **WHEN** `a_to_f_dtype=fp8`、`f_to_a_dtype=bf16`
- **THEN** 两个方向分别按 1 byte 与 2 bytes 每元素计算 payload
- **AND** weight、activation 与 KV Cache 字节数不被 transfer dtype 覆盖

#### Scenario: Transfer dtype requires conversion

- **WHEN** 任一方向的 transfer dtype 与 `activation_dtype` 不同
- **THEN** execution plan 在发送端和接收端生成显式 encode/decode compute operations
- **AND** 缺少任一转换 estimate 时在执行前失败，不得假定转换为零成本

#### Scenario: New style defaults resolve after presence checks

- **WHEN** raw input既没有legacy `af_dtype`也没有显式directional transfer dtypes，但model dtype为BF16
- **THEN** weight/activation继承model dtype，KV按KV config或BF16解析，两个transfer roles继承BF16
- **AND** default填充不会被误判为用户混用了legacy与新字段

#### Scenario: Manifest must match effective runtime roles

- **WHEN** 最终`activation_dtype=bf16`但operator manifest只声明该model使用FP16 activation keys
- **THEN** 系统在provider查询前失败并报告model、role、effective值与manifest值
- **AND** manifest不得把runtime activation静默改成FP16

#### Scenario: Legacy dtype maps consistently

- **WHEN** 输入只提供旧 `af_dtype=bf16`
- **THEN** 五个 effective roles 均解析为 BF16
- **AND** runtime summary 记录兼容映射与弃用状态

#### Scenario: FP8 has one canonical byte size

- **WHEN** FP8 用于任一受支持 role
- **THEN** 配置、memory ledger、operator query 与 payload calculation 均从同一 registry 得到每元素 1 byte

#### Scenario: Mixed legacy and explicit dtypes are rejected

- **WHEN** 输入同时设置旧 `af_dtype` 与任一新 dtype role
- **THEN** 系统在模拟前失败并要求使用单一配置形式

### Requirement: Required collectives have explicit executable plans

当Decode AF DAG的tensor layout需要collective时，对应execution domain MUST 提供唯一匹配的communication plan。该plan MUST 明确operation、触发位置、replica-local `tp_rank` participants，以及communication fabric registry中的resource group和resource lane；operation MUST 至少区分`all_reduce`、`all_gather`和`reduce_scatter`。Participants MUST 在每个选中DP instance内映射为actual domain ranks，operation MUST 进入provider query，精确的`(resource_group, resource_lane)` MUST 决定竞争。Fabric registry entry MUST 声明network、duplex、命名lane及各自capacity；AF links按方向/duplex解析lane，collective必须显式选择lane。Full-duplex group的两个directional lanes相互独立，collective只与自己选择的lane竞争；需要与两个AF方向共同竞争时必须配置half-duplex/shared lane。Collective可引用独立entry，也可显式引用与AF link相同entry。Planner MUST 为collective query提供每rank logical input/output bytes，不得预先乘算法wire-volume系数。缺少必要plan/fabric/lane时系统 MUST 失败，不得静默使用完整TP group、通用`tp_collective`或AF跨域link。

#### Scenario: Collective participants select a subset

- **WHEN** `TP=4,DP=2`的plan配置replica-local participants `[0, 2]`
- **THEN** DP0实例化为domain ranks `[0,2]`，DP1实例化为`[4,6]`
- **AND** 同一instance中的TP ranks 1和3不申请该collective资源

#### Scenario: Collective operation changes the query

- **WHEN** 两份配置仅在 `all_gather` 与 `reduce_scatter` 上不同
- **THEN** provider 收到不同 operation keys
- **AND** trace 保留各自 operation 名称

#### Scenario: Missing collective fabric fails

- **WHEN** 一个必要 collective 没有可解析的 network、resource group 或 resource lane
- **THEN** 系统在运行事件循环前失败
- **AND** 不自动改用 A-to-F 或 F-to-A resource

#### Scenario: Independent collective fabric has executable capacity

- **WHEN** collective引用一个未被AF link使用且network、shared lane与capacity均合法的fabric registry entry
- **THEN** 系统为其创建独立contention resource
- **AND** 该collective不与AF transfers竞争

#### Scenario: Collective explicitly shares the AF fabric

- **WHEN** collective plan与AF link显式引用同一resource group和同一resource lane
- **THEN** collective legs与该lane上的AF transfer legs按capacity共同竞争
- **AND** 系统不创建名称相同但彼此独立的resource

#### Scenario: Full-duplex collective lane is not inferred

- **WHEN** collective引用具有`a_to_f`和`f_to_a`两个lanes的full-duplex AF fabric
- **THEN** config必须显式选择其中一个resource lane，否则在event loop前失败
- **AND** 选择`a_to_f`时collective只与A-to-F竞争，F-to-A仍可按自身lane容量并行

#### Scenario: Collective payload remains logical at the planner boundary

- **WHEN** planner为四rank all-reduce生成query
- **THEN** query携带每rank logical input/output bytes、operation和participants
- **AND** ring/tree等wire-volume系数只由provider应用一次

#### Scenario: TP one needs no artificial collective

- **WHEN** 对应 execution group 的 `TP=1` 且 layout 不要求其他通信
- **THEN** 系统不要求配置或生成零意义 collective
