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

系统 MUST 将 Prefill、Decode A 和 Decode F 建模为独立 execution domains，并分别保存 logical instances、domain-local rank topology 与 rank-to-device mapping。每个 domain MUST 独立校验其 TP、PP、DP 以及适用的 EP 关系和 mapping 完整性；系统 MUST NOT 要求所有 P/A/F physical workers 数之和等于 `TP * PP * DP`，也 MUST NOT 将一套 global rank 顺序隐式复用于全部 domains。每个 `(logical_instance, domain_rank)` identity MUST 映射到唯一 physical worker，同一 domain 内一个 physical worker MUST NOT 被多个 logical rank identities 复用，且该 domain worker group 的每个 physical worker MUST 恰好被映射一次。EP SHALL 描述 F domain 内 expert placement 或 expert communication group，不得被无条件当作额外 physical worker 乘数。

#### Scenario: P, A, and F use different topologies

- **WHEN** Prefill、Decode A 和 Decode F domains 分别声明不同的 logical instance 数、TP/PP/DP/EP 配置和合法 device mapping
- **THEN** 系统逐 domain 接受并校验这些配置
- **AND** 任一 domain 的 rank 编号都只在该 domain 内解释

#### Scenario: One physical device backs multiple logical ranks

- **WHEN** 一个 mapping 将同一 execution domain 内的多个 logical rank identities 映射到同一 physical device 或 compute slot
- **THEN** 系统在模拟启动前拒绝该配置
- **AND** 错误信息列出冲突的 domain、logical instance、domain rank 与 physical worker

#### Scenario: Domain mapping is incomplete

- **WHEN** 一个显式多 rank domain 缺少 rank、device 或并行组映射，导致拓扑不能无歧义解析
- **THEN** 系统在模拟前拒绝配置并标明 domain 与缺失 mapping
- **AND** 系统不得按 worker group 顺序猜测 global ranks

#### Scenario: Legacy single-device configuration remains valid

- **WHEN** AF 被关闭或合法单设备 AF 配置未显式声明新 domain topology
- **THEN** 系统生成无歧义的单实例 domain view
- **AND** 既有单卡执行行为保持不变

### Requirement: AF parallel topology must be executable

系统 MUST 在仿真开始前将每个参与 AF 的 execution domain 解析为完整的 DP replica、TP group、适用的 EP group、logical rank 与 physical worker 映射。每个并行参数 MUST 对应运行时可执行的分片、复制、路由或通信语义；系统 MUST NOT 接受仅通过数值乘积或 rank 数量校验、但无法确定每个 rank 职责的配置。第一阶段 AF execution domain 的 `PP` MUST 等于 1，任何 `PP > 1` 的 AF 配置 MUST 明确失败。

#### Scenario: Unsupported AF pipeline parallelism is rejected

- **WHEN** 任一参与 AF 的 A 或 F execution domain 配置 `PP > 1`
- **THEN** 系统在创建执行事件前拒绝该配置
- **AND** 错误信息标明 domain、`PP` 值以及第一阶段仅支持 `PP=1`

#### Scenario: Resolvable TP DP and EP topology is accepted

- **WHEN** AF domains 使用 `PP=1`，且每个 logical rank 的 DP replica、TP group、适用的 EP group和 physical worker 均可唯一解析
- **THEN** 系统接受该配置并保留所有 group membership
- **AND** 系统不把 TP、DP 或 EP 降级为只参与 world-size 校验的数字

#### Scenario: Ambiguous rank responsibility is rejected

- **WHEN** 一个 AF logical rank 缺少 DP replica、TP group、适用的 EP group或 physical worker 信息，或同一 membership 可被解析为多个结果
- **THEN** 系统在仿真开始前拒绝配置
- **AND** 错误信息包含 domain、rank 与缺失或冲突的并行维度

### Requirement: Expert placement resolves routed work to F ranks

启用 MoE 的 F execution domain MUST 提供与模型 expert 集合一致的 expert placement，使每个 Router/Top-K 输出的 expert id 都能解析到合法的 F logical rank 和 physical worker。EP MUST 表示 expert 到 rank 的放置和域内 dispatch/combine 参与关系，不得被无条件解释为额外 worker 乘数。系统 MUST 在仿真前拒绝缺失 expert、未知 expert、空 placement 或引用 domain 外 rank 的配置。

#### Scenario: Complete expert placement is accepted

- **WHEN** 模型中的每个 expert 都映射到 F domain 内至少一个合法 rank，且所有映射均可解析到 physical worker
- **THEN** 系统接受 placement
- **AND** 后续 routing result 可唯一确定每个 expert 分支的候选执行 rank

#### Scenario: Routed expert has no legal rank

- **WHEN** placement 缺少模型中的某个 expert，或将其映射到 F domain 外的 rank
- **THEN** 系统在执行 AF iteration 前失败
- **AND** 错误信息列出 expert id、F domain 与无效 placement

### Requirement: Non-AF phases retain complete execution topology

当只对部分阶段启用 AF 时，系统 MUST 要求未选择阶段仍有完整执行能力：未选择 Prefill 时至少存在一个 `prefill` group，未选择 Decode 时至少存在一个 `decode_full` group。

#### Scenario: Decode-only AF retains complete Prefill

- **WHEN** `af.phases=["decode"]`
- **THEN** 系统要求至少一个 worker group 声明 `prefill`

#### Scenario: Prefill-only AF retains complete Decode

- **WHEN** `af.phases=["prefill"]`
- **THEN** 系统要求至少一个 worker group 声明 `decode_full`

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

系统 MUST 在启动模拟事件循环前完成 AF 字段、阶段、拓扑、link 和共享资源校验，并使用项目统一的配置错误类型报告可操作的字段或对象上下文。合法且已启用的 AF 配置在 executor 与所需 latency providers 可用时 SHALL 进入 AF 执行路径；若运行环境缺少 executor、provider 或必要模型元数据，系统 MUST 在产生 AF 事件前明确报告缺失依赖，不得静默退回非 AF 完整模型时延。

#### Scenario: Configuration has multiple AF errors

- **WHEN** 配置同时包含非法容量和缺失的阶段 link
- **THEN** 系统在模拟开始前拒绝配置
- **AND** 返回的错误信息至少标明一个可操作的字段或拓扑问题

#### Scenario: Valid AF configuration reaches a supported runtime

- **WHEN** 合法且已启用的 AF 配置被交给具有 executor 和所需 providers 的运行入口
- **THEN** 系统创建对应阶段的 AF 执行路径
- **AND** 系统不再以“AF executor 尚未实现”为由拒绝运行

#### Scenario: Valid AF configuration reaches a runtime without AF executor

- **WHEN** 合法且已启用的 AF 配置缺少 executor、provider 或必要模型元数据
- **THEN** 系统在产生 AF 事件前拒绝运行并标明缺失依赖
- **AND** 系统不得使用 Roofline、LLMCompass 或其他完整 iteration 后端静默模拟 AF
