## Purpose

为 AF 多 rank 模拟建立与 execution domain 一致的模型内存和 KV Cache 状态，使 TP/DP/EP 不仅改变计算图，也正确改变每个 rank 的容量、分配、释放和可执行性判断。

## ADDED Requirements

### Requirement: AF execution domains determine runtime rank state

当 `af.enabled=true` 时，系统 MUST 以选定阶段的 A/F execution domains 作为 AF runtime rank 数量、DP/TP/EP membership、physical worker 和硬件容量的唯一依据。全局 `ParallelConfig` MUST NOT 覆盖或代替 AF domain-local 并行值。每个参与 AF 计算的 logical rank MUST 有且仅有一个对应的 runtime state。

#### Scenario: Domain TP overrides unrelated global TP
- **WHEN** 全局 `TP=1` 而 A execution domain 配置 `TP=2`
- **THEN** A replica 包含两个具有独立内存与 KV 记账的 runtime ranks
- **AND** KV 与模型分片计算使用 `TP=2`

#### Scenario: Every planned rank has runtime state
- **WHEN** planner 生成一个绑定 A 或 F logical rank 的 compute node
- **THEN** 该 rank 能唯一解析到具有相同 domain、DP/TP/EP 坐标和 physical worker 的 runtime state

### Requirement: Scheduling ownership is separate from TP rank state

每个 A DP replica MUST 只有一个明确的 scheduler owner，但该 replica 中的全部必要 A TP ranks MUST 保留各自的模型与 KV 分片状态。系统 MUST NOT 因非 leader rank 不运行 scheduler 而忽略该 rank 的内存和 KV 占用。

#### Scenario: One leader schedules for a two-rank TP group
- **WHEN** 一个 A DP replica 使用 `TP=2`
- **THEN** 一个稳定 leader 负责请求调度
- **AND** 两个 TP ranks 都保留自己的模型与 KV 状态

### Requirement: Model memory follows A F component ownership

系统 MUST 按 AF 组件所有权计算每个 rank 的驻留模型内存。A ranks MUST 计入其 Attention 及明确归属 A 侧的首尾组件，每个 A DP replica 的稳定 leader MUST 额外计入 Router；F ranks MUST 按 expert placement、TP shard 和非路由 FFN 复制策略计入实际驻留的 Expert/FFN 组件。任一 rank 无法容纳其驻留模型时，系统 MUST 在执行 AF iteration 前失败。

#### Scenario: Router state belongs to the A leader
- **WHEN** 模型包含 MoE Router 且 AF 将 Attention 与 FFN 分配到不同 domains
- **THEN** 每个 A DP replica 的稳定 leader 计入一份 Router 权重
- **AND** F ranks 不计入 Router 权重

#### Scenario: A rank does not reserve F expert weights
- **WHEN** 模型的 Attention 在 A domain、routed experts 在 F domain
- **THEN** A rank 的模型内存不包含 F 侧 routed expert 权重
- **AND** F rank 只计入 expert placement 和复制策略要求的权重

#### Scenario: F expert rank cannot fit its weights
- **WHEN** 一个 F rank 的实际驻留 Expert/FFN 权重超过其设备内存
- **THEN** 系统在产生 AF 执行事件前拒绝运行
- **AND** 错误包含 domain、rank、physical worker、所需内存与可用内存

### Requirement: KV heads follow A-domain TP partition rules

系统 MUST 根据 A execution domain 的 TP size 为每个 A rank 计算 local KV heads 与每 block 字节数。当 KV head 数不小于 TP size 时，KV heads MUST 可以在 TP ranks 之间整除分片；当 KV head 数小于 TP size 时，系统 MUST 使用无歧义的 KV-head 复制组，使每个 TP rank 至少保留一个 local KV head。无法整除分片或复制的组合 MUST 在模拟前失败。

#### Scenario: KV heads are sharded across TP ranks
- **WHEN** 模型有 8 个 KV heads 且 A domain 使用 `TP=2`
- **THEN** 每个 A TP rank 计入 4 个 local KV heads

#### Scenario: One KV head is replicated
- **WHEN** 模型有 1 个 KV head 且 A domain 使用 `TP=2`
- **THEN** 两个 A TP ranks 都计入 1 个 local KV head
- **AND** 该 KV head 被标记为复制而不是被错误算作两个全局 KV heads

#### Scenario: Ambiguous KV partition is rejected
- **WHEN** KV head 数与 A TP size 既无法整除分片也无法整除复制
- **THEN** 系统在创建请求调度器前拒绝该配置

### Requirement: Logical KV operations are atomic across an A TP group

对一个 A DP replica 的逻辑 KV block 执行 allocate、free、swap-in 或 swap-out 时，系统 MUST 将同一操作一致地应用到该 replica 的全部必要 A TP ranks。如果任一 rank 无法完成操作，整个逻辑操作 MUST 失败并恢复所有已更改 rank 的原状态。F domain MUST NOT 分配或拥有这些 KV blocks。

#### Scenario: Logical block allocates one shard per TP rank
- **WHEN** scheduler 在 `TP=2` 的 A replica 上分配一个新逻辑 KV block
- **THEN** 两个 A TP ranks 各增加一个对应 local KV block
- **AND** scheduler 只记录一次逻辑 block 分配

#### Scenario: Partial allocation rolls back
- **WHEN** 一个 TP rank 可分配而另一个必要 TP rank 已无可用 block
- **THEN** 逻辑 block 分配失败
- **AND** 所有 TP ranks 的已分配 block 数与失败前一致

#### Scenario: Free removes every local shard
- **WHEN** scheduler 释放一个已分配的逻辑 KV block
- **THEN** 该 block 在对应 A TP group 的所有 local shards 中被释放

### Requirement: A replica capacity is limited by its smallest TP rank

每个 A DP replica 的可调度逻辑 GPU/CPU KV block 容量 MUST 分别等于该 replica 所有必要 TP ranks 的对应 rank-local 可用 block 数的最小值。系统 MUST NOT 对 TP ranks 的 block 数求和或只使用 leader 容量。

#### Scenario: Heterogeneous TP ranks use the smaller capacity
- **WHEN** 同一 A TP group 的两个 ranks 分别可容纳 100 和 80 个 local GPU blocks
- **THEN** scheduler 看到的 replica 逻辑 GPU block 容量为 80
- **AND** 系统不将容量报告为 100 或 180

### Requirement: DP replicas keep independent KV state and stable ownership

每个 A DP replica MUST 拥有独立的逻辑 KV 容量、TP rank shards 和 request ownership。一个 request 在某 A replica 建立 KV 状态后，其后续计算 MUST 继续使用该 replica，除非执行了明确的 KV transfer。

#### Scenario: Two DP replicas allocate independently
- **WHEN** 两个 requests 分别归属 A DP replica 0 和 replica 1
- **THEN** 一个 replica 的 KV 分配或释放不改变另一个 replica 的 rank-local block 数

#### Scenario: Decode reuses the owning replica
- **WHEN** request 已在 A DP replica 1 的 TP ranks 上建立 KV shards
- **THEN** 后续 Decode 继续由 replica 1 调度和执行
- **AND** 系统不在没有 KV transfer 的情况下改用 replica 0

### Requirement: Runtime topology inconsistencies fail before execution

系统 MUST 在启动模拟事件循环前校验 scheduler owner、planner group membership、rank runtime state、KV shard membership、model component ownership 与 expert placement 的一致性。任一不一致 MUST 明确失败，且错误 MUST 包含 domain、replica/rank 和冲突事实。

#### Scenario: Planner group differs from runtime group
- **WHEN** planner 声明的 A TP participants 与对应 A replica 的 runtime ranks 不一致
- **THEN** 系统在产生 compute 事件前失败
- **AND** 错误列出期望与实际 participants

#### Scenario: AF disabled keeps baseline state
- **WHEN** `af.enabled=false`
- **THEN** 系统继续使用现有非 AF worker、全局 `ParallelConfig` 和 KV Cache 路径
- **AND** 系统不创建 AF rank runtime state
