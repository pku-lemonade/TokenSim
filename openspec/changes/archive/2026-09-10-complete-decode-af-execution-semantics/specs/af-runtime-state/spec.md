## MODIFIED Requirements

### Requirement: AF execution domains determine runtime rank state

当 `af.enabled=true` 时，系统 MUST 以选定阶段的 execution domains 作为 AF runtime 基础 rank、DP/TP membership、physical worker 和硬件容量的唯一依据。全局 `ParallelConfig` MUST NOT 覆盖 AF domain-local 值。A 与 F domain 每个 `TP * DP` 基础 rank MUST 有且仅有一个 runtime state。F `EP=1` 时该 state 只保留基础 TP/DP membership；rank-reuse `EP=TP*DP` 时 Routed Expert EP identity SHALL 作为同一 state 的附加 membership，而不得创建额外 physical state。

#### Scenario: Domain TP overrides unrelated global TP

- **WHEN** 全局 `TP=1` 而 A execution domain 配置 `TP=2`
- **THEN** A replica 包含两个具有独立内存与 KV 记账的 runtime ranks
- **AND** KV 与模型分片使用 A domain `TP=2`

#### Scenario: Rank-reuse EP aliases existing F states

- **WHEN** F domain 使用 `TP=2`、`DP=2` 的 rank-reuse EP
- **THEN** runtime plan 包含四个 physical F states 和四个派生 EP identities
- **AND** EP membership 不创建第五个 state或重复设备容量

#### Scenario: EP one keeps the same base states

- **WHEN** F domain 使用 `TP=2`、`DP=2`、`EP=1`
- **THEN** runtime plan 同样只包含四个 F physical states
- **AND** 每个 F DP replica 的两个 states 组成其 Expert TP group

#### Scenario: Every planned rank has runtime state

- **WHEN** planner 生成绑定 A/F rank 或派生 Expert identity 的 compute node
- **THEN** 该节点唯一解析到具有相符 domain coordinates 和 physical worker 的 runtime state

### Requirement: Model memory follows A F component ownership

系统 MUST 按逐层类型、组件所有权、Expert placement、TP shard 和 `weight_dtype` 计算每个 physical rank 的驻留模型内存。A ranks MUST 计入 Attention 与首尾组件，A leader MUST 额外计入实际 MoE layers 的 Router。F ranks MUST 计入其 Dense FFN TP shards、实际驻留的 Routed Experts 以及 Shared Expert placement 要求的副本/TP shards。SwiGLU gate、up、down 三个矩阵 MUST 全部记账；MHA/GQA/MQA Attention 权重 MUST 使用实际 query 与 KV dimensions，而不得固定使用 `4H²`。同一 physical state 上因多种 membership 引用的相同 component MUST 只计入一次。

#### Scenario: Router state belongs to the A leader

- **WHEN** 模型同时包含 Dense 与 MoE layers
- **THEN** 每个 A DP owner 的 leader 只为 MoE layers 计入 Router 权重
- **AND** F ranks 不计入 Router 权重

#### Scenario: A rank does not reserve F expert weights

- **WHEN** Decode AF 将 Dense FFN、Routed Experts 与 Shared Expert 放在 F domain
- **THEN** A ranks 的模型内存不包含这些 F components
- **AND** F rank 只计入其 TP/EP mode、placement 与复制策略实际要求的权重

#### Scenario: SwiGLU uses three matrices

- **WHEN**一个 FFN 的 hidden size 为 `H`、intermediate size 为 `I`
- **THEN** 未分片 SwiGLU 权重元素数按 `3 * H * I` 计算
- **AND** Dense、Routed 与 Shared FFN 使用同一投影计数语义

#### Scenario: GQA memory uses actual KV heads

- **WHEN** query dimension 为 `Q`、KV dimension 为 `K` 且 `K < Q`
- **THEN** Attention 投影权重使用实际 Q/K/V/O shapes 计算
- **AND** 不按 MHA 的固定 `4H²` 高估 K/V 权重

#### Scenario: Shared Expert weights follow execution placement

- **WHEN** Shared Expert 被复制到多个 F TP execution groups
- **THEN**每个 physical rank 只计入其 group 内本地 TP shard
- **AND** 相同 rank 上的重复引用按稳定 component identity 去重

#### Scenario: F expert rank cannot fit its weights

- **WHEN** 修正后的 Dense/Routed/Shared 权重超过某个 F physical worker 容量
- **THEN** 系统在产生 AF execution events 前失败
- **AND** 错误包含 domain、rank、worker、component bytes、required 与 available bytes

### Requirement: KV heads follow A-domain TP partition rules

系统 MUST 根据 A execution domain TP size 为每个 A rank 计算 local KV heads，并使用 `kv_cache_dtype` 的规范字节数计算每个 KV block。当 KV head 数不小于 TP size 时，KV heads MUST 可以整除分片；当 KV head 数小于 TP size 时，系统 MUST 使用无歧义复制组，使每个 TP rank 至少保留一个 local KV head。无法分片或复制的组合 MUST 在模拟前失败，且 transfer/weight dtype MUST NOT 改变 KV block bytes。

#### Scenario: KV heads are sharded across TP ranks

- **WHEN** 模型有 8 个 KV heads、A domain `TP=2` 且 `kv_cache_dtype=bf16`
- **THEN** 每个 A TP rank 计入 4 个 local KV heads并按 2 bytes 每元素计算 block

#### Scenario: One KV head is replicated

- **WHEN** 模型有 1 个 KV head且 A domain `TP=2`
- **THEN** 两个 A TP ranks 都计入 1 个 local KV head
- **AND** runtime summary 标记复制组而非两个全局 KV heads

#### Scenario: Ambiguous KV partition is rejected

- **WHEN** KV head 数与 A TP size 既无法整除分片也无法整除复制
- **THEN** 系统在创建 request scheduler 前拒绝配置

### Requirement: Runtime topology inconsistencies fail before execution

系统 MUST 在启动模拟事件循环前校验 scheduler owner、planner group membership、基础 rank runtime state、派生 EP identity、KV shard membership、activation owner、model component placement、effective dtype 与 collective plan（含resource group/lane）的一致性。任一不一致 MUST 明确失败，且错误 MUST 包含 domain、replica/rank、operation 与冲突事实。

#### Scenario: Planner group differs from runtime group

- **WHEN** planner 声明的 A TP participants 与对应 A replica 的 runtime ranks 不一致
- **THEN** 系统在产生 compute events 前失败
- **AND** 错误列出期望与实际 participants

#### Scenario: EP identity has no base runtime state

- **WHEN** Expert placement 引用无法解析到已有 F TP/DP rank 的 EP identity
- **THEN** 系统在产生 Routed Expert event 前失败
- **AND** 不为该 identity 临时创建 runtime state

#### Scenario: Collective plan contradicts a layout

- **WHEN** planner 要求从 sharded activation 恢复 replicated layout，但 runtime plan 中没有覆盖对应 ranks 的 `all_gather`
- **THEN** 系统在事件循环前失败并报告 source/target layout 与缺失 plan

#### Scenario: AF disabled keeps baseline state

- **WHEN** `af.enabled=false`
- **THEN** 系统继续使用现有非 AF worker、全局 `ParallelConfig` 和 KV Cache 路径
- **AND** 系统不创建 AF rank runtime state

## ADDED Requirements

### Requirement: A ranks retain residual and activation ownership across F execution

Runtime plan MUST 为每个Decode token保存稳定A DP owner，并为每个跨域tensor shard保存全局hidden-index区间、`sharded`/`replicated`/`partial` layout和A TP owner。A侧 MUST 在A-to-F期间保留完成本层residual所需状态。每份F partial MUST 通过source/target interval交集解析到唯一A owner；replicated output MUST 指定stable canonical producer，数学上不同的partials MUST 保留全部producers。对应A rank MUST 仅在收齐自己必要的Routed与Shared partials后执行本地weighted combine与residual。若下一层输入layout要求复制或重新分片，runtime plan MUST 声明后续collective，而不得通过iteration barrier让非owner ranks自动获得数据。

#### Scenario: EP partials return to matching A tensor owner

- **WHEN** 两个 EP producers 分别产生同一 token 的同一 tensor shard partial
- **THEN** 两份 partial 都解析到拥有该 shard 的同一个 A TP rank
- **AND** 该 rank 在两份 partial 到齐后执行本地 combine

#### Scenario: Residual remains on A

- **WHEN** activation 被发送到 F 执行 FFN
- **THEN** residual state 仍由对应 A owner 保持
- **AND** F memory、payload 和 compute 不假设拥有一份未发送的 residual

#### Scenario: Non-owner A rank needs an explicit layout transition

- **WHEN** combine 后的 activation 是 sharded layout，而下一层 A operator 要求 replicated layout
- **THEN** 下一层开始前执行配置声明的 A TP collective
- **AND** 缺少该 transition 时 runtime validation 失败

#### Scenario: Different A and F tensor parallel sizes share a global interval space

- **WHEN** A与F execution groups使用不同TP sizes但都能精确分片hidden width
- **THEN** runtime plan以同一global interval space描述两侧shards
- **AND** planner可由interval intersections唯一生成transfer legs

### Requirement: Runtime accounting exposes effective dtype and formula sources

每个 physical rank 的 runtime summary MUST 输出 effective dtype roles、每个 component 的 shape/formula/source、去重后的 weight bytes、KV bytes/block 与 capacity。所有 byte calculations MUST 使用与 executor 和 latency provider 相同的 dtype registry；兼容 `af_dtype` 映射 MUST 在 summary 中可见。

#### Scenario: FP8 accounting agrees across subsystems

- **WHEN**某个 role 使用 FP8
- **THEN** runtime summary、operator query 和 transfer trace 均报告同一规范 dtype 与每元素 1 byte

#### Scenario: Formula correction is auditable

- **WHEN** runtime 为 GQA Attention 或 SwiGLU FFN 计算权重
- **THEN** summary 保留实际 dimensions、projection count 与最终 component bytes
- **AND** 用户可以从这些字段重建 memory total
