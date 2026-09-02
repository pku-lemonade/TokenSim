# af-execution Specification

## Purpose

为 TokenSim 提供可运行且可观测的 Prefill/Decode Attention-FFN 分离执行语义，使逐层依赖、micro-batch 流水、计算与通信资源竞争能够由确定性事件时间线验证，并与后续性能数据来源解耦。

## Requirements

### Requirement: Selected phases execute through AF blocks

系统 SHALL 仅对 `af.phases` 选中的 Prefill 或 Decode 阶段创建 AF blocks，并 SHALL 让未选阶段及 `af.enabled=false` 的请求继续使用既有完整阶段路径。进入 AF 路径的 request MUST 在该 iteration 中只属于一个 AF block。

#### Scenario: Decode-only AF bypasses Prefill

- **WHEN** AF 已启用且 `phases=["decode"]`
- **THEN** Prefill 使用既有完整 Prefill 路径
- **AND** Decode 使用 AF block 路径

#### Scenario: Both phases use AF

- **WHEN** AF 已启用且 `phases=["prefill", "decode"]`
- **THEN** Prefill 和 Decode 分别在自身 iteration 中使用 AF block 路径

#### Scenario: AF disabled preserves the baseline path

- **WHEN** `af.enabled=false`
- **THEN** 系统不创建 AF block 或 AF 资源事件
- **AND** Prefill、Decode、KV transfer 和 request 推进保持既有行为

### Requirement: AF blocks preserve request and token semantics

系统 MUST 按 scheduler 输出顺序将同一阶段的 requests 连续分组，每个 block 的 request 数不得超过 `af_block_size`，且不得为了满足 block 大小拆分单个 Prefill request。Prefill block 的 `token_count` MUST 为所含 requests 的实际 Prefill 计算 token 数之和；Decode block 的 `token_count` MUST 等于 `request_count`。

#### Scenario: Requests are grouped deterministically

- **WHEN** scheduler 按顺序给出 5 个 requests 且 `af_block_size=2`
- **THEN** 系统生成大小依次为 2、2、1 的三个 blocks
- **AND** 每个 request 只出现在一个 block 中

#### Scenario: Prefill tokens retain actual lengths

- **WHEN** 一个 Prefill block 包含多个不同计算长度的 requests
- **THEN** 该 block 的 `token_count` 等于这些 request 的 Prefill 计算 token 数之和
- **AND** 任何单个 request 均未被拆分

#### Scenario: Decode tokens equal request count

- **WHEN** 一个 Decode block 包含 4 个 requests
- **THEN** 该 block 的 `request_count` 和 `token_count` 均为 4

### Requirement: Each block follows the layer-local AF dependency chain

系统 MUST 对每个 AF block 的每一层严格执行 `A compute -> A-to-F transfer -> F compute -> F-to-A transfer`，并 MUST 仅在 F-to-A 返回原 A 侧后开始该 block 的下一层。不同 blocks 之间 SHALL 仅由数据就绪时间和共享资源竞争约束，不得被强制全局串行化。

#### Scenario: One block completes one layer in dependency order

- **WHEN** 一个 block 执行单层 AF
- **THEN** 每个 stage 的开始时间不早于前一 stage 的结束时间
- **AND** block 在 F-to-A 完成后才标记该层完成

#### Scenario: Next layer waits for the return transfer

- **WHEN** 一个 block 需要执行多层
- **THEN** 第 N+1 层 A compute 的 ready time 等于或晚于第 N 层 F-to-A 的结束时间

#### Scenario: Independent blocks may overlap

- **WHEN** 前一 block 已释放某资源且仍在后续资源上执行
- **THEN** 后一 block MAY 使用已释放资源开始相应 stage
- **AND** 系统不等待前一 block 整体完成

### Requirement: Compute resources enforce A affinity and F pool capacity

系统 MUST 让 A compute 使用持有该 request 调度与 KV 状态的 A 侧资源，并 SHALL 将 F compute 调度到 link 所绑定的 F worker pool。A/F 资源的同时服务数量不得超过对应 worker 数量；容量已占满时，ready block MUST 排队且不得产生重叠服务区间。

#### Scenario: A work remains on its owning side

- **WHEN** 一个 AF iteration 由某 A worker 提交
- **THEN** 该 iteration 中每个 block 的 A compute 均使用该 A 侧资源
- **AND** F-to-A 结果返回同一 A 侧后才继续下一层

#### Scenario: Busy F pool queues ready work

- **WHEN** 所有 F workers 均忙且另一个 block 已完成 A-to-F
- **THEN** 该 block 的 F compute 等待到某个 F worker 可用
- **AND** 其 F queue duration 等于开始服务时间减去 ready time

#### Scenario: Multiple F workers provide bounded concurrency

- **WHEN** F pool 包含多个 workers 且多个 blocks 同时 ready
- **THEN** 同时运行的 F compute 数量不超过 F worker 数量
- **AND** 空闲 F worker 可立即接收 ready block

### Requirement: AF transfers compete on normalized communication resources

系统 MUST 使用规范化 AF resource key 为通信建立共享容量。全双工资源的 A-to-F 与 F-to-A SHALL 独立竞争；半双工资源的两个方向 MUST 竞争同一容量；不同 resource key SHALL 相互独立。每次传输的总阶段时间 MUST 等于 `queue_duration + service_duration`。

#### Scenario: Full-duplex directions overlap independently

- **WHEN** A-to-F 与 F-to-A 使用同一 full-duplex resource group 的两个方向键
- **THEN** 两个方向可在各自容量允许时同时服务

#### Scenario: Half-duplex directions serialize at capacity one

- **WHEN** 相反方向的两个传输使用容量为 1 的 half-duplex 共享键
- **THEN** 两个 service 区间不得重叠
- **AND** 后获得资源的传输记录非零 queue duration

#### Scenario: Shared links contend across blocks

- **WHEN** 来自不同 blocks 或 links 的传输映射到相同 resource key
- **THEN** 它们共同竞争该 key 的容量
- **AND** queue duration 由各自 ready time 到实际 start time 的间隔产生

### Requirement: Latency providers return service time only

系统 SHALL 通过可注入的 compute provider 查询单 block、单层、单 A/F stage 的计算 service time，并通过 transfer provider 查询单次方向传输的 service time。Provider 返回值 MUST 为有限且非负的秒数，并 MUST 携带可追溯的 source、dataset、match type 与 composition 或 transfer details；这些元数据 MUST NOT 参与 executor 的资源选择或调度。Queue duration MUST 仅由执行器申请共享资源产生。AF 正式路径 MUST 不将 Roofline、LLMCompass 的完整 iteration 时延按层数或 block 数拆分后作为静默 fallback。

#### Scenario: Stub providers drive a deterministic execution test

- **WHEN** 测试注入按 stage 和方向返回固定 service time 的 providers
- **THEN** executor 按这些值推进模拟时间
- **AND** 资源等待仍由事件竞争独立产生

#### Scenario: Operator-data provider drives service time

- **WHEN** 正式 provider 为一次 compute 或 transfer 查询返回 service time 与可追溯 metadata
- **THEN** executor 只使用 service time 推进对应服务区间
- **AND** metadata 不改变资源、worker、事件依赖或开始时间

#### Scenario: Invalid provider duration fails explicitly

- **WHEN** 任一 provider 返回负数、NaN 或无限值
- **THEN** 系统终止该 AF iteration 并报告包含 block、layer 和 stage 或 direction 的错误

#### Scenario: Required estimate is unavailable

- **WHEN** provider 无法为某次 AF 查询提供受支持的结果
- **THEN** 系统明确报告缺失查询上下文
- **AND** 系统不得改用完整 iteration 时延继续运行

### Requirement: Iteration completion preserves request and KV ownership

系统 MUST 等待 iteration 中所有 AF blocks 完成后才越过 iteration barrier，并 SHALL 在 barrier 后沿既有入口让本轮每个 request 恰好推进一次。AF block 的创建、逐层执行和 barrier MUST 不额外分配、释放或转移 KV Cache blocks，且 F 侧 MUST 不取得 request 或 KV Cache 所有权。

#### Scenario: Early block waits at the iteration barrier

- **WHEN** 一个 block 早于同 iteration 的其他 blocks 完成
- **THEN** 该 block 可记录从自身完成到 iteration 完成的 barrier wait
- **AND** requests 在所有 blocks 完成前均不推进

#### Scenario: Requests advance exactly once

- **WHEN** 一个 AF iteration 的全部 blocks 完成
- **THEN** 本轮每个 request 恰好执行一次既有推进操作
- **AND** block、layer 或 stage 完成均不会额外推进 request

#### Scenario: AF execution does not alter KV block accounting

- **WHEN** 同一批 requests 分别通过 AF 路径和等价的非 AF 控制路径完成一轮
- **THEN** AF block 机制本身不增加 KV block 分配或释放次数
- **AND** KV 状态仍归原 A 侧所有

### Requirement: AF trace reconstructs queue, service, and makespan

当 `trace_enabled=true` 时，系统 MUST 为每个计算和传输事件记录 iteration、block、layer、stage 或 direction、resource、`ready_at`、`start_at`、`end_at`、`queue_duration`、`service_duration`、request 数、token 数、`source_id`、`dataset_version`、`match_type` 和 provider details；传输事件还 MUST 记录 payload bytes。Trace 中的 composition details MUST 足以重建 compute service time，且 trace MUST 能重建每个 block 的完成时间、iteration makespan、barrier wait 和各资源只按 service 区间计算的利用率。当 trace 关闭时，记录开销 MAY 被省略，但调度结果 MUST 不变。

#### Scenario: Queued event has internally consistent timestamps

- **WHEN** 一个 ready 事件等待繁忙资源后完成服务
- **THEN** `queue_duration = start_at - ready_at`
- **AND** `service_duration = end_at - start_at`

#### Scenario: Provider metadata survives trace serialization

- **WHEN** operator-data provider 返回 exact 或 interpolated composition
- **THEN** trace 事件保留 dataset、match type、来源与 component breakdown
- **AND** breakdown 的换算后总和等于事件 `service_duration`

#### Scenario: Iteration makespan uses the last block completion

- **WHEN** 多个 blocks 的完成时间不同
- **THEN** iteration makespan 等于最晚 block 完成时间减去 iteration 开始时间
- **AND** 系统不得将各 block 的独立耗时直接求和

#### Scenario: Disabling trace does not change timing

- **WHEN** 相同输入分别以 trace 开启和关闭运行
- **THEN** 两次运行的资源顺序、block 完成时间和 iteration makespan 相同

### Requirement: Deterministic end-to-end oracles validate pipelining

系统 MUST 支持从真实 AF 集群配置、scheduler 输出和注入的确定性 providers 运行到最终 iteration result 与 trace，使端到端测试能够同时断言依赖顺序、资源容量、并行重叠和最终时间，而不把 stub 数值声明为物理性能结论。

#### Scenario: Single block oracle totals seven milliseconds

- **WHEN** 单层单 block 的 A、A-to-F、F、F-to-A service time 分别为 2、1、3、1 ms 且无排队
- **THEN** block 和 iteration 的完成时间均为 7 ms

#### Scenario: Two blocks form a ten millisecond pipeline

- **WHEN** 两个单层 blocks 使用相同的 2、1、3、1 ms service time 且 A、F 与两个方向链路容量均为 1
- **THEN** 正确流水的 iteration makespan 为 10 ms
- **AND** trace 显示合法重叠而非 14 ms 的完全串行执行

#### Scenario: Bottleneck and duplex variants remain explainable

- **WHEN** 测试分别改变 A/F service time、duplex 模式或 F worker 数量
- **THEN** makespan 的变化可由 trace 中的 service、queue 和重叠区间解释
