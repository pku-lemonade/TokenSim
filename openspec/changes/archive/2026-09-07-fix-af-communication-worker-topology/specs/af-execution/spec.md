## MODIFIED Requirements

### Requirement: Each block follows the layer-local AF dependency chain

系统 MUST 为每个 AF block 的每一层生成显式执行计划，并按数据依赖执行：A compute、A domain 必要 collective、A-to-F transfer、Router/Gating compute、MoE dispatch、expert compute、F domain 必要 collective、MoE combine、F-to-A transfer。Router/Gating MUST 在 dispatch 前完成，dispatch MUST 在 expert compute 前完成；Router 与 expert MUST 是互不重叠的 compute compositions。Dispatch 与 combine MUST 始终作为 F 侧 MoE 边界的通信事件；collective MUST 仅在对应 execution domain 的 topology 或 communication plan 要求时创建。下一层 A compute MUST 等待本层 F-to-A 完成。不同 blocks 之间 SHALL 仅由数据就绪时间和共享资源竞争约束，不得被强制全局串行化。

#### Scenario: MoE communication appears in dependency order

- **WHEN** 一个单层 AF block 的 domain plan 不要求额外 collective
- **THEN** trace 顺序为 `A compute -> A-to-F -> Router/Gating compute -> dispatch -> expert compute -> combine -> F-to-A`
- **AND** 任一事件的开始时间不早于前一依赖事件的结束时间

#### Scenario: One block completes one layer in dependency order

- **WHEN** 一个 block 执行单层 AF
- **THEN** execution plan 中每个 step 的开始时间不早于其依赖 step 的结束时间
- **AND** block 在 F-to-A 完成后才标记该层完成

#### Scenario: Planned collective is inserted once

- **WHEN** A 或 F execution domain 的 communication plan 要求某个 AllReduce 或其他 collective
- **THEN** executor 在计划指定位置创建一次对应通信事件
- **AND** 未要求该 collective 的 topology 不产生该事件

#### Scenario: Next layer waits for the return transfer

- **WHEN** 一个 block 需要执行多层
- **THEN** 第 N+1 层 A compute 的 ready time 等于或晚于第 N 层 F-to-A 的结束时间

#### Scenario: Independent blocks may overlap

- **WHEN** 前一 block 已释放某资源且仍在后续资源上执行
- **THEN** 后一 block MAY 使用已释放资源开始相应 stage
- **AND** 系统不等待前一 block 整体完成

### Requirement: Compute resources enforce A affinity and F pool capacity

系统 MUST 通过 domain-local rank mapping 将 A compute 放到持有该 request 调度与 KV 状态的 A logical instance 所映射的 physical resource，并 SHALL 按 F domain 的 placement 将 Router/Gating compute 与后续 expert compute 调度到相应 F logical instance 和 physical worker pool。资源竞争 MUST 使用 physical resource identity 与容量，而不是把 P/A/F logical ranks 合并为统一 global rank；同一 physical resource 上的并发服务不得超过其容量。

#### Scenario: A work remains on its owning logical instance

- **WHEN** 一个 AF iteration 由某 A logical instance 提交
- **THEN** 该 iteration 中每个 block 的 A compute 均使用其 mapping 指向的 physical resource
- **AND** F-to-A 结果返回同一 A logical instance 后才继续下一层

#### Scenario: A work remains on its owning side

- **WHEN** 一个 AF iteration 由某 A logical instance 提交
- **THEN** 其 A compute 使用该 instance 映射的 physical resource
- **AND** request 调度与 KV 状态仍归该 A 侧所有

#### Scenario: Busy F physical pool queues ready work

- **WHEN** placement 可用的全部 F physical slots 均忙且另一个 block 的 Router 或 expert compute 已就绪
- **THEN** 该 compute 等待到合法 slot 可用
- **AND** 其 queue duration 等于开始服务时间减去 ready time

#### Scenario: Busy F pool queues ready work

- **WHEN** placement 允许的所有 F physical slots 均忙且另一个 block 的 Router 或 expert compute 已就绪
- **THEN** 该 compute 等待到合法 slot 可用
- **AND** 不会改派给 mapping 之外的空闲 worker

#### Scenario: Logical ranks do not create fictitious capacity

- **WHEN** A 或 F domain 的 placement 试图将多个 logical rank identities 映射到同一个 physical slot
- **THEN** 配置在创建 Executor 前明确失败
- **AND** 运行时不会通过 rank 共置产生无法解释的容量或显存共享

#### Scenario: Multiple F workers provide bounded concurrency

- **WHEN** F domain 映射到多个 physical workers 且多个 blocks 同时 ready
- **THEN** 同时运行的 expert compute 数量不超过映射后 physical capacity
- **AND** 合法候选集合内的空闲 slot 可立即接收 ready block

### Requirement: Latency providers return service time only

系统 SHALL 通过可注入的 compute provider 分别查询单 block、单层的 A、Router/Gating 与 Expert compute service time，并通过 transfer provider 查询每个通信 step 的 service time。Provider 返回值 MUST 为有限且非负的秒数，并 MUST 携带可追溯的 source、dataset、match type 与 composition 或 transfer details；这些元数据 MUST NOT 参与 executor 的资源选择或调度。Router composition 与 Expert composition MUST 互不重叠，二者之和 SHALL 取代旧的合并 F service time。Queue duration MUST 仅由执行器申请共享资源产生。AF 正式路径 MUST 不将 Roofline、LLMCompass 的完整 iteration 时延按层数或 block 数拆分后作为静默 fallback。

#### Scenario: Stub providers drive a deterministic execution test

- **WHEN** 测试注入按 A、Router、Expert stage 和通信 operation 返回固定 service time 的 providers
- **THEN** executor 按这些值推进模拟时间
- **AND** 资源等待仍由事件竞争独立产生

#### Scenario: Operator-data provider drives service time

- **WHEN** 正式 provider 为一次 A、Router、Expert compute 或 transfer 查询返回 service time 与可追溯 metadata
- **THEN** executor 只使用 service time 推进对应服务区间
- **AND** metadata 不改变资源、worker、事件依赖或开始时间

#### Scenario: Invalid provider duration fails explicitly

- **WHEN** 任一 provider 返回负数、NaN 或无限值
- **THEN** 系统终止该 AF iteration 并报告包含 block、layer 和 compute stage 或 communication operation 的错误

#### Scenario: Required estimate is unavailable

- **WHEN** provider 无法为 A、Router、Expert 或某次通信查询提供受支持的结果
- **THEN** 系统明确报告缺失查询上下文
- **AND** 系统不得改用完整 iteration 时延继续运行

### Requirement: AF transfers compete on normalized communication resources

系统 MUST 让 A-to-F、F-to-A、MoE dispatch、MoE combine 及 execution plan 要求的 collective 均申请规范化通信资源，并记录独立的 queue 与 service 区间。资源键 MUST 由配置和 domain communication plan 明确解析，且可让不同操作在代表同一物理网络资源时相互竞争。全双工资源的相反方向 SHALL 按配置独立竞争；半双工资源的两个方向 MUST 竞争同一容量；每次通信的总阶段时间 MUST 等于 `queue_duration + service_duration`。

#### Scenario: Dispatch and combine contend with AF transfer

- **WHEN** dispatch、combine 与 AF transfer 被映射到同一个容量为一的 physical communication resource
- **THEN** 它们的 service 区间不得重叠
- **AND** 后获得资源的事件记录对应 queue duration

#### Scenario: Collective participants determine resource use

- **WHEN** communication plan 创建一个包含明确 participant ranks 的 collective
- **THEN** executor 为其申请这些 ranks 对应的规范化资源
- **AND** trace 保留 operation、participants 和 resource identity

#### Scenario: Full-duplex directions overlap independently

- **WHEN** A-to-F 与 F-to-A 使用同一 full-duplex resource group 的两个方向键
- **THEN** 两个方向可在各自容量允许时同时服务

#### Scenario: Half-duplex directions serialize at capacity one

- **WHEN** 相反方向的两个传输使用容量为 1 的 half-duplex 共享键
- **THEN** 两个 service 区间不得重叠
- **AND** 后获得资源的传输记录非零 queue duration

#### Scenario: Shared links contend across blocks

- **WHEN** 来自不同 blocks 或 links 的任意通信 operation 映射到相同 resource key
- **THEN** 它们共同竞争该 key 的容量
- **AND** queue duration 由各自 ready time 到实际 start time 的间隔产生

### Requirement: AF trace reconstructs queue, service, and makespan

当 `trace_enabled=true` 时，系统 MUST 为每个 A compute、Router/Gating compute、Expert compute、A-to-F、dispatch、collective、combine 和 F-to-A 事件记录 iteration、block、layer、operation/stage、execution domain、logical participants、physical resource、`ready_at`、`start_at`、`end_at`、`queue_duration`、`service_duration`、request 数、token 数、payload bytes、`source_id`、`dataset_version`、`match_type` 和 provider details。Trace MUST 足以重建每个 block 的依赖链、通信竞争、完成时间、iteration makespan、barrier wait 和资源利用率；利用率 MUST 只统计 service 区间。

#### Scenario: Internal communication contributes to makespan

- **WHEN** dispatch、collective 或 combine 具有非零 service 或 queue duration
- **THEN** block 完成时间和 iteration makespan 包含这些区间
- **AND** trace 中能定位每段时延对应的 operation 与资源

#### Scenario: Queued event has internally consistent timestamps

- **WHEN** 一个 ready 事件等待繁忙资源后完成服务
- **THEN** `queue_duration = start_at - ready_at`
- **AND** `service_duration = end_at - start_at`

#### Scenario: Provider metadata survives trace serialization

- **WHEN** compute 或任一 typed communication provider 返回 provenance 与 composition/transfer details
- **THEN** trace 事件保留 dataset、match type、来源与 breakdown
- **AND** details 可重建对应事件的 `service_duration`

#### Scenario: Logical and physical identities remain distinguishable

- **WHEN** logical participants 与其映射的 physical resources 数量或编号不同
- **THEN** trace 分别记录两类 identity
- **AND** 用户无需依赖统一 global rank 即可重建 placement 和竞争

#### Scenario: Iteration makespan uses the last block completion

- **WHEN** 多个 blocks 的完成时间不同
- **THEN** iteration makespan 等于最晚 block 完成时间减去 iteration 开始时间
- **AND** 系统不得将各 block 的独立耗时直接求和

#### Scenario: Disabling trace does not change timing

- **WHEN** 相同输入分别以 trace 开启和关闭运行
- **THEN** 两次运行的资源顺序、block 完成时间和 iteration makespan 相同

### Requirement: Deterministic end-to-end oracles validate pipelining

系统 MUST 支持从真实 AF 集群配置、scheduler 输出、独立 P/A/F domain mappings 和注入的确定性 providers 运行到最终 iteration result 与 trace。端到端测试 MUST 同时断言内部通信依赖、资源容量、并行重叠、logical-to-physical placement 和最终时间，不得把 stub 数值声明为物理性能结论。

#### Scenario: Single block oracle includes MoE boundary communication

- **WHEN** 单层单 block 的 A、A-to-F、Router、dispatch、Expert、combine、F-to-A service time 分别为 2、1、1、1、2、1、1 ms 且无 collective 和排队
- **THEN** block 和 iteration 的完成时间均为 9 ms
- **AND** trace 包含全部七个事件且 Router 先于 dispatch

#### Scenario: Single block oracle totals seven milliseconds

- **WHEN** 单层单 block 的 A、A-to-F、Router、Expert、F-to-A service time 分别为 2、1、1、2、1 ms，dispatch/combine 为 0 ms，且无 collective 和排队
- **THEN** block 和 iteration 的完成时间均为 7 ms
- **AND** trace 仍包含 Router、Expert 以及零 service time 的 dispatch 与 combine 事件

#### Scenario: Shared communication resource changes makespan

- **WHEN** 两个 blocks 的 AF transfer、dispatch、combine 竞争同一个容量为一的通信资源
- **THEN** trace 显示内部通信与 transfer 之间的合法排队
- **AND** iteration makespan 等于 trace 依赖与竞争推导出的结束时间

#### Scenario: Two blocks form a ten millisecond pipeline

- **WHEN** 两个单层 blocks 使用原有 2、1、3、1 ms A/AF-transfer/合计 F compute/return service time，其中 Router 与 Expert 合计 3 ms、dispatch/combine 为 0 ms，且相关容量均为 1
- **THEN** 正确流水的 iteration makespan 为 10 ms
- **AND** trace 包含内部通信事件并显示合法重叠而非完全串行执行

#### Scenario: Bottleneck and duplex variants remain explainable

- **WHEN** 测试分别改变 A/F compute、内部通信 service time、duplex 模式或 physical F worker 数量
- **THEN** makespan 的变化可由 trace 中的 service、queue 和重叠区间解释

#### Scenario: Independent domain counts execute successfully

- **WHEN** Prefill、A 与 F 使用不同的 domain topology 和 worker 数，且每个 domain 内 logical ranks 与 physical workers 严格一对一
- **THEN** 端到端 AF iteration 成功完成
- **AND** 不触发 P/A/F worker 总数等于统一 `TP * PP * DP` 的校验

#### Scenario: AF-disabled baseline remains stable

- **WHEN** 同一模型配置设置 `af.enabled=false`
- **THEN** 不创建 AF 内部通信事件
- **AND** 既有非 AF backend 的 request、KV 和 timing 行为保持不变
