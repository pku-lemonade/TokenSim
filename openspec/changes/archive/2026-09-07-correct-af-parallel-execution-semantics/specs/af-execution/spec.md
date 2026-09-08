## MODIFIED Requirements

### Requirement: Each block follows the layer-local AF dependency chain

系统 MUST 对每个 AF block 的每一层执行以下数据依赖：为 A 与 F 工作选择合法 DP replica；在 A 域完成逐 TP rank 的 Attention 本地计算和必要 TP collective；在 A 域完成 Router/Top-K；执行恰好一次 A-to-F 域间 transfer；在 F 域完成 EP dispatch、各目标 expert rank 的本地计算与 EP combine；再执行恰好一次 F-to-A 域间 transfer。Router/Top-K MUST 在 A-to-F 前完成，下一层 A 计算 MUST 等待本层 F-to-A 完成。EP dispatch/combine MUST 是 F 域内部通信，MUST NOT 被再次建模为 A/F 域间 transfer。不同 blocks 之间 SHALL 仅由数据就绪时间和共享资源竞争约束，不得被强制全局串行化。

#### Scenario: One block follows the corrected AF order

- **WHEN** 一个单层 AF block 使用 TP 和 EP，且所有事件 service time 非零
- **THEN** trace 的依赖顺序为 `A rank compute -> required A TP collective -> A Router/Top-K -> A-to-F -> F EP dispatch -> expert rank compute -> F EP combine -> F-to-A`
- **AND** 任一事件的开始时间不早于其全部依赖事件的结束时间

#### Scenario: One block completes one layer in dependency order

- **WHEN** 一个 block 执行单层 AF
- **THEN** 每个 rank-local、通信与汇合事件的开始时间不早于其全部依赖事件的结束时间
- **AND** block 在唯一一次 F-to-A 完成后才标记该层完成

#### Scenario: AF boundary communication occurs once per direction

- **WHEN** 一个 block 完成一层 AF 执行
- **THEN** 该层只产生一次 A-to-F 和一次 F-to-A 域间 transfer
- **AND** F 域内 dispatch/combine 不产生额外 A-to-F 或 F-to-A 事件

#### Scenario: Next layer waits for the return transfer

- **WHEN** 一个 block 需要执行多层
- **THEN** 第 N+1 层 A rank compute 的 ready time 等于或晚于第 N 层 F-to-A 的结束时间

#### Scenario: Independent blocks may overlap

- **WHEN** 前一 block 已释放某资源且仍在后续资源上执行
- **THEN** 后一 block MAY 使用已释放资源开始相应事件
- **AND** 系统不等待前一 block 整体完成

### Requirement: Compute resources enforce A affinity and F placement

系统 MUST 让一个 block 在其整个 iteration 中保持选定的 A DP replica，并将 A rank compute 与 Router/Top-K 放到该 replica 的合法 TP ranks 及其映射的 physical resources 上。F 侧 expert compute MUST 只使用当前 F DP replica 和 expert placement 允许的 ranks。不同 DP replicas MUST 作为独立执行副本竞争各自映射的资源，不得因属于同一 logical instance 而锁步占用全部 ranks；同一 physical resource 上的并发服务不得超过其容量。

#### Scenario: A work remains on its selected replica

- **WHEN** 一个 AF block 被分配到某个 A DP replica
- **THEN** 该 block 各层的 A rank compute 和 Router/Top-K 均使用该 replica 的合法 ranks
- **AND** request 调度与 KV 状态仍归该 A replica 所在 A 侧所有

#### Scenario: Unselected DP replicas remain available

- **WHEN** 一个 block 只选择了某个 DP replica 且其他 replicas 的 physical resources 空闲
- **THEN** 未选 replicas 不会被该 block 锁定
- **AND** 另一个 ready block MAY 在这些 replicas 上并行执行

#### Scenario: Expert compute obeys placement

- **WHEN** Top-K routing result 只命中放置在部分 F ranks 上的 experts
- **THEN** expert compute 只出现在这些 experts 对应的合法 ranks 上
- **AND** 未命中且不参与必要通信的 F ranks 不产生 expert compute service time

### Requirement: AF trace reconstructs queue, service, and makespan

当 `trace_enabled=true` 时，系统 MUST 为每个 rank-local compute、Router/Top-K、collective、AF transfer、EP dispatch、expert compute 和 EP combine 事件记录 iteration、block、layer、operation/stage、execution domain、DP replica、logical participants、physical resources、TP/EP group、rank-local shape 或 token workload、payload bytes、`ready_at`、`start_at`、`end_at`、`queue_duration`、`service_duration`、`source_id`、`dataset_version`、`match_type` 和 provider details。Trace MUST 足以重建分支与汇合依赖、资源竞争、block 完成时间、iteration makespan、barrier wait 和只按 service 区间计算的资源利用率。当 trace 关闭时，记录开销 MAY 被省略，但调度结果 MUST 不变。

#### Scenario: Rank-local events reconstruct a synchronization point

- **WHEN** 一个 TP collective 或 EP combine 等待多个前驱 rank 事件
- **THEN** trace 可识别全部参与 ranks 与前驱完成时间
- **AND** 同步事件的 ready time 不早于最晚必要前驱的 end time

#### Scenario: Straggler contributes to makespan

- **WHEN** 某个 expert rank 因 token workload 或排队而晚于其他 expert ranks 完成
- **THEN** EP combine 等待该必要 rank
- **AND** trace 可从该分支的 service、queue 与依赖重建其对 block makespan 的贡献

#### Scenario: Queued event has internally consistent timestamps

- **WHEN** 一个 ready 事件等待繁忙资源后完成服务
- **THEN** `queue_duration = start_at - ready_at`
- **AND** `service_duration = end_at - start_at`

#### Scenario: Provider metadata survives trace serialization

- **WHEN** provider 返回 provenance 与 composition 或 transfer details
- **THEN** trace 保留 dataset、match type、来源与 breakdown
- **AND** details 可重建对应事件的 `service_duration`

#### Scenario: Iteration makespan uses the last block completion

- **WHEN** 多个 blocks 的完成时间不同
- **THEN** iteration makespan 等于最晚 block 完成时间减去 iteration 开始时间
- **AND** 系统不得将各 block 或各 rank 的独立耗时直接求和

#### Scenario: Disabling trace does not change timing

- **WHEN** 相同输入分别以 trace 开启和关闭运行
- **THEN** 两次运行的资源顺序、block 完成时间和 iteration makespan 相同

## ADDED Requirements

### Requirement: DP replicas are selected independently by global round-robin

系统 MUST 按 scheduler 输出的确定性顺序，为每个 ready AF block 从对应 execution domain 的全部可用 DP replicas 中执行全局 round-robin 选择。一个 block 在每个所需 domain 中 MUST 恰好选择一个 DP replica；同一 replica 内的 TP/EP ranks SHALL 协作执行，未选 replicas MUST 保持独立并可服务其他 blocks。相同输入和初始状态 MUST 产生相同 replica 分配序列。

#### Scenario: Blocks rotate across replicas

- **WHEN** 一个 domain 有两个 DP replicas 且四个 blocks 依次 ready
- **THEN** replica 分配按 `0, 1, 0, 1` 的全局顺序轮转
- **AND** 选择不因单个调用方或 physical worker 的局部计数重新开始

#### Scenario: Replica execution is not lockstep

- **WHEN** replica 0 正在执行一个长 block 而 replica 1 已空闲
- **THEN** 分配给 replica 1 的 ready block 可立即申请其合法资源
- **AND** replica 1 不等待 replica 0 完成

### Requirement: TP changes rank-local work and creates required collectives

当 `TP > 1` 时，系统 MUST 根据算子分片规则为每个 TP rank 生成其本地 tensor shape、计算 workload 与必要 collective 的参与关系；rank-local service time MUST 由本地 shape 查询得到，而不是在所有 TP ranks 上重复完整未分片算子时延。collective MUST 只在模型语义要求的位置产生，并 MUST 等待全部必要 TP rank 前驱；`TP=1` 时 MUST 不产生虚构的 TP collective。

#### Scenario: Tensor-parallel ranks receive local shapes

- **WHEN** 一个可沿 TP 维度分片的算子以 `TP=2` 执行
- **THEN** 两个 TP ranks 的 compute 查询分别携带其本地 shape
- **AND** 两个查询均不使用完整未分片 shape 冒充本地工作

#### Scenario: Required TP collective joins rank branches

- **WHEN** 分片算子的输出语义要求 TP collective
- **THEN** collective 明确列出参与 TP ranks，并等待最晚 rank-local compute 完成
- **AND** collective 完成后下游 Router/Top-K 才可 ready

#### Scenario: Single-rank TP creates no collective

- **WHEN** `TP=1`
- **THEN** 系统执行单个 rank-local compute
- **AND** 不产生仅用于模拟并行的零意义 TP collective

### Requirement: EP executes routed per-rank workload and waits for required experts

系统 MUST 使用 A 域 Router/Top-K 的确定性 routing result 和 F 域 expert placement，计算每个目标 expert rank 的 token workload、dispatch payload、expert compute 与 combine payload。只有命中 tokens 或因通信语义必须参与的 ranks SHALL 出现在该层 EP 分支中；expert service time MUST 基于其实际 local token workload。EP combine MUST 等待所有必要 expert 分支完成，并由最晚分支自然表现负载不均和 straggler。

#### Scenario: Uneven routing produces uneven rank workload

- **WHEN** Top-K histogram 将 12 个 token 分别路由为 rank 0 上 10 个、rank 1 上 2 个
- **THEN** 两个 expert compute 查询分别使用 10 和 2 的 local token workload
- **AND** 系统不把总 token 数平均分配给两个 ranks

#### Scenario: Empty expert rank performs no expert compute

- **WHEN** 某个 F rank 上的 experts 在当前 block 中均未被 Top-K 命中
- **THEN** 该 rank 不产生非零 expert compute 事件
- **AND** trace 不伪造平均 workload

#### Scenario: Combine waits for the slowest required expert branch

- **WHEN** 多个必要 expert ranks 的完成时间不同
- **THEN** EP combine 的 ready time 等于或晚于其中最晚完成时间
- **AND** 较早完成的分支不会使 block 提前进入 F-to-A
