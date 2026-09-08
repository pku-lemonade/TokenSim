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

系统 MUST 对每个 AF block 的每一层执行以下数据依赖：为 A 与 F 工作选择合法 DP replica；在 A 域完成逐 TP rank 的 Attention 本地计算和必要 TP collective；由 A leader 完成 Router/Top-K；执行恰好一次 A-to-F/EP-dispatch transfer；在 active F ranks 上完成 Expert 本地计算和必要 F TP collective；再执行恰好一次 F-to-A/EP-combine transfer。Router/Top-K MUST 在 A-to-F 前完成，下一层 A 计算 MUST 等待本层 F-to-A 完成。系统 MUST NOT 为同一 routed payload 生成额外的 F 域内部 dispatch/combine 通信。不同 blocks 之间 SHALL 仅由数据就绪时间和共享资源竞争约束，不得被强制全局串行化。

#### Scenario: One block follows the corrected AF order

- **WHEN** 一个单层 AF block 使用 TP 和 EP，且所有事件 service time 非零
- **THEN** trace 的依赖顺序为 `A rank compute -> required A TP collective -> A Router/Top-K -> A-to-F/EP dispatch -> expert rank compute -> required F TP collective -> F-to-A/EP combine`
- **AND** 任一事件的开始时间不早于其全部依赖事件的结束时间

#### Scenario: One block completes one layer in dependency order

- **WHEN** 一个 block 执行单层 AF
- **THEN** 每个 rank-local、通信与汇合事件的开始时间不早于其全部依赖事件的结束时间
- **AND** block 在唯一一次 F-to-A 完成后才标记该层完成

#### Scenario: AF boundary communication occurs once per direction

- **WHEN** 一个 block 完成一层 AF 执行
- **THEN** 该层只产生一次 A-to-F 和一次 F-to-A 域间 transfer
- **AND** 不产生独立的 `ep_dispatch` 或 `ep_combine` 事件

#### Scenario: Next layer waits for the return transfer

- **WHEN** 一个 block 需要执行多层
- **THEN** 第 N+1 层 A rank compute 的 ready time 等于或晚于第 N 层 F-to-A 的结束时间

#### Scenario: Independent blocks may overlap

- **WHEN** 前一 block 已释放某资源且仍在后续资源上执行
- **THEN** 后一 block MAY 使用已释放资源开始相应事件
- **AND** 系统不等待前一 block 整体完成

### Requirement: Router and EP communication follow A-domain routing

对于启用 MoE 的 AF layer，系统 MUST 在 A TP compute/collective 完成后，由所选 A DP replica 的 leader 执行 Router/Top-K，再执行恰好一次 A→F/EP-dispatch transfer。Router 结果 MUST 决定 routed token count、跨域 endpoints、F 域 expert placement 与 payload；A→F payload MUST 按 routed activations 计算。Router compute MUST 恰好计时一次，且不得被 Expert compute 或通信 provider 重复包含。所有必要 Expert branches 汇合后 MUST 执行恰好一次 F→A/EP-combine transfer，不得为同一 routed payload 再生成 F 域内部 dispatch/combine 通信。

#### Scenario: Router runs once on the selected A leader

- **WHEN** 一个 MoE AF block 完成 A TP compute 与必要 collective
- **THEN** Router/Top-K 在该 A replica 的稳定 leader 上执行一次
- **AND** Router 使用 A leader 的硬件与模型组件估算 service duration

#### Scenario: A to F carries routed activations

- **WHEN** 一个 block 的 Router 产生 `R` 个 routed tokens，hidden size 为 `H`、dtype 字节数为 `D`
- **THEN** A→F payload 为 `R * H * D`
- **AND** 缺少显式 Router histogram 时，`R` 使用 `token_count * top_k`

#### Scenario: A to F is the EP dispatch

- **WHEN** Router 已决定 active Expert ranks
- **THEN** Planner 生成一个连接 A leader 与全部 active F Expert ranks 的 A→F communication node
- **AND** Expert compute 直接等待该 node 完成，不再等待第二个 `ep_dispatch` node

#### Scenario: F to A is the EP combine

- **WHEN** 所有 active Expert branches 完成本地计算与必要 F TP collective
- **THEN** Planner 生成一个连接全部 active F Expert ranks 与 A leader 的 F→A communication node
- **AND** 不再为相同 Expert outputs 生成独立 `ep_combine` node

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

### Requirement: AF transfers compete on normalized communication resources

系统 MUST 让 A-to-F/EP-dispatch、F-to-A/EP-combine 及 execution plan 要求的 TP collective 均申请规范化通信资源，并记录独立的 queue 与 service 区间。资源键 MUST 由配置和 domain communication plan 明确解析，且可让不同操作在代表同一物理网络资源时相互竞争。全双工资源的相反方向 SHALL 按配置独立竞争；半双工资源的两个方向 MUST 竞争同一容量；每次通信的总阶段时间 MUST 等于 `queue_duration + service_duration`。

#### Scenario: AF transfers contend on a shared resource

- **WHEN** 多次 A-to-F 或 F-to-A transfer 被映射到同一个容量为一的 physical communication resource
- **THEN** 同一资源上的 service 区间不得重叠
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

### Requirement: Iteration completion preserves request and KV ownership

系统 MUST 等待 iteration 中所有 AF blocks 完成后才越过 iteration barrier，并 SHALL 在 barrier 后沿既有入口让本轮每个 request 恰好推进一次。AF block 的创建、逐层执行和 barrier MUST 不产生额外的逻辑 KV block 分配、释放或转移；既有 scheduler/block manager 对一个逻辑 KV block 的状态变化 MUST 一致反映到 request 所属 A DP replica 的全部必要 TP rank shards。Scheduler ownership MUST 保持在该 A replica 的 leader，F 侧 MUST 不取得 request 或 KV Cache 所有权。

#### Scenario: Early block waits at the iteration barrier

- **WHEN** 一个 block 早于同 iteration 的其他 blocks 完成
- **THEN** 该 block 可记录从自身完成到 iteration 完成的 barrier wait
- **AND** requests 在所有 blocks 完成前均不推进

#### Scenario: Requests advance exactly once

- **WHEN** 一个 AF iteration 的全部 blocks 完成
- **THEN** 本轮每个 request 恰好执行一次既有推进操作
- **AND** block、layer 或 stage 完成均不会额外推进 request

#### Scenario: AF blocks do not duplicate logical KV accounting

- **WHEN** scheduler 已为一批 requests 完成本轮逻辑 KV block 状态变化
- **THEN** AF block、layer、Router、Expert 或 transfer 事件不再次分配或释放逻辑 KV blocks
- **AND** scheduler 的一次逻辑变化只映射为所属 A TP group 中各 rank 的对应 shard 变化

#### Scenario: AF execution does not alter KV block accounting

- **WHEN** 同一批 requests 分别通过 AF 路径和等价的非 AF 控制路径完成一轮
- **THEN** AF block 机制本身不增加 KV block 分配或释放次数
- **AND** KV 状态仍归原 A 侧所有

#### Scenario: KV shards remain aligned after an AF iteration

- **WHEN** 一个使用 `TP>1` 的 request 完成 AF iteration
- **THEN** 该 request 的所有逻辑 KV blocks 在其 A replica 的全部必要 TP ranks 上都有对应 local shard
- **AND** F ranks 不保留该 request 的 KV block

### Requirement: AF trace reconstructs queue, service, and makespan

当 `trace_enabled=true` 时，系统 MUST 为每个 rank-local compute、Router/Top-K、TP collective、A-to-F/EP-dispatch、expert compute 和 F-to-A/EP-combine 事件记录 iteration、block、layer、operation/stage、execution domain、DP replica、logical participants、physical resources、TP/EP group、rank-local shape 或 token workload、payload bytes、`ready_at`、`start_at`、`end_at`、`queue_duration`、`service_duration`、`source_id`、`dataset_version`、`match_type` 和 provider details。Trace MUST 足以重建分支与汇合依赖、资源竞争、block 完成时间、iteration makespan、barrier wait 和只按 service 区间计算的资源利用率。当 trace 关闭时，记录开销 MAY 被省略，但调度结果 MUST 不变。

#### Scenario: Rank-local events reconstruct a synchronization point

- **WHEN** 一个 TP collective 或 F-to-A/EP-combine 等待多个前驱 rank 事件
- **THEN** trace 可识别全部参与 ranks 与前驱完成时间
- **AND** 同步事件的 ready time 不早于最晚必要前驱的 end time

#### Scenario: Straggler contributes to makespan

- **WHEN** 某个 expert rank 因 token workload 或排队而晚于其他 expert ranks 完成
- **THEN** F-to-A/EP-combine 等待该必要 rank
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

系统 MUST 使用 A 域 Router/Top-K 的确定性 routing result 和 F 域 expert placement，计算每个目标 expert rank 的 token workload，并以该结果决定 A-to-F/EP-dispatch 与 F-to-A/EP-combine 的 endpoints 和 payload。只有命中 tokens 或因通信语义必须参与的 ranks SHALL 出现在该层 EP 分支中；expert service time MUST 基于其实际 local token workload。F-to-A/EP-combine MUST 等待所有必要 expert 分支完成，并由最晚分支自然表现负载不均和 straggler。

#### Scenario: Uneven routing produces uneven rank workload

- **WHEN** Top-K histogram 将 12 个 token 分别路由为 rank 0 上 10 个、rank 1 上 2 个
- **THEN** 两个 expert compute 查询分别使用 10 和 2 的 local token workload
- **AND** 系统不把总 token 数平均分配给两个 ranks

#### Scenario: Empty expert rank performs no expert compute

- **WHEN** 某个 F rank 上的 experts 在当前 block 中均未被 Top-K 命中
- **THEN** 该 rank 不产生非零 expert compute 事件
- **AND** trace 不伪造平均 workload

#### Scenario: Return transfer waits for the slowest required expert branch

- **WHEN** 多个必要 expert ranks 的完成时间不同
- **THEN** F-to-A/EP-combine 的 ready time 等于或晚于其中最晚完成时间
- **AND** 较早完成的分支不会使 block 提前返回 A 域

### Requirement: Expert latency uses routed work and local TP shape

系统 MUST 将 Expert histogram count 解释为已包含 Top-K 展开的 routed token-expert assignments。Planner MUST 将其保留为逐 F rank、逐 Expert workload；operator-data provider MUST 使用该 workload 估算各本地 Expert 的工作并聚合。Analytical MoE 与 Dense FFN 计算 MUST 使用 F domain TP 的 rank-local 权重形状；Expert 计算不得将 routed assignment 数再次乘以 `top_k`。

#### Scenario: Top-K is expanded exactly once

- **WHEN** `T` 个原始 token、`top_k=K` 产生总量为 `T*K` 的 histogram
- **THEN** 所有 F ranks 的 Expert FLOPs 总和以 `T*K` 个 routed assignments 计算
- **AND** 公式不得再次乘以 `K`

#### Scenario: F TP shards Expert compute

- **WHEN** 同一 Expert workload 分别在 `TP=1` 与 `TP=2` 的 F domain 上使用 analytical provider
- **THEN** `TP=2` 每个 rank 的 Expert GEMM FLOPs 与权重字节按本地 intermediate shard 缩小
- **AND** F TP collective 仍作为独立 communication node 计时

#### Scenario: Per-Expert skew changes a rank query

- **WHEN** 两个 F ranks 具有相同 routed assignment 总数但不同的逐 Expert workload
- **THEN** provider 的 Expert 查询明细 MUST 保留不同的 Expert IDs 与 token counts
- **AND** rank latency MUST 由这些逐 Expert查询聚合而成

### Requirement: External Expert histograms match the current step

系统 MUST 在接受 trace 或 request 提供的 Expert histogram 时校验其总 count 等于当前 phase 实际 token 数乘以 `top_k`。Prefill 与 Decode 的 normalized histogram cache MUST 相互隔离。

#### Scenario: Histogram total is inconsistent

- **WHEN** 当前 step 有 `T` 个 token、`top_k=K`，但外部 histogram 总量不等于 `T*K`
- **THEN** 系统在构造 AF block 前报告 workload validation error
- **AND** 不得用该 histogram 继续估算通信或 Expert compute

#### Scenario: Prefill and Decode use different histograms

- **WHEN** 同一 request 同时提供 Prefill 与 Decode histograms
- **THEN** 两个 phase 分别规范化并校验自己的 histogram
- **AND** Decode 不得复用已缓存的 Prefill histogram

### Requirement: Deterministic end-to-end oracles validate pipelining

系统 MUST 支持从真实 AF 集群配置、scheduler 输出、独立 P/A/F domain mappings 和注入的确定性 providers 运行到最终 iteration result 与 trace。端到端测试 MUST 同时断言跨域通信依赖、必要 TP collective、资源容量、并行重叠、logical-to-physical placement 和最终时间，不得把 stub 数值声明为物理性能结论。

#### Scenario: Single block oracle merges EP communication into AF transfers

- **WHEN** 单层单 block 的 A、Router、A-to-F/EP-dispatch、Expert、F-to-A/EP-combine service time 分别为 2、1、1、2、1 ms，且必要 collective 为 0 ms、无排队
- **THEN** block 和 iteration 的完成时间均为 7 ms
- **AND** trace 不包含独立的 `ep_dispatch` 或 `ep_combine` 事件

#### Scenario: Shared AF communication resource changes makespan

- **WHEN** 两个 blocks 的 A-to-F 与 F-to-A transfers 竞争同一个容量为一的通信资源
- **THEN** trace 显示跨域 transfers 之间的合法排队
- **AND** iteration makespan 等于 trace 依赖与竞争推导出的结束时间

#### Scenario: Two blocks form an eleven millisecond pipeline

- **WHEN** 两个单层 blocks 使用 2、1、1、2、1 ms 的 A、Router、A-to-F、Expert、F-to-A service time，且共享 A/F compute 容量均为 1
- **THEN** 正确流水的 iteration makespan 为 11 ms
- **AND** trace 显示合法重叠而非 14 ms 的完全串行执行

#### Scenario: Bottleneck and duplex variants remain explainable

- **WHEN** 测试分别改变 A/F compute、跨域通信 service time、duplex 模式或 physical F worker 数量
- **THEN** makespan 的变化可由 trace 中的 service、queue 和重叠区间解释

#### Scenario: Independent domain counts execute successfully

- **WHEN** Prefill、A 与 F 使用不同的 domain topology 和 worker 数，且每个 domain 内 logical ranks 与 physical workers 严格一对一
- **THEN** 端到端 AF iteration 成功完成
- **AND** 不触发 P/A/F worker 总数等于统一 `TP * PP * DP` 的校验

#### Scenario: AF-disabled baseline remains stable

- **WHEN** 同一模型配置设置 `af.enabled=false`
- **THEN** 不创建 AF 分离计算或通信事件
- **AND** 既有非 AF backend 的 request、KV 和 timing 行为保持不变
