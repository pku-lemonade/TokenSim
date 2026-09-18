## MODIFIED Requirements

### Requirement: Each block follows the layer-local AF dependency chain

系统 MUST 根据模型元数据为每个 Decode AF block 的每一层选择 Dense 或 MoE DAG，而不得对所有层复用同一 MoE DAG。Dense 层 MUST 依次完成 A 域 Attention、本层必要 A TP collective、一个 logical A-to-F stage、F 域 Dense FFN、本层必要 F TP collective、一个按 ownership 的 logical F-to-A stage、A 侧 residual/layout 恢复。MoE 层 MUST 依次完成 A 域 Attention、本层必要 A TP collective、A leader Router/Top-K、一个 routed/shared-input logical A-to-F stage、F 域 Routed 与 Shared Expert branches、一个按 ownership 的 logical F-to-A partial-return stage、A 侧 weighted combine、residual 与必要 layout collective。每个 logical transfer stage MAY 展开为多个 endpoint legs；每个 leg 只等待其实际 producer，下游 rank只等待发给自己的 legs。下一层 A 计算 MUST 等待本层产生其完整输入的最后一个 A 侧节点完成。不同 blocks SHALL 仅由数据依赖和共享资源竞争约束，不得被强制全局串行化。

#### Scenario: One block follows the corrected AF order

- **WHEN** 一个单层 Decode AF block 使用 TP 和 EP，且所有事件 service time 非零
- **THEN** trace 展示与该层 Dense 或 MoE 类型匹配的 A compute、必要 collective、A-to-F、F compute、F-to-A 和 A 侧最终汇合顺序
- **AND** 任一事件的开始时间不早于其全部前驱事件的结束时间

#### Scenario: Dense layer omits MoE-only work

- **WHEN** 模型元数据将某层标记为 Dense
- **THEN** 该层执行 A Attention、A-to-F、Dense FFN、F-to-A 与 A residual/layout 恢复
- **AND** trace 不包含 Router、Top-K、routed Expert 或 weighted combine 事件

#### Scenario: MoE layer follows the complete dependency chain

- **WHEN** 模型元数据将某层标记为 MoE，且 `top_k > 1`、Shared Expert 数大于零
- **THEN** Routed Expert 与 Shared Expert branch各自等待发给自己的 A-to-F legs，每条 F-to-A leg等待其 producer，每个 A weighted combine 等待该 owner 的全部 return legs
- **AND** residual 对每个原始 token 恰好执行一次

#### Scenario: One block completes one layer in dependency order

- **WHEN** 一个 block 执行单层 Decode AF
- **THEN** 每个 rank-local、通信与汇合事件的开始时间不早于其全部前驱结束时间
- **AND** 该层只在 A 侧 combine/residual/layout 的最终节点完成后标记完成

#### Scenario: AF boundary communication occurs once per direction

- **WHEN** 一个 block 完成任一 Decode AF 层
- **THEN** 该层只产生一个逻辑 A-to-F stage 和一个逻辑 F-to-A stage
- **AND** 每个 stage MAY 记录多个 endpoint transfer legs，但不得为相同 payload 再生成第二套 EP dispatch/combine 搬运

#### Scenario: Next layer waits for the return transfer

- **WHEN** 一个 block 需要执行多层
- **THEN** 第 N+1 层 A rank compute 的 ready time 等于或晚于第 N 层最后一个 A residual/layout 节点结束时间
- **AND** Dense 层不需要伪造 weighted combine，MoE 层不能跳过 weighted combine

#### Scenario: Independent blocks may overlap

- **WHEN** 前一 block 已释放某资源且仍在后续资源上执行
- **THEN** 后一 block MAY 使用已释放资源开始相应事件
- **AND** 系统不等待前一 block 整体完成

### Requirement: Router and EP communication follow A-domain routing

对于每个 MoE Decode AF layer，系统 MUST 在该层 A TP compute/collective 完成后，由所选 A DP replica 的 leader 执行一次 Router/Top-K，并为当前层的每个原始 token 生成恰好 `top_k` 个逐 token assignments。每个 assignment MUST 保留 request/token identity、A DP owner、A tensor/layout owner、Expert id 和 routing-weight 语义；Expert histogram MUST 仅从这些 assignments 派生。A-to-F MUST 直接把 assignment 对应 activation 与 Shared Expert 所需原始 token 输入发送到合法 F endpoints。F-to-A MUST 将 Routed 与 Shared Expert partials 按其 A owner 返回对应 A ranks；F-to-A 只负责传输，不得冒充 weighted combine。各 A rank MUST 对自己拥有的 partials 执行一次本地 weighted combine 和 residual，并在下一层输入布局要求时执行明确的 A TP collective。

#### Scenario: Router runs once on the selected A leader

- **WHEN** 一个 MoE block 完成某层 A TP compute 与必要 collective
- **THEN** Router/Top-K 在该 A replica 的稳定 leader 上执行一次
- **AND** routing result 以当前层 assignments 表示，不复用其他层的 histogram

#### Scenario: Synthetic routing does not reset for every request

- **WHEN** balanced synthetic routing 为同一层四个 Decode tokens、`top_k=2` 和八个 Experts 生成 assignments
- **THEN** 它在整个 layer block 上连续、确定性地生成八个 assignments
- **AND** 不得因逐 request 重置起点而把全部 tokens 固定路由到 Experts 0 和 1

#### Scenario: A to F carries routed activations

- **WHEN** 当前层 `T` 个原始 tokens 产生 `T*K` 个 routed assignments，并确定 Shared Expert 输入 owners
- **THEN** A-to-F payload breakdown 由 assignment activation 与去重后的 Shared Expert 原始 token activation 共同构成
- **AND** payload 逐 endpoint、逐 tensor slice 计算，不得只由丢失 ownership 的 histogram 总数推导

#### Scenario: A to F is the EP dispatch

- **WHEN** 当前层 assignments 与 Shared Expert owners 已确定
- **THEN** A-to-F logical stage 的 endpoint legs 和 payload breakdown 覆盖这些目标 F ranks
- **AND** 每个 F compute直接等待发给自己的 legs，不再等待搬运相同 activation 的独立 dispatch stage

#### Scenario: F to A returns partials to their owners

- **WHEN** Routed 或 Shared Expert branch产生一个 output partial
- **THEN** F-to-A 将每份 output partial 直接发往拥有对应 token/tensor shard 的 A rank
- **AND** 该 leg不等待与其数据无关的 F branch，非 owner A rank也不因 iteration barrier 而被视为获得输入

#### Scenario: F to A is the EP combine

- **WHEN** 当前层需要从多个 F producers 返回 partial outputs
- **THEN** Planner 生成一个按 owner 分 leg 的逻辑 F-to-A return stage，承载旧称 EP-combine 的返回通信职责
- **AND** 数学 weighted combine 作为随后独立的 A 侧 compute node 执行，不得包含在 transfer service time 中

#### Scenario: A combine collapses Top-K assignments

- **WHEN** `T` 个原始 tokens 产生 `T*K` 个 routed partials 及每 token 一份 Shared Expert partial
- **THEN** A 侧 weighted combine 使用 routing-weight 语义将结果恢复为每个 token 一份 activation
- **AND** residual 和后续层工作量均以 `T` 而不是 `T*K` 计算

### Requirement: Compute resources enforce A affinity and F placement

系统 MUST 让一个 block 在整个 iteration 中保持其 A DP owner，并将 A rank compute、Router、combine、residual 与 layout collective 放到该 replica 的合法 ranks 和 physical resources 上。Dense 与 Shared FFN MUST 使用选定 F DP replica 的 TP group。`EP=1` 时 Routed Experts MUST 使用该 F TP group；rank-reuse `EP=TP*DP` 时 Routed Expert compute MUST 使用整个 F world 中实际驻留目标 Expert 的 rank。多个 blocks 若命中相同 physical F resource，MUST 按其容量自然竞争，而不得把派生 EP identity 当成额外设备。

#### Scenario: A work remains on its selected replica

- **WHEN** 一个 AF block 归属于某个 A DP replica
- **THEN** 该 block 各层的 A compute、Router、combine、residual 与 KV ownership 均保持在该 replica
- **AND** F ranks 不取得 request 或 KV Cache ownership

#### Scenario: Routed experts may span reused F ranks

- **WHEN** 一个 block 的 assignments 命中分布在多个 F TP/DP coordinates 上的 Experts
- **THEN** Expert branches 使用这些 coordinates 对应的已有 physical ranks
- **AND** 系统不为 EP 额外创建 `TP * EP * DP` workers

#### Scenario: EP one keeps work inside the selected F replica

- **WHEN** F domain 使用 `EP=1` 且一个 block 选择 F DP replica 1
- **THEN** Routed、Dense 与 Shared FFN 都只使用 replica 1 的合法 F TP ranks
- **AND** 其他 F DP replica 可服务另一个 ready block

#### Scenario: Shared physical F rank serializes conflicting work

- **WHEN** 两个不同 A owners 的 blocks 同时需要同一个 F physical rank
- **THEN** 两项服务按该 resource capacity 排队
- **AND** trace 的 queue interval 解释由 rank reuse 造成的竞争

#### Scenario: Unselected DP replicas remain available

- **WHEN** `EP=1` 的 block 只选择一个 F DP replica 且其他 replicas 空闲
- **THEN** 未选 replicas 不被该 block 锁定
- **AND** 另一个 ready block MAY 在这些 replicas 上并行执行

#### Scenario: Expert compute obeys placement

- **WHEN** Top-K assignments 只命中放置在部分 F ranks 上的 Experts
- **THEN** routed Expert compute 只出现在这些合法 ranks 上
- **AND** 未命中且不承担 Shared Expert 或必要通信的 F ranks 不产生 Expert service time

### Requirement: AF transfers compete on normalized communication resources

系统 MUST 让每个 A-to-F/F-to-A endpoint leg 与 execution plan 要求的 collective 申请各自显式解析的 communication resource，并记录独立 queue 与 service 区间。A-to-F 与 F-to-A 各自仍是每层一个 logical stage，但 MUST 以 stable stage id 关联其多个 executable legs；相同 tensor slice 发往相同 endpoint MUST 只计一次。Logical stage completion MUST 是其最晚必要 leg completion，但数据依赖 MUST 可直接指向相关 leg而不引入全 stage barrier。Collective 的 operation、participants、network、resource group 与 resource lane MUST 来自 domain communication plan。全双工相反方向 SHALL 按配置独立竞争；半双工相反方向 MUST 竞争同一 shared lane容量。系统 MUST NOT 为同一 AF payload 再创建 `ep_dispatch` 或 `ep_combine` stage。

#### Scenario: AF transfers contend on a shared resource

- **WHEN** 多个 A-to-F 或 F-to-A legs 映射到同一容量为一的 resource
- **THEN** 该 resource 上的 service intervals 不重叠
- **AND** 后获得资源的 leg 记录 queue duration

#### Scenario: Collective participants determine resource use

- **WHEN** communication plan 创建一个具有明确 operation 与 participant ranks 的 collective
- **THEN** executor 只为这些 ranks 申请配置指定的规范化 resource group/lane
- **AND** trace 保留 operation、participants、network 与完整 resource identity

#### Scenario: Missing collective resource is not replaced by an AF link

- **WHEN** 必要 collective 没有可解析的 network、resource group 或 resource lane
- **THEN** 系统在执行前失败
- **AND** 不借用 A-to-F 或 F-to-A fabric

#### Scenario: Full-duplex directions overlap independently

- **WHEN** A-to-F 与 F-to-A 使用同一 full-duplex resource group 的两个方向键
- **THEN** 两个方向可在各自容量允许时同时服务

#### Scenario: Explicit collective lane controls full-duplex contention

- **WHEN** A-to-F、F-to-A与collective同时ready，且collective显式选择full-duplex group的`a_to_f` lane
- **THEN** collective与A-to-F竞争同一lane，但可与F-to-A lane并行
- **AND** executor不得按operation名称猜测或改写collective lane

#### Scenario: Half-duplex directions serialize at capacity one

- **WHEN** 相反方向的两个 transfers 使用容量为 1 的 half-duplex shared key
- **THEN** 两个 service intervals 不重叠
- **AND** 后获得资源的 transfer 记录非零 queue duration

#### Scenario: Shared links contend across blocks

- **WHEN** 不同 blocks 或 links 的通信 operation 映射到相同 resource key
- **THEN** 它们共同竞争该 key 的容量
- **AND** queue duration 等于 ready time 到实际 start time 的间隔

### Requirement: Latency providers return service time only

系统 SHALL 通过可注入 compute provider 分别查询单 block、单层、单 rank 的 A Attention、A Router、F Dense FFN、F Routed Expert、F Shared Expert、A weighted combine、A residual、必要 layout operation以及跨dtype边界的encode/decode service time，并通过 transfer provider查询每个endpoint leg或collective的service time。Provider 返回值 MUST 为有限且非负的秒数，并 MUST 携带可追溯 source、dataset、match type 与 composition/transfer details；这些元数据 MUST NOT 参与资源选择或调度。所有 composition MUST 互不重叠，每个逻辑 operation MUST 恰好计时一次。Queue duration MUST 仅由执行器申请共享资源产生，且 AF 正式路径 MUST NOT 将完整 iteration 时延拆分为静默 fallback。

#### Scenario: Stub providers drive a deterministic execution test

- **WHEN** 测试为 Dense 与 MoE DAG 的各 compute 和 communication operation 注入固定 service time
- **THEN** executor 按这些值与 DAG 依赖推进模拟时间
- **AND** 资源等待仍由事件竞争独立产生

#### Scenario: Operator-data provider drives service time

- **WHEN** 正式 provider 为任一 A、F、conversion 或 communication query 返回 service time 与 provenance
- **THEN** executor 只使用 service time 推进该节点的 service interval
- **AND** metadata 不改变 placement、resources、dependencies 或 start time

#### Scenario: F residual is not charged per assignment

- **WHEN** `top_k=2` 的 MoE layer 在两个 F ranks 上执行 routed work
- **THEN** 两个 Expert composition 均不包含 residual add
- **AND** A residual 只按原始 token 数计时一次

#### Scenario: Dtype conversion is charged once per boundary side

- **WHEN** transfer dtype 与 activation dtype 不同
- **THEN** 对应发送 encode 与接收 decode nodes 各查询一次 compute provider
- **AND** transfer provider 不再包含或重复这两项 compute service

#### Scenario: Invalid provider duration fails explicitly

- **WHEN** 任一 provider 返回负数、NaN 或无限值
- **THEN** 系统终止该 AF iteration并报告 block、layer、rank 与 operation

#### Scenario: Required estimate is unavailable

- **WHEN** provider 无法为任一必要 compute、transfer 或 collective query 提供受支持结果
- **THEN** 系统明确报告完整缺失查询上下文
- **AND** 系统不得跳过该节点或改用完整 iteration 时延

### Requirement: AF trace reconstructs queue, service, and makespan

当 `trace_enabled=true` 时，系统 MUST 为每个 rank-local Attention/Dense/Routed/Shared/Combine/Residual/Encode/Decode/Layout compute、Router/Top-K、collective、A-to-F 与 F-to-A endpoint-leg事件记录 iteration、block、layer、layer type、operation/stage、execution domain、DP/TP/EP identity、logical stage id、logical participants、physical resources/endpoints、rank-local shape/workload、assignment 或 original-token count、endpoint payload breakdown、effective dtype role/byte width、`ready_at`、`start_at`、`end_at`、`queue_duration`、`service_duration` 与 provider provenance。Trace MUST 足以重建 assignments 到 histogram、F partials 到 A owner、分支与汇合依赖、资源竞争、block completion、iteration makespan 和只按 service interval 计算的利用率。Trace 关闭时记录开销 MAY 被省略，但调度结果 MUST 不变。

#### Scenario: Trace distinguishes transport from combine

- **WHEN** 一个 MoE layer 完成 F-to-A 和 A weighted combine
- **THEN** trace 分别记录 communication 与 compute 事件及其依赖
- **AND** 两者的 service time 不重叠记账

#### Scenario: Rank-local events reconstruct a synchronization point

- **WHEN** 一个 A rank 的 combine 等待多个 EP partials
- **THEN** trace 可识别该 owner、全部必要前驱及其最晚完成时间
- **AND** combine 的 ready time 不早于最后一份必要 partial 到达时间

#### Scenario: Queued event has internally consistent timestamps

- **WHEN** 一个 ready event 等待繁忙 resource 后完成 service
- **THEN** `queue_duration = start_at - ready_at`
- **AND** `service_duration = end_at - start_at`

#### Scenario: Provider metadata survives trace serialization

- **WHEN** provider 返回 provenance、effective dtype 与 composition/transfer details
- **THEN** trace 保留 dataset、match type、source 与 breakdown
- **AND** details 可重建该 event 的 `service_duration`

#### Scenario: Straggler contributes to makespan

- **WHEN** 某个 Routed 或 Shared Expert rank 因 workload 或排队而晚于其他必要 ranks 完成
- **THEN** 对应 F-to-A/owner combine 等待该分支
- **AND** trace 可重建其对 block makespan 的贡献

#### Scenario: Iteration makespan uses the last block completion

- **WHEN** 多个 blocks 的完成时间不同
- **THEN** iteration makespan 等于最晚 block 完成时间减去 iteration 开始时间
- **AND** 系统不得将 block、rank 或 assignment 耗时直接求和

#### Scenario: Early block waits only at the iteration barrier

- **WHEN** 一个 block 早于同 iteration 的其他 blocks 完成
- **THEN** 该 block MAY 记录从自身完成到 iteration 完成的 barrier wait
- **AND** barrier 不让下一层非 owner ranks 提前获得 activation

#### Scenario: Disabling trace does not change timing

- **WHEN** 相同输入分别以 trace 开启和关闭运行
- **THEN** 两次运行的资源顺序、block 完成时间和 iteration makespan 相同

### Requirement: DP replicas are selected independently by global round-robin

系统 MUST 按 scheduler 输出的确定性顺序，为每个 ready Decode AF block分别选择一个稳定 A DP owner和一个 F service replica。A owner MUST 承担该block全部 A work、request与KV ownership。F service replica MUST 承担Dense FFN、Shared Expert，以及`EP=1`时的Routed Expert TP work。rank-reuse `EP=TP*DP`时，F service replica仍只决定Dense/Shared分支；Routed Expert branches MUST 按placement跨整个F world执行，不得被错误限制在该replica内。未参与当前分支的replicas SHALL 保持可用，相同输入和初始状态 MUST 产生相同选择序列。

#### Scenario: Blocks rotate across replicas

- **WHEN** A domain与F service domain各有两个DP replicas且四个blocks依次ready
- **THEN** A owner与F service replica各自按全局`0, 1, 0, 1`序列轮转
- **AND** 选择不因单个调用方、physical worker或Routed Expert endpoint的局部计数重新开始

#### Scenario: Replica execution is not lockstep

- **WHEN** `EP=1`下replica 0正在执行长block而replica 1已空闲
- **THEN** 分配给replica 1的ready block可立即申请其合法A/F TP resources
- **AND** replica 1不等待replica 0完成

#### Scenario: Rank reuse crosses F replica coordinates only for routed work

- **WHEN** rank-reuse block的assignments命中位于F DP coordinates 0和1的Routed Experts
- **THEN** 两个Routed branches按placement使用这些已有F ranks
- **AND** 该block的Dense/Shared branch仍只使用选中的一个F service replica

#### Scenario: Unused F replica remains available by branch

- **WHEN** rank-reuse block的Routed assignments未命中某F rank且该rank也不属于其选中Dense/Shared TP group
- **THEN** 该rank可服务另一个ready block的合法branch
- **AND** 系统不因一个block拥有F service replica选择结果而锁定整个F world

### Requirement: TP changes rank-local work and creates required collectives

当 `TP > 1` 时，系统 MUST 根据算子与 tensor layout 的真实分片规则，为每个 TP rank 生成精确 local shape、workload 与必要 collective。所有实际分片维度 MUST 在 planning 前通过整除或显式 KV-head replication 规则校验，不得通过 `ceil` 扩大 global model。Collective MUST 只在 layout transition 要求的位置产生，使用配置的 operation/participants/fabric，并等待全部必要 rank 前驱；`TP=1` 时 MUST 不产生虚构 collective。

#### Scenario: Tensor-parallel ranks receive local shapes

- **WHEN** 一个支持的维度以 `TP=2` 合法分片
- **THEN** 两个 rank-local queries 携带精确 local shape
- **AND** 两者聚合后不超过或少于 global dimension

#### Scenario: Non-divisible shape fails before planning

- **WHEN** 8 个 query heads 配置 `TP=3` 且没有该维度的不均匀分片语义
- **THEN** 系统在生成 compute nodes 前失败
- **AND** 不将每 rank heads 向上取整为 3

#### Scenario: Required TP collective joins rank branches

- **WHEN** 分片 operator 的输出或下一 operator 的输入 layout 要求 collective
- **THEN** collective 明确列出实际 participants，并等待最晚必要 rank-local predecessor
- **AND** 下游 node 只在 collective 完成后 ready

#### Scenario: F return may require an A all gather

- **WHEN** F-to-A 按 A tensor owner 返回 sharded output，而下一层 A Attention 要求 replicated input
- **THEN** planner 在 A residual 后创建显式 `all_gather`
- **AND** 下一层全部 A TP ranks 等待该 operation

#### Scenario: Single-rank TP creates no collective

- **WHEN** `TP=1`
- **THEN** 系统执行单个 rank-local compute branch
- **AND** 不产生仅用于模拟并行的零意义 collective

### Requirement: EP executes routed per-rank workload and waits for required experts

系统 MUST 使用当前 MoE layer 的逐 token assignments 和 F Expert placement 计算每个 Routed Expert execution owner 的真实 assignment workload。`EP=1` 时一个 owner 是所选 F DP replica 的 Expert TP group；rank-reuse `EP=TP*DP` 时一个 owner 是实际驻留该 Expert 的单个 F rank。每个 assignment MUST 恰好归属一个 owner；没有 assignment 的 Routed owner MUST 不产生 routed compute。每个原始 token还 MUST 恰好进入一个 Shared Expert TP group，Shared 工作量不得乘以 `top_k`，其分支 SHALL 与 Routed branches 并行。对应 F-to-A owner legs MUST 等待生成正确结果所需的全部 Routed 与 Shared predecessors。

#### Scenario: Uneven routing produces uneven rank workload

- **WHEN** assignments 将 12 次 routed work 分别分配为 rank 0 上 10 次、rank 1 上 2 次
- **THEN**两个 routed Expert compute queries 分别使用 10 和 2 的 local workload
- **AND** 系统不按 rank 平均分配

#### Scenario: EP one executes each assignment across one TP group

- **WHEN** F domain 使用 `EP=1` 且一个 Expert assignment 命中所选 F replica
- **THEN** 该 assignment 在此 replica 的 Expert TP ranks 上执行各自 local shard
- **AND** assignment 在逻辑总工作量中仍只计一次

#### Scenario: Rank reuse executes each assignment on one F rank

- **WHEN** F domain 使用 `EP=TP*DP` 且一个 assignment 命中 Expert 7
- **THEN** 该 assignment 只在 Expert 7 placement 对应的已有 F rank 上执行
- **AND** Expert TP 为 1

#### Scenario: Shared Expert uses original tokens once

- **WHEN** `T` 个 tokens 使用 `top_k=K` 且模型有 Shared Expert
- **THEN** routed 总工作量为 `T*K` assignments，Shared Expert 总工作量为 `T` tokens
- **AND** Shared Expert 不在每个 EP rank 上重复执行全部 `T` tokens

#### Scenario: Empty expert rank performs no expert compute

- **WHEN** 某个 F rank 没有命中 routed assignments，但被选为 Shared Expert execution group 的成员
- **THEN** 该 rank 不产生 routed Expert compute 或伪造的平均 routed workload
- **AND** 若它属于 Shared Expert execution group，仍 MAY 产生语义独立的 Shared Expert local work

#### Scenario: Return transfer waits for the slowest required expert branch

- **WHEN** 多个必要 Routed 与 Shared branches 的完成时间不同
- **THEN** 对应 owner partial return 不早于其最后一个必要 producer 完成
- **AND** 较早分支不会使 block 提前进入 A combine

### Requirement: Expert latency uses routed work and local TP shape

系统 MUST 将 assignment count 解释为已经包含 Top-K 展开的 token-Expert edges。Planner MUST 按 execution owner 和 Expert id 保留该 workload；provider MUST 使用当前 EP mode 的 Expert TP local shape 估算工作。Routed Expert 计算不得再次乘以 `top_k`。Shared Expert MUST 使用原始 token 数，并将多个 Shared Experts 建模为一个等价扩大 intermediate 的 shared MLP，而不是乘成 EP replicas。

#### Scenario: Top-K is expanded exactly once

- **WHEN** `T` 个原始 tokens、`top_k=K` 产生 `T*K` assignments
- **THEN** 全部 Routed Expert owners 的总逻辑 workload 为 `T*K`
- **AND** provider 公式不再乘以 `K`

#### Scenario: F TP shards Expert compute

- **WHEN** 同一 Routed Expert workload 在 F `TP=2`、`EP=1` 上执行
- **THEN** 两个 Expert TP ranks 使用各自精确 intermediate shard
- **AND** 必要 F TP collective 作为独立 communication node 计时

#### Scenario: Rank reuse sets Expert TP to one

- **WHEN** F 使用 rank-reuse `EP=TP*DP`
- **THEN** 每个命中 Expert rank 的 query 使用完整单 Expert shape 与 `expert_tp_size=1`
- **AND** 不再按基础 F TP 值二次切分该 Expert

#### Scenario: Per-Expert skew changes a rank query

- **WHEN** 两个 owners 的 assignment 总数相同但逐 Expert counts 不同
- **THEN** provider query details 保留不同 Expert ids 与 counts
- **AND** owner latency 由这些逐 Expert查询聚合

#### Scenario: Shared Expert uses expanded shared intermediate

- **WHEN** 模型有 `S` 个 Shared Experts、每个 intermediate 为 `I`
- **THEN** Shared branch 的等价 intermediate 为 `S*I` 且 token workload 为原始 `T`
- **AND** 不创建 `S*T` 个 routed assignments

### Requirement: External Expert histograms match the current step

系统 MUST 将 Expert histogram 解释为从当前 layer assignments 派生的只读统计，而不是可替代 token ownership、Expert identity 与 routing weight 的执行计划。外部输入若同时提供 assignments 与 histogram，系统 MUST 校验 histogram 等于 assignments 的聚合且 assignment 总数等于当前 layer 原始 token 数乘以 `top_k`；只提供 histogram 的输入 MAY 用于负载报告，但 MUST NOT 驱动需要 F-to-A ownership combine 的正式执行。任何 routing cache MUST 按 request、phase、decode step 和 layer 隔离。

#### Scenario: Histogram total is inconsistent

- **WHEN** 当前 layer 有 `T` 个 tokens、`top_k=K`，但 assignments 或外部 histogram 总量不等于 `T*K`
- **THEN** 系统在构造该 layer DAG 前报告 workload validation error
- **AND** 不得用该数据继续估算通信或 Expert compute

#### Scenario: Histogram loses ownership information

- **WHEN** 正式 Decode AF 输入只提供 `{expert_id: count}` 而没有逐 token assignments
- **THEN** 系统拒绝将该 histogram 用作 F-to-A ownership 和 weighted combine 的依据
- **AND** 错误说明缺少 token owner、Expert edge 与 weight 语义

#### Scenario: Routing state is isolated by layer and step

- **WHEN** 同一 request 连续执行两个 Decode steps 或两个 MoE layers
- **THEN** 每个 step/layer 独立产生并校验 assignments 与 histogram
- **AND** 系统不得复用其他 step、phase 或 layer 的 routing result

#### Scenario: Prefill and Decode use different histograms

- **WHEN** 同一 request 同时存在 Prefill 与 Decode routing statistics
- **THEN** 两个 phase 分别从各自 step/layer assignments 聚合并校验 histogram
- **AND** Decode 不得复用已缓存的 Prefill routing state

### Requirement: Deterministic end-to-end oracles validate pipelining

系统 MUST 支持从真实 AF config、scheduler output、model layer metadata、独立 P/A/F mappings 和注入的 deterministic providers 运行到最终 iteration result 与 trace。端到端测试 MUST 覆盖 Dense/MoE layer selection、两种 EP 模式、逐 token assignments、Shared branch、ownership return、A combine/residual、必要 collectives、dtype conversion、resource capacity、block overlap 与最终 makespan。测试 MUST 从 DAG、service 和 queue intervals 推导预期时间，不得把 stub 数值声明为物理性能结论。

#### Scenario: Single block oracle merges EP communication into AF transfers

- **WHEN** 单层 MoE block 的 routed/shared、A-to-F 与 ownership return operations 使用固定 service time 且无排队
- **THEN** trace 恰好包含一个逻辑 A-to-F dispatch stage 和一个逻辑 F-to-A partial-return stage
- **AND** block completion 等于完整 DAG 关键依赖路径上的 service time 之和
- **AND** trace 不包含独立 `ep_dispatch` 或 `ep_combine` transport events

#### Scenario: Shared AF communication resource changes makespan

- **WHEN** 相同两 block fixture 分别让 A-to-F 与 F-to-A stages 竞争同一容量为一的 resource 和使用互不竞争的 resources
- **THEN** 共享配置的 trace 显示合法 queue intervals，且两次 makespan 差异可由这些 intervals 精确重建
- **AND** compute provider service durations 在两次运行中保持相同

#### Scenario: Two blocks form an eleven millisecond pipeline

- **WHEN** 兼容的旧 deterministic fixture 为更新后的完整 Dense/MoE DAG 提供固定 service durations，并为新增语义节点设置明确测试值
- **THEN** iteration makespan 等于 trace 推导的 11 毫秒最晚 block completion
- **AND** trace 展示合法 overlap，既不强制两个 blocks 完全串行，也不允许同一容量为一的 service intervals 重叠

#### Scenario: Mixed Dense and MoE block follows distinct DAGs

- **WHEN** 一个 block 依次执行一层 Dense 和一层含 Shared Expert 的 MoE，且每个 provider operation 有固定 service time
- **THEN** trace 只在 MoE 层包含 Router、assignments 和 weighted combine
- **AND** block completion 可从两层最终 A-side nodes 重建

#### Scenario: Rank reuse oracle keeps physical count fixed

- **WHEN** F 配置 `TP=2`、`DP=2`、`EP=4`
- **THEN** oracle 只观察到四个 F physical compute resources
- **AND** routed skew 与 queueing 可由这四个 resources 上的 events 解释

#### Scenario: Ownership return oracle cannot end at A leader only

- **WHEN** A `TP=2` 且 Routed partials 需要两个 A tensor owners
- **THEN** F-to-A endpoint legs覆盖两个 A ranks，各 rank 等待自己的 partials
- **AND** 下一层不能仅因 A leader 收到数据而开始

#### Scenario: Bottleneck and duplex variants remain explainable

- **WHEN** 测试改变 A/F compute、cross-domain service、duplex mode 或 physical F rank 数
- **THEN** makespan 变化可由 trace 的 service、queue 与 overlap intervals 解释

#### Scenario: Independent domain counts execute successfully

- **WHEN** Prefill、A 与 F 使用不同合法 topology 和 worker 数，且每个 domain 基础 ranks 与 physical workers 一一映射
- **THEN** Decode AF iteration 成功完成
- **AND** 系统不要求 P/A/F worker 总数等于一套 global parallel product

#### Scenario: AF-disabled baseline remains stable

- **WHEN** 回归套件以同一 model/config 和 workload 设置 `af.enabled=false`
- **THEN** 系统不创建 AF assignments、ownership、compute、communication、provider queries 或 trace events
- **AND** 既有非 AF backend 的 request、KV 与 timing 行为保持不变

## ADDED Requirements

### Requirement: Routing assignments preserve token and combine identity

每个 MoE Decode layer 的 routing result MUST 是一组逐 token assignments，而不是只有 histogram。每个 assignment MUST 保留 iteration/step、layer、request/token identity、稳定 A DP owner、A tensor/layout owners、Expert id 与 routing weight。相同输入和 routing seed MUST 产生相同 assignments；synthetic balanced routing MUST 在整个 layer block 上连续生成，不能在每个 request 开头重置。

#### Scenario: One token keeps two weighted edges

- **WHEN** 一个 Decode token 以 `top_k=2` 路由到 Experts 3 和 7
- **THEN** routing result 保留两条具有相同 token identity、不同 Expert id 与各自 weight 的 assignments
- **AND** histogram 只能聚合出两个 counts，不能替代这两条 edges

#### Scenario: Balanced routing advances across the layer block

- **WHEN** 四个 Decode tokens、`top_k=2`、八个 Experts 使用 balanced synthetic routing
- **THEN** 系统在同一 layer block 上连续产生八条 assignments
- **AND** 不因逐 request 重置而让四个 tokens 全部命中 Experts 0 和 1

### Requirement: Shared Expert executes once for every original token

MoE Decode layer 中每个原始 token MUST 进入一个 Shared Expert logical branch；该 branch SHALL 与 Routed Expert branches 并行。Shared workload MUST 按原始 `T` 个 tokens 计算，不受 `top_k` 或 EP identities 数量放大；多个 Shared Experts MUST 通过扩大 shared intermediate 表达。Shared output MUST 在 A owner 侧与 routed weighted result 汇合。

#### Scenario: Routed and shared workloads use different counts

- **WHEN** `T` 个 tokens 使用 `top_k=K` 且模型含 Shared Experts
- **THEN** Routed 总 workload 为 `T*K` assignments，Shared 总 workload 为 `T` tokens
- **AND** 两个 branches 可在资源允许时重叠

#### Scenario: Shared output participates in the A join

- **WHEN** 某 token 的全部 Routed partials 已返回但 Shared partial 尚未返回
- **THEN** 该 token owner 的 combine/residual 尚不可 ready
- **AND** Shared partial 到达后才完成最终求和

### Requirement: F results return to owning A ranks

F-to-A MUST 是按 token/tensor ownership 的 partial return，而不是 mathematical combine。每个 F output MUST 按目标 A layout 切分为 owner-specific legs；每个 A TP rank只等待并本地 gather 属于自己的 Routed 与 Shared partial shards，再执行 weighted combine 与 residual。非 owner ranks MUST 通过显式 layout collective 获取其所需数据，iteration barrier 不得替代数据移动。

#### Scenario: Full Expert output is scattered by A tensor shard

- **WHEN** rank-reuse Expert 产生一个 hidden width 为 `H` 的完整 output且 A 使用 `TP=2`
- **THEN** F-to-A 为两个 A tensor owners 各返回一个精确 `H/2` shard
- **AND** 两个 payload shard 聚合后恰好为一个 `H` output，不重复完整 tensor

#### Scenario: One A rank waits for all of its EP partials

- **WHEN** 同一 token shard 的 Top-K partials 来自两个 F producers
- **THEN** 对应 A owner 在两份 partial 都到达后执行 local combine
- **AND** 另一个 A rank 的到达状态不能替代该依赖

### Requirement: Decode payloads follow layer and ownership semantics

Planner MUST 从具体 tensor items 与 endpoint legs 计算 payload，而不得只用一个无 owner 的总 token count。每个 source/target tensor shard MUST 以全局hidden-index区间和`sharded`、`replicated`或`partial` layout描述；跨不同TP sizes的legs MUST 由source/target区间交集产生。Replicated source MUST 由稳定canonical producer发送，数学上不同的partial sources MUST 全部保留。Dense A-to-F MUST 携带原始token activation；MoE A-to-F MUST 携带每条routed assignment的activation和Shared branch所需原始token input。Dense F-to-A MUST 携带每token output；MoE F-to-A MUST 携带Routed partials与Shared output partials。相同source slice到相同endpoint的同一logical copy MUST 去重。每个方向 MUST 使用其effective transfer dtype；若与activation dtype不同，边界转换 MUST 作为独立compute nodes计时并按source/target worker、stable tensor item identity、slice interval和dtype pair去重。

#### Scenario: MoE return preserves Top K partials until combine

- **WHEN** `T` 个 tokens、`top_k=K` 在 A 侧 combine
- **THEN** F-to-A 在 combine 前保留 `T*K` Routed partial identities 及每 token一份 Shared partial identity
- **AND** 返回后 activation 数才由 A weighted combine 恢复为 `T`

#### Scenario: Endpoint payload is the sum of unique tensor items

- **WHEN** 多个 logical consumers 引用同一 source tensor slice 与同一 destination endpoint
- **THEN** 该 endpoint leg 的 payload bytes 只计一次物理 copy
- **AND** 不同 destination endpoints 仍分别进入 payload breakdown

#### Scenario: Different A and F TP sizes map by global intervals

- **WHEN** 相同hidden tensor分别从`A_TP=2`映射到`F_TP=4`，或从`F_TP=4`返回`A_TP=2`
- **THEN** 每条leg只携带source与target全局区间的非空交集
- **AND** 所有unique intervals拼接后恰好覆盖global hidden width且无缺口或重复

#### Scenario: Replicated output selects one canonical producer

- **WHEN** F TP all-reduce后多个ranks持有同一replicated output
- **THEN** 每个target tensor item只由该group最低domain rank的canonical producer发送
- **AND** 其他replicas不重复发送相同tensor item

#### Scenario: Partial outputs are not deduplicated as replicas

- **WHEN** 多个F producers持有同一hidden interval的不同mathematical partials
- **THEN** A owner接收所有必要partials并在combine/reduction语义中保留各自identity
- **AND** payload去重不得删除这些数值不同的贡献

#### Scenario: Conversion is deduplicated around fan out

- **WHEN** 同一source tensor item从BF16编码为FP8并发送到两个不同endpoints
- **THEN** source worker执行一次该item的encode，两个endpoint legs复用encoded item
- **AND** 每个target worker分别对自己收到的item执行一次decode

#### Scenario: Distinct tensor items do not share conversion

- **WHEN** 两个不同tokens或两个不同Top-K partials在同一stage、同一worker上具有相同hidden interval与dtype pair
- **THEN** stable tensor item identity使它们各自生成encode/decode work
- **AND** 去重不得把数值不同的items合并为一次转换
