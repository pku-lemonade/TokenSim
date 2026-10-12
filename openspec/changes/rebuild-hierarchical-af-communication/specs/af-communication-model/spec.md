## Purpose

规定 AF 通信的逻辑数据量、算法阶段、纯服务估值和缓存边界，使 flat 与 physical 使用一致的数据契约，并在可控制的仿真粒度下保留正确的依赖、传输次数和完成语义。

## ADDED Requirements

### Requirement: Collective payload and rounds have one meaning

Collective payload MUST 表示每 rank 逻辑输入字节 M，估值 MUST 另行记录输出、wire bytes、当前子组 N、算法与轮次。同质 ring AllGather、ReduceScatter、AllReduce 的每 rank 总发送量 MUST 分别为 (N−1)M、(N−1)M/N、2(N−1)M/N，轮次分别为 N−1、N−1、2(N−1)。节点内共享域 MUST 按各轮全部实际有向发送字节求和后使用节点总带宽计时，每条发送计一次，不以单 rank 流量直接除以整节点带宽。算法显式展开轮次时 MUST 不重复计轮次固定延迟。测量表使用完整 buffer 口径时 MUST 明确转换，不能直接对同一 M 套不同含义系数。

#### Scenario: AllGather local payload is conserved

- **WHEN** N=4 且每 rank 输入 M=100 bytes
- **THEN** 每 rank 输出 400 bytes，ring 发送 300 bytes，经历 3 轮
- **AND** flat 与 physical 使用相同的算法流量规则；纯服务仅在资源容量及聚合口径也相同时一致

### Requirement: Analytical service uses explicit latency and bandwidth

新的基础解析通信服务 MUST 以明确固定延迟、实际 wire bytes 和峰值方向带宽计算，不使用 H/half_bandwidth_bytes 或未校准 collective_efficiency。oversubscription MUST 仅作用于数据带宽项；显式容量已覆盖的同一瓶颈 MUST 不重复施加超售惩罚。旧字段兼容性 MUST 明确标注，不得影响其他尚未迁移消费者。无实际网络工作的零字节/单 rank 操作 MUST 不申请网络资源或收费，实际本地 copy/reorder/reduction MUST 单独保留。

#### Scenario: Oversubscription leaves startup unchanged

- **WHEN** 单链路 alpha=2 微秒，M/B=10 微秒，超售系数从 1 变为 2
- **THEN** 解析服务从 12 微秒变为 22 微秒
- **AND** 不将 alpha 加倍

#### Scenario: Deprecated H cannot alter the new estimate

- **WHEN** 仅改变旧 H 或未校准 collective_efficiency 字段
- **THEN** 新基础解析模型结果不变，输出说明该字段在本模型中的兼容状态

### Requirement: Algorithms expose supported hierarchical phases

系统 MUST 使用确定的基础 ring 与层级模板，明确 subgroup、shard ownership、wire bytes、阶段依赖和所需资源。首版层级 AllReduce SHALL 为节点内 ReduceScatter、节点间 shard AllReduce、节点内 AllGather。层级 AllGather SHALL 先由对应逻辑 local slot 跨节点聚合各自输入分片，再在节点内聚合结果，不能重复跨节点发送每个节点的完整 buffer。ReduceScatter MUST 先按目标节点切分输入，在节点内归约到 local slot，再由对应 slot 跨节点归约散发至目标 rank。local slot MUST 按当前子组稳定顺序确定，通过 placement 映射物理 GPU，不要求物理卡号相同。每节点参与数相同及分片整除的支持域 MUST 在执行前校验，不支持布局明确失败。P2P/AF endpoint legs MUST 遵循实际 ownership，不得隐式新增通用同 local_rank 转发、gather/scatter 或重复 EP dispatch/combine。

#### Scenario: Hierarchical reduction exposes distinct phases

- **WHEN** 均匀两节点 group 执行层级 AllReduce
- **THEN** trace/计划可识别节点内归约分片、对应 shard 跨节点归约及节点内聚合
- **AND** 每阶段的数据量由上一阶段实际输出推导

#### Scenario: Uneven unsupported layout fails

- **WHEN** 节点参与 rank 数不满足模板支持域
- **THEN** 系统执行前报告布局不支持，不能通过总 rank 数近似生成错误阶段

#### Scenario: Hierarchical AllGather transfers each shard across nodes

- **WHEN** 两节点各 4 ranks，每 rank 输入 100 bytes
- **THEN** 跨节点 AllGather 后每 rank 持有 200 bytes，随后节点内聚合得到每 rank 800 bytes
- **AND** ring 跨节点总发送量为 800 bytes，不是先节点内复制后的 3200 bytes

### Requirement: Domain service and phase service remain distinct

节点内 transmission MUST 使用 S=alpha+M/B_local 并独占本节点 local 资源；跨节点 transmission MUST 使用 S=alpha_inter+M/min(B_source_tx,B_target_rx)，原子占用源 tx 和目标 rx，固定延迟按声明范围计一次，不展开额外物理段。串行算法阶段 MUST 分别估值、申请和释放资源，并行 subgroup MUST 根据依赖计算完成。模型 MUST 披露域级独占 FIFO、粗 phase 资源持有及未建模 SM/HBM 竞争的近似。

#### Scenario: Serial stages retain an intermediate queue

- **WHEN** 三个显式串行阶段服务为 2、5、3，第一阶段 t=0 开始，第二阶段资源 t=6 可用，其他无等待
- **THEN** 执行区间为 [0,2]、[6,11]、[11,14]，中间排队为 4
- **AND** 第三阶段仅在算法或显式路由要求时存在

#### Scenario: Node aggregate counts each directed send once

- **WHEN** 单节点四 rank ring AllGather 每 rank 输入 100 bytes，节点带宽 B，忽略固定延迟
- **THEN** 每 rank 发送 300 bytes，节点域总发送 1200 bytes，数据服务为 1200/B
- **AND** 不以 300/B 代表整个节点服务，不因同时接收再翻倍

#### Scenario: Inter-node bandwidth uses both reservations

- **WHEN** 源节点 tx 为 50 GB/s，目标节点 rx 为 25 GB/s
- **THEN** 单次跨节点传输的数据项使用 25 GB/s，服务期间同时持有两项方向资源

### Requirement: Message packing preserves readiness and ownership

系统 MUST 仅合并算法真实打包为一次发送的数据，要求同一 operation 实例、phase、source/target GPU、通信域资源 与兼容 ready/consumer completion 边界；合并后 wire bytes MUST 求和，启动成本按真实发送次数计算。MUST NOT 跨独立调用、block/iteration 合并，或将节点内外传输合并，或阻塞原本可以早完成的消费者。路由 assignment、shared 输入复用和 weighted combine MUST 保留既有语义。统计聚合 MUST 不改变执行粒度。

#### Scenario: Mixed local and remote destinations stay distinct

- **WHEN** 同一个 A 分别发送给本节点 F 与远端 F
- **THEN** 两条消息保留各自域资源、服务与排队

#### Scenario: Similar messages with different readiness remain distinct

- **WHEN** 两条同端点消息属于独立调用或具有不同下游完成边界
- **THEN** 不因 payload/资源相同而只执行一次或等待合并

### Requirement: Cache reuse excludes queue and execution state

系统 MUST 扩展既有有界服务/计划缓存，不新建同职责的重复缓存。缓存 MUST 仅复用不可变静态模板、静态通信域资源需求、阶段依赖和逐阶段纯 service；服务估值键 MUST 包含所有影响纯服务的参数及模型/数据版本，计划键或不可变配置指纹 MUST 额外覆盖 topology、placement、通信域资源映射、participants 及顺序、layout、routing/locality 特征和阶段依赖。相同 payload 但不同资源或依赖 MUST 不得误用同一具体计划。现有服务键对 routing_edges、tensor_slices 和执行 node_id 的排除 MUST NOT 直接复制为计划键规则；影响流量、实际节点资源映射和依赖的字段 MUST 规范化纳入计划键，只排除纯调用身份。

每次调用 MUST 重新绑定本次 operation/phase 身份，逐阶段独立创建 ready/start/end、申请并释放资源和计算排队；不得用缓存整体时长替代阶段调度。缓存 MUST NOT 包含绝对时间戳、外部队列等待、资源可用时间、事件或执行状态。内部时间表 MAY 缓存，但 MUST 以阶段开始为零点、假设输入已齐且无外部竞争。

配置 MUST 使用单次运行不可变快照；带宽、延迟、topology、资源映射、locality、算法或数据版本变化时 MUST 创建新 runtime/provider 和缓存，禁止跨不兼容快照复用。缓存 MUST 按需构造且容量有界，不扫描全部 shape 配置，不通过静默截断通信任务降低事件规模。

#### Scenario: Cache hits still contend

- **WHEN** 两次相同查询同时命中缓存并请求同一独占资源
- **THEN** 两次调用均执行，第二次按实际竞争记录排队
- **AND** 关闭缓存不改变时间线或结果

#### Scenario: Same plan, different queue state

- **WHEN** 两次调用静态计划相同，但通信域资源空闲时刻不同
- **THEN** 两次调用可复用同一个不可变服务计划
- **AND** ready/start/end 与实际排队独立计算

#### Scenario: Cached phases retain intermediate queueing

- **WHEN** 节点内阶段服务为 3 微秒，节点间阶段服务为 8 微秒，后者资源直到 t=10 才空闲，任务 t=0 就绪
- **THEN** 节点内阶段执行于 [0,3]，节点间排队为 7 微秒、执行于 [10,18]
- **AND** 缓存命中不把整体 11 微秒服务合并为一次资源占用

#### Scenario: Relative timetable is rebound per invocation

- **WHEN** 同一个无外部竞争相对时间表被两个不同调用复用
- **THEN** 计划保留从零起算的相对偏移，但各调用使用自己的事件和 operation/phase 身份
- **AND** 不复用上一调用的绝对时间戳或完成状态

#### Scenario: Equal payload with different resources selects different plans

- **WHEN** 两次查询 payload 相同，但目标节点、通信域资源映射、routing/locality 或阶段依赖不同
- **THEN** 计划键或配置指纹区分两次查询
- **AND** 纯服务参数相同时可复用服务估值，但不能复用错误的资源计划

#### Scenario: Reconfiguration creates a fresh cache

- **WHEN** 节点内带宽由 300 GB/s 改为 600 GB/s，或其他服务/计划配置改变
- **THEN** 新 runtime/provider 使用新快照和缓存重新估值
- **AND** 旧 runtime 的不可变配置及缓存不被原地修改

### Requirement: Service provenance and validation state limitations

纯服务 MUST 携带 timing_scope、来源、模型/数据版本和证据。启用 host 分离计费时 MUST 仅接受明确 transport_service，拒绝可能含 host 成本的 eager/full/unknown；不得对每个内部阶段重复加同一逻辑调用的 host 费用。验证报告 MUST 分别覆盖域级 FIFO 相对带宽共享参考的敏感性、粗 phase 延迟释放和小消息固定延迟/去 H 误差，说明实测证据可得性，并记录任务/事件数、CPU/wall、内存及 trace 成本。

#### Scenario: Ambiguous measurement fails with separated host costs

- **WHEN** 分离 host 计费配置收到 eager_interval 或未知作用域测量
- **THEN** 在使用估值前明确失败并给出数据来源与作用域

#### Scenario: No hardware measurements are available

- **WHEN** 只有手算及细粒度参考模拟可用于对照
- **THEN** 报告将结论限定为自洽性和近似敏感性，不宣称实测精度

### Requirement: Coarse phases include internal contention

粗 phase MUST 按算法轮次、真实发送次数和域资源建立输入已齐、无外部竞争的零起点相对时间表。同一节点 local、同源 tx 或同目标 rx 上的发送 MUST 按稳定顺序独占服务，不相交资源 MAY 并行。内部各轮 MUST 累计实际有向 wire bytes，每条发送计一次；只有真实一次打包才共享固定启动成本。phase 服务 MUST 为内部完成时间，外部资源需求 MUST 为 canonical resources 的去重并集；首版粗 phase 在整个服务期间持有该并集，MUST 披露延迟释放的保守性，不重复申请同一资源。

#### Scenario: Two internal sends share one local domain

- **WHEN** phase 内两条发送各需 5 微秒，均使用同一节点 local 资源，其他条件无额外等待
- **THEN** phase 服务为 10 微秒，不能取单条最大值 5 微秒
- **AND** 对该 local 资源只发起一次外部预留

#### Scenario: Internal disjoint domains can overlap

- **WHEN** phase 内两项各需 5 微秒的传输分别使用 node0→node1 和 node2→node3，且无相互依赖
- **THEN** 无外部竞争的内部完成时间为 5 微秒，不将独立发送串行相加

### Requirement: Measurements match algorithm and stage scope

使用实测阶段服务时 MUST 匹配算法版本、participants/拓扑、payload 口径和阶段范围。仅有整个 collective 的 transport_service 总时长 MUST 不被按字节比例分摊或重复用于每个阶段。缺少匹配数据 MUST 遵守显式 provider fallback：允许解析时报告解析来源，measured-only 时失败。

#### Scenario: Whole collective measurement cannot price each phase

- **WHEN** physical 多阶段算法只有一个整个 collective 的 20 微秒测量记录
- **THEN** 不将 20 微秒应用于每阶段，也不臆造各阶段比例
- **AND** 按显式 fallback 使用解析估值或明确失败

### Requirement: Payload and participant validation prevents silent repair

bytes MUST 为非负整数并符合 dtype/shard 边界，group_size MUST 等于已验证的实际 participants 数；空 group、重复或缺失 participant MUST 失败，单 rank MUST 不被提升为两 rank。wire bytes MUST 明确表示算法数据与显式元数据，未建模的协议冗余不得被宣称已包含，逐域流量统计 MUST 与应用逻辑发送量分列。

#### Scenario: Singleton group does not fall through to a two-rank estimate

- **WHEN** 一个合法单 rank collective 输入非零且无网络需求
- **THEN** 不生成 ring 网络传输或查询两 rank 估值
### Requirement: Hierarchical All-to-All exposes locality without requiring exact routing

选择层级 All-to-All 时，模型 MUST 显式生成源节点内重排与节点间远端交换阶段，记录 logical payload、逐域 wire bytes、依赖及资源。目标节点二次分发 MUST 默认禁用，只能由显式节点聚合路由启用；直接交付最终目标 rank 时 MUST NOT 增加该阶段。源节点重排 MUST 按所选计划的实际搬运量计费，已经位于正确中转 rank 的数据不重复搬运，纯布局标记不自动产生网络字节。仅本地交付与远端重排的流量 MUST 分列，不将所有 local payload 都假定为零通信或都算作重排。

top_k MUST 只决定 expert activation 总 payload，不解释为 remote_fraction。无精确 token 路由时 MUST 支持 local_fraction/remote_fraction（有限且在 [0,1]，同时给出时总和为 1）及显式统计假设。可选 load_imbalance MUST 表示固定总量下的热点分布，不能隐式替代或修改 remote_fraction；改变 locality 的实验 MUST 明确修改 fraction。热点用于改变域间负载或内部排程，不凭空增加 wire bytes；缺少分布证据 MUST 标为统计假设，不伪造精确目的 rank。

结果 MUST 标记 exact_routing、statistical_locality 或 uniform_fallback，保留既有精确 AF routing/ownership。统计模式 MUST 不要求逐 token expert placement；uniform fallback MUST 基于当前参与者布局明确说明同 rank、本节点其他 rank 和远端节点的处理，不能仅凭物理 node_size 假设所有卡参与。local_fraction 表示本节点比例（含无需网络的同 rank 部分），wire bytes MUST 按统计假设另行推导。participants、当前层级 fan 与 wire_bytes MUST 分别记录。规则布局支持域沿用算法要求，不强制满节点。

#### Scenario: Default route delivers directly

- **WHEN** 规则多节点 All-to-All 选择同逻辑 local slot 中转，跨节点后已到最终目标 rank
- **THEN** 计划具有实际需要的源节点内重排和远端交换
- **AND** 不生成目标节点内二次分发

#### Scenario: Redistribution is explicit

- **WHEN** 用户显式选择将远端数据收到节点级中间 rank 再转交的路由
- **THEN** 计划增加目标节点 local 阶段，记录其字节和依赖

#### Scenario: Uniform locality fallback

- **WHEN** 未提供路由或 fraction，两个节点各有 4 个当前参与 rank，物理 node_size 为 8
- **THEN** 均匀目的 rank 假设下本节点比例为 1/2，同 rank 比例为 1/8，远端比例为 1/2
- **AND** 标为 uniform_fallback，不假定每节点 8 个 rank 参与，也不将同 rank 部分计为网络发送

#### Scenario: Locality changes node-level cost

- **WHEN** 两查询 num_tokens、top_k、hidden_size 相同，但 remote_fraction 不同
- **THEN** 总逻辑 payload 相同，远端 wire bytes 和服务可不同

#### Scenario: Hotspot does not create payload

- **WHEN** remote_fraction 和总 payload 固定，但统计远端流量集中到更少目标节点
- **THEN** 总远端 wire bytes 不变，目标 rx 负载和排队可以变化
- **AND** 结果明确热点假设，不将该变化伪装为 locality 改变

#### Scenario: Exact routing is opt-in

- **WHEN** 未提供 expert placement 或 token routing
- **THEN** 统计计划不要求或伪造逐 token 目的 rank
- **AND** 已有精确 AF legs 仍按真实 ownership 执行，不重复生成 EP dispatch/combine
