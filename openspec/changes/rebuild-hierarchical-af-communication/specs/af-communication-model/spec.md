## Purpose

规定 AF 通信的逻辑数据量、算法阶段、纯服务估值和缓存边界，使 flat 与 physical 使用一致的数据契约，并在可控制的仿真粒度下保留正确的依赖、传输次数和完成语义。

## ADDED Requirements

### Requirement: Collective payload and rounds have one meaning

Collective payload MUST 表示每 rank 逻辑输入字节 M，估值 MUST 另行记录输出、wire bytes、当前子组 N、算法与轮次。同质 ring AllGather、ReduceScatter、AllReduce 的每 rank 总发送量 MUST 分别为 (N−1)M、(N−1)M/N、2(N−1)M/N，轮次分别为 N−1、N−1、2(N−1)。算法显式展开轮次时 MUST 不重复计轮次固定延迟。测量表使用完整 buffer 口径时 MUST 明确转换，不能直接对同一 M 套不同含义系数。

#### Scenario: AllGather local payload is conserved
- **WHEN** N=4 且每 rank 输入 M=100 bytes
- **THEN** 每 rank 输出 400 bytes，ring 发送 300 bytes，经历 3 轮
- **AND** flat 与单层同质 physical 在相同算法和参数下具有一致纯服务估值

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

### Requirement: Path service and phase service remain distinct

单个 transmission 的路径服务 MUST 使用 S≈Σalpha_segment+M/min(B_segment) 的整路径流水近似，并将各物理段一次性纳入该任务资源需求。串行算法阶段 MUST 分别估值及排队，不能对所有阶段统一取最小带宽。并行 subgroup 的完成 MUST 根据依赖事件计算；粗粒度 collective 阶段 MUST 按实际轮次和资源流量计费，不能把多 NIC 非均匀负载平均成池化带宽。模型 MUST 披露整路径/粗阶段预留及未建模 SM/HBM 竞争的近似。

#### Scenario: Serial stages retain an intermediate queue
- **WHEN** 三个串行算法阶段服务为 2、5、3，第一阶段 t=0 开始，第二阶段资源 t=6 可用，其余无等待
- **THEN** 三阶段区间分别为 [0,2]、[6,11]、[11,14]，中间排队为 4
- **AND** 第二阶段路径内部不再增加中途排队

### Requirement: Message packing preserves readiness and ownership

系统 MUST 仅合并算法真实打包为一次发送的数据，要求同一 operation 实例、phase、source/target GPU、path 与兼容 ready/consumer completion 边界；合并后 wire bytes MUST 求和，启动成本按真实发送次数计算。MUST NOT 跨独立调用、block/iteration 合并，或将节点内外传输合并，或阻塞原本可以早完成的消费者。路由 assignment、shared 输入复用和 weighted combine MUST 保留既有语义。统计聚合 MUST 不改变执行粒度。

#### Scenario: Mixed local and remote destinations stay distinct
- **WHEN** 同一个 A 分别发送给本节点 F 与远端 F
- **THEN** 两条消息保留各自路径、资源、服务与排队

#### Scenario: Similar messages with different readiness remain distinct
- **WHEN** 两条同端点消息属于独立调用或具有不同下游完成边界
- **THEN** 不因 payload/路径相同而只执行一次或等待合并

### Requirement: Cache reuse excludes queue and execution state

缓存 MUST 仅复用静态模板、路径、计划和纯 service；key 或不可变指纹 MUST 覆盖模型/数据版本、topology、placement、NIC mapping、算法、payload、dtype、layout 与 routing identity。每次调用 MUST 独立创建 ready/start/end、请求资源并完成，缓存不得包含队列等待、事件或执行状态。计划 MUST 按需构造，不扫描全部 shape 配置；不得通过静默截断通信任务降低事件规模。

#### Scenario: Cache hits still contend
- **WHEN** 两次相同查询同时命中缓存并请求同一独占资源
- **THEN** 两次调用均执行，第二次按实际竞争记录排队
- **AND** 关闭缓存不改变时间线或结果

### Requirement: Service provenance and validation state limitations

纯服务 MUST 携带 timing_scope、来源、模型/数据版本和证据。启用 host 分离计费时 MUST 仅接受明确 transport_service，拒绝可能含 host 成本的 eager/full/unknown；不得对每个内部阶段重复加同一逻辑调用的 host 费用。验证报告 MUST 分别覆盖整路径提前占接收端与小消息流水/去 H 误差，说明实测证据可得性，并记录任务/事件数、CPU/wall、内存及 trace 成本。

#### Scenario: Ambiguous measurement fails with separated host costs
- **WHEN** 分离 host 计费配置收到 eager_interval 或未知作用域测量
- **THEN** 在使用估值前明确失败并给出数据来源与作用域

#### Scenario: No hardware measurements are available
- **WHEN** 只有手算及细粒度参考模拟可用于对照
- **THEN** 报告将结论限定为自洽性和近似敏感性，不宣称实测精度

### Requirement: Coarse phases include internal contention

粗粒度 phase MUST 使用算法轮次、真实发送次数与实际方向资源计算无外部竞争的内部时间表；共享接口的发送 MUST 受独占/独立 lane 容量约束，不相交路径可并行。phase 服务 MUST 为该时间表的完成时间，外部资源需求 MUST 为使用资源的去重并集，重复字节与启动成本由内部时间表计费。不能用单消息最大时长忽略 phase 内部共享 NIC，也不能对同一资源重复申请造成自我等待。

#### Scenario: Two internal sends share one NIC
- **WHEN** phase 包含两条各需 5 微秒的传输，共用容量一的 NIC tx，其他条件无额外等待
- **THEN** phase 服务为 10 微秒而不是 5 微秒
- **AND** 对该 NIC 只发起一次外部预留

### Requirement: Measurements match algorithm and stage scope

使用实测阶段服务时 MUST 匹配算法版本、participants/拓扑、payload 口径和阶段范围。仅有整个 collective 的 transport_service 总时长 MUST 不被按字节比例分摊或重复用于每个阶段。缺少匹配数据 MUST 遵守显式 provider fallback：允许解析时报告解析来源，measured-only 时失败。

#### Scenario: Whole collective measurement cannot price each phase
- **WHEN** physical 多阶段算法只有一个整个 collective 的 20 微秒测量记录
- **THEN** 不将 20 微秒应用于每阶段，也不臆造各阶段比例
- **AND** 按显式 fallback 使用解析估值或明确失败

### Requirement: Payload and participant validation prevents silent repair

bytes MUST 为非负整数并符合 dtype/shard 边界，group_size MUST 等于已验证的实际 participants 数；空 group、重复或缺失 participant MUST 失败，单 rank MUST 不被提升为两 rank。wire bytes MUST 明确表示算法数据与显式元数据，未建模的协议冗余不得被宣称已包含，逐段流量统计 MUST 与应用逻辑发送量分列。

#### Scenario: Singleton group does not fall through to a two-rank estimate
- **WHEN** 一个合法单 rank collective 输入非零且无网络需求
- **THEN** 不生成 ring 网络传输或查询两 rank 估值
