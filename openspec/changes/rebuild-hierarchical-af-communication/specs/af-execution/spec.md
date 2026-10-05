## MODIFIED Requirements

### Requirement: AF transfers compete on normalized communication resources

系统 MUST 让每个 A-to-F/F-to-A endpoint leg 与 execution plan 要求的 collective 申请各自显式解析的 communication resources，并记录独立 queue 与 service 区间。A-to-F 与 F-to-A 各自仍是每层一个 logical stage，但 MUST 以 stable stage id 关联多个 executable legs；相同 tensor slice 发往相同 endpoint MUST 按既有 payload 语义只计一次。Logical stage completion MUST 是其最晚必要 leg completion，数据依赖 MUST 可直接指向相关 leg 而不引入全 stage barrier。Collective 的 operation、participants、network、算法阶段和资源需求 MUST 来自 domain communication plan。flat_link MUST 保留显式 resource group/lane 语义；physical MUST 使用实际路径的方向接口、共享总线及上联资源，不能为同一瓶颈再申请旧全局 fabric lane。全双工相反方向 SHALL 按配置独立竞争；半双工相反方向 MUST 竞争同一 shared resource 容量。系统 MUST NOT 为同一 AF payload 再创建 ep_dispatch 或 ep_combine stage。

#### Scenario: AF transfers contend on a shared resource
- **WHEN** 多个 A-to-F 或 F-to-A legs 映射到同一容量为一的 resource
- **THEN** 该 resource 上 service intervals 不重叠，后获得资源的 leg 记录 queue duration

#### Scenario: Collective participants determine resource use
- **WHEN** communication plan 创建具有明确 operation 与 participant ranks 的 collective
- **THEN** flat_link 使用计划指定 group/lane，physical 使用阶段解析的实际资源集合
- **AND** trace 保留 operation、participants、network 与完整 resource identity

#### Scenario: Missing collective resource is not replaced by an AF link
- **WHEN** 必要 collective 在当前模式中缺少可解析 network 或所需资源
- **THEN** 系统执行前失败，不借用 A-to-F 或 F-to-A fabric

#### Scenario: Full-duplex directions overlap independently
- **WHEN** 两条传输使用同一 full-duplex 资源的两个方向且无其他共享瓶颈
- **THEN** 两方向可在各自容量允许时同时服务

#### Scenario: Explicit collective lane controls full-duplex contention
- **WHEN** flat_link 的 A-to-F、F-to-A 与 collective 同时 ready，collective 显式选择 a_to_f lane
- **THEN** collective 与 A-to-F 竞争该 lane，可与 F-to-A 并行
- **AND** executor 不按 operation 名称改写 lane

#### Scenario: Half-duplex directions serialize at capacity one
- **WHEN** 相反方向 transfers 使用容量一的 half-duplex shared key
- **THEN** service intervals 不重叠，后获得资源者记录非零 queue

#### Scenario: Shared links contend across blocks
- **WHEN** 不同 blocks 或 links 的通信映射到相同 resource key
- **THEN** 它们竞争该容量，queue 等于 ready 到实际 start 的间隔

#### Scenario: Independent physical paths avoid a global lane
- **WHEN** 两项 physical 通信没有共同物理资源或明确额外并发上限
- **THEN** 它们可以同时执行，不被旧逻辑 fabric 标签串行化

## ADDED Requirements

### Requirement: Physical reservations are atomic and deterministic

physical transmission MUST 只在全部所需方向资源可用时原子预留，在完成时释放；等待期间 MUST 不持有部分路径资源，启动后 MUST 无额外物理段排队。算法阶段 MUST 依赖前驱完成并分别预留。仲裁 MUST 按 ready 时间与稳定序号确定；后到请求不得越过资源集合相交的更早等待者，不相交任务可先执行。非法或永不可满足需求 MUST 进入等待前失败；失败/取消 MUST 清理全部等待与持有资源。

#### Scenario: Blocked path does not hold a free source
- **WHEN** 一条路径的目标 NIC 忙而源 NIC 空闲
- **THEN** 等待传输不持有源 NIC，其他不违反仲裁顺序的可行传输可使用它

#### Scenario: Cancellation frees the complete reservation
- **WHEN** 多资源传输等待中或运行中被取消
- **THEN** 全部等待记录/已占资源被移除或释放，后继不被错误完成
- **AND** 后续合法请求能够继续执行

### Requirement: Communication traces separate dependency queue and service

Trace MUST 记录 model version、operation/phase/message identity、path/resource identities、logical/wire bytes、算法与估值 provenance，以及每任务 ready/start/end。ready MUST 表示该任务依赖及必要提交已满足，资源等待 MUST 为 start−ready；dependency wait MUST 不重复算入 queue/service。host submit/completion MUST 沿逻辑调用边界保留独立队列和服务，transport 完成即释放网络。串行阶段累计等待与服务，并行任务 MUST 用依赖图时间线重建 completion/makespan。完整 trace 历史、索引和延迟导出能力 MUST 保留。

#### Scenario: Consumer waits for its required completion
- **WHEN** endpoint transport 已完成但非零 host completion 尚未完成
- **THEN** 网络资源已释放，下游消费者继续等待该调用 completion

#### Scenario: Parallel task queues are not summed into makespan
- **WHEN** 两条独立路径并行服务并具有不同 queue intervals
- **THEN** 阶段 completion 使用必要任务的最终完成时间，不把所有任务的 queue/service 串行相加

### Requirement: Stage readiness and arbitration are reproducible

每个 subgroup phase MUST 等待全部实际输入依赖到齐，不得按最早 rank 的就绪时间提前执行，也不得加入无关 subgroup 的全局 barrier。同一模拟时刻 MUST 在处理该时刻释放/取消与依赖就绪后按稳定逻辑 identity 仲裁，结果 MUST 不受容器迭代顺序、缓存冷热或 trace 导出影响。

#### Scenario: One late participant delays only its dependent subgroup
- **WHEN** subgroup 的一个必要输入晚到且另一独立 subgroup 已就绪
- **THEN** 前者等待输入，后者在资源允许时执行

#### Scenario: Cache and trace settings do not reorder ties
- **WHEN** 相同输入下同时 ready 的竞争任务分别以缓存/trace 开关运行
- **THEN** 排队顺序和 completion 时间一致

### Requirement: Network migration preserves host scopes and supported entry points

新通信契约 MUST 覆盖当前支持的 AF prefill、decode 与模型边界 collective。host_resource_scope MUST 保留既有 per_worker/shared 语义，node_id 不得隐式引入节点 host 锁。禁用 host timing MUST 不新增 host service，且不得改变网络 payload、算法阶段和物理资源。

#### Scenario: Disabling host timing keeps the network plan
- **WHEN** 相同拓扑/输入只切换 host timing enabled
- **THEN** 网络阶段、payload 和路径资源一致；端到端差异来自 host 阶段及其引起的实际竞争变化
