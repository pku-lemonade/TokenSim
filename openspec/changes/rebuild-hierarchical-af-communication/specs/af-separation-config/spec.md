## MODIFIED Requirements

### Requirement: Resource groups define communication competition

在 flat_link 模式，系统 SHALL 将相同 resource_group 的 links 归一化为同一通信资源；全双工 group SHALL 形成独立 A→F 与 F→A 键，半双工 SHALL 形成方向共享键，不同 group SHALL 为独立资源。相同逻辑 resource_group 声明的 network、duplex 与对应容量 MUST 一致，否则拒绝配置。在 physical 模式，group/lane MUST 仅作为兼容逻辑标签，实际竞争 MUST 根据节点 local、inter_tx/inter_rx 通信域资源解析；逻辑 group 相同不能强制独立通信域任务串行，group 不同不能隔离实际共享通信域。旧 link 容量字段 MUST 保留校验，并在配置摘要明确其在 physical 下不创建额外 lane；首版 MUST 不新增隐含全局并发上限。

#### Scenario: Full-duplex links share directional competition domains

- **WHEN** flat_link 的多条全双工 link 使用相同 resource_group
- **THEN** 所有 A→F 使用同一个方向键，所有 F→A 使用另一个方向键

#### Scenario: Half-duplex links share one competition domain

- **WHEN** flat_link 的多条半双工 link 使用相同 resource_group
- **THEN** 两方向映射到同一共享资源键

#### Scenario: Shared resource definitions conflict

- **WHEN** 相同 resource_group 的 links 声明不同 network、duplex 或容量
- **THEN** 系统拒绝配置并标明冲突 group 与 links

#### Scenario: Physical sharing follows domain identities

- **WHEN** 两个不同逻辑 group 的传输使用同一个节点的 local 或 inter_tx/inter_rx
- **THEN** 两者竞争该通信域资源，不因标签不同获得额外带宽

### Requirement: Required collectives have explicit executable plans

当受支持的 AF prefill/decode DAG 或模型边界 tensor layout 需要 collective 时，对应 execution domain MUST 提供唯一匹配的 communication plan，明确 operation、触发位置与 replica-local tp_rank participants；operation MUST 至少区分 all_reduce、all_gather 与 reduce_scatter。Participants MUST 在选中 DP instance 内映射为 actual domain ranks，operation MUST 进入 provider query。flat_link 计划 MUST 显式选择 fabric registry 的 resource group/lane；registry MUST 声明 network、duplex、命名 lane 与各自正容量，精确 group/lane MUST 决定竞争。physical 计划 MUST 根据实际参与端点、拓扑与固定算法解析阶段、分片和通信域资源，并在运行前完成支持域校验；逻辑 lane 不再代替通信域映射。Planner 对外 provider query MUST 提供每 rank logical input/output bytes，算法 wire-volume 转换 MUST 由共享模型恰好执行一次，不能在 planner 和 provider 重复相乘。缺少必要 plan、flat_link 所需 fabric/lane 或 physical 所需通信域映射时 MUST 失败，不得静默采用完整 TP group、通用 tp_collective 或 AF 跨域 link。

#### Scenario: Collective participants select a subset

- **WHEN** TP=4、DP=2 的 plan 配置 replica-local participants [0,2]
- **THEN** DP0 映射为 [0,2]，DP1 映射为 [4,6]
- **AND** 每实例的其他 TP ranks 不因 group 名称被加入 collective

#### Scenario: Collective operation changes the query

- **WHEN** 两份配置仅在 all_gather 与 reduce_scatter 上不同
- **THEN** provider 收到不同 operation key，trace 保留对应名称

#### Scenario: Missing collective fabric fails

- **WHEN** 必要 collective 缺少当前模式要求的 fabric/lane 或通信域映射
- **THEN** 系统在 event loop 前失败，不自动改用 A-to-F 或 F-to-A resource

#### Scenario: Independent collective fabric has executable capacity

- **WHEN** flat_link collective 引用未被 AF link 使用且 network、shared lane、capacity 合法的 registry entry
- **THEN** 系统创建独立 contention resource，不与 AF transfers 竞争

#### Scenario: Collective explicitly shares the AF fabric

- **WHEN** flat_link collective 与 AF link 显式引用同一 group/lane
- **THEN** 两者按该容量竞争，不创建同名独立资源

#### Scenario: Full-duplex collective lane is not inferred

- **WHEN** flat_link collective 引用具有 a_to_f 与 f_to_a lane 的 full-duplex fabric
- **THEN** 必须显式选择 lane，否则 event loop 前失败
- **AND** 选择 a_to_f 时只与 A-to-F 竞争，F-to-A 按其容量独立执行

#### Scenario: Collective payload remains logical at the planner boundary

- **WHEN** planner 为四 rank AllReduce 生成 query
- **THEN** query 保留每 rank logical input/output bytes、operation、participants
- **AND** wire-volume 系数仅由共享模型应用一次

#### Scenario: TP one needs no artificial collective

- **WHEN** execution group 的 TP=1 且 layout 不要求其他通信
- **THEN** 不要求配置或生成零意义 collective

#### Scenario: Physical collective resources match domain mappings

- **WHEN** collective 与 AF endpoint leg 使用同一节点 local 或方向域资源
- **THEN** 两者使用同一通信域资源标识竞争，逻辑 lane 标签不能绕过该瓶颈
