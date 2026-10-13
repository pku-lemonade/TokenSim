# af-communication-correctness Specification

## Purpose

规定 AF 通信纠错后的可观察行为：独立子组按实际输入启动，阶段测量只匹配适用的硬件与 backend，flat All-to-All 校验其输入并明确支持范围，AF physical 使用固定 8 卡节点与唯一 placement 节点归属。通过反例、失败清理及缓存一致性保证测试通过具有具体可解释的边界。

## Requirements

### Requirement: Collective phases respect subgroup input readiness

系统 MUST 将每个 collective 参与者的输入 producer 绑定到真实物理 endpoint。阶段 MUST 等待它实际消耗的输入和前置阶段完成；同一 collective 中不相关参与者的未就绪输入 MUST NOT 阻塞可独立执行的阶段。子组就绪后仍 MUST 遵守对应通信域 FIFO。缺失、重复或歧义的输入映射 MUST 在执行前报错，不得猜测 rank 对应关系。

#### Scenario: Ready node starts while another node is late

- **WHEN** 两节点各有四个参与 ranks、执行层级 AllReduce，第一节点输入 t=0 全部就绪，第二节点一个输入 t=100 µs 才就绪，资源空闲且 host timing 关闭
- **THEN** 第一节点的本地 ReduceScatter 在 t=0 启动，第二节点本地阶段不得早于 t=100 µs 启动
- **AND** 跨节点 shard AllReduce 仍等待其所需的各节点本地归约完成

#### Scenario: AllGather waits only for its slot inputs

- **WHEN** 两节点各四 rank 的 AllGather 中，slot 0 的两个输入均已就绪、slot 1 有一个输入延迟，slot 0 所需资源空闲
- **THEN** slot 0 的跨节点阶段可以开始，slot 1 的阶段继续等待自身输入
- **AND** 节点内最终聚合仍等待其全部所需跨节点阶段完成

#### Scenario: Input reordering preserves local dependencies

- **WHEN** ReduceScatter 需要显式本地输入重排且某节点输入尚未就绪
- **THEN** 该节点的重排及后续本地归约等待其输入，其他节点已就绪的独立阶段正常开始
- **AND** 未就绪或失败的 producer 不得被当作有效数据

### Requirement: Collective completion and failure remain explicit

逻辑 collective 的完成 MUST 表示全部必需阶段及本地输出重排已成功完成，下游 MUST NOT 因一个子组提前完成而读取不完整输出。trace MUST 保留真实输入依赖、阶段依赖、ready/start/end 及 queue/service；数据依赖等待 MUST NOT 被算作资源排队。输入或子阶段失败及取消 MUST 传播为本次操作失败，并清理等待与占用，不得伪造成功完成。

#### Scenario: Early subgroup does not release consumers

- **WHEN** 第一节点本地阶段已完成，而另一节点的输入或后续跨节点阶段尚未完成
- **THEN** 依赖逻辑 collective 完整输出的消费者继续等待
- **AND** trace 的完成事件位于全部必需阶段之后

#### Scenario: Queue begins after subgroup readiness

- **WHEN** 子组输入 t=10 µs 就绪、对应通信域 t=16 µs 可用、服务时间 3 µs
- **THEN** ready=10、start=16、end=19 µs，queue=6 µs，service=3 µs
- **AND** 输入就绪前的等待不计入这 6 µs 的资源队列

#### Scenario: Late producer fails during an earlier phase

- **WHEN** 一个本地阶段已运行，而另一节点所需 producer 失败，或 collective 被取消
- **THEN** collective 报告失败，其子阶段释放占用并取消尚未完成的等待
- **AND** 后续独立通信可以使用这些资源，仿真不因遗留等待挂起

### Requirement: Phase measurements have hardware and backend applicability

阶段测量 MUST 保留来源设备、来源数据包 backend、数据版本与来源标识，并携带可验证的实际参与硬件及 backend 适用范围。匹配 MUST 同时满足适用范围和原有算法、阶段、dtype、payload、通信域参数及流量时序条件。设备别名 MUST 按实际设备标识归一化；P/A/F 角色本身 MUST NOT 替代设备身份。未声明异构适用范围的同质设备测量 MUST NOT 覆盖异构阶段。若运行时声明 transport backend，测量 MUST 具有匹配的显式 transport backend；未知 transport backend MUST NOT 被声称为已精确匹配。

#### Scenario: A measurement cannot price a different F device

- **WHEN** 仅 A 设备数据包有 7 µs 的阶段测量，纯 F 阶段其他签名相同，但 F 的实际设备不同
- **THEN** F 阶段不使用该 7 µs 记录，也不将其标为 exact
- **AND** 允许解析回退时使用 F 阶段解析服务，table-only 模式报告没有适用测量

#### Scenario: Hardware aliases refer to the same real device

- **WHEN** A 与 F 使用不同配置别名但解析为同一实际设备，backend 及其他签名均匹配
- **THEN** 适用的同质测量可以匹配，角色名差异不造成错误隔离

#### Scenario: Backend mismatch cannot hit an exact measurement

- **WHEN** 设备相同，但实际数据包 backend 或显式 transport backend 与记录不符
- **THEN** 记录不参与 exact 估值，按配置解析回退或报缺失测量
- **AND** 输出能够说明实际 backend 和测量适用范围

#### Scenario: Mixed hardware requires an explicit combination

- **WHEN** 一个跨节点阶段包含两种设备，只有其中一种设备的同质测量
- **THEN** 该测量不能用于整阶段
- **AND** 仅硬件/backend 组合及排列与该阶段一致的显式异构记录可匹配

#### Scenario: Multiple applicable records conflict

- **WHEN** 多个不同测量记录同时匹配同一阶段及适用范围
- **THEN** 系统报告冲突，不能以载入顺序选择服务时间
- **AND** 重复载入的同一来源记录可以去重，但不得合并不同来源与适用范围

### Requirement: All-to-All validates inputs before fallback or no-op

flat 与 physical All-to-All MUST 校验 payload 为非负整数字节且满足传输 dtype 对齐，参与 endpoints 非空、唯一、已映射，group_size 与真实参与数一致。local/remote fraction MUST 为有限且在 [0,1] 的非布尔数，两者同时提供时和 MUST 为 1；load_imbalance MUST 为有限且至少为 1 的非布尔数，destination redistribution MUST 为布尔值。校验 MUST 在零数据、单 rank、解析回退和缓存命中之前生效。

#### Scenario: Negative fraction is rejected in both modes

- **WHEN** All-to-All 指定 remote_fraction=-1
- **THEN** flat 与 physical 均报告配置错误，即使 payload 为 0 或只有一个参与 rank

#### Scenario: FP16 payload alignment is checked

- **WHEN** All-to-All 的传输 dtype 为 FP16，payload 为 3 bytes
- **THEN** flat 与 physical 均拒绝该不对齐输入
- **AND** 不得对其按 3*(N−1)/N 给出服务时间

#### Scenario: Invalid statistical parameters cannot bypass validation

- **WHEN** fractions 不互补，或 fraction/imbalance 为 NaN、infinity、布尔值，或 redistribution 为非布尔值
- **THEN** 两种模式均报告具体字段错误，不静默使用均匀回退

### Requirement: Flat All-to-All supports only an explicit uniform contract

flat All-to-All SHALL 仅支持 local_fraction 与 remote_fraction 均未提供、load_imbalance=1、destination_redistribution=false 的均匀 direct 契约。合法但超出此范围的参数 MUST 明确拒绝，并提示使用 physical 模式；不得忽略配置后输出 uniform_fallback。其 payload M MUST 表示每 rank 逻辑输入，per-rank wire bytes MUST 为 M*(N−1)/N；输出 MUST 明确字节口径、算法和均匀假设。有效零字节或单 rank 输入 MUST 返回零网络服务且不申请网络资源。physical 模式 MUST 保留现有统计 locality 能力。

#### Scenario: Uniform four rank estimate preserves the flat formula

- **WHEN** flat All-to-All 使用四 rank、每 rank 输入 400 bytes、默认统计参数、alpha=2 µs、B=50 GB/s
- **THEN** per-rank wire bytes 为 300 bytes，服务时间为 2.006 µs
- **AND** 输出 algorithm=direct、routing_provenance=uniform_fallback 和 wire_bytes_scope=per_rank

#### Scenario: Explicit locality is unsupported in flat mode

- **WHEN** flat All-to-All 指定合法 remote_fraction=0.75，或任一显式 local/remote fraction
- **THEN** 系统明确报告 flat 无节点划分语义，不忽略该参数
- **AND** physical 对合法布局及统计配置仍可构造对应流量计划

#### Scenario: Nonuniform load or redistribution is unsupported in flat mode

- **WHEN** flat All-to-All 设置 load_imbalance=2 或 destination_redistribution=true
- **THEN** 系统报告超出支持域并建议 physical，不返回假定均匀的估值

#### Scenario: Valid zero or singleton operation is a no-op

- **WHEN** 所有输入合法且统计参数为默认，payload 为 0 或 N=1
- **THEN** 服务时间和 wire bytes 均为 0，执行不占用网络资源

### Requirement: Communication caches isolate applicability from runtime state

静态服务和计划缓存 MUST 区分实际硬件/backend 适用范围、配置、测量内容与统计参数。同一流量在不同输入就绪时刻的调用 MUST 独立计算依赖和通信域排队；缓存 MUST NOT 包含 producer 完成事件、ready/start/end、queue 或资源可用状态。缓存开启与关闭 MUST 得到相同估值、trace 依赖及完成结果；非法或不支持输入 MUST NOT 通过旧缓存结果绕过拒绝行为。

#### Scenario: New invocation has fresh input events

- **WHEN** 两次调用的静态 collective 相同，但第二次有一个 producer 更晚就绪
- **THEN** 第二次的静态缓存可以命中，阶段只等待第二次实际 producer 事件
- **AND** 两次调用的 queue 和完成时间分别计算

#### Scenario: Changing measurement applicability invalidates reuse

- **WHEN** wire bytes 相同但设备/backend、适用范围或测量数据内容改变
- **THEN** 不复用不适用的旧 exact 估值
- **AND** 开关缓存后 match_type、来源、服务时间与执行结果一致

### Requirement: AF physical placement uses fixed eight GPU nodes

AF-enabled physical 模式 MUST 显式提供 communication_domains 与正整数 node_count 库存，node_size SHALL 缺省为 8 且 MUST 等于 8。系统 MUST 在配置说明中给出解析后的 dedicated/mixed/manual 策略，并统一通过 build_domain_placement 构建 AF 放置。缺少通信域、缺少库存或非 8 卡配置 MUST 在启动前报告字段错误及配置迁移方向，MUST NOT 根据网络名、活跃 worker 数或 collective 参与数猜测节点边界、A/F 放置或库存。本补充契约收紧前置提案中 AF physical 的 node_size 支持范围。

每节点 MUST 保持 8 个槽位，允许空槽；placement MUST 完整覆盖真实 worker，卡位 MUST 为非负整数、无重复且处于声明库存内。dedicated/mixed/manual 的角色槽位与显式覆盖语义 MUST 保留，空槽 MUST NOT 被虚构为通信参与者。

#### Scenario: Missing physical placement is rejected

- **WHEN** AF physical 配置缺少 communication_domains，或提供通信域但未声明 node_count
- **THEN** 系统在事件循环前报错，并提示提供 8 卡节点、节点库存及放置策略
- **AND** 不退回旧网络内 worker 数推导节点的放置路径

#### Scenario: Non eight GPU physical node size is rejected

- **WHEN** AF physical 的 communication_domains.node_size 配置为 4 或其他非 8 数值
- **THEN** 系统在启动前报告仅支持 8 卡物理节点
- **AND** 四 rank 参与 collective 仍可使用合法的 8 卡节点部分占用布局

#### Scenario: Partially occupied nodes preserve their boundaries

- **WHEN** 声明两个 8 卡节点，manual 将 A 放在 slot 0、F 放在 slot 8，其余槽位为空
- **THEN** A 属于 node0、F 属于 node1，A→F 使用 node0 inter_tx 和 node1 inter_rx
- **AND** 对 300 MB payload、零固定延迟与默认节点间带宽，纯服务时间为 6 ms，不得使用节点内 300 GB/s 得到 1 ms

#### Scenario: Mixed and dedicated policies keep declared slots

- **WHEN** 两 A 与六 F 使用一节点 mixed 且 mixed_a_slots=2，或使用两节点 dedicated
- **THEN** mixed 分配 node0 的两个 A 槽位与六个 F 槽位，dedicated 将 A/F 分别放入独立节点
- **AND** 节点未占满不改变 8 卡边界，角色预留空槽不被另一角色借用

### Requirement: AF communication shares one validated placement node identity

系统 MUST 提供 TopologyPlacement.node_id(worker_id) 或等价的统一节点读取接口；physical provider 与执行器资源管理 MUST 从同一份不可变 placement 读取节点归属，MUST NOT 各自按默认值重新解释物理卡位。所有 physical 构建入口，包括独立构造 provider/resource registry，MUST 验证 level-0 大小、communication_domains.node_size 和固定值 8 一致，验证完整映射、卡位唯一性与库存。接口 MUST 拒绝缺失 worker 映射，不隐式回退为 worker_id。输出的放置和通信资源归属 MUST 一致。

#### Scenario: Legacy network grouping cannot merge physical nodes

- **WHEN** 旧 network 分组按两个单 worker 网络生成 level-0 大小 1，而调用者尝试在 AF physical 中沿用该 placement
- **THEN** 系统在启动前拒绝不一致的 placement 并提示显式迁移
- **AND** 不把两个旧拓扑节点同时解释成 node0 的 local 通信域

#### Scenario: Provider and executor use the same node identity

- **WHEN** 合法 8 卡 placement 将物理 worker 放在 slots 0、7、8、15
- **THEN** topology、provider 与资源管理分别得到一致的节点编号 0、0、1、1
- **AND** 相同节点竞争同一个 local FIFO，跨节点传输申请对应源 tx 与目标 rx

#### Scenario: Invalid physical slots fail before execution

- **WHEN** placement 缺少 worker、卡位重复、卡位为非法值、卡位超出 node_count*8，或 level-0 大小与 8 不一致
- **THEN** runtime/provider/resource registry 在执行前报告放置错误，不创建可继续执行的错误通信映射
