## Purpose

为 AF 仿真提供可配置的物理节点、GPU 放置与节点级通信域容量契约，明确节点内共享服务和节点间独立收发资源的竞争边界，使同一库存下的不同放置实验可复现、可解释，避免将节点级聚合假设误称为单 GPU 硬件规格。

## ADDED Requirements

### Requirement: Node communication domains define shared FIFO capacity

系统 MUST 支持可配置 node_size，默认 8。每节点 MUST 提供一个节点内共享资源，默认有效聚合带宽 300 GB/s，所有实际节点内网络传输按该资源独占 FIFO 服务，不按 rank pair 复制带宽。每节点 MUST 另有独立 inter_tx 与 inter_rx 共享资源，节点间带宽 MUST 与节点内带宽分别配置。首版 MUST NOT 拆分并行链路或动态均分资源带宽，不要求 GPU endpoint、GPU-to-NIC、PCIe、rail 或交换机端口图。fabric 的 direct/switched 标签 MUST 只表达抽象语义，不改变上述共享容量规则。

#### Scenario: Local transfers share one node budget

- **WHEN** 同一节点的 A、B、C 三项传输同时就绪，节点内带宽为 300 GB/s
- **THEN** 三项任务按稳定 FIFO 顺序独占 300 GB/s 服务
- **AND** 不将其拆为各两条 50 GB/s 链路，也不为每个 rank pair 分配 300 GB/s

#### Scenario: Separate nodes keep independent local domains

- **WHEN** 两项节点内通信分别发生在 node0 和 node1
- **THEN** 它们使用不同节点内资源，可以同时服务

### Requirement: Inter-node transfers reserve source transmit and destination receive

每个跨节点 transmission MUST 原子使用源节点 inter_tx 和目标节点 inter_rx，有效数据带宽 MUST 为两者配置带宽的较小值，固定延迟按声明的阶段范围计一次。同源发送或同目标接收 MUST 竞争相应资源。节点间 MUST NOT 退化成全集群一个共享队列；反向传输 MUST 使用另一组方向资源。节点内与节点间资源默认独立，跨节点传输 MUST NOT 隐式占用节点内资源。

#### Scenario: Source or destination sharing creates contention

- **WHEN** node0→node1 与 node0→node2 同时就绪，或 node0→node2 与 node1→node2 同时就绪
- **THEN** 前一组竞争 node0 的 inter_tx，后一组竞争 node2 的 inter_rx

#### Scenario: Opposite directions can overlap

- **WHEN** node0→node1 与 node1→node0 无其他共同资源
- **THEN** 它们分别占用 node0.tx/node1.rx 与 node1.tx/node0.rx，可以并行

#### Scenario: Disjoint transfers avoid a global queue

- **WHEN** node0→node1 与 node2→node3 同时就绪且各自资源空闲
- **THEN** 两项传输可以同时服务，不竞争隐含的全局 50 GB/s 资源

### Requirement: Bandwidth units and modeling assumptions are explicit

带宽 MUST 区分 bit/s、byte/s、单向与双向汇总，内部归一化为 bytes/s，GB/s 使用十进制。节点内默认 300 GB/s MUST 标记为节点级有效聚合建模假设，不声称是整机实测或每 GPU NVLink 规格。节点间 50 GB/s MAY 作为显式实验参数，MUST NOT 默认为集群总带宽。带宽 MUST 有限且正，延迟 MUST 有限且非负并记录来源。已聚合容量 MUST NOT 再乘链路数；首版配置 MUST 不要求 per-link 带宽或 link_count。

#### Scenario: Changing local bandwidth leaves inter-node configuration intact

- **WHEN** 仅将节点内带宽从 300 改为 600 GB/s
- **THEN** 节点间 tx/rx 配置不变
- **AND** 新配置建立独立不可变快照及资源，不原地修改正在执行的运行

#### Scenario: Units and aggregate capacity are not duplicated

- **WHEN** 节点间配置使用 400 Gb/s 或等价的 50 GB/s
- **THEN** 归一化结果为每节点每方向 50×10^9 bytes/s
- **AND** 不再乘物理节点卡数或链路数

### Requirement: AF placement is deterministic and validated

系统 MUST 支持手工 worker→node/GPU、dedicated/mixed 自动规则与手工覆盖，输出完整 placement、策略、种子（若使用）与配置指纹。dedicated SHALL 分离 A/F 节点，mixed SHALL 使用声明角色槽位；手工槽位 MUST 先预留，自动确定性填充其余槽位，尾节点允许空槽。每 GPU 首版 MUST 仅承载一个 A 或 F worker，TP/EP rank reuse 不增加 worker。启动前 MUST 校验占卡、槽位、通信域存在性、group/shard 和 weights+KV+buffers 显存预算；放置 MUST 不改变选中 experts 或 ownership。

#### Scenario: Mixed placement with an override

- **WHEN** 8 卡节点采用 2A+6F 策略且指定合法槽位覆盖
- **THEN** 自动放置保留覆盖并确定性填充其余 worker，重复运行映射一致

#### Scenario: Invalid placement fails before execution

- **WHEN** 重复占卡、通信域缺失、槽位策略冲突或显存不足
- **THEN** 执行前失败并报告具体 worker、节点及原因

### Requirement: Regular collective layout is the initial supported domain

层级 collective 模板 MUST 校验各参与节点的参与 rank 数相同、阶段分片可整除；不支持的非均匀布局 MUST 在执行前失败，不隐式填充。物理 node_size MUST 不等于每个 collective 的强制参与数；该规则 MUST 不被泛化为 P2P、单 rank 或所有角色 placement 的满节点要求。

#### Scenario: Partial nodes can form a regular collective

- **WHEN** 两个 8 卡物理节点各有 4 个 rank 参与当前 collective
- **THEN** 布局满足相同参与数约束，不要求补入其余 GPU

#### Scenario: Uneven participation is rejected

- **WHEN** 当前层级 collective 的两个参与节点分别提供 8 与 5 个 rank
- **THEN** 执行前报告不支持，不生成补齐传输或隐式二次分发

### Requirement: Domain identities and compatibility are unambiguous

资源 MUST 使用含 node_id、domain 和方向的 canonical identity，别名归一化且冲突早失败。系统 MUST 保留 physical 与 flat_link：physical 使用节点通信域，flat_link 保留逻辑 group/lane。单次运行的拓扑、放置和容量 MUST 不可变，修改时重建资源及缓存，输出模型版本和迁移假设。

#### Scenario: Equal domain names on different nodes are independent

- **WHEN** node0 与 node1 均有 inter_tx
- **THEN** canonical identity 不同，不因同名而共享容量

#### Scenario: Legacy virtual topology is not silently repacked

- **WHEN** 旧实验声明 33 卡同域，新配置采用 8 卡节点
- **THEN** 必须明确选择新放置并输出实际节点，不能标记为同一配置版本

### Requirement: Placement comparisons expose the complete hardware inventory

自动放置 MUST 仅使用声明库存，不隐式增加 GPU 或节点。实验 MUST 输出总/已用 GPU、空槽、节点数以及各节点通信域容量；同库存对比 MUST 保持资源定义相同，仅修改允许的 placement。物理端口库存 MAY 作为外部元数据，但 MUST 不成为首版必需映射。

#### Scenario: More active nodes do not hide extra capacity

- **WHEN** 两份配置 worker 数相同但使用节点数不同
- **THEN** 报告展示节点、空槽和通信域容量差异，不把全部收益归因于 A/F 摆放
