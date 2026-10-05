## Purpose

为 AF 仿真提供可复现的物理节点、GPU、NIC、方向带宽与放置契约，使节点内外通信使用真实声明的共享资源，并使不同 A/F 放置实验能够在相同逻辑工作量下比较。

## ADDED Requirements

### Requirement: Physical topology declares directional interface capacities

系统 MUST 支持可配置 node_size（默认 8）、节点内全互通交换结构、独立 GPU-to-NIC 接入、每节点多 NIC 与节点间交换结构。默认交换结构 SHALL 无阻塞，可显式声明共享上联。带宽 MUST 表示具体接口的聚合方向容量；全互通不得隐含每 GPU pair 各自获得一份聚合峰值。全双工 MUST 区分 tx/rx，半双工 MUST 共享资源。NVLink、接入总线和 NIC MUST 仅按声明的硬件共享关系竞争。

#### Scenario: GPU fanout shares aggregate bandwidth
- **WHEN** 一个 GPU 同时向两个节点内 GPU 发送数据
- **THEN** 两项传输竞争同一个源 NVLink tx 资源，接收端按各自 rx 竞争
- **AND** 不复制源端聚合带宽

#### Scenario: Interface independence is explicit
- **WHEN** NVLink 与 NIC 接入未声明共享硬件瓶颈
- **THEN** 系统不因同一 GPU identity 而强制两类传输串行

### Requirement: Bandwidth units and NIC mapping are explicit

系统 MUST 区分 bit/s 与 byte/s、单向与双向聚合规格。A100 SXM 与 H100 SXM 预设的每 GPU NVLink 单向聚合带宽 SHALL 分别为 300 与 450 GB/s；200 与 400 Gb/s NIC 每端口每方向 SHALL 分别为 25 与 50 GB/s。GPU-to-NIC 接入带宽、固定延迟、NIC 数量与映射 MUST 独立声明。每次传输 MUST 使用映射确定的 NIC，首版不自动 striping 或池化多个 NIC 带宽。固定延迟 MUST 标记来源且不得重复计费。

#### Scenario: Shared NIC constrains two GPUs
- **WHEN** 两个 GPU 映射同一个 400 Gb/s NIC，接入均不更慢
- **THEN** 它们竞争该 NIC 对应方向的 50 GB/s 容量
- **AND** 其他空闲 NIC 不提高这两条消息的带宽

#### Scenario: Slower access link limits one path
- **WHEN** NIC 为 50 GB/s 而该 GPU 接入链路为 25 GB/s
- **THEN** 该传输路径的数据项使用不超过 25 GB/s 的带宽

### Requirement: AF placement is deterministic and validated

系统 MUST 支持手工 worker→node/GPU 配置、dedicated/mixed 自动规则和手工覆盖，并输出完整物化 placement、策略、种子（若使用）与配置指纹。dedicated SHALL 分离 A/F 节点；mixed SHALL 使用配置的每节点角色槽位。手工槽位 MUST 先预留，自动分配其余 worker；尾节点允许空槽。每 GPU 首版 MUST 仅承载一个 A 或 F worker，TP/EP rank reuse 不增加 worker。启动前 MUST 校验占卡冲突、节点槽位、group/shard 合法性、NIC/path 存在性以及 weights+KV+buffers 显存预算。逻辑组 MUST 独立于物理节点，放置不得改变路由选中 experts 或 ownership。

#### Scenario: Mixed placement with an override
- **WHEN** 用户选择 8 卡节点的 2A+6F 策略并指定一个合法 worker 槽位覆盖
- **THEN** 自动分配保留该槽位并确定性填充其余 worker
- **AND** 重复运行产生相同完整映射

#### Scenario: Invalid placement fails before execution
- **WHEN** 两个 worker 占用同一 GPU，或显存预算不足，或缺失 NIC 路径
- **THEN** 配置在执行前失败并报告具体 worker、节点与原因

### Requirement: Topology modes retain explicit compatibility

系统 MUST 保留 af.transfer_topology_mode 的 physical 与 flat_link。physical MUST 使用物化端点与路径资源，flat_link MUST 使用逻辑链路参数和 lane。新物理模型版本、配置迁移与旧虚拟拓扑假设 MUST 出现在实验输出或关联 manifest 中。单次运行拓扑/放置配置 MUST 不可变，改变配置必须重建相关快照、资源与缓存。

#### Scenario: Legacy virtual topology is not silently repacked
- **WHEN** 旧实验声明 33 卡同 NVLink 域而新实验采用 8 卡节点
- **THEN** 用户必须提供或选择明确的新放置配置，系统输出实际节点分配
- **AND** 不将两种物理假设标记为同一配置版本

### Requirement: Resource identities and directed paths are unambiguous

物理资源 MUST 使用包含节点、设备/端口和方向的 canonical identity，别名 MUST 归一化，定义冲突 MUST 早失败。路径 MUST 遵守固定 NIC 映射、声明的有向可达性及 rail/网络分区，反向路径 MUST 单独解析；同节点默认使用本地 GPU 路径且不占 NIC。带宽 MUST 有限且正，延迟 MUST 有限且非负。

#### Scenario: Equal local NIC names on different nodes are independent
- **WHEN** 两节点均声明本地 nic0
- **THEN** 两个 NIC 使用不同资源标识，只有显式共享上联才产生共同竞争

#### Scenario: Disconnected rails fail
- **WHEN** 固定源/目标 NIC 所在网络没有声明可达连接
- **THEN** 在运行前失败，不自动假设存在跨 rail 交换路径

### Requirement: Placement comparisons expose the complete hardware inventory

自动 placement MUST 仅使用声明库存，不为满足角色比例隐式增加设备。手工覆盖与选择的策略槽位冲突 MUST 明确失败。放置实验 MUST 输出总/已用 GPU、空槽、节点数、NIC 数与带宽；比较同一库存时 MUST 保持所有硬件资源定义相同，仅修改允许的 placement。资源数量变化 MUST 在结论中单独说明。

#### Scenario: Fixed GPU count does not hide more NICs
- **WHEN** 两份放置配置使用相同 worker 数但占用节点/NIC 数不同
- **THEN** 报告同时展示节点/NIC 和空槽差异，不将全部收益归因于 A/F 摆放
