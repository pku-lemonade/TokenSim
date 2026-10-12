## Why

当前通信估值的 collective payload、轮次和 wire bytes 口径不一致，分层设计又混用了通信域和逐端口路径要求。需要统一成可解释、可配置的节点级通信域 FIFO 模型，让服务估值与实际竞争资源一致，同时保留既有缓存收益，支持比较 A/F 放置及路由局部性的影响。

## What Changes

- 统一每 rank 逻辑输入、输出、算法轮次与逐通信域 wire bytes；保留 `ring_traffic_factor` 兼容辅助函数，修正 AllGather、flat collective 和 oversubscription 计算。
- **BREAKING**：新的基础解析模型采用明确固定延迟与有效方向带宽，移除对 H/half_bandwidth_bytes 和未校准 collective_efficiency 的依赖，通信结果预期改变。
- physical 模式默认每节点 8 卡，可配置。每节点一个节点内共享 FIFO 资源，默认有效聚合带宽 300 GB/s；节点间使用各节点独立的 tx/rx 共享 FIFO 资源，带宽与节点内分别配置。300 GB/s 是节点级建模假设，不代表 NVIDIA 每 GPU 的硬件规格。
- 同一资源上的任务独占全部配置带宽并按 FIFO 排队，不拆分并行链路、不动态均分带宽；跨节点任务原子申请源 tx 和目标 rx，不相交资源可以并发。逐端口、GPU-to-NIC、PCIe、rail 与物理交换图退出本次必做范围。
- 提供手工、dedicated/mixed 自动放置与覆盖，输出确定的 worker→GPU→node 映射，校验库存、分组和显存。规则 collective 布局要求各参与节点的参与 rank 数相同、分片可整除，不要求每个 collective 占满物理节点。
- 按算法阶段分别计费、排队与释放资源。分层 All-to-All 默认源节点内重排及跨节点远端交换；目标节点二次分发只在显式选择相应路由时加入。MoE 默认支持统计 locality，top_k 决定 payload，remote_fraction 决定跨节点比例，不强制逐 token expert 路由。
- 扩展既有有界服务/计划缓存，缓存逐阶段纯服务、依赖和静态资源需求；区分服务键与计划键，配置不可变，每次调用独立创建事件并计算排队，禁止缓存队列或绝对完成时间。
- 保留 flat_link、host 调用边界、既有 AF routing/ownership 和 trace 能力；覆盖受支持的 prefill/decode、模型边界、确定性、取消恢复、近似敏感性及仿真性能验证。

## Capabilities

### New Capabilities

- `af-physical-topology`：节点级通信域、方向容量与可复现 A/F 放置。
- `af-communication-model`：payload、算法阶段、域服务公式、消息粒度与缓存契约。

### Modified Capabilities

- `af-execution`：通信域原子预留、FIFO、阶段依赖与 trace，保留 flat_link 的逻辑 lane。
- `af-separation-config`：physical 按节点通信域解析竞争与计划；逻辑 group/lane 的执行容量限定为 flat_link。

## Impact

涉及 TokenSim/comm/collectives.py、hardware/links.py、hardware/placement.py、AF 配置加载、planning.py、operator_provider.py、resources.py、executor.py、model_executor.py，以及配置样例、测试和用户文档。不新增外部运行时依赖，具体字段布局沿用现有 schema。

默认精度目标是系统级趋势和瓶颈分析。输出标记 exact_routing、statistical_locality 或 uniform_fallback，允许 locality/热点敏感性对照；缺少硬件测量与真实路由时不宣称绝对时延精度。现有精确 AF 路由保持原语义。

依赖已实现的 integrate-deepseek-v3-core-runtime（提交 d24df24），其规范尚未归档同步；同步时先同步前置变更，再应用本变更的 af-execution 修改，不自动归档其他变更。AF-disabled 与无关计算路径保持兼容，旧 33 卡虚拟同域配置不静默转换成真实 8 卡节点，既有用户实验不被覆盖。
