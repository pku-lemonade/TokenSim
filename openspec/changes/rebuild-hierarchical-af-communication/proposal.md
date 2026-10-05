## Why

当前通信估值中 collective payload、轮次与物理流量系数的口径不一致，物理拓扑估值与实际排队资源也未完全对应。需要建立可解释的节点、端口、算法阶段和资源预留模型，支持比较 A/F 放置方式，并保留现有执行器缓存带来的性能收益。

## What Changes

- 统一每 rank 逻辑输入、算法 wire bytes、输出大小与轮次；修正 AllGather、flat collective 和 oversubscription 计算。
- **BREAKING**：新的基础解析模型使用固定延迟与峰值方向带宽，移除其对 H/half_bandwidth_bytes 和未校准 collective_efficiency 的依赖，通信结果预期改变。
- 扩展 physical 模式：默认可配置 8 卡节点，节点内交换结构、GPU-to-NIC 接入、多个 NIC、节点间交换结构及可选共享上联；明确聚合方向带宽。
- 提供 dedicated/mixed 自动放置与手工覆盖，输出确定的 worker→GPU→node→NIC 映射，校验资源、分组和显存。
- 固定首版通信算法模板，分别调度算法阶段；单次传输内部原子预留整条路径，使用流水近似。分别记录依赖等待、资源排队和纯服务。
- 限制消息合并边界，缓存静态计划和纯通信服务；每次调用独立排队。提供时间线、模型版本与证据来源。
- 保留 flat_link 对照模式，新增公式、排队、近似误差、放置对比及模拟器性能验证。
- 覆盖受支持的 AF prefill/decode 与模型边界通信；约束 coarse phase 内部共享资源计费、测量阶段匹配、确定性仲裁及固定硬件库存下的放置对比。

## Capabilities

### New Capabilities

- `af-physical-topology`: 分层网络、方向容量、NIC 映射与 A/F 放置。
- `af-communication-model`: payload、算法阶段、服务公式、消息粒度与缓存契约。

### Modified Capabilities

- `af-execution`: physical 通信的原子路径预留、阶段依赖与 trace 契约，保留 flat_link 逻辑 lane 语义。
- `af-separation-config`: 将逻辑 group/lane 的竞争约束限定为 flat_link，physical 根据实际路径解析竞争资源与 collective 计划。

## Impact

涉及 TokenSim/comm/collectives.py、hardware/links.py 与 placement.py、AF 配置加载、planning.py、operator_provider.py、resources.py、executor.py、model_executor.py，以及配置样例、测试和说明文档。新增配置的具体字段布局在实现中遵循现有 schema，行为契约由本提案确定；不新增外部运行时依赖。

依赖已实现的 integrate-deepseek-v3-core-runtime（提交 d24df24），其规范尚未归档同步。本变更不自动归档其他变更；同步时须先同步该前置变更，再应用本变更的 af-execution 修改，避免旧 delta 覆盖新资源语义。AF-disabled 与无关计算路径保持兼容；旧的虚拟 33 卡同节点配置不能静默解释成真实 8 卡节点。已有实验文件不在本次规划修改范围。
