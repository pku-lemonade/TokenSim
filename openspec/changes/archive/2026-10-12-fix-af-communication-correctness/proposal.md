## Why

通信实现提交 `4f42e71` 的复查确认了三个缺陷：collective 全局输入等待阻塞已就绪子组、阶段测量缺少硬件/backend 适用范围，以及 flat All-to-All 忽略 locality/imbalance 并漏校验。459 项既有测试没有覆盖这些反例，需要以明确行为规范和回归场景修复。

原三个问题已在工作区实施，相关任务与验证记录保留。后续复查发现旧拓扑按网络内 worker 数推导节点大小，而 provider/resource registry 默认按 8 卡切分；两个真实节点可被误归为同一通信域。本次追加用户确认的固定 8 卡放置与统一节点归属修复，新增任务尚待实施。

## What Changes

- 将 collective 的 rank 输入就绪依赖绑定到对应阶段；独立子组及时启动，下游仍等待完整输出，排队继续由通信域 FIFO 计算。
- 为阶段测量保留来源设备/backend，规范实际阶段适用范围与异构组合；匹配和静态缓存均包含这些信息，禁止跨设备误标 `exact`。
- 统一 All-to-All 的 payload、dtype、participants 和统计参数校验。**BREAKING**：flat 模式显式拒绝其无法表达的非默认 locality/imbalance/目标节点分发配置，不再静默忽略。
- 新增三个缺陷的反例、失败清理和缓存隔离测试；更新已知问题记录，分别记录修复证据。
- **BREAKING**：AF physical 模式固定每节点 8 个物理 GPU 槽位，要求显式 `communication_domains` 与节点库存，并记录解析后的放置策略；缺少放置配置、缺少节点库存或非 8 卡配置在启动前报错并给出迁移提示。
- AF physical 放置统一进入 `build_domain_placement()`；新增 `TopologyPlacement.node_id(worker_id)`，provider 与执行器资源管理共用 placement 的节点归属，校验节点边界、重复占卡及库存。
- 为旧拓扑误合并节点、部分占用的 8 卡节点及 A/F 放置增加反例；既有 4 卡物理节点测试迁移为 8 卡节点内四 rank 参与，保留原通信依赖验收。

## Capabilities

### New Capabilities

- `af-communication-correctness`：规定 collective 阶段输入就绪、测量适用范围、flat All-to-All 输入与支持域，相关缓存隔离，以及固定 8 卡 AF physical placement 的纠错契约。

### Modified Capabilities

无。本变更新增补充契约，保留既有分层通信提案与主规范；`af-communication-model` 仍位于前置变更的 delta 中，尚未同步，不能假定它已是主规范。

## Impact

- 前置基线：`rebuild-hierarchical-af-communication` 与提交 `4f42e71`；问题记录见 `TokenSim/validation/known_communication_issues.md`。
- 实施涉及 `af/planning.py`、`af/executor.py`、`af/operator_provider.py`、`af/estimate_cache.py`、`comm/traffic.py`，`hardware/placement.py`、`hardware/topology.py`、`hardware/domains.py`、`af/resources.py`、AF 配置校验，以及必要的阶段/查询数据结构、专项测试和文档。
- 与未提交的 `align-vllm-afd-protocol-and-timing` 有文件交集，实施时须保留其控制/校准语义并隔离提交范围。
- AF physical 的 8 卡节点默认值收紧为固定约束；节点域 FIFO、现有通信粒度与已披露的粗阶段资源占用近似保持原语义。
