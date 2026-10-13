## Context

动机见 `proposal.md`。基线为 `4f42e71`：通信域独占 FIFO、缓存静态计划/纯服务，8 卡节点默认布局。原三个反例已在工作区修复，1–5 组任务完成；后续完整工作区复查通过 551 项仓库测试和 46 项通信专项，但这些记录未覆盖本次新增的旧拓扑节点归属问题。

原三个缺陷按实施前符号定位：`af/planning.py:1973` 汇集全部 producer 为 collective predecessors；`af/executor.py:775` 在进入内部阶段前等待全部 predecessors；`comm/traffic.py` 内部阶段已具有 subgroup 与阶段前驱。`af/operator_provider.py` 载入记录时未保留完整设备/backend 来源，phase 匹配也未纳入适用范围；flat All-to-All 分支仅计算均匀系数。

本次追加的缺陷位于 `hardware/placement.py` 的 `synthesize_topology()` 与旧入口：它按网络内 worker 数设置 level-0 大小。`af/operator_provider.py` 和 `af/resources.py` 则各自默认用 `device_index // 8` 定义通信节点；旧入口节点大小小于 8 时未被拒绝，两个拓扑节点可能合并。用户确认物理节点始终为 8 卡，允许空槽，并确认统一 AF 放置入口、显式库存和共享节点归属。

已有通信 delta 尚未同步为主规范，本次以独立 `af-communication-correctness` 补充契约承接，不修改或重建前置变更。工作区还包含 AFD 控制/校准实现，行号可能变化，实施须同时按符号定位。

## Goals / Non-Goals

**Goals:**

- 将静态 producer 映射与每次调用的运行时事件分离，在保留真实数据依赖的前提下解除独立子组的全局等待。
- 使 exact 测量与实际阶段的设备/backend 组合一致；缺失适用测量时沿用解析回退或 table-only 错误。
- 为 flat All-to-All 建立可解释且可测试的支持域，拒绝无法表达的统计配置。
- 固定 AF physical 的 8 卡节点边界，以 placement 为 provider、资源管理及 trace 节点归属的共同来源，旧配置必须显式迁移。

**Non-Goals:**

- 不优化通信算法、拆分并行带宽或改变 FIFO 策略，不扩展不规则布局支持。
- 不引入逐 Expert/逐物理链路状态，不改变粗 phase 持有资源并集的既有近似。
- 不顺带实施 AFD 控制协议、校准、全链路 timing 精度调整或自动同步/归档前置变更。

## Decisions

### 1. 显式保存参与 endpoint 到输入 producer 的静态映射

planner 在 collective 节点保留可校验的输入绑定，例如 `(physical_endpoint, producer_node_id)`，并与 domain rank、participant 顺序一致。引用仍参与 DAG 缺失引用、拓扑排序及环检测，不能简单删除全部 predecessors 后绕过 DAG 验证。额外控制/host 前驱与数据 producer 前驱区分；不属于 rank 输入的强制依赖继续等待。

executor 为本次 DAG 解析 producer 完成事件。physical 分层 collective 的协调过程不再先等待全部输入；入口阶段只等待其 participants 的输入事件，后续阶段等待已有 phase predecessors。需要输入重排时，重排阶段承担输入等待。单阶段、flat 或未具备输入映射的旧接口保留完整输入 barrier，不猜测 subgroup；正式 planner 产生的分层 collective 必须具有完整绑定。

这样复用现有阶段图与通信域资源，无须增加逐 token 或逐链路执行粒度。拒绝仅移除 barrier 的方案，因为它会允许阶段读取未生产的数据。保留逻辑节点完成事件，只有全部必需阶段与输出本地操作成功结束才释放下游；本次不新增消费者的局部输出完成 API。

### 2. 运行时等待与 host 提交不进入静态缓存

阶段的 ready 是其真实数据及控制/host 依赖全部完成的时刻；资源队列从 ready 开始计算。静态缓存只含阶段形状、流量和服务参数，producer 事件由每次执行重新创建并绑定。trace 记录真实 producer/phase 前驱，完成标记不重复计服务。

host timing 开启时，不得重新以全部输入完成作为早期子组的提交门槛。沿用既有提交次数及作用范围，将软件提交就绪与数据就绪分开；如果既有协议明确要求全输入控制 barrier，则保留并在 trace 中标注为显式协议依赖，不能把它混为 subgroup 资源排队。测试分别覆盖 host timing 关闭的反例和开启时的提交计费守恒。与 AFD native protocol 的交互仅保持已声明语义。

producer 失败、phase 失败和取消通过本次执行的失败通道传播；等待输入的阶段和已启动阶段都参加清理。producer 的 done event 即使因 finally 被触发，也必须伴随成功状态校验，避免失败被误当输入就绪。验收检查资源释放、active stage 清理和后续独立操作继续运行。

### 3. 测量来源与适用范围分开保存

加载每条记录时保留实际 `device_id`、数据包 backend、dataset/source identity。同质旧记录缺少硬件 scope 时，可从其所属设备包推导为“仅同一实际设备及包 backend”，并标明推导来源；不得推导成任意设备可用。显式异构记录增加按 phase participants 稳定顺序排列的 `(device_id, package_backend)` 组合。rank 数字及 P/A/F 角色不进入硬件身份；设备别名先归一化。

数据包 backend 与 transport backend 是不同字段，不将包名自动解释为 NCCL/DeepEP。运行时查询如显式指定 transport backend，记录必须同样显式声明并匹配；未指定时仅表达 transport backend 未声明，不声称验证了具体传输实现。旧包缺少 transport 字段的同质记录可在运行时也未声明 transport backend 的范围内沿用，并输出此限制。

phase 估值接收不可变适用上下文，或在静态 phase 上携带该上下文；保持 phase 计划与估值使用相同 participant 顺序。完整 scope 与测量内容加入 provider namespace/phase cache key。去重按规范化来源、适用范围及内容识别同一记录；不同适用记录保留，多个同时命中的记录继续报冲突。拒绝只在记录加载时按 A/F 角色筛选的方案，因为同一角色可包含多设备、同一设备可对应不同角色。

### 4. 共用 All-to-All 基础校验，flat 严格限定均匀契约

从 physical All-to-All 中抽出无拓扑副作用的基础参数校验，flat 与 physical 都在 no-op 和缓存返回前执行。错误分为非法参数和合法但超出模式支持范围。维持原有 physical 流量/热点计算，本次不改变 imbalance 定义。

flat 仅接受两 fractions 都为 None、imbalance=1、redistribution=false；即使显式 fraction 恰好等于均匀比例，也拒绝，因为 flat 没有本地/远端节点分区语义。支持路径沿用每 rank `M*(N−1)/N` 与 link alpha-beta 估值，补充 `wire_bytes_scope=per_rank`。合法 0-byte/N=1 操作先通过全部校验，再返回零网络服务且执行不持有网络资源。

相比猜测 placement 并复用 physical 计划，明确拒绝更符合 flat 的单链路含义，也不会暗中改成节点聚合带宽口径。错误提示给出 physical 模式及相关配置迁移方向。

### 5. AF physical 固定 8 卡，并使用唯一放置入口

本次收紧 AF physical 的节点约束：每节点始终包含 8 个 GPU 槽位，活跃 worker 数和当前 collective 参与数不改变节点容量。`af.communication_domains` 必须显式声明，`node_count` 必须提供正整数库存，`node_size` 缺省为 8 且仅接受 8；放置策略按已声明的 dedicated/mixed/manual 语义执行，缺省 manual 必须在解析后配置说明中可见。此补充契约收紧前置通信提案中 AF physical 的可配置 node_size，不重写通用非 AF 拓扑或 flat_link 的布局含义。

`build_topology_placement()` 的 AF-enabled physical 分支统一调用 `build_domain_placement()`，不得退回按网络内 worker 数生成节点边界的旧放置逻辑。即使配置同时声明 legacy topology，也以显式通信域的 8 卡放置为 AF 节点归属依据，并在配置说明中报告该选择；原 link 元数据仅作为声明过的兼容信息。缺少 communication_domains、缺少 node_count 或非 8 卡配置均在构建 runtime/provider/resource registry 前校验并报告字段和迁移方向；不得自动猜测 A/F 分节点还是混放、推导库存或悄悄重排旧实验。

添加 `TopologyPlacement.node_id(worker_id)`，从该 worker 的物理卡位及已验证的 level-0 边界读取节点。provider 与 `AFResourceRegistry` 均从同一份 placement 生成节点映射，不再分别按本地默认 node_size 重新解释卡位。独立构造 provider/resource registry 的 physical 入口也必须验证显式配置和 placement，不能绕过 runtime 校验。构建入口验证 placement 完整覆盖真实 workers、所有卡位非负整数、无重复、未超库存，且 `placement.topology.levels[0].size == communication_domains.node_size == 8`；只记录实际存在的 worker，不填充空槽为虚构 rank。运行时 placement 保持不可变，已有 provider/executor 映射一致性检查保留。

例如两节点各放一张卡，A 在 slot 0、F 在 slot 8，两个节点仍各有 8 个槽位；A→F 使用 node0 inter_tx 与 node1 inter_rx。若 A/F 按 mixed/manual 放在 slot 0 与 slot 1，则共用 node0 local。dedicated 不借用 A 节点空槽给 F，mixed 不借用预留角色槽位，manual 不改变用户显式卡位。对 300 MB 的合法跨节点 P2P，零固定延迟和默认 50 GB/s 带宽给出 6 ms，不能因仅有两个 worker 而使用 300 GB/s local 得到 1 ms。

原测试中两节点各四 rank 的 collective 场景仍然有效，但节点库存必须改成每节点 8 个槽位，并把第二节点 ranks 放在 slot 8 起；不得把 collective 四参与者数量当作物理 node_size=4。迁移相关 fixture 时保留原迟到输入、阶段依赖、失败清理、scope/cache 与流量断言；此文档修订不自动完成实现任务。

## Risks / Trade-offs

- [DAG 模板缓存遗留旧输入绑定] → 静态绑定参与模板身份或升级模板版本，验证模板缓存开关与多次调用，运行时事件始终在调用内创建。
- [解除全局等待造成失败传播漏洞] → 让输入等待参与统一失败通道，用早阶段运行中晚 producer 失败/取消的反例检验清理。
- [旧测量缺失元数据] → 仅推导所属设备包的同质 scope，异构和显式 transport backend 不猜测；table-only 提供可定位错误。
- [flat 配置过去静默被忽略] → 明确标注兼容性变化，基础默认均匀配置保留原数值并提供 physical 迁移说明。
- [与未提交 AFD 改动交叠] → 修复提交只纳入本次代码块及测试，保留用户工作；单独验证通信修复快照，另报完整工作区回归。
- [旧 AF physical 配置曾隐式获得 placement] → 现在必须明确提供 8 卡通信域、节点库存和放置策略；给出 dedicated/mixed/manual 配置示例，不据旧 network 名猜物理节点。
- [固定 8 卡令原测试 fixture 不再合法] → 将 node_size=4 fixture 迁移为 8 卡节点内四 rank 参与，重新验证原依赖、流量和缓存语义，保留历史测试记录的时间边界。
- [测试通过被误读为硬件精度保证] → 原三个缺陷的测试通过不代表追加 placement 修复已实施；各阶段独立记录反例与验证，模型近似继续披露。

## Migration Plan

1. 先为三个问题加入修复前失败的最小回归，并记录与现有无关工作区的边界。
2. 落地输入绑定/阶段就绪、测量 scope 与缓存身份、All-to-All 校验/支持域，各自以相关场景验收。
3. 更新配置与测量元数据说明、已知问题状态及新验证日志；不得用历史 459/575 测试通过记录替代修复验证。
4. 执行通信专项和仓库回归，核对静态服务不因排队变化，严格校验本变更。无新增依赖或自动归档动作。
5. 如需回滚，仅回滚本次修复提交并恢复对应契约/已知问题状态，保留前置分层通信实现及用户 AFD 工作。
6. 追加 placement 修复先记录旧入口误合并节点的失败反例，再统一固定 8 卡入口与 node_id 接口，迁移旧配置及 fixture，重新执行相关回归和严格校验；1–5 组历史完成状态不替代第 6 组验收。
