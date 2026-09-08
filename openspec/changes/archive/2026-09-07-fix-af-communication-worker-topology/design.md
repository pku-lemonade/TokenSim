## Context

参见 `proposal.md` 的问题背景。当前 `ClusterConfig.workers()` 先展开全部 worker groups，再按单一 `ParallelConfig.world_size` 赋 `global_rank/tp_rank/pp_rank/dp_rank`；这使物理设备数量、逻辑执行身份和模型并行 rank 成为同一个概念。当前 `AFExecutor._run_block()` 又把逐层流程固定为四段，虽然后端 provider 已识别 `all_reduce`、`moe_dispatch`、`moe_combine`，运行时没有构造这些查询和资源事件。

本设计需要兼容非 AF 路径及无歧义的单设备 AF 配置。总任务书要求优先支持 PD 分离下的 Decode AF：Prefill 可由独立 P domain 完整执行，Decode 由 A/F domains 协作。真实 Ada/Groq 数据当前不可用，因此本 change 只保证事件机制、查询契约与确定性时序正确，不声明物理精度。

## Goals / Non-Goals

**Goals:**

- 给 P、Decode A、Decode F 建立彼此独立的 domain-local topology 和 rank-to-physical mapping。
- 让 physical resource capacity 成为排队依据，让 logical rank 只表达执行与通信语义。
- 以显式逐层 execution plan 调度 dispatch、combine 和 topology 要求的 collective。
- 所有通信统一产生 provider query、资源申请、queue/service 区间和 trace 事件。
- 用完整端到端 oracle 验证依赖、重叠、竞争、placement 和 makespan。

**Non-Goals:**

- 不采集、补造或校准 Ada/Groq kernel 与通信性能数据。
- 不改变 scheduler、request progression 或 KV Cache ownership。
- 不重写既有非 AF worker/parallel 执行引擎；只在边界增加兼容适配。
- 不模拟协议重传、随机网络拥塞或 packet-level 行为。

## Decisions

### 1. 分离 physical resource、logical instance 与 domain rank

新增内部配置视图：

- `PhysicalWorkerRef`：稳定标识 `worker_group + worker_index`，指向实际计算 slot 和容量。
- `ExecutionDomainConfig`：标识 `prefill`、`decode_attention` 或 `decode_ffn`，持有独立 `DomainParallelConfig`。
- `LogicalInstanceConfig`：domain 内可被调度的执行实例，包含一组 domain-local rank members。
- `DomainRankMapping`：将 domain-local rank 坐标和 logical instance 映射到 `PhysicalWorkerRef`。

`DomainParallelConfig` 保存 TP/PP/DP，并在 F domain 保存 EP/expert placement。TP、PP、DP 决定该 domain 需要覆盖的 rank 坐标；EP 描述 expert group 与 placement，不自动乘入 physical worker 数。配置加载后为每个 domain 单独检查 rank 唯一性、坐标覆盖、logical instance 完整性和 physical reference 有效性。第一版采用严格一对一 placement：每个 `(logical_instance, domain_rank)` identity 映射到一个唯一 physical worker，同一 domain 内不得复用 physical worker，并且该 domain worker group 的全部 workers 必须各被映射一次。因此 A/F 各自的 logical rank identity 总数等于各自 physical worker 数，但 A 与 F 两侧数量无需相等。

不再在 `ClusterConfig.workers()` 中对全部 groups 调用统一 `validate_worker_count`，也不再按展开顺序给 P/A/F 赋同一套 global ranks。既有 `ParallelRankInfo` 继续服务非 AF 路径；AF 路径使用 domain-local identity，避免一次性重写整个引擎。

备选方案是给每类 worker 继续分配全局连续 rank 区间。该方案仍会让 domain topology 依赖无业务意义的全局编号，并容易再次用总 worker 数做错误校验，因此不采用。

### 2. 配置归一化先于运行时，旧格式只自动迁移无歧义单实例

配置 parser 将显式 execution domains 归一化为统一内部对象。`af.enabled=false` 保持现状；未声明新 topology 的单设备 AF 配置生成每端一个 logical instance、一个 local rank 的默认 view。显式多 rank AF 配置若缺少 mapping，直接给出 domain、rank 与字段上下文，不按 worker group 顺序猜测。

AF link 在归一化后引用 A/F domain id，同时保留现有 endpoint group、network 和 resource group。这样 link 决定跨域连接，domain 决定域内 topology，worker group 决定物理容量，三者职责不混合。

备选方案是为所有旧多 rank 配置自动生成 mapping。现有配置没有足够信息判断 P/A/F rank 的边界和复用关系，自动迁移可能产生看似成功但语义错误的结果，因此只兼容无歧义场景。

### 3. Executor 消费 execution plan，不直接推断 collective

新增 `AFExecutionPlanBuilder`。它接收 phase、layer/model metadata、A/F domain topology、expert placement 和 link，输出有序 typed steps：

1. `a_compute`
2. A domain communication plan 指定的 collective（零个或多个）
3. `a_to_f`
4. `router_compute`
5. `moe_dispatch`
6. `expert_compute`
7. F domain communication plan 指定的 collective（零个或多个）
8. `moe_combine`
9. `f_to_a`

Router/Gating 在 F domain 上完成 token-to-expert 路由决策，因此必须先于 dispatch；expert compute 只能在 dispatch 将 token 送达目标 expert 后开始。Dispatch/combine 是 MoE F 边界的必需步骤；collective 的 operation、位置、participants 和 payload rule 来自 domain communication planner，不在 `_run_block()` 中按 TP/EP 数量临时猜测。Planner 可复用 `parallel.py` 的 group 构造语义，但输出纯描述对象，不直接推进 SimPy 时间。

Compute provider 将原有 `AFComputeQuery(stage="f")` 拆成 `stage="router"` 与 `stage="expert"`。Router composition 只包含 Router/Gating 及 dispatch 前必要 elementwise；Expert composition 只包含 grouped/fused expert 及 dispatch 后必要 elementwise。Composition validator 必须保证两边没有重复 component，二者之和替代旧 F compute 的纯计算总量。Dispatch、combine 和 collective 仍由 transfer provider 提供 service time，不能混入任一 compute composition。

Executor 逐 step 等待依赖、申请资源、调用 provider、推进 service time、释放资源并写 trace。下一层只依赖本层最后一个 F-to-A step；不同 blocks 继续由 SimPy 自然并行。

备选方案是在当前四段流程中插入固定一次 AllReduce。不同 TP/EP placement 所需 collective 的位置和次数不同，固定插入会把一种拓扑误写成通用规则，因此不采用。

### 4. 所有通信使用统一 typed query 和 resource registry

扩展 `AFTransferQuery`，至少包含：

- `operation`：`af_transfer`、`moe_dispatch`、`moe_combine` 或具体 collective。
- phase、layer、block、source/target domain。
- logical participant ranks 与映射后的 physical endpoints。
- direction、network、resource identity、payload bytes、topology details。

`AFCommunicationResourceResolver` 将 step 解析为一个或一组规范化资源键。跨 A/F transfer 继续遵守 link 的 full/half duplex 规则；dispatch/combine/collective 根据 participants 的 physical endpoints、network 和 resource group 解析。代表同一物理 fabric 的不同 operation 得到同一竞争域，容量由规范化配置唯一确定。第一版一个 step 按 planner 给出的资源集合原子申请，排序资源键后申请以避免死锁。

Provider 只计算 service duration 和 provenance。Executor 在资源申请前记录 `ready_at`，获得全部资源后记录 `start_at`，从而让 contention 等待只产生于模拟事件层。

备选方案是为 dispatch/combine 使用独立、不共享的虚拟链路。这会掩盖它们与 AF transfer 或 collective 对同一物理网络的竞争，无法满足 makespan 与利用率验证，因此不采用。

### 5. Compute placement 先选 logical target，再解析 physical capacity

A block 保留 request owner 的 A logical instance；A compute 和 F-to-A 返回均引用该 identity。F 侧依据 expert placement 与 F domain plan 选择 logical instance/ranks，再通过 mapping 得到 physical compute resources。`AFResourceRegistry` 以 physical identity 建立 SimPy capacity，而不是以 logical rank 数创建容量。

当一个 logical instance 由多个 ranks 协同执行时，每个 rank 使用其唯一 physical resource，planner 返回全部参与 resources，Executor 原子申请后执行该 stage。配置校验在事件循环前拒绝 rank 共置、一个 rank 多重映射、未映射 worker 或跨 worker group 引用。确定性 tie-break 使用配置顺序和稳定 id，避免测试依赖字典顺序。

备选方案是优先找任意空闲 physical worker。该做法会绕过 rank/expert placement，得到不属于当前 communication group 的 worker，因此不采用；只能在 mapping 允许的候选集合内按稳定策略选择空闲 slot。

### 6. Trace 同时保留逻辑参与者和物理竞争对象

扩展 `AFTraceEvent`，新增 operation、source/target domain、logical participants、physical endpoints 与 resource identities。每个 dispatch/combine/collective 都使用和 compute/AF transfer 相同的时间字段。Makespan 仍取 iteration 最后完成 block 的时间；资源利用率只累计 service 区间，queue 单独统计。

Provider breakdown/provenance 原样进入 details，但不参与调度。这样 trace 可以分别回答“谁在逻辑上参与通信”和“哪些物理资源发生了竞争”，不会再用一个 global rank 字段表达两件事。

### 7. 端到端测试以确定性 provider 为时间真值

测试从完整 YAML/config 进入真实 engine/AF executor，使用固定 service provider 与小型生成数据包两类 oracle。核心断言包括：

- 单 block 的七段基础链总和包含 Router、dispatch、Expert 和 combine，且顺序为 `A -> A-to-F -> Router -> dispatch -> Expert -> combine -> F-to-A`。
- topology 指定 collective 时事件恰好出现，未指定时不出现。
- 多 block 共享网络时 queue、重叠、resource utilization 和 makespan 可由 trace 手算重建。
- P/A/F 可以使用彼此不同的 worker 数和 domain topology，但每个 domain 内 rank identities 与 workers 必须一对一。
- rank 共置、重复 physical mapping 和未映射 worker 在模拟启动前失败。
- mapping 缺失在事件循环前失败；AF disabled 保持回归行为。

真实性能准确性需要以后用目标硬件 kernel 和通信测量作 ground truth；本 change 的 deterministic provider 只验证模拟机制。

## Risks / Trade-offs

- [Risk] 配置模型增加 domain、instance 和 mapping 后更复杂 → 提供单设备归一化、字段级错误和更新后的示例配置；拒绝有歧义的隐式多 rank 推断。
- [Trade-off] 第一版禁止 logical rank 共置，暂不模拟共置引入的计算份额、显存划分和通信容量共享；未来若有明确需求再通过新 change 扩展。
- [Risk] 多资源原子申请可能引入死锁或不公平 → 对资源键稳定排序并集中封装 acquire/release；端到端加入交叉竞争场景。
- [Risk] `parallel.py` 现有逻辑依赖 worker 上的全局 rank 字段 → 保留非 AF 适配器，AF planner 使用 domain-local view，逐步隔离而不修改非 AF 行为。
- [Risk] collective plan 或 payload 公式不完整会导致漏算或重复计算 → 每个 step 使用唯一 operation id，planner 测试检查序列与 participants，provider composition 继续明确排除通信。
- [Risk] 既有 compute provider 只识别 `stage="f"`，直接切换会造成数据查找失败或 Router/Expert 重复计时 → 在同一实现变更中迁移 query 类型、stub、正式 provider、composition manifest 与测试，正式 AF 路径不保留静默的合并 F fallback。
- [Trade-off] 第一版要求显式多 rank mapping，配置较冗长，但可避免以 worker 顺序猜测拓扑造成静默错误。
- [Trade-off] 模拟到共享资源事件而非 packet level，能表达容量竞争与排队，但不能预测协议级拥塞和抖动。

## Migration Plan

1. 先引入 execution domain 配置类型、归一化与独立校验，同时保持旧非 AF 和单设备 AF 行为。
2. 更新 AF link 解析，使其绑定 domain；迁移仓库中的多 rank AF 示例为显式 mapping。
3. 引入 execution plan 与 typed communication steps，将合并 F compute query 拆成互不重叠的 Router/Expert compositions，并扩展 provider query、resource resolver 和 trace schema。
4. 将 Executor 切换为计划驱动，按 Router -> dispatch -> Expert -> combine 调度，补齐 collective 后运行完整端到端回归。
5. 若需要回滚，可恢复旧 Executor 路径和旧 link adapter；新配置字段在旧版本中不受支持，因此代码回滚需同时回滚示例配置。本 change 不修改 operator-data 文件格式中的现有实测记录。

## Open Questions

- Ada/Groq 的 dispatch、combine 和 collective 实测数据、适用 payload 范围及校准误差仍待获取；在此之前只能验证机制和模型自洽性，不影响本 change 的接口与任务拆分。
