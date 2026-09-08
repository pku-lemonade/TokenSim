## Context

问题动机见 `proposal.md` - Why，行为契约见 `specs/af-runtime-state/spec.md` 和 `specs/af-execution/spec.md`。

当前 `ClusterConfig.workers()` 会从 execution-domain mapping 给 physical worker 设置 rank 坐标，但 `LLMEngine` 只把每个 A DP replica 的 leader 创建为 `LLMWorker`；其他 A/F ranks 是不拥有 request 或 KV Cache 的 `AFComputeWorker`。Leader 构造 `CacheConfig` 时使用 engine 全局 `ParallelConfig`，而 AF planner 使用 `ExecutionDomainConfig.parallel`。因此同一次 AF 运行中同时存在两套拓扑解释。

现有 `CacheConfig` 同时负责模型权重字节、KV 每 token 字节和可用 block 数，其边界默认一个 worker 既是 scheduler owner 又是一个完整模型 rank。该假设不适用于 A/F 分离后的 A TP group 和 F EP/TP ranks。现有 block manager/scheduler 仍需要看到一个稳定的“逻辑 KV Cache”接口，因此不应为每个 TP rank 重复运行 scheduler。

## Goals / Non-Goals

**Goals:**

- 让同一份编译后 AF 拓扑同时驱动 scheduler ownership、rank 内存、KV 分片、planner group 和运行前校验。
- 保留每个 A DP replica 一个 scheduler owner，同时为全部参与计算的 A/F ranks 建立真实内存状态。
- 在不改变 scheduler 逻辑 block 语义的前提下，将每次 KV 状态变化原子映射到 A TP group 的 rank-local shards。
- 让模型驻留内存反映 A/F 组件所有权与 F expert placement，并为容量结果提供可对账的 rank 明细。

**Non-Goals:**

- 不将 Router 进一步拆成 gating GEMM、softmax 与 Top-K 独立 DAG nodes；它仍是 A leader 上的一个可追溯 compute node。
- 不将 A compute 继续拆成 QKV、Attention core 和 output projection 独立 DAG nodes。
- 不改变当前“每个 F DP replica 复制完整 expert placement”的模式，不实现 vLLM-style 跨 DP EP group。
- 不实现 `PP>1`、token-level Router 输出、EP traffic matrix 或新的 KV transfer 时延模型。

## Decisions

### 1. 在 engine 创建 worker 前编译唯一 AFRuntimePlan

新增一个只读 `AFRuntimePlan`，它由已校验的 execution domains、logical instances、rank mappings、physical worker 硬件和模型元数据在启动时一次编译得到。它至少保留：

- `AFRankBinding`：以 `(domain, domain_rank)` 为键，记录 DP/TP/EP 坐标、replica、TP/EP groups 和 physical identity。
- `AFReplicaRuntime`：以 `(domain, dp_rank)` 为键，记录 ranks、稳定 leader 和 A replica 的逻辑 KV 视图。
- `AFPhysicalRuntimeState`：以 physical identity 为键，记录设备容量、驻留组件、模型字节、KV 字节和 block 状态。

rank binding 与 physical state 分开，因为同一 physical worker 可能被不同阶段的 domain binding 引用。物理内存以 physical identity 去重：同一份共享权重只驻留一次，不同组件的字节需累加。

`LLMEngine`、AF planner、容量校验和 trace/runtime summary 都消费该 plan；AF 路径不再从全局 `ParallelConfig` 重新推导 rank 事实。

备选方案是继续让 planner、worker 和 cache 各自读 `ClusterConfig`。这会保留多套解释和本次问题的根因，因此不采用。

### 2. Scheduler owner 只是控制者，不再代表整个 TP group 的物理状态

每个 A DP replica 继续选 `tp_rank=0` 作为 scheduler owner，并保留一份 request queue、scheduler 和 logical block manager。其他 A TP ranks 不创建重复 scheduler，但必须关联各自的 `AFPhysicalRuntimeState`。

若 Prefill 和 Decode 使用不同 A domain，但两个 replicas 具有相同 TP/DP 大小并逐 rank 映射到同一组 physical workers，则它们是同一物理 scheduler/KV 状态的 domain aliases：Engine 只创建一份 scheduler、request queue 和 logical block manager，并为两个 domain 同时维护 request ownership。映射或并行大小不同的 replicas 不允许共享 leader physical worker，启动时明确失败。

F ranks 仍不拥有 request 和 KV Cache，但其 physical state 必须记录驻留的 Expert/FFN 权重并参与模型可装载校验。`AFComputeWorker` 可保持无 scheduler 的轻量执行对象，但“无 scheduler”不再等于“无设备内存状态”。

备选方案是把所有 TP ranks 都创建为 `LLMWorker`。这会重复请求队列、调度决策和 logical block 分配，与“一个 replica 共同执行一份请求”矛盾，因此不采用。

### 3. 模型权重记账使用显式组件所有权策略

从现有 `CacheConfig._estimate_rank_model_params()` 拆出 AF 专用的组件驻留计算，输入是模型元数据、AF domain role、TP/EP membership 和 expert placement，输出是按 stable component id 去重的 rank-local bytes 明细。策略为：

- A Attention 权重在同一 A TP group 中按 TP 分片。
- Router 权重只计入每个 A DP replica 的 `tp_rank=0, ep_rank=0` leader，与 A→F 前在 A 域执行 Router/Top-K 的依赖一致。
- embedding/lm-head 归 A 侧，按已支持的 TP 权重策略分片；权重 tying 使用相同 component id 去重。
- F routed-expert 权重只驻留于 expert placement 指定的 EP shard，并在该 shard 的 TP group 内分片。
- shared expert 或其他每个 EP branch 都需要的非路由 FFN 权重在 EP shards 间复制，在每个 shard 的 TP ranks 间分片。

任一必要模型组件缺少 shape、dtype 或 ownership 元数据时，多 rank AF 路径明确失败，不使用“整模型字节除以 rank 数”静默补齐。分析公式的来源与输入保留在 runtime summary 中，以区分拓扑正确性和物理校准精度。

备选方案是只将 A domain TP 传给 leader 的现有 `CacheConfig`。它仍然忽略非 leader 容量、F expert 内存和异构 TP ranks，只修正了一个除数，因此不采用。

### 4. A replica 向 scheduler 暴露逻辑 Cache 视图，内部维护每 rank shard

为每个 A DP replica 建立一个 composite KV/block-manager 视图，保留 scheduler 所依赖的现有逻辑 allocate/free/swap 契约。其内部为每个 TP rank 维护 local block pool，并保留：

```text
logical block id
  -> tp rank 0 local block id
  -> tp rank 1 local block id
  -> ...
```

allocate 与 swap-in 先按稳定 rank 顺序 reserve，所有 ranks 成功后才 commit；任一 rank 失败即逆序 rollback。free 和 swap-out 使用已保存的 mapping 更新所有 shards。孤立 local shard、缺失 local shard 或不一致的 reference state 均视为 simulation-state error，不自动修复。

replica 的逻辑 GPU/CPU block 容量分别为各必要 TP rank 在扣除驻留模型内存和 watermark 后 local block 容量的最小值。Scheduler 不能看到各 rank 容量的和，也不能只看 leader。KV connector 继续处理逻辑 blocks，composite 视图负责将目的端状态映射为 TP shards；本 change 不增加各 shard 的独立传输时延。

备选方案是先在 leader 的既有 block manager 分配，再用 observer 向其他 ranks 复制记账。observer 失败时难以原子回滚，且 preemption/swap 容易遗漏事件，因此不采用。

### 5. KV-head 策略同时输出 local heads 和 replication group

KV-head 计算使用 A domain TP size，规则与 vLLM 的分片/复制边界对齐：

- `num_kv_heads >= tp_size` 时要求 `num_kv_heads % tp_size == 0`，每 rank 获得 `num_kv_heads / tp_size` 个 heads。
- `num_kv_heads < tp_size` 时要求 `tp_size % num_kv_heads == 0`，每 rank 获得一个 head，并记录哪些 ranks 是同一全局 KV head 的复制组。

策略输出 `(global_head_ids, replica_factor, bytes_per_block)`，而不只输出一个整数 local-head count，使 runtime summary 能区分“真实分片”与“小于 TP 时的复制”。

备选方案是继续使用 `ceil(num_kv_heads / tp_size)`。该公式会接受无法形成合法分片组的配置，且无法表达 replication group，因此不采用。

### 6. Planner 消费 runtime plan，但不分配 KV 或占用设备

`ParallelExecutionPlanner` 的 topology 查询改为从 `AFRuntimePlan` 获得 A/F replica、rank groups、physical endpoints 和 expert residency。Planner 仍只生成 DAG，不运行 scheduler、不修改 KV block、不保留 mutable device state，也不占用任一 TP rank。

Scheduler 的处理顺序为：

```text
scheduler 选出 requests 并更新逻辑 KV
  -> replica KV 视图原子更新全部 A TP shards
  -> 生成本轮 AF block snapshot
  -> planner 使用同一 runtime plan 生成逐 rank DAG
  -> executor 执行 DAG
```

这保留“Scheduler 决定这一轮选谁，Planner 决定选中工作如何在 ranks 上执行”的边界。

### 7. 静态校验与动态 invariant 分层执行

在 engine 启动前执行静态校验：

- 每个 rank binding 有唯一 physical state，每个 A replica 有唯一 leader。
- A TP/DP 与 F TP/DP/EP groups 与 execution domain 完全一致。
- KV-head 分片/复制合法，每个必要模型组件有所有者，每个 expert 只在 placement 允许的 F ranks 驻留。
- 每个 physical worker 的模型字节不超过设备容量，A replica 有非负逻辑 KV 容量。

在 block 状态变化后执行动态 invariant：

- 每个 logical block 恰好有一个属于每个必要 A TP rank 的 local shard。
- 同一 replica 的 rank-local allocate/free/swap 状态一致，且不超过各自容量。
- request owner 与持有其 KV shards 的 A replica 一致。

AF trace 仍记录计算/通信事件；内存与 KV 对账信息以 runtime summary 输出，包含每 physical worker 的 component bytes、KV bytes/block、local block capacity/usage 以及每 replica 的 logical capacity/usage。不在每个 compute event 中重复序列化整份内存快照。

### 8. AF-disabled 和 TP=1 保留现有路径

`af.enabled=false` 时不编译 AF runtime plan，不替换现有 `CacheConfig`/block manager，全局 `ParallelConfig` 仍是非 AF 路径的事实来源。AF `TP=1, DP=1, EP=1` 使用同一新边界，但 composite KV 视图退化为一个 local shard，逻辑容量和 rank-local 容量相同。

不引入用户可选的“旧 AF 内存语义”开关：同一配置只能对应一种容量结果，避免两套错误/正确模式长期共存。

### 9. Router 位于 A 域，跨域传输同时完成 EP dispatch/combine

每层在 A TP compute 与必要 collective 完成后，由所选 A replica 的 `tp_rank=0, ep_rank=0` leader 执行一个 `router_topk` compute node。该节点使用 A leader 的硬件和驻留 Router 权重，并输出 planner 已采样或输入提供的 expert histogram。

Router 完成后执行一次 A→F transfer。A→F payload 按 routed activation 计算：存在 histogram 时使用其中 token count 之和，否则使用 `token_count * top_k`，最终字节数为 `routed_count * hidden_size * dtype_size`。该 transfer 的物理 endpoints 是 A leader 与所有 active F Expert ranks，它本身就是跨域 EP dispatch；不存在再把同一 payload 从虚构的 F ingress 搬到 Expert ranks 的第二跳。

所有 Expert branches 完成本地计算与必要 F TP collective 后，执行一次 F→A transfer。它的 endpoints 是 active F Expert ranks 与 A leader，payload 仍按 routed assignments 计算，并同时承担 EP gather/combine；A leader 收到 partial expert results 后得到原 token 顺序的下一层输入。因而一层只有两次跨域数据搬运，不再额外生成 `ep_dispatch`/`ep_combine` communication nodes。

该边界来自 Step-3 AFD 的直接 RDMA 拓扑：Attention 到 FFN 的路径标为 TP gather/EP scatter，反向路径标为 TP scatter/EP gather；vLLM 的 MoE prepare/finalize 同样将 Top-K 结果直接交给 dispatch，并由 combine 返回结果。二者都不支持“先完成一次相同 payload 的 A→F，再在同一默认链路完整 dispatch 一遍”的解释。

### 10. Routed assignment 是 Expert 计算量的唯一 token 单位

`expert_histogram` 的每个 count 表示一个 token 被分配给一个 Expert 的一次 routed assignment，因此其总量已经包含 Top-K 展开。Planner 将每个 F EP rank 的 histogram 继续细分为逐 Expert workload；provider 按每个 Expert 的 routed count 查询或插值后聚合为该 rank 的 Expert latency。`moe.num_tokens` 的含义固定为 routed assignments，`top_k` 只保留路由配置/数据集身份，不再乘入 FLOPs、activation bytes 或 active Expert 数。

F TP 使用同一批 routed assignments，但每个 TP rank 只执行自己的权重分片。MoE analytical 公式使用 `ceil(intermediate_size / tp_size)`；Dense FFN 的 gate/up 输出维度、SwiGLU 中间维度和 down projection 输入维度也按 TP 切分。TP collective 仍独立计时，不能用“增加 collective”代替“缩小本 rank compute”。

外部 histogram 在 workload 边界按当前 phase 的实际 token 数校验：Prefill 使用本轮 `prefill_compute_len`，Decode 每步使用 1；总 routed assignments 必须等于 `token_count * top_k`。缓存按 `(request, phase)` 隔离，防止 Prefill histogram 被 Decode 错误复用。

## Risks / Trade-offs

- [Risk] composite block manager 涉及 allocate、append、free、preemption 和 swap 的共享边界，漏一个转移就会造成 shard 失配 → 先对现有 block-manager 全部状态转移建立失败测试，所有变化统一走组级 transaction API，每次变化后运行 shard invariant。
- [Risk] 同一 physical worker 在不同阶段的组件集合可能重叠 → 以 stable component id 去重权重，以 physical identity 统一累计内存，增加共享/不共享两类测试。
- [Risk] 分析模型权重公式仍可能与某个真实 checkpoint 有偏差 → 本 change 保证所有权与分片语义，同时输出公式来源；实测校准留给 operator/model-data 后续 change。
- [Risk] AF 多 rank 的最大可调度请求数会变化，使历史 benchmark 不可直接对比 → 在输出中保留 runtime-state schema/version 和每 rank 容量明细，文档明确这是语义修正而非性能回归。
- [Trade-off] 组级 KV transaction 比单 block pool 有更多状态，但它是表达 TP 必须同时拥有 KV shards 的必要成本。

## Migration Plan

1. 先以只读方式编译 `AFRuntimePlan` 并与现有 planner rank mappings 对账；在没有切换 KV 路径前建立 topology 失败测试。
2. 实现 component ownership、physical memory ledger 与 KV-head shard policy，用手算 TP/EP 示例验证每 rank bytes 和模型可装载结果。
3. 实现 A replica composite KV/block-manager 及 allocate/free/swap 原子 transaction，使用专门单元测试覆盖成功、中途失败与 rollback。
4. 将 A leader scheduler 与 KV connector 切换到 replica 逻辑视图，将 planner 切换为消费同一 runtime plan，然后启用启动和动态 invariants。
5. 更新 runtime summary、AF 文档与 `TP=2, DP=2, EP=2` 示例预期，运行 AF、block manager、KV transfer、并行、MoE 和 AF-disabled 回归。

如需回滚，以整个 change 为单位恢复旧 AF worker/cache 绑定；运行期临时规避方式是关闭 AF 或将 AF 限制为经过验证的单 rank 拓扑，不保留一个长期可选的旧语义开关。
