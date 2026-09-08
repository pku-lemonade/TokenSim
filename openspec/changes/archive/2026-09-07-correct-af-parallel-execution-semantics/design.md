## Context

问题动机见 `proposal.md`，行为契约见本 change 的三份 delta specs。当前 `AFExecutionPlanBuilder.build_layer()` 返回线性 `AFExecutionStep` 列表：一个 compute step 同时列出 logical instance 的全部 ranks 和 physical workers；`AFExecutor` 随后一次性申请这些 workers，却只用第一个 worker 构造一次包含完整 `token_count` 的 `AFComputeQuery`。因此 TP/EP ranks 表面上参与了步骤，实际没有各自的 shape、workload 或完成时间。

当前 `LogicalInstanceConfig.rank_mapping` 要求每个 instance 重复覆盖 domain 的完整 `world_size=TP*PP*DP`，F instance 又按 `block_id % instance_count` 选择。这同时表达了“DP rank”和“额外完整实例”两套复制概念，导致 DP ranks 被锁在一个 compute step 中。`expert_parallel_size` 不参与 rank coordinate，`ExpertRouting` 与 `ExpertPlacement` 已能生成 histogram 和 expert-to-rank 信息，但 AF planner 尚未消费它们。当前层流程还把 Router 放到 F domain，并把 `moe_dispatch`/`moe_combine`再次构造成 A/F cross-domain transfer。

本 change 是 `fix-af-communication-worker-topology` 的后续语义修正。前一 change 建立的 domain-local physical mapping、typed communication 和 trace 字段继续保留，但其线性执行链与 instance 定义将被替换。参考边界为本地 vLLM `main@53e211d2923192ea1f7442de93d59cfbb150e089`：借鉴 scheduler output ownership、MoE placement 及 dispatch/combine 抽象；该版本没有可直接复用的 Attention-FFN disaggregation 实现，因此 AF 数据依赖以 Step-3 为准，不套用 vLLM 的 PD/KV-transfer 流程。

## Goals / Non-Goals

**Goals:**

- 建立一个唯一负责解释 TP/DP/EP 语义的 planner 边界，使配置、执行事件和 trace 可以逐 rank 对账。
- 把 DP replica、TP rank、EP rank、logical identity 和 physical capacity 分开建模，并允许 TP 与 EP 在 F replica 内叠加执行。
- 用不可变 DAG 表达 fan-out、同步和 fan-in，使 straggler 通过真实依赖影响 makespan。
- 保持 scheduler 决定 token/request 顺序，A 侧保持 request 与 KV ownership；planner 只展开一次已确定的工作。
- 让 provider 只估算 planner 给出的局部工作或单次通信，Executor 只负责 DAG 与资源时间。

**Non-Goals:**

- 第一阶段不实现 PP stage partition、pipeline send/recv、bubble 或 interleaved schedule；AF domain 只允许 `PP=1`。
- 不修改非 AF 执行引擎的并行语义，也不把 AF planner 推广成全系统通用图编译器。
- 不模拟 token 级排列、packet、重传或随机拥塞；Router histogram 是本阶段所需的最细 workload 粒度。
- 不声称当前 Ada/Groq 算子和链路估计具有实测物理精度。

## Decisions

### 1. 用规范化 topology view 消除 logical instance 与 DP 的双重复制

配置解析后生成只读 `AFDomainTopology`。`LogicalInstanceConfig` 在 AF 路径中被明确解释为一个 DP replica，而不是再次包含完整 DP world：

- A domain 的一个 replica 包含 `TP` 个 ranks，坐标为 `(dp_rank, tp_rank, pp_rank=0)`。
- F domain 的一个 replica 包含 `EP * TP` 个 ranks，坐标为 `(dp_rank, ep_rank, tp_rank, pp_rank=0)`；同一 expert shard 内的 TP ranks 协作完成该 expert 的 tensor-parallel 计算。
- `logical_instances` 数量必须等于 `data_parallel_size`；每个 instance 只覆盖自己的 `dp_rank`，domain rank 在整个 domain 内唯一。
- A domain 的 `expert_parallel_size` 必须为 1；F domain 的 expert placement 在每个 DP replica 内完整复制，保证不同 DP replicas 可以独立服务 block。

为此给 F rank mapping 增加显式 `ep_rank`，并用 capability-aware 的 `required_rank_count()` 取代 AF 对通用 `DomainParallelConfig.world_size` 的直接依赖。EP 只有在 F MoE domain 中才增加所需 rank，不能作为所有 domain 的无条件乘数。解析器验证坐标全集、每 replica 的 TP/EP group、expert coverage 和一对一 physical mapping；`PP>1` 在这里立即失败。

备选方案是保留每个 logical instance 覆盖 `TP*PP*DP` 的结构，再在运行时从中挑部分 ranks。该方案仍有两个 DP 层级，难以定义 round-robin、KV ownership 和资源独立性，因此不采用。另一个备选方案是像部分 vLLM EP 部署那样让 expert placement 跨 DP group；这会使本 change 已确认的“DP replica 独立运行”不成立，因此第一阶段选择每 replica 完整复制 experts。

### 2. 全局 round-robin 在创建并发进程前确定，A ownership 生命周期内保持不变

新增确定性 replica selector，cursor 按 `(phase, domain)` 维护，并按 scheduler 输出顺序分配，不能依据 SimPy 进程谁先获得 CPU 来推进。这样相同输入始终得到相同 replica 序列。

A replica 在 request 首次 admission 时由全局 round-robin 选择，并写入 request ownership；后续 Decode iterations 必须复用该 owner，避免在没有 KV transfer 的情况下移动 KV。`AFBlockBuilder` 只把相同 A owner 的 requests 放入一个 block。F 不持有 request/KV，可在每个 block 创建时从可用 F DP replicas 全局轮询选择。选择结果存入不可变的 `AFBlockPlacement`，同一 block 各层不再按 `block_id` 临时取模。

备选方案是运行到某层时选择当前最空闲 replica。它会让输出依赖事件调度细节，且无法给出已确认的 global round-robin 行为；以后如需 load-aware 策略，应作为显式可选 policy 加入。

### 3. Planner 输出逐 rank DAG，而不是持有多 rank 的线性 step

以 `ParallelExecutionPlanner` 取代当前线性 builder。它接收规范化 topology、block placement、模型/层元数据、TP sharding policy、Router histogram、expert placement 和通信资源描述，按层输出不可变 `AFExecutionDAG`。每个 `ExecutionNode` 至少包含：

- 稳定 `node_id`、layer、operation、node kind 与 predecessor ids；
- domain、DP replica、logical rank、TP/EP group 与 physical worker；
- rank-local tensor/workload descriptor；
- 对通信节点已确定的 participants、endpoints、resource keys、payload 和 direction。

一个 compute node 只对应一个 physical compute slot和一次 rank-local provider query。同步通过多个 predecessor 指向一个 collective/combine node表达，不再通过“同时锁住所有 compute workers，然后调用一次 provider”近似。DAG 按 layer 惰性生成，上一层 F-to-A 完成后才生成/释放下一层，避免 `layers * blocks * ranks` 的整轮图常驻内存。

备选方案是在 `AFExecutionStep` 上继续增加 rank workload 字典。Executor 仍需识别 fan-out/fan-in 并编写各类特判，planner 与 executor 的职责会再次混合，因此不采用。

### 4. TP sharding policy 自动产生 local work 和语义所需 collective

planner 通过集中式 sharding policy 将模型全局 shape 转换为每个 TP rank 的 local descriptor，并生成算子语义要求的 collective。A 侧至少覆盖 QKV/Attention/output projection 的支持路径：本地算子分别查询，row-parallel 输出在需要时生成 TP collective；Router 只能在该层 A hidden state 完整可用后开始。`TP=1` 直接省略 collective。

F 侧每个被命中的 expert shard先在其 TP ranks 上 fan-out 本地 expert compute；若 `TP>1`，该 expert branch 完成必要 TP collective 后才算完成，随后参与 EP combine。现有 `DomainCollectiveConfig` 不再决定“是否应有 collective”，只允许为 planner 已确定的 collective 绑定 network/resource profile 或受控覆盖 payload；配置不得任意插入与模型语义冲突的 collective。

备选方案是把完整算子时延除以 TP。该方法无法表达非均匀 shape、通信、不同 rank 设备或 straggler，也无法使用 operator-data 的局部 shape 查询，因此不采用。

### 5. Router/Top-K 固定在 A leader，routing result 是 planner 输入而非 provider 输出

每个 A DP replica 选 `tp_rank=0` 作为稳定 Router leader。它在 A TP 结果就绪后申请自己的 compute resource，执行独立 Router/Top-K query，并产生规范化 expert histogram。对于 trace-provided 或 synthetic routing，系统可在规划前知道 histogram，但 DAG 仍保持 Router 完成后 dispatch 才 ready 的因果关系；仿真提前知道结果不等于提前完成计算。

`ExpertRouting.histogram_for_step()` 在 request 对象仍可用时执行，结果以不可变 workload snapshot 附到 block/layer planning context，避免 Executor 后续读取可变 request。Provider 只估算 Router kernel，不生成或修改 expert ids。

备选方案是让所有 A TP ranks 重复执行完整 Router。除非实现 token partition 和跨 rank Top-K merge，否则这会重复计时；第一阶段采用单 leader，未来可以通过新的 router sharding policy扩展。

### 6. 一层只含两个跨域节点，EP 通信严格留在 F domain

单层 DAG 采用以下结构：

```text
A TP rank computes
        \|/
 required A TP collective
         |
 A Router/Top-K (leader)
         |
 one A-to-F transfer
         |
 F-internal EP dispatch
      /  |  \
 expert branches (each may contain TP ranks + TP collective)
      \  |  /
 F-internal EP combine
         |
 one F-to-A transfer
```

A-to-F 是一个逻辑 DAG node，表示 routed activations 从 A replica 跨到所选 F replica 的 ingress；F-to-A 表示 combine 后的最终 token activations 返回 A replica。它们可以包含多个 physical endpoints，但 trace 中每层各只有一个 cross-domain operation id。F ingress 使用稳定 rank，EP dispatch/combine 再使用 F domain 的内部 fabric，不复用 `direction=a_to_f/f_to_a` 来伪装域内通信。

payload 由 planner 统一计算且只计算一次：A-to-F 与 EP dispatch 基于 routed assignments（`token_count * top_k`）和 dtype/hidden size，但代表两个不同物理 hop；EP combine 基于 expert outputs，F-to-A 只返回 combine 后每 token 一份 hidden activation。所有公式和中间值写入 node metadata，Provider 不得再次乘 Top-K。

备选方案是让每个目标 expert rank各生成一次 A-to-F。该模型可以表达 direct fan-out，但与当前已确定的“一次 A-to-F 后 F 内部 dispatch”拓扑不同，且会再次把 AF boundary 与 EP dispatch 混为一类事件，因此不采用。

### 7. EP workload 由 histogram 与 placement 精确聚合，最慢必要分支自然成为 straggler

planner 复用 `ExpertRouting` 和 `ExpertPlacement` 的纯数据能力，但为每个选定 F DP replica 建立独立 placement view。对每个 expert id，将 histogram count 加到其 `ep_rank`；只为非零 workload 的 expert shards 建 compute branches。每个 shard 内再按 TP sharding policy生成 rank-local work。空 shard不产生非零 expert compute，也不参与不必要的同步。

EP dispatch node 的 targets/payload 来自非零 shard，EP combine 的 predecessors 是所有必要 expert branch 的结束节点。Executor 无需单独计算 `max(expert_latency)`；DAG join 的 ready time自然等于最晚 predecessor end time，因此计算慢、网络排队和设备竞争都能形成 straggler。

备选方案是复用 `roofline.py` 中直接返回一个 `_estimate_moe_straggler_latency()` 总量。它可用于粗粒度非 AF 路径，但会隐藏每 rank queue/resource 竞争，不能用于本 AF 执行图。

### 8. Executor 成为通用 DAG runner，资源表只按 physical identity 与显式 fabric 建立

Executor 对 ready nodes 启动 SimPy process：等待 predecessors，解析并按稳定顺序申请 node 声明的资源，调用对应 provider，推进 service duration，释放资源并记录完成事件。rank compute只申请一个 compute slot；collective/transfer 可原子申请一个或多个已解析通信资源。失败传播继续终止同一 iteration 的未完成 nodes，并保留第一个带 node context 的异常。

`AFResourceRegistry` 移除运行时 `ffn_pool` 随机取 worker和缺失 resource 时动态创建 fallback 的行为。所有 compute slots 与 A/F boundary、A-TP、F-TP、F-EP fabric capacities 必须在启动前解析。相同 physical fabric 可通过同一规范化 resource key 显式共享竞争；未声明共享时不默认把 F-internal communication 绑定到 AF link。

备选方案是给每种 operation 固定一个虚拟 resource。它会掩盖同一 fabric 上的真实竞争，也可能让配置中的 topology 对 makespan 无效，因此不采用。

### 9. Provider 查询携带局部事实，Trace 同时保存 DAG 和时间事实

扩展 compute query，加入 operation、domain、DP/TP/EP coordinates、logical/physical identity、local tensor shape、local token workload、expert ids 和 shard metadata。扩展 communication query 的 direction/domain scope，明确区分 `af_a_to_f`、`af_f_to_a`、`tp_collective`、`ep_dispatch`、`ep_combine`。Provider 只返回 service duration、payload确认值和 provenance；若返回 payload，必须与 planner 值一致，否则失败，避免双方各算一次。

Trace 以 `node_id` 和 `predecessor_ids` 保存 DAG关系，并记录 replica/group/rank、local work、participants、resource ids、ready/start/end、queue/service 和 provider details。Makespan 仍从最终 node 时间得到，不汇总各 rank duration。提供 trace invariant checker，用于验证每个 node 的时间关系、每层恰好两个 cross-domain nodes、join ready time 和资源容量。

## Risks / Trade-offs

- [Risk] 将 logical instance 重解释为 DP replica 会使上一 change 的显式多 rank YAML 不兼容 → 提供配置迁移器/示例，把每个旧 instance 拆为一个 `dp_rank` 的 rank 子集；无法唯一拆分时字段级失败。
- [Risk] `TP*EP` 的 F replica 会增加 rank 与 DAG node 数 → 按层惰性生成并在层结束后释放图；性能测试覆盖大 block/layer/rank 组合。
- [Risk] Router histogram 只有计数，无法验证 token 顺序或同一 token 的 Top-K identity → 第一阶段只承诺 per-expert/rank workload 与 payload；需要 token-level fidelity 时另立 change。
- [Risk] 全局 round-robin cursor 若在并发 process 中推进会产生不确定结果 → 所有 placement 在提交顺序中预分配并写入 block snapshot，运行时只读。
- [Risk] A request owner 与 block round-robin 冲突可能移动 KV → A 只在首次 admission 选择并终身绑定；已有 owner 的 eligible set 只有该 replica，F 才按 block轮询。
- [Risk] 自动 TP collective 规则与模型算子布局不匹配会漏算或重复 → 用版本化 sharding policy 和 component ownership 表驱动；planner 与 provider 的 invariant 测试确保每个 operation 恰好一次。
- [Risk] 把 F-internal EP fabric 与 AF boundary fabric 分开后，缺少配置可能使旧场景无法运行 → 单 rank/无 EP 场景无需内部 fabric；多 rank 场景必须显式声明或使用可验证的同 fabric alias，禁止运行时猜测。
- [Trade-off] 每个 F DP replica复制完整 expert placement需要更多模拟设备，但换取 DP 真正独立；跨 DP expert parallel 留待未来独立模式。
- [Trade-off] 本设计比单段 F latency 产生更多事件，但这是表现负载不均、资源竞争和 straggler 的必要成本。

## Migration Plan

1. 先增加 capability-aware topology normalization 与校验，补充 `ep_rank`/group view，并在任何运行入口前拒绝 AF `PP>1`；同步迁移多 rank fixtures 和示例。
2. 将 A request admission 接入持久 DP owner，将 F block placement 接入全局 round-robin；保留单 rank配置得到相同 placement 的兼容路径。
3. 引入新的 DAG、node、local-work 和 block-placement 数据对象，并用 planner 单元测试锁定坐标、依赖、payload 与每层一次双向 AF transfer。
4. 接入 TP sharding、Router snapshot、ExpertRouting 与 per-replica ExpertPlacement；先用确定性 stub 验证不均衡 workload 和 straggler join。
5. 扩展 compute/communication provider query 与 operator-data lookup，随后切换 DAG Executor、resource registry 和 trace；同一提交中更新 stub providers，避免保留静默旧 query fallback。
6. 运行配置、planner、provider、executor、MoE、并行与 AF-disabled 回归，并增加可手算的 TP+DP+EP 端到端 oracle。
7. 先同步/归档 `fix-af-communication-worker-topology`，再归档本 change，保证主 specs 先获得 domain/typed communication 基础，再应用本 change 的语义替换。

回滚以完整回退本 change 的代码和配置示例为单位；不保留两套可同时启用的 AF 执行语义。若上线后发现多 rank路径问题，可临时限制 AF 为单 rank或关闭 AF，非 AF 路径不受影响。

## Open Questions

- Ada/Groq 上 A/F boundary、F-internal EP fabric 与 TP collective 的实测 bandwidth、fixed latency 和有效 payload 区间仍待采集；这些数值只影响 operator-data package 与校准，不改变本设计的 DAG、资源和查询边界。
