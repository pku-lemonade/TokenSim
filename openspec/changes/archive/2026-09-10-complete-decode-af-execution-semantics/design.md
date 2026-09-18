## Context

见 `proposal.md - Why`。当前 Decode AF 已有 block、rank-local node、resource contention 和 provider 接口，但四类事实混在了一起：

- `AFBlock.expert_histogram` 是 block 级聚合值，却被所有 MoE layers 当成执行计划复用；它没有 token owner、routing weight 或返回路径。
- `ParallelExecutionPlanner.build_layer()` 对所有 layers 固定生成 Router/Expert DAG，F-to-A 只连接 A leader，且一个 communication node 同时代表多个 endpoints。
- F domain 的基础 ranks 当前按 `TP * EP * DP` 展开；runtime、placement 和 planner 因而把 EP identity 误当作额外设备。
- executor、runtime 和 analytical generator各自解析 dtype/shape，collective operation/participants/fabric 尚未完全进入 query、DAG 与 resource selection。

本设计以 vLLM 的执行语义为参照，但保留 TokenSim 的离散事件建模边界：planner 决定数据、依赖与资源；provider 只估算 service time；executor 只执行 DAG 并产生 queue time。目标验收配置使用 `af.phases=["decode"]`。Prefill 继续走现有完整模型 aggregate path；既有 AF-Prefill 能力不做准确性重构，只适配本次引入的公共 rank、dtype、query 与 fabric 契约。

## Goals / Non-Goals

**Goals:**

- 让每个 Decode layer 的 Dense/MoE DAG、工作量、数据 owner、返回路径和完成点与真实数学语义一致。
- 让 TP、DP、EP 都改变实际 rank workload、placement、communication 和 contention，而不是只改变配置数字。
- 建立可追踪的逐 token routing assignment、Shared Expert branch、A-side combine/residual 和 endpoint-level transfer legs。
- 统一 dtype、shape、memory 和 operator-data 公式，使同一 tensor 在各子系统中有一致的元素数与 byte width。
- 在进入 event loop 前拒绝本版本无法无歧义执行的 topology、shape、dtype、placement 和 collective plan。

**Non-Goals:**

- 不实现 AF execution domain 的 `PP>1`；AF micro-batch overlap 不是模型 Pipeline Parallelism。
- 不实现 `1 < EP < TP*DP` 或独立二维 `TP * EP * DP` physical grid。
- 不增加可配置多级 AF pipeline；该吞吐实验等本 change 的工作量和资源语义稳定后另行设计。
- 不重写 Prefill 为逐 rank DAG，不改善 Prefill 变长 Attention 近似。
- 不实现 DeepSeek MLA/MFA 专用公式，也不承诺公式生成值等同目标硬件实测。

## Decisions

### 1. 先编译不可变 execution plan，再运行事件

在提交 iteration 前，将 config、model metadata 和 operator manifest编译为不可变 plan。Plan至少包含：逐 layer type、基础 rank/runtime bindings、F EP mode、Expert/Shared/Dense placement、tensor layouts、collective plans 和 effective dtype roles。Planner 只能消费该 plan，不能在运行中再次猜测 group、shape 或 fallback。

这样可以在任何 service event创建前发现不一致，也让 trace、runtime summary 和 provider queries共享同一事实来源。替代方案是在 planner/provider 内分别推导；当前代码已经证明这种方式会产生 `ceil`、dtype bytes 和 EP world-size 分歧，因此不采用。

### 2. F world 固定为 TP x DP，EP 是两种明确的执行视图

F 基础 world始终是 `F_TP * F_DP` physical ranks：

| 配置 | Routed Expert 执行 | Expert TP | Expert 复制/放置 |
|---|---|---:|---|
| `EP=1` | 选定 F DP replica 的 TP group | `F_TP` | 每个 F DP replica持有完整 Expert 集合的 TP shards |
| `EP=F_TP*F_DP` | 跨全部现有 F ranks 的 rank-reuse EP group | `1` | Expert 分散到整个 F world |

rank-reuse identity 使用稳定映射 `ep_rank = dp_rank * F_TP + tp_rank`。每个 block 分别通过全局 round-robin 选择稳定 A DP owner 和 F service replica。A owner 承担全部 A work、request 与 KV；F service replica 承担 Dense/Shared，并在 `EP=1` 时也承担 Routed Expert TP work。rank-reuse 时只有 Routed Expert branches 可按 placement 跨多个 F DP coordinates，F service replica 不限制这些 branches，也不锁定整个 F world。`1 < EP < F_TP*F_DP` 和更大的 EP 值首版失败。

选择这个受限模型，是因为它直接对应用户确认的 vLLM-style reuse，且不会用一个参数同时表达两种物理拓扑。替代方案是新增独立 `expert_tensor_parallel_size` 和 arbitrary EP groups；它更灵活，但会显著扩大本次 placement、collective 与迁移范围。

### 3. Routing assignment 是执行事实，histogram 只是派生统计

新增逐 layer、逐 step 的不可变 `LayerRoutingSnapshot`，内部保存 `RoutingAssignment`。Assignment至少包含 request/token identity、A DP owner、A tensor/layout owner、Expert id 和 routing weight。Histogram由 snapshot聚合产生，只用于负载显示、外部数据核对和 trace summary。

Synthetic balanced routing以整个 layer block为输入连续生成 `T*K` assignments，不能对每个 Decode request重置 Expert游标。外部只提供 histogram 时，可用于统计报告，但正式 ownership return/weighted combine 路径必须失败，因为 histogram不能恢复 token-to-expert edges和 weights。

替代方案是在 block 上继续缓存一个 histogram并给所有 layers复用；它无法描述不同 layers 的路由，也无法正确返回 partial，因此不采用。

### 4. Planner 生成不同 layer DAG，并以 A-side terminal join结束

Planner从 runtime plan读取 `layer_type`，分别生成：

```text
Dense:
A Attention -> required A collective -> A-to-F legs
            -> F Dense TP work -> required F collective -> F-to-A owner legs
            -> A residual/layout -> optional A layout collective

MoE:
A Attention -> required A collective -> A leader Router/Top-K
            -> A-to-F routed/shared legs
            -> Routed Expert branches ---------+
            -> Shared Expert TP branch --------+-> F-to-A owner legs
            -> per-owner weighted combine + shared sum -> A residual
            -> optional A layout collective
```

`ExecutionNode.stage`扩展为 Attention、Router、Dense FFN、Routed Expert、Shared Expert、Combine、Residual、Cast 和 Layout。`RankLocalWork`显式区分 original token count、assignment count、逐 Expert workload和 local tensor shape。DAG保留一个显式 A-side terminal join或多个 terminal ids，由 executor等待全部 terminals；不能再依赖“最后 append 的 node”代表完成。

Residual始终留在 A owner。Dense层不生成 Router/weighted combine；MoE层必须将 `T*K` Routed partials按 weight缩回 `T` outputs，并把每 token一份 Shared output相加，然后 residual只执行一次。

### 5. 一个 logical AF stage 展开为独立 endpoint legs

A-to-F 和 F-to-A 每层各只有一个 stable logical stage id，避免重复模拟 AF transport；但 stage内包含多个可调度 `TransferLeg`。每条 leg保存 source/target physical worker、token/tensor item ids、owner/layout、dtype、bytes和直接 predecessors。每个 tensor shard使用全局hidden-index半开区间`[start, end)`描述，不能只写“第几个TP rank”；source与target的不同TP sizes通过区间交集生成legs。

每条 A-to-F leg只等待产生其 payload的 A node，对应 F branch只等待发给自己的 legs。每条 F-to-A leg只等待自己的 F producer，对应 A owner只等待发给自己的 Routed/Shared partials。Logical stage completion为必要 legs 的最晚 completion，只用于 layer/trace summary，不作为无条件全局 barrier。

Source layout明确区分`sharded`、`replicated`和`partial`。`sharded` source只发送与target区间相交的slice；`replicated` source由group中最低domain rank的稳定canonical producer发送，避免重复完整tensor；`partial`表示数学求和前每个producer的贡献，所有必要producer都必须发送其相交slice，不能去重掉不同partials。经过all-reduce的输出是`replicated`，经过reduce-scatter的输出是`sharded`。

完整 Expert output若为 `H`，按目标 A TP layout切成精确 owner shards再返回；不能把完整 `H` 复制到每个 A rank。A-to-F也使用同一interval/layout规则，因此`A_TP != F_TP`时仍能确定每个source发送哪段、每个target等待哪些legs。相同source tensor slice到相同endpoint在同一stage只传一次；发送到不同endpoints仍是不同物理copies。

替代方案是一个 multi-endpoint communication node返回一个总时延；它不能表达逐 leg contention、producer-ready time或每个 A owner的局部汇合，因而不采用。

### 6. Shared Expert 是一个按原始 token 执行的 TP branch

Shared Expert位于 F侧，使用选定 F DP replica的 TP group，与 Routed branches并行。每个 token逻辑上进入一次 Shared branch；TP ranks各自执行 local shard不算逻辑重复。多个 Shared Experts合并为一个等价 shared intermediate：`shared_intermediate = num_shared_experts * per_expert_intermediate`。

Shared output按 A tensor ownership返回，在 A combine点与 routed weighted result相加。这样能同时避免按 `top_k` 放大 Shared workload和在每个 EP rank重复完整 Shared MLP。

### 7. Dtype 分角色，并把跨格式转换作为显式 compute

建立公共 canonical dtype registry，例如 FP8=1 byte、BF16/FP16=2 bytes、FP32=4 bytes。Effective runtime roles的配置归属为：

- model/runtime config: `weight_dtype`、`activation_dtype`、`kv_cache_dtype`
- AF link: `a_to_f_dtype`、`f_to_a_dtype`

解析器必须在填充defaults前记录raw field presence。若raw input显式提供旧`af_dtype`和任一新role field，立即失败；若只提供旧字段，将其canonical值映射到五个roles。无legacy字段时，`weight_dtype`与`activation_dtype`缺省继承现有model dtype，`kv_cache_dtype`缺省继承现有KV配置或activation dtype，两个link dtype缺省继承activation dtype。Operator manifest不覆盖runtime semantics；它声明数据覆盖的dtype keys，并且必须与最终effective roles一致，否则在仿真前失败。

Runtime memory、KV ledger、compute keys和transfer payload只读取各自role。Legacy映射、defaults、来源和manifest一致性结果都进入runtime summary。

若transfer dtype不同于activation dtype，planner在发送端与接收端生成encode/decode compute nodes并查询provider。Encode按`(logical stage, source worker, stable tensor item id, slice interval, from dtype, to dtype)`去重一次，其encoded item可被多个fan-out legs复用；Decode按`(logical stage, target worker, stable tensor item id, slice interval, from dtype, to dtype)`去重一次。因此同一个item slice发送到两个F endpoints时是一次source encode、两次physical transfers、每个target各一次decode；不同token或不同mathematical partial即使区间相同也不能共用转换。转换不能默认为零成本或隐藏进带宽公式。Operator-data schema需要升版，让GEMM/Expert keys同时保留weight/activation dtype，Attention再保留KV dtype。

### 8. TP shape严格校验，collective plan真正控制执行

Plan compiler在 runtime/planning前校验 hidden、query width、Attention heads、FFN intermediate及支持的shard axes。只有“KV heads小于 TP且 TP可被 KV heads整除”的复制规则是显式例外。Provider只接收已验证的 exact local shape，不再用 `ceil`修补配置。

每个必要collective必须匹配一条domain communication plan。`participants`固定解释为replica-local `tp_rank` offsets，并在选中的每个DP instance中映射为actual domain ranks；它不是global rank列表。其`operation`直接成为typed query operation。

新增独立communication fabric registry，entry至少包含`resource_group`、`network`、`duplex`和命名lane及其`capacity`。每个communication operation最终解析到精确的`(resource_group, resource_lane)`：AF transfer按方向与duplex解析lane，collective plan必须显式声明lane。Full-duplex group的`a_to_f`与`f_to_a`是两个独立lane；collective选择其中一个时只与该方向竞争。若实验需要collective与两个方向共同竞争，必须使用half-duplex/shared lane配置，不能由executor猜测。AF links与collective plans都引用registry entry；为兼容旧配置，合法AF link inline resource定义先规范化进同一registry。缺失、冲突或越界plan在event loop前失败。

Planner为collective query提供每rank logical input/output bytes与layout，不预先乘ring/tree等算法的wire-volume系数。Transfer provider根据operation、participants、topology和link profile计算service/wire details；这样不会在planner和provider两次放大payload。

### 9. Provider composition互斥，executor只产生 queue time

Compute provider按以下互斥 operations查询：Attention、Router、Dense FFN、Routed Expert、Shared Expert、Weighted Combine、Residual、Encode/Decode、Layout。Routed workload为 assignments count且不能再次乘 `top_k`；Shared/Residual为 original token count；Combine保留输入 partial edges与输出 token count。运行时operator-data lookup仍遵守主规范：先exact match，再做允许的单维内插；离线analytical generator只能把带来源的估算记录写入版本化package，不能在runtime exact/interpolation miss后充当隐式第三种fallback。

Transfer provider按每条 endpoint leg或collective返回 service time，不选择 placement、不创建新 transport、不排队。Executor按node predecessors申请 compute/communication resources并产生 queue duration。Trace记录 logical stage id、leg endpoints、owners、layer type、workload、effective dtype、provider provenance、ready/start/end和queue/service，使 makespan可由最晚 block terminal重建。

### 10. 离线公式与内存记账使用同一模型形状

离线analytical generator对SwiGLU Dense/Routed/Shared FFN都按 gate、up、down三个矩阵计算权重与工作量。MHA/GQA/MQA使用真实 Q heads、KV heads与head dimension计算Q/K/V/O，不再固定 `4H^2`。Decode Attention对每个request分别使用 `effective_context = min(context_len, window_size)`（无window时为完整context）再聚合，不能用batch最大值替代所有requests。生成结果必须先进入版本化operator-data package，runtime才能通过exact或合法插值使用。

Runtime component ledger使用stable component identity去重，确保同一physical state因TP/EP多种membership引用时只计一次权重。

### 11. AF-Prefill 只做公共契约适配，不进入 Decode 语义重构

`phases=["decode"]`是本change的目标与性能验收配置，此时Prefill明确走现有完整模型aggregate backend。为避免公共config/runtime/query迁移意外删除主规范已有能力，显式选择`prefill` AF的合法配置仍通过一个compatibility adapter运行现有AF-Prefill planner；adapter负责消费新的基础rank view、effective dtype和typed provider字段，但不引入本change的Decode routing assignments、A-side ownership combine或新的Prefill公式精度承诺。

因此“保留AF-Prefill”只表示公共接口迁移后仍能选择并运行，不表示本change修复其执行准确性。旧`TP*EP*DP`配置无论用于哪个phase仍需迁移到新EP模式；合法迁移后的AF-Prefill必须通过现有行为回归。

## Risks / Trade-offs

- [Operator-data schema升级会使旧package不兼容] -> 提升schema version，在加载时给出明确迁移错误；提供生成器迁移与fixture同步任务，不静默猜测缺失dtype roles。
- [EP仅支持两个离散模式，表达能力有限] -> 错误中列出允许值；未来需要arbitrary EP时新增显式group/Expert-TP配置，而不是重新解释现字段。
- [Endpoint legs会增加DAG和trace体积] -> 只为非零unique tensor item创建leg，使用logical stage id聚合展示；trace关闭时省略序列化但不改变调度。
- [严格TP校验会拒绝过去能运行的配置] -> 这是有意的breaking correction；错误展示global dimension、TP与规则，并迁移现有fixtures。
- [Encode/decode数据可能暂时不完整] -> 缺少estimate时明确失败；不把零成本当默认。Deterministic tests可显式注入0用于验证非性能语义。
- [Dense/Shared TP group与rank-reuse Routed group共享F ranks，可能产生更多排队] -> 这是实际resource competition，应由trace解释，不通过创建虚假副本规避。
- [AF-Prefill仍保留旧精度层级] -> Decode-only结果明确标记`decode_af_rank_local`与`prefill_aggregate`；公共契约通过compatibility adapter维持可运行性，AF-Prefill准确性改造另立change。

## Migration Plan

1. 引入公共 dtype registry与扩展后的config/query types；保留legacy-only `af_dtype`映射，先让现有调用方可迁移。
2. 修改F world/EP校验与rank mappings：迁移旧 `TP*EP*DP` fixtures为 `EP=1`或`EP=TP*DP`，在此边界后不再接受旧拓扑。
3. 编译新的runtime group/component/ownership plan，并校验memory、TP shape、Expert placement和collective fabric。
4. 引入逐layer routing snapshots与assignments，再拆分Dense/MoE DAG及endpoint legs。
5. 扩展executor、trace和providers，接通Shared、A combine/residual、cast/layout与多terminal completion。
6. 升级operator-data schema、generator、sample package和analytical formulas，再更新engine/benchmark/result fidelity字段。
7. 接入AF-Prefill compatibility adapter，只适配新的rank/dtype/query contract而不切换到Decode DAG。
8. 运行deterministic oracles、Decode-only Prefill回归、AF-Prefill保留回归、AF-disabled回归和全量测试。

若迁移期间需要回退，应整体回退本change的config/schema与runtime实现，并继续使用迁移前的旧fixtures；新旧EP topology或operator-data schema不得在一个运行中混用。
