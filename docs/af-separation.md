# AF 分离配置与验证

## 当前实现范围

本版本已经把 Attention-FFN（AF）分离配置接入可运行的 SimPy executor。它支持 Prefill、Decode 或两阶段 AF，能够明确 A/F worker group、通信网络、全/半双工以及物理竞争域，并输出可重建流水行为的结构化 trace。

当前 executor 需要调用方注入 compute 与 communication latency provider。并行 planner 把 TP、DP 和 EP 配置展开为逐 rank DAG；provider 只估算一个 rank 的局部计算或一次已规划通信。仓库测试使用带 `test-stub` source ID 的确定性 provider 验证调度机制；这些数值不是 Groq 或其他硬件的物理性能结论。缺少 provider 或必要模型元数据时，`benchmark.py` 会在产生 AF 事件前明确失败，不会静默回退到完整 iteration 时延。

## 配置模型

### AFConfig

集群配置未提供 `af` 时，默认值为：

```json
{
  "enabled": false,
  "phases": ["prefill", "decode"],
  "af_block_size": 1,
  "links": [],
  "trace_enabled": true
}
```

字段语义：

| 字段 | 语义 |
| --- | --- |
| `enabled` | 是否启用 AF；默认关闭，旧配置行为不变 |
| `phases` | AF 作用阶段，可为 Prefill、Decode 或两者 |
| `af_block_size` | 一个 AF micro-batch 最多包含的 request 数，必须是严格正整数 |
| `links` | 每个所选阶段的 A/F 拓扑和通信资源 |
| `trace_enabled` | 是否收集并输出 AF 细粒度时间线 |

`af_block_size` 按 request 计数。Decode 每个 request 每轮贡献一个 token；Prefill 的 `token_count` 是 block 内 request 的实际 Prefill 计算 token 数之和。第一版不拆分单个长 Prefill request。

现有 CLI `--block_size` 始终表示 KV Cache token block。它与 `af_block_size` 是两个独立命名空间，互不覆盖。

## Executor 运行语义

scheduler 选出的同阶段 requests 按 `af_block_size` 连续组成 blocks。每个 block 对模型的每一层严格执行：

```text
A TP rank-local compute
-> required A TP collective
-> A leader Router/Top-K
-> one A-to-F / EP dispatch
-> routed expert rank-local compute and required F TP collective
-> one F-to-A / EP combine
```

Router/Top-K 固定在所选 A replica 的 `tp_rank=0, ep_rank=0` leader，并等待 A TP compute/collective 完成。Router 产生的 routed workload 同时决定 Expert 负载、跨域 endpoints 和 payload；存在 histogram 时 routed token count 为 histogram 计数之和，否则为 `token_count * top_k`。A-to-F 直接从 A leader 发到 active F Expert ranks，本身就是 EP dispatch；F-to-A 从这些 ranks 返回 partial Expert results，本身就是 EP gather/combine。Planner 不再生成重复搬运同一 payload 的 F-internal dispatch/combine 节点。每层始终只有一个 `af_a_to_f` 和一个 `af_f_to_a`，且两个方向都按 routed token count 计算 payload。同一 block 只有在 F-to-A 返回 A 域后才能进入下一层。A 域持有 request/KV ownership，首次 admission 后不能在没有 KV transfer 的情况下换 A DP replica；F replica 按全局 round-robin 在 block 提交顺序中预选。

## AF 并行拓扑

AF 第一阶段支持 TP、DP 和 F 域 EP，明确不支持 PP。任何 A/F execution domain 的 `pipeline_parallel_size > 1` 都会在创建执行事件前失败。

- 每个 `LogicalInstanceConfig` 表示一个 DP replica，而不是重复包含整个 DP world。
- A replica 有 `TP` 个 ranks，坐标为 `(dp_rank, ep_rank=0, tp_rank, pp_rank=0)`。
- F replica 有 `EP * TP` 个 ranks，坐标为 `(dp_rank, ep_rank, tp_rank, pp_rank=0)`。
- A 域必须为 `EP=1`。F 域在每个 DP replica 内完整复制 expert placement，支持 `linear` 或 `round_robin`。
- `domain_rank` 在整个 domain 唯一，每个 physical worker 只能映射一个 rank。

完整可加载示例见 `data/clusters/af/tp2_dp2_ep2.json`。其中 A 域 `TP=2, DP=2`，所以有 4 个 ranks；F 域 `TP=2, DP=2, EP=2`，所以有 8 个 ranks。旧配置若让每个 instance 都覆盖 `TP*DP` ranks，需要拆成两个 instances：`a0` 只保留 `dp_rank=0` 的 TP ranks，`a1` 只保留 `dp_rank=1` 的 TP ranks；F 域同理，每个 instance 只保留自己的 `EP*TP` ranks。

### Runtime plan、Scheduler 与 KV 状态

启用 AF 后，Engine 在创建 worker 前把 execution domains 编译成唯一的 `AFRuntimePlan`。它是 topology 的事实来源，记录每个 domain rank 的 DP/TP/EP 坐标、物理 worker、TP/EP group、scheduler leader、模型组件和 Expert 驻留。Planner 只读取这张表生成 DAG；Planner 不运行请求调度、不分配 KV，也不占用一个 TP rank。

每个 A DP replica 只有 `tp_rank=0` leader 运行一份 Scheduler，但这个 replica 的每个 TP rank 都有独立模型和 KV 账本。Scheduler 看到一个逻辑 KV block，内部状态为：

Prefill 和 Decode 可以声明为两个 A domain，并逐 rank 复用同一组物理 TP workers。只要两个 domain 的 TP/DP 大小和物理 rank 映射完全一致，它们会共享同一份 Scheduler、request queue 和 KV 状态，同时保留各自的 domain ownership；若拓扑不等价，Engine 会在启动前明确拒绝，避免两套逻辑状态误占同一组物理卡。

```text
logical block (request, block index)
  -> A TP rank 0 local shard
  -> A TP rank 1 local shard
  -> ...
```

allocate、append、fork、free、swap-in、swap-out、preemption 和 connector release 都更新完整 shard 集。分配或 swap 的任一 rank 失败时，已更新 ranks 会回滚。F ranks 只记录实际驻留的 Expert/FFN 权重，不创建 request queue 或 KV Cache。

一个 A replica 的逻辑容量取所有必要 TP ranks 的最小可用容量。例如两个 ranks 分别有 100 和 80 个 local blocks，Scheduler 只能使用 80 个逻辑 blocks；不能只看 leader 的 100，也不能相加为 180。

KV heads 使用 A-domain TP，而不是全局 `parallel_config`。当 KV heads 不少于 TP 时必须整除分片；当 KV heads 少于 TP 时，TP 必须能被 KV head 数整除，并建立明确复制组。例如 1 个 KV head、TP=2 时两个 ranks 都保存 head 0，但全局仍只有一个 KV head。

模型内存同样按组件归属记账：A 保存 Attention 和首尾组件，每个 A DP replica 的 `tp_rank=0, ep_rank=0` leader 额外保存 Router；F ranks 按 EP placement 保存 routed experts，并保存需要复制的 dense/shared FFN，不保存 Router。相同物理 worker 上相同 component id 只计一次，不同组件累加。模型放不下、元数据缺失或 topology 不一致时，Engine 在事件循环前失败。

结果 JSON 的 `af_runtime_state` 使用带版本的 schema，包含：

```text
schema/version
physical_workers[].component_bytes/model_bytes/kv_bytes_per_block/rank_usage
replicas[].local_gpu_blocks/local_cpu_blocks/logical_gpu_usage/logical_cpu_usage
```

这份 summary 用于逐卡核对内存与 KV；计算和通信时间线仍在 `af_trace` 中，两者不重复记录。

每个物理 worker 只有一个共享 compute slot。未选择 AF 的完整 Prefill/Decode 与 AF compute 也申请该 slot，复合 capability worker 因而不会重复提供算力。所有 iterations 共用 Engine 生命周期内的 worker、F pool 和通信资源注册表，跨 block、跨 link、跨 iteration 的竞争不会因重建资源而消失。

iteration 等待所有 blocks 完成 barrier 后，才由原 worker 入口对本轮每个 request 调用一次 `step()` 并更新 scheduler。block、layer 和 stage 完成本身不会推进 request，也不会分配、释放或迁移 KV blocks。Prefill 与 Decode 使用不同 A groups 时，阶段间 KV 仍通过现有 connector 转移；逐层 A-to-F/F-to-A link 不替代该 KV transfer。

Trace 的每个事件包含 DAG node/predecessors、DP/TP/EP rank/group、logical/physical participants、local shape/workload、payload、resource、ready/start/end、queue/service 和 provider provenance。内置 invariant checker 验证依赖与 join 时间、资源容量、每层双向边界节点及 payload 一致性。Iteration summary 从最终节点时间与 service intervals 重建 block completion、barrier wait、makespan 和资源利用率，不对各 rank duration 求和。`trace_enabled=false` 只关闭记录，不改变资源选择或模拟时间。

### Worker capability

worker group 支持以下 capability：

| Capability | 职责 |
| --- | --- |
| `prefill` | 完整、非 AF Prefill |
| `decode_full` | 完整、非 AF Decode |
| `af_attention` | AF 的 A 阶段，拥有 request/KV 状态 |
| `af_ffn` | AF 的 F 阶段，不拥有 KV Cache |

新格式使用稳定 `name` 和非空 `capabilities`。硬件名称只描述设备，不能用 `Ada`、`Groq` 等名称推断 A/F 职责。

旧 role 继续支持，并按真实现有语义归一化：

| Legacy role | 归一化 capability |
| --- | --- |
| `prefill` | `prefill` |
| `decode` | `decode_full` |
| `hybrid` | `prefill`, `decode_full` |

旧 group 缺少名称时会按输入顺序生成 `legacy-<role>-<index>`。同时写 role 和 capabilities 时，完整阶段 capability 必须一致；可以额外声明 AF capability。

### AF link

全双工 link 示例：

```json
{
  "name": "attention-ffn",
  "phases": ["prefill", "decode"],
  "attention_group": "attention-pool",
  "ffn_group": "ffn-pool",
  "network": "af_net",
  "resource_group": "af-fabric",
  "duplex": "full",
  "a_to_f_capacity": 1,
  "f_to_a_capacity": 1
}
```

半双工使用 `duplex="half"` 和 `half_duplex_capacity`，不得同时提供方向容量。每个所选阶段必须恰好由一条 link 覆盖；第一版不定义同阶段多 link 的路由策略。A/F 端点必须不同，并分别声明 `af_attention` 与 `af_ffn`。

只选择一个 AF 阶段时，另一阶段必须保留完整执行能力：Decode-only 需要 `prefill`，Prefill-only 需要 `decode_full`。

仓库样例：

- `data/clusters/af/prefill_decode_full_duplex.json`：Prefill 和 Decode 共用一条全双工 AF link。
- `data/clusters/af/decode_half_duplex.json`：仅 Decode AF，Prefill 完整执行，A/F 两个方向共享半双工资源。

## 通信竞争语义

`network` 选择链路性能配置，`resource_group` 标识物理竞争域。相同 `resource_group` 的 links 必须具有完全相同的 network、duplex 和容量。

规范化资源键为：

```text
full duplex:
  (resource_group, "a_to_f")
  (resource_group, "f_to_a")

half duplex:
  (resource_group, "shared")
```

同一资源键上的传输竞争容量；不同资源键独立。`capacity` 表示并发槽位，而不是带宽倍数。executor 分别记录：

```text
queue_duration   = transfer_start - ready_at
service_duration = fixed_latency + message_bytes / effective_bandwidth
total_duration   = queue_duration + service_duration
```

链路利用率只累计 `service_duration`。Groq 第一版网络模型是确定性系统级近似，不引入随机拥塞、乱序、重传或随机尾延迟。

## 计算时延数据契约

### provider 选择

`af.latency_provider` 决定 AF executor 的 compute/transfer provider 来源：

| 取值 | 含义 |
| --- | --- |
| `injected`（默认） | 调用方注入 provider；仓库测试用带 `test-stub` source ID 的确定性 provider 验证调度机制 |
| `operator_table` | Engine 用硬件目录构建正式 provider：计算查 `data/operator_data/<device>/<backend>` 的算子表，缺失 shape 回退到设备族解析模型；通信用 `data/topologies/links.yaml` 的链路类和分层 collective 模型 |

`operator_table` 模式下还可以设置 `af.operator_backend`（覆盖全局 `--operator_backend`，例如 `vllm`、`trtllm`、`sglang`）和 `af.latency_fallback`（`table_first` / `table_only` / `analytical_only`，覆盖全局 `--latency_fallback`）。worker group 的 `hardware` 必须是 `data/devices` 中的 `device_id`，`networks` 的值必须是链路类 ID：

```json
{
  "networks": {"af_net": "ib_ndr400"},
  "worker_groups": [{"name": "attention-pool", "capabilities": ["af_attention"], "hardware": "h100_sxm", "num_workers": 4, "network": "af_net"}],
  "af": {"enabled": true, "latency_provider": "operator_table", "operator_backend": "vllm"}
}
```

设备缺失、后端不存在或网络名没有对应链路类时，Engine 在事件循环前失败。`benchmark.py` 只在 `injected` 模式下要求调用方提供 provider；`operator_table` 模式与注入的 provider 不能同时使用。

算子表的格式、查表顺序（精确匹配 → 受控插值 → 有界外推 → 解析模型）与数据来源见 [operator-latency-model.md](operator-latency-model.md)；每次查询返回：

```text
latency_us
match_type = exact | interpolated | extrapolated | analytical
source_id
```

`match_type` 表示查询方式，`source_id` 表示证据来源，两者不能混用。AF 事件记录的 `match_type` 是该阶段所有算子中最差的一项，`dataset_version` 是设备数据包版本（无数据包时为 `analytical:<device_id>`）。结果 JSON 的 `af_latency_providers` 记录两个 provider 的描述，`af_operator_stats` 与 step 级 backend 一样给出 match type 计数和缺失 shape。

### A/F 组合边界

A 阶段（`a_rank_compute`，按 A domain 的 TP 切分 rank 本地形状）：

- RMSNorm、QKV GEMM `(T, (q+2kv)/tp, h)`、RoPE；
- Attention 核心：Prefill 按 request 长度分组查 `context_attention`，Decode 按 context 长度分组查 `generation_attention`，使用真实 KV head 维度；
- 输出投影 GEMM `(T, h, q/tp)`、Residual Add。

Router 阶段（A leader，`router_topk`）：F 侧 RMSNorm 和 Router GEMM `(T, experts, h)`；dense 模型只有 RMSNorm。

F 阶段（`expert_rank_compute`，按 F domain 的 TP 切分）：

- MoE：一次 `moe` 表查询。`T` 是本 rank 已展开 Top-K 的 routed assignments，平衡的 EP group 会把 `N` 个输入 token 变成每 rank `N * top_k / ep_size` 个 assignments，所以查表用等效输入 token 数 `T * ep_size / top_k`，与 step 级 backend 的 MoE 语义一致；
- dense FFN：gate/up GEMM、SwiGLU、down GEMM；
- Residual Add。

Attention 核心不重复包含 QKV 和输出投影；MoE expert 不包含 Router 或通信；shared experts 只进入 runtime plan 的显存账本，不单独计时。A→F/EP dispatch、F→A/EP combine 和必要的 TP collective 是独立通信事件，一层不会再额外生成同 payload 的 F-internal dispatch/combine。

### 通信 service time

- `af_a_to_f` / `af_f_to_a`：`service = latency_us + payload_bytes / (bandwidth_per_direction * efficiency(payload))`，链路类来自 AF link 的 `network` 在 `networks` 中映射到的 ID；
- `tp_collective`：参与 rank 的物理 worker 映射到拓扑 device index，交给分层 alpha-beta collective 模型计算 all-reduce（设备数据包有实测 NCCL 曲线时优先），没有拓扑放置时退化为同一链路类上的 ring；
- payload 由 planner 唯一计算，provider 只返回 service time，不重算 Top-K、不计算 queue time。全/半双工、容量和多个 AF blocks 的竞争仍由 AF executor 的 SimPy resource 产生。

Groq 等 SRAM 架构的峰值、链路参数同样写在 `data/devices` 与 `data/topologies` 的 YAML 中，并带 `source_id` 与证据等级；峰值不能冒充算子实测。

## 端到端验证矩阵

AF executor 的测试从真实配置加载，经 Engine、worker、scheduler、资源申请和 barrier 运行到最终 request result 与 trace。测试使用固定 stub compute/transfer provider，但不绕过主执行流程；事件时间线是机制验收的最终判定。

| 场景 | 主要断言 |
| --- | --- |
| 单 block、单层 | A TP→A Router→A-F/EP Dispatch→Expert→F-A/EP Combine 依赖顺序严格正确；每层只有两次跨域 payload 搬运 |
| 单 block、多层 | 每层边界清晰，层间状态不丢失，iteration barrier 正确 |
| 双 block 流水 | 独立计算/通信按依赖重叠，makespan 与 oracle 一致 |
| A 慢、F 快 | A 是稳态瓶颈，F 不产生虚假并发 |
| A 快、F 慢 | F 队列与等待符合容量，request 不重复推进 |
| 全双工 | A→F 与 F→A 使用不同资源键，可同时服务 |
| 半双工 | 两个方向竞争同一资源键，排队区间可检查 |
| 多个 F worker | 容量和 worker 占用独立，任务只执行一次 |
| Prefill | token_count 为 request 实际 Prefill token 总和 |
| Decode | 每 request 每轮一个 token，轮末 barrier 正确 |
| 非 AF 回归 | AF 默认关闭，原 Prefill/Decode、KV、MoE 和 latency 行为不变 |
| TP=2, DP=2, EP=2 | 局部 shape、replica 选择、KV shards、A/F component bytes、Expert 驻留、边界传输次数、join 与 makespan 可核对 |

每项必须断言：

- 依赖顺序和无越界重叠；
- 合法的计算/通信并行；
- resource group 竞争与容量；
- `queue_duration`、`service_duration` 及其总和；
- iteration barrier；
- 每个请求恰好推进一次；
- trace 区间完整、无负时长；
- 最终 makespan 和请求完成状态。

### 固定双 block oracle

```text
A compute = 2 ms
A Router  = 1 ms
A -> F    = 1 ms
F Expert  = 2 ms
F -> A    = 1 ms
```

容量为 1 时，单 block 为 7 ms，两个 block 完全串行为 14 ms。正确流水应为：

```text
block 0: A[0,2]  Router[4,5]  A->F[5,6]  Expert[6,8]   F->A[8,9]
block 1: A[2,4]  Router[5,6]  A->F[6,7]  Expert[8,10]  F->A[10,11]
```

两个 blocks 同时提交时，它们的 A compute 先进入同一个 A worker 队列，因此 block 0 的 Router 在两个 A compute 之后取得该 worker。最终 makespan 必须精确为 11 ms。该 oracle 证明执行机制正确，不证明真实硬件时延准确。

当前确定性 E2E 矩阵的已验证结果如下，时间均来自 SimPy event records：

| 场景 | makespan | 关键 queue/service 与通过条件 |
| --- | ---: | --- |
| 单 block，2/1/1/2/1 ms | 7 ms | A、A Router、A-to-F/EP dispatch、F Expert、F-to-A/EP combine 五段非零 service 严格相接，不生成独立 dispatch/combine 事件 |
| 双 block，2/1/1/2/1 ms | 11 ms | 两次 A compute 和 Router 竞争同一 A worker；block 1 的 Expert 等待 F worker |
| A 慢 4 ms，F 快 1 ms，3 blocks | 13 ms | 主要等待落在共享 A slot，A service 区间不重叠 |
| A 快 1 ms，F 慢 4 ms，3 blocks | 13 ms | 主要等待落在 F pool，F service 区间不重叠 |
| full-duplex，双向各 3 ms | 10 ms | A-to-F 与 F-to-A 在独立方向键上合法重叠 |
| half-duplex，双向各 3 ms | 13 ms | 相反方向串行，两个 return transfer 各等待 3 ms |
| 同 resource group 的两条 links | 9 ms | 相同方向键跨 link 竞争 |
| 不同 resource groups 的两条 links | 6 ms | 两条 link 的方向资源独立 |
| 1 个 F worker，两个 blocks 的 Router/Expert 为 1/2 ms | 5 ms | Router 在 A worker 串行，两个 Expert 在同一 F worker 串行 |
| 2 个 F workers，两个 blocks 的 Router/Expert 为 1/2 ms | 4 ms | Router 仍在 A worker 串行，两个 Expert 分配到不同 F workers 并行 |

测试还覆盖 Prefill-only、Decode-only、两阶段、不同 A groups 的阶段间 KV transfer、trace on/off、AF disabled、provider 缺失/unsupported/非法值，以及失败后同一 SimPy environment 的恢复运行。

分析报告还应检查通信隐藏候选条件：

```text
T_C < T_A
T_C < T_F
m >= ceil(2 * (1 + T_C / max(T_A, T_F)))
```

公式只用于解释和筛选实验点；3 或 4 个 micro-batch 不是固定配置，最终判定以离散事件 trace 为准。

## 精度校验与数据要求

### 来源等级

| 等级 | 来源 | 可作出的声明 |
| --- | --- | --- |
| A | 官方规格、论文算子级或层级实测 | 在完整记录实验口径后直接使用 |
| B | 公开端到端基准 | 检查整体量级和趋势，不拆成单算子真值 |
| C | 经过公开结果校准的 roofline | 离线补齐 shape，声明为校准估算 |
| D | 项目初始假设 | 只用于敏感性分析 |

所有硬件参数和时延通过 `source_id` 映射到来源元数据。C/D 数据不得声明物理准确。

### 计算数据校验

calibration set 与未参与拟合的 holdout set 必须隔离。按 GEMM、Prefill Attention、Decode Attention、MoE、逐元素算子分别报告：

- shape coverage；
- 平均和最大相对误差；
- P50、P90、P95 相对误差；
- bias；
- 瓶颈分类一致性；
- worst-error shapes。

### 通信数据校验

microbenchmark 数据必须覆盖：

- payload size；
- A→F 和 F→A；
- 全双工和半双工；
- 单传输与并发传输；
- 本地、跨设备和跨节点；
- resource group 竞争；
- A→F/EP dispatch、F→A/EP combine 与必要的 TP collective。

`service_duration` 对照链路实测，`queue_duration` 对照确定性事件 oracle，total 必须等于二者之和。没有目标链路数据时只能声明模型自洽，不能声明物理准确。

### 公开与真实系统对齐

每项基准必须记录模型版本、数值精度、batch/并发、输入输出长度、芯片数量、是否包含主机通信、软件版本、拓扑和指标定义。未知信息必须标记未知，不能自行补齐。

- BERT-base 在 Groq TSP 上 batch=1、序列长度 128、约 130 us，可作口径匹配的模型级校验。
- Mixtral 8x7B 约 812 token/s 缺少完整芯片数、编译映射和服务并发信息，只能作弱范围检查。
- 匹配真实系统 trace 时比较 A/F kernel、transfer、queue/wait、TTFT、TPOT 和吞吐。

### 敏感性分析

分别扰动算子延迟、有效带宽和固定延迟。报告必须给出：

- AF 实验结论保持不变的参数区间；
- 结论发生变化的边界；
- 对结论影响最大的输入；
- 已知误差来源；
- 仍需补充的数据。

## 自动化与后续 change 边界

Codex 可以实现并运行配置测试、确定性 E2E oracle、trace invariant、表格一致性、calibration/holdout 切分、误差报告和敏感性分析脚本。Codex 不能生成目标硬件的物理 ground truth；算子 benchmark、链路 benchmark 和真实系统 trace 必须来自实际测量或可追溯公开资料。

`add-af-executor` 实现 AF block/executor、共享 SimPy 资源、结构化 trace 和确定性 E2E harness。正式性能数据能力由仓库统一的算子表栈提供（`TokenSim/af/operator_provider.py`）：

- 设备目录 `data/devices`、链路类 `data/topologies/links.yaml`、算子数据包 `data/operator_data`；
- `OperatorTableAFComputeProvider` 通过与 step 级 backend 共享的 `OperatorResolver` 查表、插值、外推并回退解析模型；
- `TopologyAFTransferProvider` 用链路类和 collective 模型给出通信 service time。

正式 provider 通过现有 compute/transfer query 与 `AFServiceEstimate` 接口接入。`source_id`、dataset、match/composition metadata 只用于可追溯性和误差分析，不参与 executor 的资源选择或调度。运行时不依赖 AIConfigurator 或 InferSim；AIConfigurator 的实测表通过离线导入进入 `data/operator_data`。

仓库测试通过 `HardwareContext.in_memory` 构建微型设备目录和 `TestModel`，AF 集群在生产的 HardwareContext / ModelSpec / CacheConfig 代码路径上运行，不再需要任何原生扩展占位。
