# AF 分离配置与验证

## 当前实现范围

本版本已经把 Attention-FFN（AF）分离配置接入可运行的 SimPy executor。它支持 Prefill、Decode 或两阶段 AF，能够明确 A/F worker group、通信网络、全/半双工以及物理竞争域，并输出可重建流水行为的结构化 trace。

当前 executor 需要调用方注入 compute 与 communication latency provider。Decode AF planner 按逐层模型元数据把 Dense 与 MoE 层展开为逐 rank DAG；provider 只估算一个 rank 的局部计算或一次已规划通信。Decode-only AF 下，Prefill 继续走完整模型 aggregate 路径，结果分别标记为 `decode_af_rank_local` 与 `prefill_aggregate`。仓库测试使用带 `test-stub` source ID 的确定性 provider 验证调度机制；这些数值不是 Groq 或其他硬件的物理性能结论。缺少 provider 或必要模型元数据时，`benchmark.py` 会在产生 AF 事件前明确失败，不会静默回退到完整 iteration 时延。

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

scheduler 选出的同阶段 requests 按 `af_block_size` 连续组成 blocks。Decode AF 对每层读取真实 layer type，并生成不同依赖链：

```text
Dense:
A Attention -> A collective -> A-to-F endpoint legs
            -> F Dense FFN -> F collective -> F-to-A owner legs
            -> A residual/layout

MoE:
A Attention -> A collective -> A leader Router/Top-K
            -> A-to-F routed/shared endpoint legs
            -> F Routed Expert branches ----+
            -> F Shared Expert TP branch ---+-> F-to-A owner legs
            -> A weighted combine/shared sum -> A residual/layout
```

Router/Top-K 固定在所选 A replica 的 leader，并等待 A TP compute/collective 完成。每个 MoE layer、phase 和 Decode step 都冻结独立的逐 token routing snapshot；每条 assignment 保留 request/token、A owner、Expert 与 routing weight，histogram 只能从 assignments 派生并用于统计，不能替代正式执行数据。

A-to-F 是 Routed dispatch 和 Shared input 的唯一跨域传输，F-to-A 是 Routed/Shared partial return 的唯一跨域传输。每个方向在一层中只有一个 logical stage，但会按 producer、endpoint、token item 和 global hidden interval 展开为多条可调度 legs；每条 leg 只等待自己的 producer。F-to-A 不做数学 combine：partial 到达对应 A tensor owner 后，A rank 本地执行 weighted combine、Shared sum 和一次 residual。不同 TP sizes 通过 `[start, end)` hidden interval 的交集切分，不把完整 hidden 重复发给每个 rank，也不生成第二套 `ep_dispatch`/`ep_combine` 通信。

A 域持有 request、KV、activation owner 和 residual，首次 admission 后不能在没有 KV transfer 的情况下换 A DP replica。A owner 与 F service replica 使用彼此独立的全局 round-robin；rank-reuse EP 下只有 Routed branches 可按 Expert placement 跨 F world，Dense/Shared 仍使用选中的 F DP replica。

Shared Expert 位于 F 域，对每个原始 token 恰好执行一次，与 Routed branches 并行。`top_k=K` 时 Routed 总工作量是 `T*K` assignments，而 Shared、combine、residual 和下一层仍是 `T` tokens。

## AF 并行拓扑

AF 第一阶段支持 TP、DP 和 F 域 EP，明确不支持 PP。任何 A/F execution domain 的 `pipeline_parallel_size > 1` 都会在创建执行事件前失败。

- 每个 `LogicalInstanceConfig` 表示一个 DP replica，而不是重复包含整个 DP world。
- A replica 有 `TP` 个 ranks，坐标为 `(dp_rank, ep_rank=0, tp_rank, pp_rank=0)`。
- F replica 同样只有 `TP` 个基础 ranks；EP 只解释这些已有 ranks，不增加 physical worker。
- A 域必须为 `EP=1`。F 域仅支持 `EP=1` 或 `EP=F_TP*F_DP`：前者在每个 F DP replica 内复制完整 Expert 集并以 F TP 切分单个 Expert，后者用 `ep_rank=dp_rank*F_TP+tp_rank` 把 Expert 放到现有 F world，单个 Expert 的 TP 为 1。
- `domain_rank` 在整个 domain 唯一，每个 physical worker 只能映射一个 rank。

完整可加载示例见 `data/clusters/af/decode_tp2_dp2_ep1.json` 与 `data/clusters/af/decode_tp2_dp2_ep4_rank_reuse.json`。两者的 A 域与 F 域都是 `TP=2, DP=2`，所以各有 4 个 physical ranks；将 F EP 从 1 改为 4 只改变 Expert placement 和 Expert TP，不会把 F worker 数从 4 乘到 16。每个 logical instance 只包含自己 `dp_rank` 下的两个 TP ranks。

从仓库根目录运行最小 Decode-only Engine 验证：

```powershell
python -m unittest tests.test_af_engine_integration.AFEngineIntegrationTest.test_decode_only_connects_runtime_plan_dtypes_and_providers -v
```

该命令使用确定性测试 provider，经真实 Engine、Scheduler 和 AF executor 执行一次完整 Prefill 与后续 Decode。断言 Prefill 保持 `prefill_aggregate`，Decode 使用 `decode_af_rank_local`，并检查逐层 metadata、五类 dtype、provider query、request/KV 单次推进和最终结果标记。测试数值只验证执行语义，不代表目标硬件性能。

所有实际 TP shard axis 必须精确整除。唯一例外是 KV heads 少于 A TP 且复制组可无歧义划分；例如 8 个 query heads 配 TP=3 会在 runtime plan 前失败，而不是用 `ceil` 伪造第 9 个 head。

### Dtype 角色

AF 分别配置 `weight_dtype`、`activation_dtype`、`kv_cache_dtype`、`a_to_f_dtype` 和 `f_to_a_dtype`。前三个是 runtime/model 角色，后两个通常写在 link 上。所有值经过同一 registry 规范化，FP8/BF16/FP16/FP32 分别按 1/2/2/4 bytes 记账。旧 `af_dtype` 只在没有任何新字段时映射到五个角色并标记弃用；与新字段混用会失败。传输 dtype 与 activation dtype 不同时，DAG 显式加入发送 encode 和接收 decode compute，链路 provider 不隐藏这部分时间。

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

全双工 fabric 与 link 示例：

```json
{
  "communication_fabrics": [{
    "resource_group": "af-fabric",
    "network": "af_net",
    "duplex": "full",
    "lanes": [
      {"name": "a_to_f", "capacity": 1},
      {"name": "f_to_a", "capacity": 1}
    ]
  }],
  "links": [{
    "name": "attention-ffn",
    "phases": ["decode"],
    "attention_group": "attention-pool",
    "ffn_group": "ffn-pool",
    "network": "af_net",
    "resource_group": "af-fabric",
    "a_to_f_dtype": "fp8",
    "f_to_a_dtype": "bf16"
  }]
}
```

半双工 fabric 只声明一个 `shared` lane。每个所选阶段必须恰好由一条 link 覆盖；第一版不定义同阶段多 link 的路由策略。A/F 端点必须不同，并分别声明 `af_attention` 与 `af_ffn`。

必要 TP collective 由 execution domain 的 `communication_plan` 显式声明 `operation`、触发 stage、replica-local `participants`、`network`、`resource_group` 和 `resource_lane`。`participants` 在每个 DP replica 内映射为实际 domain ranks。Planner 传每 rank logical bytes，provider只应用一次 all-reduce/all-gather/reduce-scatter wire-volume 规则。缺 fabric/lane 时立即失败，不能借用 A-to-F 链路；collective 若显式选择与 AF 相同的 lane，才会与该方向竞争。

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

### 离线生成与运行时查询

后续 operator-data change 必须将数据生成和 TokenSim 运行分开：

1. `shape_manifests/<experiment-id>.yaml` 定义目标 shape。
2. 离线生成器根据实测、workload 公式、硬件参数和校准系数生成数据。
3. `*_perf.parquet` 只保存查询主键、`latency_us` 和 `source_id`。
4. `*_analysis.parquet` 保存 FLOPs、搬运量、理论时间、校准系数和瓶颈。
5. `generation_meta.yaml` 保存硬件参数、公式版本、单位、来源和行数。
6. TokenSim 运行时只查表，不重新计算 roofline。

第一版分表：

| 表 | 主要主键 | 允许的插值 |
| --- | --- | --- |
| GEMM | dtype、m、n、k | 相同 dtype/n/k 下对 m 一维内插 |
| Prefill Attention | dtype、batch、input_seq_len、head 配置、window | 对 input_seq_len 一维内插 |
| Decode Attention | weight/activation/KV dtype、batch、context_len、head 配置、window | 对单 request 的 context_len 一维内插 |
| MoE | dtype、phase、distribution、tokens、模型/专家/TP/EP | 对 num_tokens 一维内插 |
| 逐元素算子 | op、dtype、tokens、hidden size | 第一版不得跨 op 或 dtype 插值 |

查询顺序固定为精确匹配、表覆盖范围内受控一维插值、明确缺失错误。第一版禁止外推，也禁止运行时公式或默认时间兜底。返回结果至少包含：

```text
latency_us
match_type = exact | interpolated
source_id
```

`match_type` 表示查询方式；`source_id` 表示证据来源，两者不能混用。缺失错误必须输出完整主键，交由离线生成器补表。

### A/F 组合边界

A 阶段查询：

- Attention（内部只组合一次 Q/K/V、Attention core 和 output projection）；
- MoE layer 的 A leader Router/Top-K；
- MoE weighted combine/Shared sum；
- 每个原始 token 一次 residual；
- 必要的 layout 与跨 dtype encode/decode。

F 阶段查询：

- Dense layer 的 Dense FFN；
- MoE layer 的逐 Expert Routed work；
- 按原始 token 执行的 Shared Expert TP branch；
- 必要的跨 dtype encode/decode。

所有 composition 互斥：F Expert 不包含 Router、residual 或通信，Routed workload 已是 assignments 数，Shared/residual 使用原始 token 数。A→F dispatch、F→A partial return 和必要 TP collective 是独立通信事件；F→A 不执行 weighted combine，也不会再生成同 payload 的 F-internal dispatch/combine。

GQA/MQA 必须使用真实 KV head 维度。Decode Attention 先对每个 request 使用自己的 `min(context_len, window_size)`（无 window 时用完整 context），再聚合工作量，不能以平均值或 batch 最大值替代所有 requests。MoE 满足 `sum(T_e)=T*k`，其中 `T_e` 是已展开的 routed assignments；Expert 公式直接使用 `T_e`，不再乘一次 `k`。外部 assignments 与 histogram 同时存在时必须完全匹配；只有 histogram 时只能报告统计，不能驱动 ownership combine。

Groq 第一版候选公开峰值包括 FP16 188 TFLOPs、INT8 750 TOPS、230 MB SRAM、80 TB/s SRAM 带宽和 480 GB/s 芯片网络带宽。每项写入 `hardware.yaml` 前必须保存原始来源，并区分公开规格、项目假设和待确认参数；峰值不能冒充算子实测。

AIConfigurator 只作为分表 schema、查询和元数据组织参考；InferSim 只作为 FLOPs、内存搬运和通信量组合参考。两者不是运行时依赖，也不是物理精度 ground truth。

## 端到端验证矩阵

AF executor 的测试从真实配置加载，经 Engine、worker、scheduler、资源申请和 barrier 运行到最终 request result 与 trace。测试使用固定 stub compute/transfer provider，但不绕过主执行流程；事件时间线是机制验收的最终判定。

| 场景 | 主要断言 |
| --- | --- |
| 单 block、Dense 层 | 无 Router/Expert/weighted combine，A Attention→F Dense→A residual 依赖正确 |
| 单 block、MoE 层 | assignments、Routed/Shared 并行、owner return、A combine/residual完整 |
| 单 block、多层 | Dense/MoE DAG分别选择，下一层等待全部 A-side terminals |
| 双 block 流水 | 独立计算/通信按依赖重叠，makespan 与 oracle 一致 |
| A 慢、F 快 | A 是稳态瓶颈，F 不产生虚假并发 |
| A 快、F 慢 | F 队列与等待符合容量，request 不重复推进 |
| 全双工 | A→F 与 F→A 使用不同资源键，可同时服务 |
| 半双工 | 两个方向竞争同一资源键，排队区间可检查 |
| 多个 F worker | 容量和 worker 占用独立，任务只执行一次 |
| Prefill | token_count 为 request 实际 Prefill token 总和 |
| Decode | 每 request 每轮一个 token，轮末 barrier 正确 |
| 非 AF 回归 | AF 默认关闭，原 Prefill/Decode、KV、MoE 和 latency 行为不变 |
| TP=2, DP=2, EP=1/4 | Expert-TP 与 rank-reuse 两种模式的局部 shape、replica 选择、KV shards、组件驻留、owner return、join 与 makespan 可核对 |

每项必须断言：

- 依赖顺序和无越界重叠；
- 合法的计算/通信并行；
- resource group 竞争与容量；
- `queue_duration`、`service_duration` 及其总和；
- iteration barrier；
- 每个请求恰好推进一次；
- trace 区间完整、无负时长；
- 最终 makespan 和请求完成状态。

### Deterministic DAG oracle

确定性测试为每种 typed operation 注入固定 service time，但期望 makespan 必须从完整 DAG 和实际资源队列推导，不能沿用旧五段式求和。MoE oracle 至少包含：

```text
A Attention ranks -> A collective -> Router
                    -> A-to-F endpoint legs
                    -> Routed branches ----+
                    -> Shared TP branch ---+-> F-to-A owner legs
                    -> per-owner weighted combine -> residual -> optional layout
```

Dense oracle以 Dense FFN branch 替换 Router/Routed/Shared/weighted combine。单 block 的完成时间是全部 A-side terminal 的最晚结束时间；多 block iteration 的 makespan 是最晚 block completion，而不是所有 rank、assignment 或 block service time 的和。两个 ready blocks 只有在依赖或同一 `(resource_group, resource_lane)` / physical compute slot 上冲突时才串行，其他 branches 应允许重叠。

每个 oracle 都从 trace 重新计算以下事实：每个 node 的 ready time 等于全部直接前驱的最晚结束时间，`queue=start-ready`，`service=end-start`，同容量资源的 service intervals 不非法重叠，owner combine 等待自己的全部 Routed/Shared partials，下一层等待前一层全部 A-side terminals。修改 service time、duplex、lane capacity、F rank 数或 routing skew 后，makespan 差异必须能由这些区间精确解释。固定值只验证执行机制，不代表真实硬件性能。

兼容回归还覆盖 Decode-only 下的完整 Prefill aggregate path、显式 AF-Prefill compatibility、trace on/off、AF disabled、provider 缺失/unsupported/非法值、request/KV 单次推进，以及失败后同一 SimPy environment 的恢复运行。

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
- A→F routed/shared endpoint legs、F→A owner-specific partial-return legs与必要的 TP collective。

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

`add-af-executor` 实现 AF block/executor、共享 SimPy 资源、结构化 trace 和确定性 E2E harness。`add-operator-data-latency-provider` 已增加以下正式性能数据能力，详细使用方式见 [operator-data-latency.md](operator-data-latency.md)：

- Groq `hardware.yaml`、shape manifest、离线生成器和 Parquet 数据；
- 实测 kernel 匹配、插值、公式校准和 A/F 延迟组合 provider；
- kernel/link 数据采集、校准和真实系统对照。

正式 provider 通过现有 compute/transfer query 与 `AFServiceEstimate` 接口接入。`source_id`、dataset、match/composition metadata 只用于可追溯性和误差分析，不参与 executor 的资源选择或调度。运行时使用 `pyarrow` 加载离线生成的 Parquet，但不依赖 AIConfigurator 或 InferSim。

仓库完整回归在 Windows 上使用测试模块内的 `TransformerRoofline` 占位实现，因为仓库携带的原生扩展是 Linux CPython 3.11 `.so`，不能由本机 CPython 3.13 直接加载。占位只解除平台导入限制；Prefill、Decode、KV transfer、并行、MoE、trace timestamp 和 latency baseline 的 Python 行为仍由完整测试套件执行。
