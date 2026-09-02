# AF 分离配置与验证

## 当前实现范围

本版本已经把 Attention-FFN（AF）分离配置接入可运行的 SimPy executor。它支持 Prefill、Decode 或两阶段 AF，能够明确 A/F worker group、通信网络、全/半双工以及物理竞争域，并输出可重建流水行为的结构化 trace。

当前 executor 需要调用方注入 compute 与 transfer latency provider。仓库测试使用带 `test-stub` source ID 的确定性 provider 验证调度机制；这些数值不是 Groq 或其他硬件的物理性能结论。缺少 provider 或必要模型元数据时，`benchmark.py` 会在产生 AF 事件前明确失败，不会静默回退到 Roofline 或 LLMCompass 的完整 iteration 时延。

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
A compute -> A-to-F transfer -> F compute -> F-to-A transfer
```

同一 block 只有在 F-to-A 返回原 A worker 后才能进入下一层。不同 blocks 使用独立 SimPy process，可以在依赖和资源容量允许时形成计算/通信流水。A compute 固定使用持有 request 与 KV 的 source A worker；F compute 从 link 对应的 F pool 获取稳定 worker token。F-only worker 不拥有 scheduler、Request 或 KV Cache。

每个物理 worker 只有一个共享 compute slot。未选择 AF 的完整 Prefill/Decode 与 AF compute 也申请该 slot，复合 capability worker 因而不会重复提供算力。所有 iterations 共用 Engine 生命周期内的 worker、F pool 和通信资源注册表，跨 block、跨 link、跨 iteration 的竞争不会因重建资源而消失。

iteration 等待所有 blocks 完成 barrier 后，才由原 worker 入口对本轮每个 request 调用一次 `step()` 并更新 scheduler。block、layer 和 stage 完成本身不会推进 request，也不会分配、释放或迁移 KV blocks。Prefill 与 Decode 使用不同 A groups 时，阶段间 KV 仍通过现有 connector 转移；逐层 A-to-F/F-to-A link 不替代该 KV transfer。

Trace 的每个 compute/transfer 事件包含 iteration、block、layer、stage 或 direction、resource、worker、ready/start/end、queue/service、request/token 数和 `source_id`，transfer 还包含 payload bytes。Iteration summary 包含 block completion、barrier wait、makespan 和只按 service intervals 计算的资源利用率。`trace_enabled=false` 只关闭记录，不改变资源选择或模拟时间。

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
| Decode Attention | dtype、batch、context_len、head 配置、window | 对 context_len 一维内插 |
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

A 阶段组合：

- QKV GEMM；
- Attention 核心；
- 输出投影 GEMM；
- 必要的 RMSNorm、RoPE、Residual Add 和 data reorder。

F 阶段组合：

- Router/Gating GEMM；
- MoE grouped/fused expert 计算；
- 必要的逐元素算子。

Attention 核心不重复包含 QKV 和输出投影；MoE expert 不包含 Router、dispatch 和 combine；softmax 与 fused SwiGLU 只在其所属聚合算子中计时一次。dispatch、combine、AllReduce、A→F 和 F→A 始终是独立通信事件。

GQA/MQA 必须使用真实 KV head 维度。Decode `context_len` 使用长度桶上界或 batch 最大 KV 长度，不使用平均长度。MoE 满足 `sum(T_e)=T*k`，第一版以 `balanced` distribution 建表。

Groq 第一版候选公开峰值包括 FP16 188 TFLOPs、INT8 750 TOPS、230 MB SRAM、80 TB/s SRAM 带宽和 480 GB/s 芯片网络带宽。每项写入 `hardware.yaml` 前必须保存原始来源，并区分公开规格、项目假设和待确认参数；峰值不能冒充算子实测。

AIConfigurator 只作为分表 schema、查询和元数据组织参考；InferSim 只作为 FLOPs、内存搬运和通信量组合参考。两者不是运行时依赖，也不是物理精度 ground truth。

## 端到端验证矩阵

AF executor 的测试从真实配置加载，经 Engine、worker、scheduler、资源申请和 barrier 运行到最终 request result 与 trace。测试使用固定 stub compute/transfer provider，但不绕过主执行流程；事件时间线是机制验收的最终判定。

| 场景 | 主要断言 |
| --- | --- |
| 单 block、单层 | A→A-F→F→F-A 依赖顺序严格正确，每个事件一次 |
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
A -> F    = 1 ms
F compute = 3 ms
F -> A    = 1 ms
```

容量为 1 时，单 block 为 7 ms，两个 block 完全串行为 14 ms。正确流水应为：

```text
block 0: A[0,2]  A->F[2,3]  F[3,6]  F->A[6,7]
block 1: A[2,4]  A->F[4,5]  F[6,9]  F->A[9,10]
```

最终 makespan 必须精确为 10 ms。该 oracle 证明执行机制正确，不证明真实硬件时延准确。

当前确定性 E2E 矩阵的已验证结果如下，时间均来自 SimPy event records：

| 场景 | makespan | 关键 queue/service 与通过条件 |
| --- | ---: | --- |
| 单 block，2/1/3/1 ms | 7 ms | 四段 service 严格相接，queue 均为 0 |
| 双 block，2/1/3/1 ms | 10 ms | block 1 的 A 与 block 0 的 F 重叠；早完成 block barrier wait 为 3 ms |
| A 慢 4 ms，F 快 1 ms，3 blocks | 13 ms | 主要等待落在共享 A slot，A service 区间不重叠 |
| A 快 1 ms，F 慢 4 ms，3 blocks | 13 ms | 主要等待落在 F pool，F service 区间不重叠 |
| full-duplex，双向各 3 ms | 10 ms | A-to-F 与 F-to-A 在独立方向键上合法重叠 |
| half-duplex，双向各 3 ms | 13 ms | 相反方向串行，两个 return transfer 各等待 3 ms |
| 同 resource group 的两条 links | 9 ms | 相同方向键跨 link 竞争 |
| 不同 resource groups 的两条 links | 6 ms | 两条 link 的方向资源独立 |
| 1 个 F worker，两个 3 ms F stages | 6 ms | 同时服务数为 1 |
| 2 个 F workers，两个 3 ms F stages | 3 ms | worker 1/2 在 0 ms 同时开始，容量不越界 |

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
- 内部 dispatch/combine 与 AllReduce。

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
