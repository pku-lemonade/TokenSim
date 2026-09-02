## Context

当前 `afd@967692f` 的集群配置由 `ClusterConfig.from_file()` 读取 JSON，并通过 Pydantic dataclass 构造 `WorkerGroupConfig`。真实 worker role 是 `prefill`、`decode`、`hybrid`；worker group 只有 `role`、`hardware`、`num_workers` 和 `network`，无法表达 Attention/FFN 分离职责、AF 作用阶段或物理通信竞争域。`benchmark.py --block_size` 已用于 KV Cache，不能复用为 AF micro-batch 粒度。

需求优先级以总任务书为准：AF 能力覆盖 Prefill 和 Decode，并允许实验按配置只启用其中一个阶段。Phase 1 文档中的 Decode-only 边界保留为一种合法实验配置，不再作为系统能力上限。动机与范围见 `proposal.md`，行为契约见 `specs/af-separation-config/spec.md`。

本 change 只建立配置、归一化和校验契约。AF block 状态机、执行器、通信 SimPy resource、离线算子数据生成器和分表查询接口在后续 changes 中实现；运行入口在此期间必须显式拒绝已启用 AF 的运行，不能静默使用完整 iteration 时延。

## Goals / Non-Goals

**Goals:**

- 建立可由 JSON 和 Python 调用方共同使用的规范化 AF 配置模型。
- 显式选择 Prefill、Decode 或两阶段 AF，并为每个阶段绑定唯一 A/F 拓扑。
- 将计算职责从旧 role 解耦为可组合 capability，同时保持现有配置可加载。
- 使用 link 和 `resource_group` 表达通信端点、网络与竞争域，为后续排队模拟提供确定输入。
- 在模拟开始前完成字段、阶段、拓扑、网络引用和共享资源一致性校验。
- 固定 AF 离线算子数据与运行时查询的架构边界，避免后续退回旧 Roofline/LLMCompass 或笼统的 A/F 固定时延。

**Non-Goals:**

- 不在本 change 中实现 AF block、executor、队列、屏障、时间线或 SimPy resource。
- 不实现 Groq 硬件参数、shape manifest、离线算子数据生成器、Parquet 数据、插值器或 transfer provider。
- 不填入 Ada/Groq 的未提供性能数值，也不模拟芯片内部指令或周期级微架构。
- 不实现同一阶段多个 AF links 之间的负载均衡策略。
- 不改变现有模型并行度、KV Cache 分配或非 AF 时延语义。

## Decisions

### Decision 1: AFConfig selects phases and owns explicit links

`ClusterConfig` 新增默认构造的 `af` 字段，核心数据模型为：

```python
@dataclass
class AFConfig:
    enabled: bool = False
    phases: tuple[Literal["prefill", "decode"], ...] = ("prefill", "decode")
    af_block_size: int = 1
    links: list[AFLinkConfig] = Field(default_factory=list)
    trace_enabled: bool = True


@dataclass
class AFLinkConfig:
    name: str
    phases: tuple[Literal["prefill", "decode"], ...]
    attention_group: str
    ffn_group: str
    network: str
    resource_group: str
    duplex: Literal["full", "half"] = "full"
    a_to_f_capacity: int | None = None
    f_to_a_capacity: int | None = None
    half_duplex_capacity: int | None = None
```

总任务书要求两阶段能力，因此 `phases` 的规范默认值是两阶段；`enabled=false` 保证旧配置行为不变。Decode-only 实验显式写为 `phases: ["decode"]`。

容量放在 link 而非 AFConfig 顶层，因为不同阶段可以使用不同网络和设备池。A/F worker 数量仍来自 worker groups，避免重复拓扑真相。

### Decision 2: Use generic AF stage capabilities plus phase binding

定义 capability 集合：

| Capability | 含义 |
|---|---|
| `prefill` | 非 AF 的完整 Prefill |
| `decode_full` | 非 AF 的完整 Decode |
| `af_attention` | 可承担 AF 的 A 阶段并拥有该阶段 request/KV 状态 |
| `af_ffn` | 可承担 AF 的 F 阶段，不拥有 KV Cache |

`af_attention` 与 `af_ffn` 不写死 Prefill 或 Decode；具体阶段由 link 的 `phases` 绑定。这样同一物理 A/F 设备池可覆盖两阶段，也可用两条 link 将两阶段映射到不同设备。

备选方案是定义 `prefill_attention`、`decode_attention` 等四个 capability。该方案会重复阶段信息并增加组合数量，因此不采用。

### Decision 3: First version requires exactly one link per selected phase

配置归一化后，对每个 `af.phases` 中的阶段建立 `phase -> link` 映射，并要求恰好一条 link 覆盖该阶段。link 的 A/F 端点必须是不同 group，并分别声明 `af_attention`/`af_ffn`。

这是第一版的确定性约束。允许同阶段多 link 会立即引入按请求、block、队列长度或最早完成时间选路的策略，而这些不属于配置 change。后续若增加多 link 调度，必须单独扩展规格。

若只选择一个 AF 阶段，未选择阶段必须存在完整执行 capability。若 Prefill 和 Decode 由不同 A group 承担，后续执行器在阶段边界复用现有请求派发与 KV transfer 机制；该阶段迁移不是 A→F/F→A link，也不得计入 AF 层内通信。

### Decision 4: AF block size counts requests in both phases

`af_block_size` 表示每个 AF block 的最大 request 数，builder 按 scheduler 顺序连续分组。Decode 每个 request 每轮贡献一个 token；Prefill block 的 token 数是所含 request 的实际 Prefill 计算 token 数之和。第一版不切分单个长 Prefill request。

这个定义保持一个配置字段在两阶段中的稳定含义，同时让时延 provider 使用实际 `token_count` 和 shape 反映 Prefill 长度差异。备选的 token-budget Prefill chunking 会改变请求推进和 KV 分配语义，留给独立 change。

### Decision 5: Normalize current legacy roles before cross-field validation

配置加载分三步：

1. 将原始 group 归一化为 `name`、`capabilities`、`hardware`、`num_workers`、`network` 和可选兼容 role。
2. 校验单对象字段、唯一性和网络引用。
3. 在 ClusterConfig 层校验阶段覆盖、link 端点和共享资源一致性。

以当前代码为准的旧 role 映射为：

| Legacy role | Normalized capabilities |
|---|---|
| `prefill` | `prefill` |
| `decode` | `decode_full` |
| `hybrid` | `prefill`, `decode_full` |

旧 group 缺少 `name` 时按输入顺序生成确定性名称，例如 `legacy-prefill-0`。若同时提供 role 和 capabilities，role 对应的完整阶段 capabilities 必须一致；额外的 AF capabilities 可以显式添加，但不能由旧 role 推断。文件加载和 Python 构造调用同一规范化/校验入口。

### Decision 6: resource_group is the physical competition identity

每条 link 通过 `network` 选择性能配置，通过 `resource_group` 声明物理竞争域。相同 resource group 的 link 必须具有相同 network、duplex 和有效容量。

规范化资源键如下：

```text
full duplex:
  (resource_group, "a_to_f")
  (resource_group, "f_to_a")

half duplex:
  (resource_group, "shared")
```

后续 executor 按资源键只创建一次 SimPy resource。相同键上的所有 block/links 竞争容量；不同键独立。`capacity` 表示并发槽位而非带宽倍数。

通信时间必须拆分为：

```text
queue_duration   = transfer_start - ready_at
service_duration = fixed_latency + serialization_time
total_duration   = queue_duration + service_duration
```

链路利用率只统计 service duration。第一版不在并发数变化时动态重算剩余传输时间；若容量大于 1，应将其解释为多个独立通道或已由实测有效带宽吸收并发影响。

### Decision 7: Keep AF and KV block namespaces separate

`af_block_size` 只存在于 AFConfig。现有 CLI `--block_size` 保留为 KV Cache 兼容入口，并在 `benchmark.py` 内使用 `kv_block_size` 局部命名传给请求与 BlockManager。AF 配置不得覆盖或进入 KV block 路径。

### Decision 8: AF uses offline operator tables and structured runtime lookup

现有 Roofline/LLMCompass 接口返回完整 Prefill/Decode iteration 时间，粒度和数据均不适合 `(AF block, phase, layer, A/F stage)` 查询。它们只保留给非 AF baseline 和回归，不作为 AF fallback；禁止把完整 iteration 时间按 layer/block 线性均分。

后续性能模型 change 将数据生成与模拟运行分成两层：离线层根据实测数据、workload 公式、Groq 硬件参数和少量可解释校准系数生成算子延迟表；TokenSim 运行时只按设备、算子类型、backend、数据版本和 shape 查询表格，不重新计算 roofline。每类算子同时生成运行时 `*_perf.parquet` 与可追溯的 `*_analysis.parquet`，两者使用相同主键；`generation_meta.yaml` 记录硬件参数、公式版本、单位、来源和行数，`shape_manifests/<experiment-id>.yaml` 定义待生成的实验 shape。

第一版 Groq 硬件 profile 以计算时延文档列出的公开值作为候选输入：FP16 188 TFLOPs、INT8 750 TOPS、片上 SRAM 230 MB、片上 SRAM 带宽 80 TB/s、芯片网络带宽 480 GB/s。operator-data change 在写入 `hardware.yaml` 前必须逐项保留原始来源并区分公开规格、项目假设和待确认参数；这些峰值只定义理论上界，不能被称为实测算子延迟。

运行时分表和查询主键固定为：

| 数据表 | 关键查询维度 | 允许的一维插值 |
|---|---|---|
| `gemm_perf.parquet` | `gemm_dtype, m, n, k` | 相同 `dtype/n/k` 下的 `m` |
| `context_attention_perf.parquet` | dtype、batch、`input_seq_len`、head/KV head、head dim、window | 其他维度相同的 `input_seq_len` |
| `generation_attention_perf.parquet` | dtype、batch、`context_len`、head/KV head、head dim、window | 其他维度相同的 `context_len` |
| `moe_perf.parquet` | dtype、phase、distribution、tokens、模型/专家/TP/EP 参数 | 其他维度相同的 `num_tokens` |
| `elementwise_perf.parquet` | op、dtype、tokens、hidden size | 由后续模型 change 明确，第一版不得跨 op/dtype 插值 |

查询顺序为：

```text
structured operator query
  -> exact key match
  -> controlled one-dimensional interpolation within table coverage
  -> explicit missing-key error
```

第一版禁止外推，也不允许在 TokenSim 内静默使用公式或默认时间补缺；缺失时输出完整主键，由离线生成器补表。查询结果至少返回 `latency_us`、`match_type`（`exact` 或 `interpolated`）和 `source_id`。`match_type` 描述如何命中，`source_id` 独立追溯数据依据，二者不得混为一个来源枚举。

算子边界遵循实际执行结构：A stage 聚合 QKV GEMM、Attention 核心、输出投影 GEMM 和必要的 RMSNorm/RoPE/Residual Add/data reorder；F stage 聚合 Router GEMM、MoE grouped/fused expert 计算和必要逐元素算子。Attention 核心不重复包含 QKV/输出投影；MoE 表不包含 Router、dispatch 或 combine；softmax 和 fused SwiGLU 只在其聚合算子中计时一次。dispatch、combine、AllReduce、A→F 和 F→A 都是独立通信事件，不进入计算表。`other_residual` 只能绑定具体 shape 与 `source_id`，禁止用与 shape 无关的 A/F 固定常数。

workload 口径以计算时延文档为准：MHA 可用 `h_q=h_kv=h`，GQA/MQA 必须使用实际 KV head 维度；Decode `context_len` 使用长度桶上界或 batch 最大 KV 长度，不取平均；MoE 满足 `sum(T_e)=T*k`，第一版以 `balanced` distribution 建表。离线生成器的初始估算可使用 `max(FLOPs/peak_compute, bytes/peak_bandwidth)`，但该公式只负责生成带来源的数据，不是 TokenSim 运行时 fallback。

参考边界：

- AIConfigurator（核对版本 `095f58a51c4ca8e61b66ec108d86f223f8d559ce`）：借鉴按设备/算子/backend/版本组织的分表 schema、shape 查询、覆盖检查、插值和元数据来源管理。
- InferSim（核对版本 `ae3f73d5030310aa056ac2e9c5b9c1912f91b0da`）：借鉴 FLOPs、内存搬运、MFU/profile 与通信量组合的轻量离线建模方式。
- 不直接依赖或嵌入两者的部署搜索、调度器和完整运行框架；实现需适配 TokenSim 的离散事件边界。

### Decision 9: Configuration is runtime-inert until executor support lands

本 change 完成后，合法 AF 配置可以解析、归一化和校验，但不能被静默当作非 AF 运行。`benchmark.py` 在创建/启动 SimPy 事件循环前检测 `af.enabled=true` 并抛出明确的阶段性 `ConfigurationError`。后续 executor change 替换此保护并按所选 phase 进入 AF 路径。

### Decision 10: Validate AF with a four-layer evidence chain

AF 的最终验收必须包含完整端到端测试，但不能只依赖端到端结果。端到端偏差无法单独判断来自调度依赖、资源竞争、计算模型、通信模型还是指标记账，因此后续 executor 和 operator-data changes 必须按以下四层验证，并保留每层可复现的输入、trace 和报告。当前配置 change 只交付验证矩阵与数据契约，不实现这些尚不存在的执行器、provider 或硬件采集工具。

**第一层：确定性小规模端到端 oracle。** 使用固定返回值的 stub compute/transfer providers，但从真实配置加载入口运行到最终 trace 和结果，验证机制而非物理精度。标准双 block 用例固定为：

```text
A compute = 2 ms
A -> F transfer = 1 ms
F compute = 3 ms
F -> A transfer = 1 ms
```

单 block 完成时间应为 7 ms；两个 block 完全串行应为 14 ms；在容量为 1、依赖和流水重叠正确时，两个 block 的 makespan 应精确为 10 ms。测试必须断言依赖顺序、计算/通信重叠、资源竞争、queue/service 时间、iteration barrier、每个请求恰好推进一次，以及 trace 区间和 makespan。验证矩阵至少包含：单 block 单层、单 block 多层、双 block 流水、A 慢/F 快、A 快/F 慢、全双工、半双工、多个 F worker、Prefill、Decode 和非 AF 回归。

除事件 trace 的精确 oracle 外，流水报告还需检查分析性隐藏条件。令 `T_A`、`T_F`、`T_C` 为单个 micro-batch 的 A、F 和单向通信时间，完全隐藏通信的候选条件为 `T_C<T_A`、`T_C<T_F` 且 `m>=ceil(2*(1+T_C/max(T_A,T_F)))`。该公式只用于解释和筛选实验点，3 或 4 个 micro-batch 不是固定配置，最终结论以离散事件时间线为准。

**第二层：计算模型保真度。** 优先使用可获得的同设备、同 shape 算子级或层级实测。数据按文档统一分级：A 为官方规格或论文算子/层级实测，可直接使用并记录原始设置；B 为公开端到端基准，只检查整体量级；C 为经公开结果校准的 roofline，用于离线补齐目标 shape；D 为项目初始假设，只能用于敏感性分析。用于校准的数据与未参与拟合的 holdout 必须隔离，并按 GEMM、Prefill/Decode Attention、MoE 和逐元素算子报告 coverage、平均/最大相对误差、P50/P90/P95、bias、瓶颈分类一致性和最差 shape。证据可信度顺序为：

```text
目标设备同 shape 实测
> 官方/论文算子或层级实测
> 公开端到端量级校验
> 公开结果校准公式
> 显式项目假设
```

**第三层：通信模型保真度。** 在目标链路上对不同 payload size、A->F/F->A、全/半双工、单传输/并发传输、本地/跨设备/跨节点、resource group 竞争以及内部 dispatch/combine 通信执行 microbenchmark。`service_duration` 与链路实测比较；`queue_duration` 与确定性事件 oracle 比较；总通信时间必须等于两者之和。没有目标硬件链路数据时，只能声明模型自洽，不能声明物理准确。

**第四层：公开与真实系统端到端校准。** 在模型、软件版本、workload 和拓扑匹配的条件下，将真实系统 trace 与 TokenSim 对齐，比较 A/F kernel 时间、transfer 时间、queue/wait 时间、TTFT、TPOT 和吞吐。BERT-base 在 Groq TSP 上 batch=1、序列长度 128、约 130 us 的论文结果可作口径匹配的模型级校验；Mixtral 8x7B 约 812 token/s 因芯片数、编译映射和服务并发未知，只能作弱范围检查。任何公开结果都必须记录模型版本、精度、batch/并发、输入输出长度、芯片数、主机通信和指标定义，未知项不得自行补齐。

所有硬件参数和时延必须通过 `source_id` 映射到 A/B/C/D 来源元数据，并与查询的 `match_type` 分开记录。缺少目标硬件数据时，公式结果必须标记为 C 或 D 及其适用范围，不得写成“准确”。后续还必须分别扰动算子延迟、有效带宽和固定延迟，报告 AF 结论稳定区间和发生变化的边界。AIConfigurator 和 InferSim 只提供 schema 与建模思路参考。Codex 可以自动化 oracle E2E、trace invariant、表格一致性、数据集切分、误差报告和敏感性分析，但不能生成物理 ground truth。

## Risks / Trade-offs

- [Risk] 两阶段共用一个 request-count block size 会导致 Prefill block 的 token 数差异较大 → provider 始终接收实际 token 数和 shape；后续可独立引入 token-budget/chunked Prefill。
- [Risk] 相同 resource group 的配置可能在多处重复 → 归一化时强制 network、duplex、capacity 完全一致并生成单一资源描述。
- [Risk] 有效 AF 配置暂时不能运行模拟 → 入口显式失败，避免生成错误但看似合理的 baseline 结果。
- [Risk] 新旧字段并存增加短期复杂度 → 下游统一读取 capabilities，role 仅作为兼容输入和既有非 AF 路径视图。
- [Risk] 两阶段使用不同 A group 需要 KV 状态迁移 → 后续 executor 复用现有阶段间 KV transfer，且与层内 AF link 分开统计。
- [Trade-off] 每阶段只允许一个 link 限制多池调度实验 → 换取第一版配置和执行的确定性，多 link 路由另立规格。
- [Trade-off] 固定容量槽位不模拟动态带宽共享 → 保留 queue/service 分项和实测有效带宽接口，后续按实验需要扩展。
- [Risk] 只看端到端总时间可能掩盖多个误差相互抵消 → 同时保留确定性机制 oracle、component holdout、链路 microbenchmark 和真实 trace 四层证据。
- [Risk] 尚无目标硬件数据时公式可能被误称为准确 → 强制使用 A/B/C/D 来源等级和 `source_id`，C/D 数据只能声明为校准估算或项目假设。
- [Risk] 运行时公式 fallback 会掩盖 shape 覆盖缺口 → 第一版禁止外推和静默 fallback，返回完整缺失主键并由离线生成器补表。
- [Risk] 算子聚合边界重叠会重复计时 → 分离 GEMM、Attention 核心、MoE expert、逐元素和通信表，并为 fused 操作规定唯一归属。

## Migration Plan

1. 增加 AF phase/capability 常量、`AFLinkConfig`、`AFConfig` 和严格字段校验，默认保持 AF 关闭。
2. 扩展 worker group/worker 配置并实现 `prefill`/`decode`/`hybrid` 的真实 legacy 映射。
3. 实现 phase-to-link、端点 capability、未选阶段完整能力及 network 引用校验。
4. 按 resource group 归一化物理通信资源键并校验共享定义一致性。
5. 在配置聚合出口导出新类型，明确 KV block 局部命名，并加入 AF 执行尚未落地时的运行保护。
6. 添加两阶段 AF、Decode-only AF、全/半双工与共享 resource group 样例和测试，再运行完整回归。

回滚时移除新 AF/link/capability 字段与运行保护即可；由于默认关闭且旧 role 输入仍保留，既有配置不需要迁移。

## Open Questions

- Groq 公开参数中哪些可作为最终设备配置、哪些仍需标记为项目假设，以及校准系数和表格覆盖范围，由后续 operator-data change 通过来源元数据确定。
- elementwise 表允许插值的唯一 shape 维度、非均匀 MoE distribution schema 和 ragged Attention schema 由后续版本化数据 change 确定。
- 并发传输的动态带宽共享是否需要事件中途重算，由后续通信实验根据数据决定。
- Prefill token-budget 或 chunked request 的 AF block 语义由后续 Prefill 优化 change 决定。
