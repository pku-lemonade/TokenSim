# operator-data-latency-provider Specification

## Purpose

为 TokenSim 的 AF 计算与通信提供基于版本化算子数据的可追溯 service time，使实测 kernel、校准公式、受控插值和算子组合能够在不混入排队逻辑的前提下驱动端到端仿真与精度验证。

## Requirements

### Requirement: Operator-data package is versioned and validated

系统 MUST 加载包含 `schema_version`、`dataset_version`、device/backend 标识、`hardware.yaml`、`generation_meta.yaml`、shape manifests、runtime perf 表和对应 analysis 表的 operator-data package。Perf 表 SHALL 使用微秒字段 `latency_us` 和可解析的 `source_id`；系统 MUST 在开始仿真前校验 schema、必填列、键类型、单位、有限非负时延、来源引用和 exact key 唯一性。

#### Scenario: Valid package is accepted

- **WHEN** 数据包的版本、文件、列、单位、键和来源引用均合法
- **THEN** 系统完成一次确定性加载并暴露其 dataset identity

#### Scenario: Duplicate exact key is rejected

- **WHEN** 同一 runtime perf 表存在两个完全相同的查询键
- **THEN** 系统在仿真开始前明确失败并报告表名与重复键
- **AND** 系统不得任取第一条记录

#### Scenario: Invalid latency or schema is rejected

- **WHEN** 数据包含未知 schema version、缺少必填列或 `latency_us` 为负数、NaN、无限值
- **THEN** 系统明确报告文件、字段和原因

### Requirement: Offline generation prioritizes measured kernels

离线生成器 SHALL 以可追溯实测 kernel 样本作为首选事实，并 MAY 使用由 FLOPs、内存搬运、硬件参数和 calibration 系数构成的公式生成缺失 shape 的候选值。每一行 MUST 标明测量或公式来源、设备/backend、生成方法和证据等级；峰值参数、公开估计或公式结果 MUST NOT 被标记为目标硬件实测值。

#### Scenario: Measured sample overrides analytical candidate

- **WHEN** 同一规范化 shape 同时存在合格实测样本和公式候选值
- **THEN** runtime perf 表采用实测结果
- **AND** analysis 与 generation metadata 保留候选值和选择依据

#### Scenario: Formula fills an unmeasured manifest shape

- **WHEN** shape manifest 要求的 shape 没有合格实测样本但满足已校准公式的适用域
- **THEN** 生成器输出带公式、校准版本和证据等级的估算记录
- **AND** 该记录不得声明为 measured

### Requirement: Runtime lookup is exact, bounded, and deterministic

运行时 SHALL 先执行 exact match；仅在 exact miss 时，才允许在所有离散键相同的记录之间进行确定性单维插值。第一版 MUST NOT 外推，也 MUST NOT 跨 device、backend、dtype、算子类别或其他离散键插值。无合法匹配时系统 MUST 报告完整规范化 key、表名和失败原因，且 MUST NOT 静默回退到 Roofline、LLMCompass、默认时延或完整 iteration 时延。

#### Scenario: Exact match is preferred

- **WHEN** exact 记录存在且同一插值轴附近也存在其他记录
- **THEN** 查询返回 exact 记录的 `latency_us`、`source_id` 和 `match_type=exact`

#### Scenario: Supported interior interpolation succeeds

- **WHEN** exact 记录不存在且目标轴值严格位于两个合法邻点之间
- **THEN** 查询只使用这两个确定性邻点插值
- **AND** 返回 `match_type=interpolated` 及两个来源和权重

#### Scenario: Out-of-range query fails without fallback

- **WHEN** 目标轴值落在已知数据范围之外或缺少合法邻点
- **THEN** 查询明确失败并报告完整 key
- **AND** 系统不执行外推或旧模型 fallback

### Requirement: Interpolation axes are operator-specific

系统 MUST 按算子类别限制插值轴：GEMM 仅可在相同 weight/activation dtype、`n/k` 及其他离散键下沿 `m` 插值；Prefill Attention 仅沿 `input_seq_len`；Decode Attention 仅沿单个 request 的 `context_len`；Routed 与 Shared Expert 仅沿各自 `num_tokens`。Weighted combine、residual、encode/decode 与其他 elementwise 第一版 MUST NOT 跨 operation 或 dtype 插值，且只有 package schema 明确声明合法单维轴时才可插值。

#### Scenario: GEMM differs on two dimensions

- **WHEN** GEMM query 需要同时改变 `m` 与 `n` 才能连接两个数据点
- **THEN** 系统拒绝插值并报告不一致的离散键

#### Scenario: Decode attention interpolates context length only

- **WHEN** 两个邻点除 `context_len` 外所有规范化键相同
- **THEN** Decode Attention query 可在其内部区间插值

#### Scenario: Routed and shared workloads do not mix

- **WHEN** Routed Expert 与 Shared Expert records 具有相同 token 数但不同 operation identity
- **THEN** provider 不得跨两个类别插值
- **AND** 各自只沿自身 `num_tokens` 轴寻找合法邻点

### Requirement: A and F service times use non-overlapping operator composition

Compute provider SHALL 接收 A Attention、A Router、F Dense FFN、F Routed Expert、F Shared Expert、A weighted combine、A residual、必要 layout operation 及传输 dtype 不同时的 encode/decode rank-local queries，并 MUST 使用执行计划给出的 layer type、original token/assignment workload、local tensor shape、effective weight/activation dtype、设备与算子元数据组合 service time。每次 query MUST 返回可重建总和的有序 breakdown。Router 仅覆盖 A 域 gating/Top-K；Routed Expert 仅覆盖其 assignments；Shared Expert 仅覆盖每 token 一次的 shared work；weighted combine 与 residual 仅在 A 侧计时；encode/decode 仅覆盖对应 tensor 元素的一次格式转换。所有 composition MUST 互不重叠，transfer 与 collective MUST NOT 进入 compute service time。

#### Scenario: TP-local attention composition reconstructs service time

- **WHEN** A rank 查询的所有必要 Attention 原子算子按精确 rank-local shape 成功匹配
- **THEN** 返回 service time 等于有序 breakdown 各项换算后时长之和
- **AND** query 与 breakdown 均保留 logical rank、actual heads 与 local shape

#### Scenario: Attention composition reconstructs service time

- **WHEN** A rank 的 Attention query 成功解析
- **THEN** Q/K/V、Attention core 与 output projection 各自恰好出现一次
- **AND** residual 不被提前包含在 Attention composition 中

#### Scenario: Dense layer uses Dense FFN composition

- **WHEN** execution query 标记当前 layer 为 Dense
- **THEN** provider 组合 A Attention、F Dense gate/up/SwiGLU/down 与 A residual/layout work
- **AND** 不查询 Router、Routed Expert、Shared Expert 或 weighted combine

#### Scenario: MoE layer separates routed shared and combine work

- **WHEN** `T` 个 tokens、`top_k=K` 执行含 Shared Expert 的 MoE layer
- **THEN** Routed composition 使用 `T*K` assignments，Shared composition 使用 `T` tokens，A combine/residual 使用 `T` tokens
- **AND** 任一 residual 不出现在 F Expert breakdown 中

#### Scenario: Expert service uses routed local tokens

- **WHEN** 两个 F Expert ranks 分别收到不同数量的 routed assignments
- **THEN** provider 分别使用各自逐 Expert local workload 查询 service time
- **AND** provider 不以总 assignment 数平均值替代真实 workload

#### Scenario: Router is estimated as A-domain work

- **WHEN** 系统查询 MoE Router/Top-K service time
- **THEN** query 标明 A domain、A owner、执行 rank、original token count 与 Expert count
- **AND** Router breakdown 不出现在 F composition 中

#### Scenario: Transfer conversion is timed separately

- **WHEN** A-to-F 使用 FP8 而 activation 使用 BF16
- **THEN** provider 分别查询发送端 quantize 与接收端 dequantize composition
- **AND** 两项转换不被隐藏在 transfer bandwidth service time 中，也不被重复计入 Expert compute

#### Scenario: MoE composition excludes communication

- **WHEN** Dense 或 MoE layer 的全部 compute queries 成功组合
- **THEN** breakdown 不包含 AF transfer 或 collective
- **AND** 所有通信由独立 typed communication query 估算

#### Scenario: Required component is missing

- **WHEN**任一必要 composition component 在版本化package中无 exact match或合法插值
- **THEN** query 明确失败并列出 layer type、operation、rank、shape/workload 与完整 key
- **AND** 系统不得在runtime调用analytical formula、返回部分总和或使用完整未分片算子的fallback

### Requirement: Communication provider returns service time independently of contention

Transfer provider SHALL 接收planner已确定的typed endpoint-leg query或collective query，其中包含logical stage id、operation、layout、participants、physical source/target、唯一tensor items、effective transfer dtype、network/resource group/lane identity与topology。AF endpoint-leg query MUST 携带实际wire payload bytes；collective query MUST 携带每rank logical input/output bytes，provider再依据operation、participants与topology计算一次wire-volume和service details。Operation MUST 区分A-to-F、F-to-A、`all_reduce`、`all_gather`和`reduce_scatter`；A-to-F MAY 在一个logical stage中承担routed dispatch和Shared input transport，F-to-A MUST 表示partial return而不是mathematical weighted combine。Provider MUST 为每条executable leg或collective返回独立service time与payload breakdown，且 MUST NOT 修改placement、owners、participants、资源、依赖或queue state。同一logical AF stage的完成时间由executor取必要legs的最晚完成时间。

#### Scenario: Directional dtype controls transfer bytes

- **WHEN**相同元素数分别通过 FP8 A-to-F 与 BF16 F-to-A 传输
- **THEN** provider query 分别携带 1 byte 与 2 bytes 每元素的 payload breakdown
- **AND** provider 不用 compute dtype 覆盖 transfer dtype

#### Scenario: AF transfer uses planned payload once

- **WHEN** planner 在一个 A-to-F 或 F-to-A logical stage 下提交多条 unique endpoint-leg queries
- **THEN** provider 为每条 leg仅估算其 endpoint payload一次，并保留共同 stage id
- **AND** 不附加第二个 dispatch/combine stage 或反向 transport

#### Scenario: Transfer service uses payload and link parameters

- **WHEN** typed communication query 匹配固定延迟、有效带宽与版本化链路参数
- **THEN** provider 根据 endpoint payload bytes 返回统一单位的 service time 与来源
- **AND** provider 不返回或修改 queue duration

#### Scenario: TP collective preserves participants

- **WHEN** 调用方提交一个明确 `all_gather`、participants、fabric 与 resource lane 的 query
- **THEN** provider 使用这些精确字段匹配通信模型
- **AND** 不把它归一化为无法区分算法的通用 `tp_collective`

#### Scenario: Collective wire volume is derived once

- **WHEN** collective query携带每rank logical tensor bytes、operation、participants和topology
- **THEN** provider只在自身内部应用一次对应collective算法的wire-volume规则
- **AND** result details同时保留logical bytes与derived wire bytes以便审计

#### Scenario: EP communication uses routed payload

- **WHEN** planner 以 A-to-F 承载 routed dispatch、以 F-to-A 承载 routed partial return
- **THEN** provider 按 assignments、owners、unique endpoint legs 与各方向 dtype 计算两个 logical AF stages 下各 legs 的实际 payload
- **AND** 不再要求或生成搬运相同数据的独立 `ep_dispatch`、`ep_combine` query

#### Scenario: Identical service queries experience different queue time

- **WHEN** 两次相同 typed queries 竞争同一容量为一的 resource
- **THEN** provider 返回相同 service time
- **AND** 后一次的非零 queue time仅由 executor 产生

### Requirement: Provider results preserve provenance and units

每个成功结果 MUST 包含秒为单位的有限非负 service duration、`source_id`、`dataset_version`、`match_type` 和结构化 details；插值结果 MUST 包含邻点来源与权重，组合结果 MUST 包含每个 component 的规范化 key、原始 `latency_us`、effective dtype role/byte width、匹配类型和来源。微秒到秒的换算 MUST 恰好执行一次，元数据 MUST NOT 改变资源选择或事件顺序。

#### Scenario: Microseconds convert once at provider boundary

- **WHEN** exact perf row 的 `latency_us` 为 2500
- **THEN** provider 返回 service duration 0.0025 秒
- **AND** breakdown 仍保留原始 2500 微秒值

#### Scenario: Effective dtype remains auditable

- **WHEN** 一个 compute 或 transfer result 使用 FP8
- **THEN** details 标明具体 dtype role、canonical name 与每元素 1 byte
- **AND** 用户可从 element count 与 byte width 重建 payload 或package中公式来源记录的bytes

#### Scenario: Row order does not alter result

- **WHEN** 合法 package 中的 table rows 以不同顺序加载
- **THEN** 相同 query 得到相同 service duration、匹配来源与 breakdown 顺序

### Requirement: Calibration and validation remain evidence-separated

离线流程 MUST 以可复现 manifest 将 calibration 与 holdout 样本隔离，并 SHALL 按算子类别报告 coverage、mean/max relative error、P50/P90/P95、bias、bottleneck consistency 和 worst shapes。Holdout 样本 MUST NOT 参与拟合或插值参数选择；缺少目标硬件或真实系统 ground truth 时，报告 MUST 将结论限定为数据一致性或模型自洽性。

#### Scenario: Holdout data is excluded from fitting

- **WHEN** 同一实验生成 calibration 与 holdout 报告
- **THEN** generation metadata 可证明 holdout 样本未参与校准
- **AND** 误差指标仅在预测完成后对 holdout 计算

#### Scenario: Ground truth is unavailable

- **WHEN** 数据集只有公开估计或公式生成值而没有目标设备实测
- **THEN** 报告明确标注证据限制
- **AND** 报告不得宣称目标硬件物理准确

### Requirement: End-to-end oracle uses a generated data package

系统 MUST 提供从 shape manifest 与源数据生成 package、加载正式 providers、运行 Decode AF iteration 到最终 trace/result 的端到端验证。验证 MUST 覆盖 Dense/MoE layer selection、逐 layer assignments、Routed/Shared Expert、F-to-A ownership return、A weighted combine/residual、directional dtypes、collective operation/participants/fabric、exact/interpolated/missing 查询、资源竞争和 makespan，并保留正常 Prefill 与 AF-disabled 回归。

#### Scenario: Generated package drives complete AF execution

- **WHEN** 端到端 fixture 运行同时包含 Dense 与 MoE layers 的多-block Decode iteration
- **THEN** makespan 可由 trace 中全部 compute/communication service、queue 和 overlap 重建
- **AND** 每个 service event 可追溯到版本化package record（含离线公式生成record）或这些records之间的合法插值

#### Scenario: Generated package drives AF execution

- **WHEN** 端到端 fixture 从源样本生成合法 package 并运行 Decode-only AF 多 block iteration
- **THEN** makespan 可由 Dense/MoE compute、AF transfer、必要 collective、queue 与 overlap 重建
- **AND** 不要求独立的 EP dispatch/combine communication records

#### Scenario: Missing internal communication record fails explicitly

- **WHEN** F-to-A partial return、A weighted combine、Shared Expert 或必要 collective 无可用 estimate
- **THEN** iteration 在对应 node 明确失败并报告完整 query context
- **AND** 不得跳过 operation 或使用旧完整 iteration backend

#### Scenario: Normal Prefill remains on the complete path

- **WHEN** 配置仅为 Decode 启用 AF
- **THEN** Prefill 不查询 AF operator-data DAG
- **AND** Decode provider 扩展不改变既有 Prefill request progression

#### Scenario: Existing execution baselines remain stable

- **WHEN** 回归套件以 `af.enabled=false` 运行
- **THEN** 不创建新 assignment、ownership 或 AF provider queries
- **AND** 非 AF timing 行为保持兼容

### Requirement: Effective dtype roles are explicit in operator keys

Operator-data schema 与离线analytical generator MUST 区分 weight、activation、KV Cache、A-to-F 和 F-to-A dtype roles。GEMM 与 Expert keys MUST 保留实际 weight/activation dtype，Attention keys MUST 额外保留 KV Cache dtype，communication keys MUST 使用对应方向 dtype。所有路径 MUST 共享同一规范 dtype registry 与 byte size。离线公式生成的结果 MUST 先写入版本化package；runtime lookup仍只允许exact match或主规范允许的单维插值。

#### Scenario: Weight and activation dtype differ

- **WHEN** 权重使用 FP8 而 activation 使用 BF16
- **THEN** GEMM/Expert query key 同时保留两个 dtype roles
- **AND** 离线公式生成record中的bytes分别按各自元素大小计算

#### Scenario: Unknown dtype fails consistently

- **WHEN** 任一 role 使用 registry 未知 dtype
- **THEN** package validation 在模拟前失败并标明 role 与 dtype
- **AND** executor 不使用默认 2 bytes 继续运行

### Requirement: Offline analytical generation preserves supported model shapes

离线analytical generator MUST 对支持的 MHA/GQA/MQA 使用实际 query-head、KV-head 与 head dimensions计算 Q/K/V/O 权重、FLOPs、activation 和 KV bytes；对 SwiGLU Dense/Routed/Shared FFN MUST 使用 gate、up、down 三个矩阵；对 Decode Attention MUST 按每个 request 的有效 context length 组合工作量，并在 `window_size > 0` 时使用受窗口限制的长度。所有实际 TP-sharded dimensions MUST 由已校验的精确 local shape 给出，不得在generator内使用 `ceil` 修复非法配置。本版本 MUST NOT 将这些公式用于 MLA/MFA。生成结果 MUST 携带公式、校准版本与证据等级并写入版本化package后才能被runtime查询。

#### Scenario: SwiGLU includes gate up and down

- **WHEN** hidden size 为 `H`、local intermediate size 为 `I`、token workload 为 `T`
- **THEN** GEMM FLOPs按三个投影计算并包含对应 activation bytes
- **AND** weight bytes 与 runtime ledger 使用相同三矩阵 shape

#### Scenario: GQA uses fewer KV heads than query heads

- **WHEN** `num_kv_heads < num_attention_heads`
- **THEN** Q/K/V/O package records和离线公式bytes使用实际不同dimensions
- **AND** 不按 MHA 的相同 Q/K/V width 计算

#### Scenario: Decode window caps each request independently

- **WHEN** 同一 Decode block 的 requests context lengths 不同且 `window_size > 0`
- **THEN** 每个 request 的 Attention work使用自身受窗口限制的有效长度后再聚合
- **AND** 不使用 batch 最大长度替代全部 requests

#### Scenario: Provider rejects implicit uneven TP shards

- **WHEN** query 的 local shape 只能由非整除 global dimension 向上取整得到
- **THEN** provider 拒绝该 query并报告 shape inconsistency
- **AND** 不产生超过 global dimension 的总 rank workload

### Requirement: Combine and residual workloads use original token ownership

Weighted combine provider MUST 按每个 A tensor owner 实际收到的 routed partials 计算 routing-weight multiply 与 `top_k` reduction，并按每个原始 token 至多加入一次 Shared Expert partial。Residual provider MUST 只按该 owner 的原始 token tensor shard 计算一次。两者的输出 token 数 MUST 保持为原始 `T`，不得以 `T*K` 进入下一层。

#### Scenario: Top K two collapses to one output per token

- **WHEN** 一个 A owner 为 `T` 个 tokens 收到 `2T` 个 routed partial shards
- **THEN** weighted combine query 保留 `2T` 条输入 edge 与 `T` 个 output tokens
- **AND** residual query 的 token count 为 `T`

#### Scenario: Shared output is summed once

- **WHEN** MoE layer 同时有 Routed 与 Shared Expert branches
- **THEN** combine composition 对每个 token 加入一份 Shared output shard
- **AND** routing weight 只作用于 Routed partials，Shared output 不被乘以 `top_k`
