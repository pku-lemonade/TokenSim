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

系统 MUST 按算子类别限制插值轴：GEMM 仅可在相同 `dtype/n/k` 及其他离散键下沿 `m` 插值；Prefill Attention 仅沿 `input_seq_len`；Decode Attention 仅沿 `context_len`；MoE 仅沿 `num_tokens`。Elementwise 第一版 MUST NOT 跨 op 或 dtype 插值，且只有数据包 schema 明确声明其合法单维轴时才可插值。

#### Scenario: GEMM differs on two dimensions

- **WHEN** GEMM 查询需要同时改变 `m` 与 `n` 才能连接两个数据点
- **THEN** 系统拒绝插值并报告不一致的离散键

#### Scenario: Decode attention interpolates context length only

- **WHEN** 两个邻点除 `context_len` 外所有规范化键相同
- **THEN** Decode Attention 查询可在其内部区间插值

### Requirement: A and F service times use non-overlapping operator composition

Compute provider SHALL 分别接收 A rank compute、A 域 Router/Top-K 和 F expert rank compute 查询，并 MUST 使用执行计划给出的 rank-local tensor shape、local token workload、dtype、设备与算子元数据组合对应 service time。每次查询 MUST 返回可重建总和的有序 breakdown。A rank composition SHALL 覆盖该 TP rank 的 QKV、Attention core、output projection 及必要 elementwise；Router composition SHALL 只覆盖 A 域 Router/Gating、Top-K 与 dispatch 前必要 elementwise；Expert composition SHALL 只覆盖目标 F rank 的 grouped/fused experts 与必要 elementwise。三类 composition MUST 互不重叠，任何逻辑 operation MUST 恰好计时一次；dispatch、combine、collective 与 A-to-F/F-to-A MUST NOT 进入 compute service time。

#### Scenario: TP-local attention composition reconstructs service time

- **WHEN** A rank 查询的所有必要原子算子按 rank-local shape 成功匹配
- **THEN** 返回的 service time 等于 breakdown 各项换算后时长之和
- **AND** query 和 breakdown 均保留 logical rank 与 local shape

#### Scenario: Attention composition reconstructs service time

- **WHEN** A rank 查询的全部 Attention 原子算子按 rank-local shape 成功匹配
- **THEN** breakdown 中 QKV、Attention core 与 output projection 各自恰好出现一次
- **AND** 返回的 service time 等于这些本地 component 与必要 elementwise 的换算后总和

#### Scenario: Router is estimated as A-domain work

- **WHEN** 系统查询一个 AF layer 的 Router/Top-K service time
- **THEN** query 标明 A execution domain、选定 DP replica 与执行 ranks
- **AND** Router breakdown 不出现在任何 F expert composition 中

#### Scenario: Expert service uses routed local tokens

- **WHEN** 两个 F expert ranks 分别收到不同数量的 routed tokens
- **THEN** provider 分别使用各自 local token workload 查询 service time
- **AND** provider 不以总 token 数平均值替代实际 workload

#### Scenario: MoE composition excludes communication

- **WHEN** Router 与 Expert 查询分别组合其本地计算 component
- **THEN** 两个 breakdown 互不重复，且不包含 TP collective、EP dispatch、EP combine 或 AF transfer
- **AND** 所有通信均由独立 typed communication query 估算

#### Scenario: Required component is missing

- **WHEN** A rank、Router 或 Expert composition 的任一必要原子算子无法 exact match 或合法插值
- **THEN** 对应查询明确失败并列出 stage、rank、local shape/workload、缺失 component 与完整 key
- **AND** 系统不得返回部分总和或完整未分片算子的 fallback

### Requirement: Communication provider returns service time independently of contention

Transfer provider SHALL 接收已确定 operation 和参与者的 typed communication query，至少包含 source/target execution domain、logical participant ranks、physical endpoints、network/resource identity、payload bytes 以及适用的 group 或 topology 元数据。Operation MUST 区分 A-to-F、F-to-A、TP collective、EP dispatch 和 EP combine。Provider SHALL 根据 query 与版本化链路参数返回单次通信的 service time，MUST NOT 选择 DP replica、推导 expert placement、修改参与 ranks、生成额外 transfer 或模拟排队与容量竞争；这些执行语义和等待 MUST 由调用方确定。

#### Scenario: AF transfer uses planned payload once

- **WHEN** 调用方提交一次具有确定 endpoints 和 payload 的 A-to-F 或 F-to-A query
- **THEN** provider 返回该单次 transfer 的 service time 与 provenance
- **AND** provider 不附加 dispatch、combine 或反向 transfer

#### Scenario: Transfer service uses payload and link parameters

- **WHEN** typed communication query 匹配到固定延迟、有效带宽和版本化链路参数
- **THEN** provider 根据 payload bytes 返回按统一单位计算的 service time 与链路来源
- **AND** provider 不返回或修改 queue duration

#### Scenario: TP collective preserves participants

- **WHEN** 调用方提交一个包含明确 TP group 和 participant ranks 的 collective query
- **THEN** provider 使用这些参与者和 payload 匹配通信模型
- **AND** provider 不扩大或缩小 participant 集合

#### Scenario: EP communication uses routed payload

- **WHEN** 调用方分别提交 EP dispatch 和 EP combine query
- **THEN** provider 按各自 operation、expert participants 与实际 payload 返回独立 service time
- **AND** 两次查询不会被折叠为 A/F 域间 transfer

#### Scenario: Identical service queries experience different queue time

- **WHEN** 两次相同 typed communication query 竞争同一容量为一的资源
- **THEN** provider 对两次查询返回相同 service time
- **AND** 后一次的非零 queue time 仅由 executor 产生

### Requirement: Provider results preserve provenance and units

每个成功结果 MUST 包含秒为单位的有限非负 service duration、`source_id`、`dataset_version`、`match_type` 和结构化 details；插值结果 MUST 包含邻点来源与权重，组合结果 MUST 包含各 component 的规范化 key、原始 `latency_us`、匹配类型和来源。微秒到秒的换算 MUST 恰好执行一次，元数据 MUST NOT 改变资源选择或事件顺序。

#### Scenario: Microseconds convert once at provider boundary

- **WHEN** exact perf 行的 `latency_us` 为 2500
- **THEN** provider 返回的 service duration 为 0.0025 秒
- **AND** breakdown 仍保留原始 2500 微秒值

#### Scenario: Row order does not alter result

- **WHEN** 合法数据包中的表行以不同顺序加载
- **THEN** 相同查询得到相同 service duration、匹配来源和 breakdown 顺序

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

系统 MUST 提供从 shape manifest 与源数据生成数据包、加载正式 providers、运行带独立 domain topology 的 AF iteration 到最终 trace/result 的端到端验证。验证 SHALL 覆盖 exact、interpolated、missing/out-of-range、非法 schema、单位换算、A/F 无重复组合，以及 `af_transfer`、dispatch、combine、必要 collective 的 typed query、provider metadata、资源竞争和 makespan；验证 MUST 保留 AF-disabled 回归。

#### Scenario: Generated package drives complete AF execution

- **WHEN** 端到端 fixture 从源样本生成合法数据包并运行包含 dispatch、combine 和计划 collective 的多 block AF iteration
- **THEN** 最终 makespan 可由 trace 中全部 compute/communication service、queue 和重叠区间重建
- **AND** 每个 service 事件可追溯到数据包记录或合法插值

#### Scenario: Generated package drives AF execution

- **WHEN** 端到端 fixture 从源样本生成合法数据包并运行多 block AF iteration
- **THEN** 最终 makespan 可由 trace 中 compute、AF transfer、dispatch、combine、collective 的 service、queue 和重叠区间重建
- **AND** 每个 service 事件可追溯到数据包记录或合法插值

#### Scenario: Missing internal communication record fails explicitly

- **WHEN** dispatch、combine 或必要 collective 的 typed query 无 exact 或合法插值结果
- **THEN** iteration 在对应事件处明确失败并报告完整 query context
- **AND** 系统不得跳过该通信或使用旧完整 iteration backend

#### Scenario: Existing execution baselines remain stable

- **WHEN** 回归套件运行更新后的确定性 AF oracle 和 AF-disabled 场景
- **THEN** AF trace 包含规划要求的内部通信
- **AND** 非 AF backend 行为不因 typed communication query 扩展而改变
