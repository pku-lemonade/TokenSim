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

Compute provider SHALL 从原子算子查询结果组合单 block、单层的 A 或 F service time，并 MUST 返回可重建总和的有序 breakdown。A 侧 SHALL 覆盖 QKV GEMM、Attention core、output projection GEMM 及必要的 RMSNorm、RoPE、Residual Add、data reorder；F 侧 SHALL 覆盖 Router/Gating GEMM、grouped/fused MoE expert 及必要 elementwise。Attention core MUST NOT 重复包含 QKV 或 output projection；MoE expert MUST NOT 重复包含 Router、dispatch 或 combine；softmax 与 SwiGLU 在 fused kernel 与独立算子之间 MUST 恰好计时一次。

#### Scenario: Attention composition reconstructs service time

- **WHEN** A 查询的所有必要原子算子均成功匹配
- **THEN** 返回的 service time 等于 breakdown 各项换算后时长之和
- **AND** QKV 与 output projection 各出现一次且不包含在 Attention core 中

#### Scenario: MoE composition excludes communication

- **WHEN** F 查询组合 Router、fused expert 和必要 elementwise
- **THEN** dispatch/combine 与 A-to-F/F-to-A transfer 不计入 F compute service time

#### Scenario: Required component is missing

- **WHEN** A 或 F composition 的任一必要原子算子无法 exact match 或合法插值
- **THEN** 整个 compute 查询明确失败并列出缺失 component 与完整 key
- **AND** 系统不得返回部分总和

### Requirement: Communication provider returns service time independently of contention

Transfer provider SHALL 根据 direction、source/target device、network、payload bytes 及可追溯链路参数返回通信 service time，并 SHALL 采用 `fixed_latency + payload_bytes / effective_bandwidth` 或数据包中明确版本化的等价实测模型。Provider MUST NOT 模拟排队、容量竞争、重传或随机拥塞；这些等待仍由 executor 的共享资源产生。

#### Scenario: Transfer service uses payload and link parameters

- **WHEN** A-to-F 查询匹配到固定延迟和有效带宽
- **THEN** 返回值包含 payload bytes、链路 source 与按统一单位计算的 service time

#### Scenario: Identical service queries experience different queue time

- **WHEN** 两次相同 transfer 查询竞争同一容量为一的链路资源
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

系统 MUST 提供从 shape manifest 与源数据生成数据包、加载正式 providers、运行 AF iteration 到最终 trace/result 的端到端验证。验证 SHALL 覆盖 exact、interpolated、missing/out-of-range、非法 schema、单位换算、A/F 无重复组合、direction/payload、provider metadata 和通信竞争，并 MUST 保留既有 AF stub oracle 与 AF-disabled 回归。

#### Scenario: Generated package drives AF execution

- **WHEN** 端到端 fixture 从源样本生成合法数据包并运行多 block AF iteration
- **THEN** 最终 makespan 可由 trace 中 compute/transfer service、queue 和重叠区间重建
- **AND** 每个 service 事件可追溯到数据包记录或合法插值

#### Scenario: Existing execution baselines remain stable

- **WHEN** 回归套件运行既有 7/10 ms stub provider 场景和 AF-disabled 场景
- **THEN** 其时序与非 AF backend 行为不因 operator-data provider 引入而改变
