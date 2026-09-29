## Purpose

使 DeepSeek-V3/MLA 成为 TokenSim 标准模型执行能力，通过统一 serving、调度、KV、AF、operator-table 和结果路径运行，而不依赖脚本中的专用 executor。

## ADDED Requirements

### Requirement: DeepSeek-V3 runs through the standard TokenSim entry path

系统 SHALL 允许用户通过标准模型与集群配置选择 DeepSeek-V3，并通过与其他模型相同的 workload、scheduler、KV cache、LLMEngine、AF executor 和结果导出路径运行。标准路径 MUST NOT 导入 `scripts` 下的 runtime factory 或 V3 专用 executor。

#### Scenario: Standard benchmark runs V3 decode

- **WHEN** 用户以标准 benchmark 入口选择 V3 ModelSpec、合法 AF cluster 和 decode workload
- **THEN** 系统完成请求调度、逐 token decode、KV 记账和结果导出
- **AND** 执行不依赖 `scripts.deepseek_v3_runtime.V3Executor`

#### Scenario: Script and standard entry share core semantics

- **WHEN** 兼容脚本与标准入口使用相同模型、集群、请求和随机种子
- **THEN** 两者调用同一个核心 model executor
- **AND** 请求完成时间、通信字节、operator provenance 和 makespan 一致

### Requirement: Whole-model operations are core execution nodes

系统 MUST 在核心模型执行生命周期中表达 embedding、transformer layers、final norm、LM head、必要 boundary collective 和 token selection。Boundary operations MUST 使用与层内操作相同的 provider、physical resources、通信语义与 provenance，不得由 V3 脚本直接调用 resolver 或自行构造第二套 trace。

#### Scenario: Complete decode contains model boundaries

- **WHEN** 一个 V3 decode step 成功执行
- **THEN** 核心执行结果包含 embedding、61 层、final norm、LM head 和 token selection
- **AND** TP>1 时需要的 boundary collective 由通用 runtime plan产生

#### Scenario: Boundary work contends with layer work

- **WHEN** boundary operator 与另一 ready block 的 layer operator 映射到相同 physical worker
- **THEN** 两者按同一 resource capacity 排队
- **AND** 不存在脚本级旁路执行时间

### Requirement: Model metadata selects MLA and Dense/MoE layer semantics

通用执行器 SHALL 根据 ModelSpec 自动选择 MLA Attention、Dense FFN 或 MoE Router/Routed/Shared Expert DAG。V3 的前三层 MUST 为 Dense，后 58 层 MUST 为 MoE；MLA KV cache MUST 使用 latent cache width。执行器 MUST NOT 通过模型名称硬编码 V3 层语义。

#### Scenario: V3 mixed layers execute correctly

- **WHEN** V3 完成一个 decode step
- **THEN** layers 0-2 不包含 Router/Expert work
- **AND** layers 3-60 包含配置声明的 Routed 与 Shared Expert work

#### Scenario: Another MLA model uses the same core path

- **WHEN** 非 V3 模型声明合法 MLA dimensions 和不同层结构
- **THEN** 通用执行器依据其 ModelSpec 编译执行计划
- **AND** 不要求新增模型专用 executor

### Requirement: Unsupported V3 features remain explicit

系统 MUST 明确区分完整 decode、受限 prefill、checkpoint execution 和 MTP。尚未实现的能力 MUST 在仿真前失败，不得把 decode-only 或 analytical 结果标记为完整 V3 serving/硬件实测。

#### Scenario: Unsupported mode fails before execution

- **WHEN** 用户要求未实现的 MTP、checkpoint inference 或未支持的 prefill 形式
- **THEN** 标准入口在创建事件前报告具体不支持项
- **AND** 不输出伪装为完整 serving 的成功结果

### Requirement: Legacy regression coverage migrates to the current backend

系统 SHALL 将三个遗留 MLA、完整 decode 与 operator batching 测试模块迁入当前
operator-table API，保留对应的公式、latent KV、TP 权重、混合层、路由、边界依赖、
分组 workload、exact lookup 和 missing-shape 行为覆盖。系统 MUST NOT 通过 skip
这些模块或恢复无实现的兼容桩宣称全量回归通过。

#### Scenario: Current calibration replaces retired shape rules explicitly

- **WHEN** 测试通过当前 package 的 table/default calibration 定价
- **THEN** 校准只作用于 analytical fallback，实测值不重复加固定开销
- **AND** 非法 multiplier/offset 显式失败
- **AND** 文档明确旧 shape-scoped CalibrationRule 已退役，按表校准不具有同等 shape 选择能力


### Requirement: Synthetic routing is independent of AF partitioning

相同 seed、phase、layer、request 与 token identity 的 synthetic expert 选择 MUST 不受 AF block 大小、batch 成员、iteration 编号及调用顺序影响。Top-k MUST 无重复且权重和为 1；owner 与执行标识仍按本轮 binding 填入。

#### Scenario: Repartitioning preserves token routes

- **WHEN** 相同 token 集合以不同 block 大小、顺序或拆分调用生成 uniform 或 hotspot routing
- **THEN** 每个 token 的 expert 和权重一致
- **AND** 不因重新分块改变 routed workload
