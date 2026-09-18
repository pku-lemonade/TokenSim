## MODIFIED Requirements

### Requirement: Execution domains separate topology from physical placement

系统 MUST 将 Prefill、Decode A 和 Decode F 建模为独立 execution domains，并分别保存 logical instances、domain-local rank topology 与 rank-to-device mapping。每个 domain MUST 独立校验其 TP、PP、DP 以及适用的 EP 关系和 mapping 完整性；系统 MUST NOT 将一套 global rank 顺序隐式复用于全部 domains。参与本变更 Decode AF 的 A 与 F domain 均使用 `PP=1`，因此 A 与 F 的基础 world size 分别为各自的 `TP * DP`。同一 domain 内每个基础 rank MUST 恰好映射一个 physical worker；F rank-reuse EP identity MUST 叠加在这些基础 ranks 上，不得再乘出额外 workers。

#### Scenario: P, A, and F use different topologies

- **WHEN** Prefill、Decode A 和 Decode F domains 分别声明不同的 topology 和合法 device mapping
- **THEN** 系统逐 domain 接受并校验这些配置
- **AND** 任一 domain 的 rank 编号都只在该 domain 内解释

#### Scenario: F rank reuse does not allocate extra workers

- **WHEN** F domain 配置 `TP=2`、`DP=2` 并使用 rank-reuse EP
- **THEN** F 基础 world 和 physical runtime state 数均为 4
- **AND** 派生 EP identity 不创建第五个 state，也不要求 `TP * EP * DP` 个 workers

#### Scenario: Derived EP identity resolves to one base rank

- **WHEN** 系统编译 rank-reuse F domain
- **THEN** 每个 EP identity 唯一解析到一个已有 TP/DP rank 和 physical worker
- **AND** 一个 physical worker 只记账一次容量与资源竞争

#### Scenario: One physical device backs multiple logical ranks

- **WHEN** 一个 mapping 将同一 execution domain 内的多个基础 rank identities 映射到同一 physical worker
- **THEN** 系统在模拟启动前拒绝配置
- **AND** 错误列出冲突的 domain、rank 与 physical worker

#### Scenario: Domain mapping is incomplete

- **WHEN** 一个显式多 rank domain 缺少 rank、device 或并行组 mapping
- **THEN** 系统在模拟前拒绝配置并标明 domain 与缺失 mapping
- **AND** 系统不得按 worker group 顺序猜测 global ranks

#### Scenario: Legacy single-device configuration remains valid

- **WHEN** AF 被关闭或合法单设备 AF 配置未显式声明新 domain topology
- **THEN** 系统生成无歧义的单实例 domain view
- **AND** 既有单卡执行行为保持不变

### Requirement: AF parallel topology must be executable

系统 MUST 在仿真开始前将每个参与 AF 的 execution domain 解析为完整的 DP replica、TP group、适用的 rank-reuse EP group、logical rank、physical worker、tensor layout 与必要 collective plan。每个并行参数 MUST 对应可执行的分片、复制、路由或通信语义。参与本变更 Decode AF 的 domain 的 `PP` MUST 等于 1。除明确支持的 KV-head 复制外，每个实际由 TP 分片的维度 MUST 满足对应算子的整除规则；系统 MUST NOT 使用 `ceil` 隐式增加 heads、hidden、query width 或 intermediate width。

#### Scenario: Unsupported AF pipeline parallelism is rejected

- **WHEN** 任一参与 Decode AF 的 A 或 F execution domain 配置 `PP > 1`
- **THEN** 系统在创建 execution events 前拒绝配置
- **AND** 错误说明 AF micro-batch 流水与 PP 是不同概念且本版本只支持 `PP=1`

#### Scenario: Resolvable TP DP and EP topology is accepted

- **WHEN** Decode AF domains 使用 `PP=1`，且每个基础 rank 的 DP、TP、适用 EP、layout 与 physical worker 均可唯一解析
- **THEN** 系统接受配置并保留所有 group membership
- **AND** TP、DP 或 EP 均不退化为只参与数值校验的参数

#### Scenario: Ambiguous rank responsibility is rejected

- **WHEN** 一个 AF rank 缺少 DP replica、TP group、适用 EP group、layout 或 physical worker，或者同一 membership 有多个解析结果
- **THEN** 系统在仿真开始前拒绝配置
- **AND** 错误包含 domain、rank 与冲突维度

#### Scenario: Non-divisible tensor shape is rejected

- **WHEN** Attention heads、query width、hidden shard 或 FFN intermediate 等实际分片维度不满足配置的 TP 规则
- **THEN** 系统在创建 runtime plan 前拒绝配置
- **AND** 错误列出 domain、operator、global dimension、TP size 与整除规则

#### Scenario: Supported KV head replication remains valid

- **WHEN** KV head 数小于 A TP size 且 TP size 可由 KV head 数整除
- **THEN** 系统按明确复制组接受配置
- **AND** 不将该例外推广到其他 tensor dimensions

### Requirement: Expert placement resolves routed work to F ranks

启用 MoE 的 F execution domain MUST 使用以下两种且仅两种模式之一。`EP=1` 时，Routed Experts MUST 在所选 F DP replica 的 TP group 上执行，Expert TP 等于 F TP，并在 F DP replicas 之间复制 Expert 集合。rank-reuse 模式下，`EP` MUST 等于 F 基础 world size `TP * DP`，每个已有 F rank 对应一个 EP identity，Expert TP 等于 1，Routed Expert placement 跨整个 F world；该模式不得增加 physical workers。`1 < EP < TP*DP` 和 `EP > TP*DP` MUST 在本版本中失败。Dense FFN 与 Shared Expert MUST 独立使用所选 F DP replica 的 TP group。每个 Routed Expert MUST 映射到合法执行 owner；每个原始 token MUST 只选择一个 Shared Expert TP group。

#### Scenario: EP one uses expert tensor parallelism

- **WHEN** F domain 配置 `TP=2`、`DP=2`、`EP=1`
- **THEN** 一个 block 的 Routed Experts 在选中的两 rank F TP group 上执行且 Expert TP 为 2
- **AND** 两个 F DP replicas 各自持有同一完整 Expert 集合的 TP shards

#### Scenario: EP equals the reused F world

- **WHEN** F domain 配置 `TP=2`、`DP=2`、`EP=4`
- **THEN** 四个现有 F ranks 形成 rank-reuse EP group且 Expert TP 为 1
- **AND** 系统仍只使用四个 physical workers

#### Scenario: Intermediate EP size is rejected

- **WHEN** F domain 配置 `TP=2`、`DP=2`、`EP=2`
- **THEN** 系统在模拟前拒绝配置
- **AND** 错误列出仅支持 `EP=1` 或 `EP=TP*DP=4`

#### Scenario: Complete expert placement is accepted

- **WHEN** 模型中的每个 Routed Expert 都映射到当前 EP 模式允许的 F ranks，且一次 assignment 可唯一确定执行 owner 或 TP group
- **THEN** 系统接受 placement
- **AND** placement 可解析出全部 compute participants 与 physical resources

#### Scenario: Routed expert has no legal rank

- **WHEN** placement 缺少模型中的某个 Routed Expert，或引用 F domain 外 rank
- **THEN** 系统在执行 AF iteration 前失败
- **AND** 错误列出 Expert id、F domain 与无效 placement

#### Scenario: Shared Expert placement is executable once per token

- **WHEN** 模型声明一个或多个 Shared Experts
- **THEN** 配置解析出一个所选 F DP replica 的 TP execution group，并将多个 Shared Experts 表示为该 group 上扩大的 shared intermediate
- **AND** 每个原始 token 逻辑上执行一次 shared branch，而不是在所有 EP identities 上重复完整工作

### Requirement: Non-AF phases retain complete execution topology

当只对部分阶段启用 AF 时，系统 MUST 要求未选择阶段仍有完整执行能力。`af.phases=["decode"]` 时 Prefill MUST 使用既有完整模型 aggregate 路径，不创建 Decode A/F blocks 或跨域 transfers；结果 MUST 将该精度等级标为 `prefill_aggregate`。本变更 MUST NOT 删除或改变既有 AF-Prefill capability，只是不重构或验收其逐 rank 执行语义。

#### Scenario: Decode-only AF retains complete Prefill

- **WHEN** `af.phases=["decode"]`
- **THEN** 系统要求至少一个 worker group 声明 `prefill` 并通过既有完整 Prefill 路径计算
- **AND** Prefill 不使用 Decode AF planner、ownership returns 或 Decode AF links

#### Scenario: Prefill approximation is observable

- **WHEN** Decode 使用 rank-local AF DAG 而 Prefill 使用既有完整路径
- **THEN** 结果区分 `decode_af_rank_local` 与 `prefill_aggregate`
- **AND** 系统不把 Prefill aggregate latency 宣称为逐 rank resource trace

#### Scenario: Prefill-only AF retains complete Decode

- **WHEN** `af.phases=["prefill"]`
- **THEN** 系统要求至少一个 worker group 声明 `decode_full`
- **AND** 既有 AF-Prefill capability 仍可按迁移后的合法配置被选择

### Requirement: Invalid or unsupported AF configuration fails before simulation

系统 MUST 在启动模拟事件循环前完成 AF 字段、阶段、拓扑、EP 模式、TP shape、dtype、Expert/component placement、tensor ownership、collective participants、network、resource group 和 resource lane 校验，并使用项目统一配置错误类型报告可操作上下文。合法且已启用的 AF 配置在 executor 与所需 providers 可用时 SHALL 进入 AF 路径；缺少必要依赖时 MUST 明确失败，不得静默退回完整 iteration latency。

#### Scenario: Configuration has multiple AF errors

- **WHEN** 配置同时包含非法容量和缺失阶段 link
- **THEN** 系统在模拟开始前拒绝配置
- **AND** 错误至少标明一个可操作字段或拓扑问题

#### Scenario: Valid AF configuration reaches a supported runtime

- **WHEN** 合法 Decode AF 配置被交给具有 executor、model metadata 和所需 providers 的入口
- **THEN** 系统创建对应 Decode AF 执行路径
- **AND** 系统不再以“AF executor 尚未实现”为由拒绝运行

#### Scenario: Valid AF configuration reaches a runtime without AF executor

- **WHEN** 合法且已启用的 AF 配置缺少 executor、provider、dtype、collective fabric 或必要模型元数据
- **THEN** 系统在产生 AF events 前拒绝运行并标明缺失依赖
- **AND** 系统不得使用完整 iteration backend 静默继续

#### Scenario: Orthogonal F TP EP DP mapping is rejected

- **WHEN** 配置或 mapping 试图按 `TP * EP * DP` 创建 F workers
- **THEN** 系统拒绝该配置并指出本版本的两种 EP 模式
- **AND** 不进入 runtime state 编译

## ADDED Requirements

### Requirement: AF dtypes describe distinct storage and transfer roles

系统 MUST 分别解析effective `weight_dtype`、`activation_dtype`、`kv_cache_dtype`、`a_to_f_dtype`与`f_to_a_dtype`。解析器 MUST 在填充defaults前记录raw field presence：显式legacy `af_dtype`与任一显式新role field混用 MUST 失败；只提供legacy字段时 MUST 将其映射到五个roles并产生可观测弃用信息；无legacy字段时，weight/activation MUST 缺省继承现有model dtype，KV MUST 缺省继承现有KV config或activation dtype，两个AF link directions MUST 缺省继承activation dtype。Operator manifest MUST 声明与最终effective compute/storage roles一致的数据覆盖，但不得覆盖runtime配置。所有roles MUST 通过同一个canonical dtype registry解析。

#### Scenario: Directional transfer dtypes differ

- **WHEN** `a_to_f_dtype=fp8`、`f_to_a_dtype=bf16`
- **THEN** 两个方向分别按 1 byte 与 2 bytes 每元素计算 payload
- **AND** weight、activation 与 KV Cache 字节数不被 transfer dtype 覆盖

#### Scenario: Transfer dtype requires conversion

- **WHEN** 任一方向的 transfer dtype 与 `activation_dtype` 不同
- **THEN** execution plan 在发送端和接收端生成显式 encode/decode compute operations
- **AND** 缺少任一转换 estimate 时在执行前失败，不得假定转换为零成本

#### Scenario: New style defaults resolve after presence checks

- **WHEN** raw input既没有legacy `af_dtype`也没有显式directional transfer dtypes，但model dtype为BF16
- **THEN** weight/activation继承model dtype，KV按KV config或BF16解析，两个transfer roles继承BF16
- **AND** default填充不会被误判为用户混用了legacy与新字段

#### Scenario: Manifest must match effective runtime roles

- **WHEN** 最终`activation_dtype=bf16`但operator manifest只声明该model使用FP16 activation keys
- **THEN** 系统在provider查询前失败并报告model、role、effective值与manifest值
- **AND** manifest不得把runtime activation静默改成FP16

#### Scenario: Legacy dtype maps consistently

- **WHEN** 输入只提供旧 `af_dtype=bf16`
- **THEN** 五个 effective roles 均解析为 BF16
- **AND** runtime summary 记录兼容映射与弃用状态

#### Scenario: FP8 has one canonical byte size

- **WHEN** FP8 用于任一受支持 role
- **THEN** 配置、memory ledger、operator query 与 payload calculation 均从同一 registry 得到每元素 1 byte

#### Scenario: Mixed legacy and explicit dtypes are rejected

- **WHEN** 输入同时设置旧 `af_dtype` 与任一新 dtype role
- **THEN** 系统在模拟前失败并要求使用单一配置形式

### Requirement: Required collectives have explicit executable plans

当Decode AF DAG的tensor layout需要collective时，对应execution domain MUST 提供唯一匹配的communication plan。该plan MUST 明确operation、触发位置、replica-local `tp_rank` participants，以及communication fabric registry中的resource group和resource lane；operation MUST 至少区分`all_reduce`、`all_gather`和`reduce_scatter`。Participants MUST 在每个选中DP instance内映射为actual domain ranks，operation MUST 进入provider query，精确的`(resource_group, resource_lane)` MUST 决定竞争。Fabric registry entry MUST 声明network、duplex、命名lane及各自capacity；AF links按方向/duplex解析lane，collective必须显式选择lane。Full-duplex group的两个directional lanes相互独立，collective只与自己选择的lane竞争；需要与两个AF方向共同竞争时必须配置half-duplex/shared lane。Collective可引用独立entry，也可显式引用与AF link相同entry。Planner MUST 为collective query提供每rank logical input/output bytes，不得预先乘算法wire-volume系数。缺少必要plan/fabric/lane时系统 MUST 失败，不得静默使用完整TP group、通用`tp_collective`或AF跨域link。

#### Scenario: Collective participants select a subset

- **WHEN** `TP=4,DP=2`的plan配置replica-local participants `[0, 2]`
- **THEN** DP0实例化为domain ranks `[0,2]`，DP1实例化为`[4,6]`
- **AND** 同一instance中的TP ranks 1和3不申请该collective资源

#### Scenario: Collective operation changes the query

- **WHEN** 两份配置仅在 `all_gather` 与 `reduce_scatter` 上不同
- **THEN** provider 收到不同 operation keys
- **AND** trace 保留各自 operation 名称

#### Scenario: Missing collective fabric fails

- **WHEN** 一个必要 collective 没有可解析的 network、resource group 或 resource lane
- **THEN** 系统在运行事件循环前失败
- **AND** 不自动改用 A-to-F 或 F-to-A resource

#### Scenario: Independent collective fabric has executable capacity

- **WHEN** collective引用一个未被AF link使用且network、shared lane与capacity均合法的fabric registry entry
- **THEN** 系统为其创建独立contention resource
- **AND** 该collective不与AF transfers竞争

#### Scenario: Collective explicitly shares the AF fabric

- **WHEN** collective plan与AF link显式引用同一resource group和同一resource lane
- **THEN** collective legs与该lane上的AF transfer legs按capacity共同竞争
- **AND** 系统不创建名称相同但彼此独立的resource

#### Scenario: Full-duplex collective lane is not inferred

- **WHEN** collective引用具有`a_to_f`和`f_to_a`两个lanes的full-duplex AF fabric
- **THEN** config必须显式选择其中一个resource lane，否则在event loop前失败
- **AND** 选择`a_to_f`时collective只与A-to-F竞争，F-to-A仍可按自身lane容量并行

#### Scenario: Collective payload remains logical at the planner boundary

- **WHEN** planner为四rank all-reduce生成query
- **THEN** query携带每rank logical input/output bytes、operation和participants
- **AND** ring/tree等wire-volume系数只由provider应用一次

#### Scenario: TP one needs no artificial collective

- **WHEN** 对应 execution group 的 `TP=1` 且 layout 不要求其他通信
- **THEN** 系统不要求配置或生成零意义 collective
