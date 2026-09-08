## ADDED Requirements

### Requirement: AF parallel topology must be executable

系统 MUST 在仿真开始前将每个参与 AF 的 execution domain 解析为完整的 DP replica、TP group、适用的 EP group、logical rank 与 physical worker 映射。每个并行参数 MUST 对应运行时可执行的分片、复制、路由或通信语义；系统 MUST NOT 接受仅通过数值乘积或 rank 数量校验、但无法确定每个 rank 职责的配置。第一阶段 AF execution domain 的 `PP` MUST 等于 1，任何 `PP > 1` 的 AF 配置 MUST 明确失败。

#### Scenario: Unsupported AF pipeline parallelism is rejected

- **WHEN** 任一参与 AF 的 A 或 F execution domain 配置 `PP > 1`
- **THEN** 系统在创建执行事件前拒绝该配置
- **AND** 错误信息标明 domain、`PP` 值以及第一阶段仅支持 `PP=1`

#### Scenario: Resolvable TP DP and EP topology is accepted

- **WHEN** AF domains 使用 `PP=1`，且每个 logical rank 的 DP replica、TP group、适用的 EP group和 physical worker 均可唯一解析
- **THEN** 系统接受该配置并保留所有 group membership
- **AND** 系统不把 TP、DP 或 EP 降级为只参与 world-size 校验的数字

#### Scenario: Ambiguous rank responsibility is rejected

- **WHEN** 一个 AF logical rank 缺少 DP replica、TP group、适用的 EP group或 physical worker 信息，或同一 membership 可被解析为多个结果
- **THEN** 系统在仿真开始前拒绝配置
- **AND** 错误信息包含 domain、rank 与缺失或冲突的并行维度

### Requirement: Expert placement resolves routed work to F ranks

启用 MoE 的 F execution domain MUST 提供与模型 expert 集合一致的 expert placement，使每个 Router/Top-K 输出的 expert id 都能解析到合法的 F logical rank 和 physical worker。EP MUST 表示 expert 到 rank 的放置和域内 dispatch/combine 参与关系，不得被无条件解释为额外 worker 乘数。系统 MUST 在仿真前拒绝缺失 expert、未知 expert、空 placement 或引用 domain 外 rank 的配置。

#### Scenario: Complete expert placement is accepted

- **WHEN** 模型中的每个 expert 都映射到 F domain 内至少一个合法 rank，且所有映射均可解析到 physical worker
- **THEN** 系统接受 placement
- **AND** 后续 routing result 可唯一确定每个 expert 分支的候选执行 rank

#### Scenario: Routed expert has no legal rank

- **WHEN** placement 缺少模型中的某个 expert，或将其映射到 F domain 外的 rank
- **THEN** 系统在执行 AF iteration 前失败
- **AND** 错误信息列出 expert id、F domain 与无效 placement
