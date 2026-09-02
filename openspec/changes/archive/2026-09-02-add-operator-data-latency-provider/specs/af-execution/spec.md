## MODIFIED Requirements

### Requirement: Latency providers return service time only

系统 SHALL 通过可注入的 compute provider 查询单 block、单层、单 A/F stage 的计算 service time，并通过 transfer provider 查询单次方向传输的 service time。Provider 返回值 MUST 为有限且非负的秒数，并 MUST 携带可追溯的 source、dataset、match type 与 composition 或 transfer details；这些元数据 MUST NOT 参与 executor 的资源选择或调度。Queue duration MUST 仅由执行器申请共享资源产生。AF 正式路径 MUST 不将 Roofline、LLMCompass 的完整 iteration 时延按层数或 block 数拆分后作为静默 fallback。

#### Scenario: Stub providers drive a deterministic execution test

- **WHEN** 测试注入按 stage 和方向返回固定 service time 的 providers
- **THEN** executor 按这些值推进模拟时间
- **AND** 资源等待仍由事件竞争独立产生

#### Scenario: Operator-data provider drives service time

- **WHEN** 正式 provider 为一次 compute 或 transfer 查询返回 service time 与可追溯 metadata
- **THEN** executor 只使用 service time 推进对应服务区间
- **AND** metadata 不改变资源、worker、事件依赖或开始时间

#### Scenario: Invalid provider duration fails explicitly

- **WHEN** 任一 provider 返回负数、NaN 或无限值
- **THEN** 系统终止该 AF iteration 并报告包含 block、layer 和 stage 或 direction 的错误

#### Scenario: Required estimate is unavailable

- **WHEN** provider 无法为某次 AF 查询提供受支持的结果
- **THEN** 系统明确报告缺失查询上下文
- **AND** 系统不得改用完整 iteration 时延继续运行

### Requirement: AF trace reconstructs queue, service, and makespan

当 `trace_enabled=true` 时，系统 MUST 为每个计算和传输事件记录 iteration、block、layer、stage 或 direction、resource、`ready_at`、`start_at`、`end_at`、`queue_duration`、`service_duration`、request 数、token 数、`source_id`、`dataset_version`、`match_type` 和 provider details；传输事件还 MUST 记录 payload bytes。Trace 中的 composition details MUST 足以重建 compute service time，且 trace MUST 能重建每个 block 的完成时间、iteration makespan、barrier wait 和各资源只按 service 区间计算的利用率。当 trace 关闭时，记录开销 MAY 被省略，但调度结果 MUST 不变。

#### Scenario: Queued event has internally consistent timestamps

- **WHEN** 一个 ready 事件等待繁忙资源后完成服务
- **THEN** `queue_duration = start_at - ready_at`
- **AND** `service_duration = end_at - start_at`

#### Scenario: Provider metadata survives trace serialization

- **WHEN** operator-data provider 返回 exact 或 interpolated composition
- **THEN** trace 事件保留 dataset、match type、来源与 component breakdown
- **AND** breakdown 的换算后总和等于事件 `service_duration`

#### Scenario: Iteration makespan uses the last block completion

- **WHEN** 多个 blocks 的完成时间不同
- **THEN** iteration makespan 等于最晚 block 完成时间减去 iteration 开始时间
- **AND** 系统不得将各 block 的独立耗时直接求和

#### Scenario: Disabling trace does not change timing

- **WHEN** 相同输入分别以 trace 开启和关闭运行
- **THEN** 两次运行的资源顺序、block 完成时间和 iteration makespan 相同
