## MODIFIED Requirements

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
