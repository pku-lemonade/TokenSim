## MODIFIED Requirements

### Requirement: A and F service times use non-overlapping operator composition

Compute provider SHALL 接受 `AFComputeQuery.stage` 为 `a`、`router` 或 `expert`，并从原子算子查询结果组合单 block、单层的对应 service time；每次查询 MUST 返回可重建总和的有序 breakdown。A composition SHALL 覆盖 QKV GEMM、Attention core、output projection GEMM 及必要的 RMSNorm、RoPE、Residual Add、data reorder。Router composition SHALL 只覆盖 Router/Gating 以及 dispatch 前必要的 elementwise；Expert composition SHALL 只覆盖 grouped/fused MoE expert 以及 dispatch 后必要的 elementwise。Router 与 Expert compositions MUST 互不重叠，二者之和 SHALL 取代旧的合并 F composition；任何逻辑 operation MUST 恰好计时一次。Attention core MUST NOT 重复包含 QKV 或 output projection；Expert composition MUST NOT 包含 Router、dispatch 或 combine；dispatch、combine 与 A-to-F/F-to-A transfer SHALL 继续由 transfer provider 估算，不得进入 compute service time。

#### Scenario: Attention composition reconstructs service time

- **WHEN** A 查询的所有必要原子算子均成功匹配
- **THEN** 返回的 service time 等于 breakdown 各项换算后时长之和
- **AND** QKV 与 output projection 各出现一次且不包含在 Attention core 中

#### Scenario: MoE composition excludes communication

- **WHEN** Router 查询组合 Router/Gating 与 dispatch 前必要 elementwise，且 Expert 查询组合 grouped/fused expert 与 dispatch 后必要 elementwise
- **THEN** 两个 breakdown 不包含重复 component，且其 service time 之和等于旧 F compute 的纯计算总和
- **AND** dispatch/combine 与 A-to-F/F-to-A transfer 不计入 Router 或 Expert compute service time

#### Scenario: Required component is missing

- **WHEN** A、Router 或 Expert composition 的任一必要原子算子无法 exact match 或合法插值
- **THEN** 对应 compute 查询明确失败并列出 stage、缺失 component 与完整 key
- **AND** 系统不得返回部分总和或退回旧的合并 F 查询

### Requirement: Communication provider returns service time independently of contention

Transfer provider SHALL 接受 typed communication query，至少包含 operation、source/target execution domain、logical participant ranks、映射后的 physical endpoints、network/resource identity、payload bytes 及适用的 topology 参数。Operation MUST 区分 `af_transfer`、`moe_dispatch`、`moe_combine` 和具体 collective。Provider SHALL 根据这些字段和可追溯链路参数返回通信 service time，并 SHALL 采用版本化实测记录、`fixed_latency + payload_bytes / effective_bandwidth` 或数据包中明确版本化的等价模型。Provider MUST NOT 模拟排队、容量竞争、重传或随机拥塞；这些等待仍由 executor 的共享资源产生。

#### Scenario: AF transfer service uses payload and link parameters

- **WHEN** A-to-F 查询匹配到固定延迟和有效带宽
- **THEN** 返回值包含 operation、payload bytes、链路 source 与按统一单位计算的 service time

#### Scenario: Transfer service uses payload and link parameters

- **WHEN** 任一 typed communication query 匹配到固定延迟、有效带宽和版本化链路参数
- **THEN** 返回值包含 payload bytes、链路 source 与按统一单位计算的 service time
- **AND** provider 不返回或修改 queue duration

#### Scenario: Dispatch query preserves expert participants

- **WHEN** executor 为一次 MoE dispatch 查询 service time
- **THEN** query 明确包含发送 ranks、目标 expert ranks、payload 与 physical resource identity
- **AND** provider 结果标明 `operation=moe_dispatch`

#### Scenario: Combine and collective remain distinct operations

- **WHEN** executor 查询 MoE combine 或 execution plan 指定的 AllReduce
- **THEN** provider 分别按 `moe_combine` 或具体 collective operation 匹配模型
- **AND** 不把两者折叠为通用 A-to-F/F-to-A transfer

#### Scenario: Identical service queries experience different queue time

- **WHEN** 两次相同 typed communication query 竞争同一容量为一的资源
- **THEN** provider 对两次查询返回相同 service time
- **AND** 后一次的非零 queue time 仅由 executor 产生

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
