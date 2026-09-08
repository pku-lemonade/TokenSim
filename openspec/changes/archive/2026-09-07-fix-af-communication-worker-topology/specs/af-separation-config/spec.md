## MODIFIED Requirements

### Requirement: Worker groups declare explicit capabilities

系统 SHALL 支持 worker group 使用唯一 `name` 和非空 `capabilities` 声明职责。允许的 capability MUST 限于 `prefill`、`decode_full`、`af_attention` 和 `af_ffn`；同一 group 内重复或未知 capability MUST 被拒绝。`num_workers` MUST 表示该 group 中可独立竞争的物理计算 slot 数量，而不是 A/F 共用的 TP/PP/DP/EP 总数；其值 MUST 为正整数，group 引用的默认网络 MUST 存在于集群网络配置中。对每个启用的 AF execution domain，其全部 logical rank identities 与该 domain worker group 的 physical workers MUST 形成严格一对一映射。

#### Scenario: Physical worker count is independent of parallel ranks

- **WHEN** A group 与 F group 分别声明 2 个和 4 个 physical workers，且各自 execution domain 分别生成并一对一映射 2 个和 4 个 logical rank identities
- **THEN** 系统保留 2 个和 4 个可竞争计算 slot
- **AND** 系统不将两组 physical worker 数相加后与任一 domain 的 world size 比较

#### Scenario: Shared complete Prefill and AF Attention group is accepted

- **WHEN** 一个 worker group 同时声明 `prefill` 和 `af_attention`
- **THEN** 系统保留两个 capability
- **AND** 是否在 Prefill 阶段使用 AF 只由 `af.phases` 和 link 绑定决定

#### Scenario: Unknown capability is rejected

- **WHEN** 一个 worker group 声明不在允许集合中的 capability
- **THEN** 系统拒绝该配置
- **AND** 错误信息包含 group 名称和未知 capability

#### Scenario: Duplicate group names are rejected

- **WHEN** 两个 worker group 使用相同的 `name`
- **THEN** 系统拒绝该配置
- **AND** 错误信息标明重复的 group 名称

### Requirement: Each selected phase has one split topology

当 `af.enabled=true` 时，每个所选阶段 MUST 恰好由一条 AF link 覆盖。该 link 的 `attention_group` 与 `ffn_group` MUST 引用不同 worker group，前者 MUST 声明 `af_attention`，后者 MUST 声明 `af_ffn`。每个端点 MUST 引用与其职责相符的 execution domain；系统 MUST 不根据 Ada、Groq 或其他硬件名称推断职责或 rank topology。

#### Scenario: Decode AF uses separate A and F domains

- **WHEN** Decode link 将 A group 和 F group 分别绑定到合法的 `decode_attention` 与 `decode_ffn` execution domain
- **THEN** 系统接受两个 domain 各自的 topology 与 device mapping
- **AND** A 与 F 的 worker 数 MAY 彼此不同，但每个 domain 内 logical rank identity 数 MUST 等于其映射的 physical worker 数

#### Scenario: One link covers both phases

- **WHEN** 一条合法 link 的 `phases` 同时包含 `prefill` 和 `decode` 且端点 domain 覆盖相应职责
- **THEN** 系统接受同一组 A/F 设备承担两个阶段

#### Scenario: Separate links cover separate phases

- **WHEN** 一条合法 link 只覆盖 Prefill，另一条合法 link 只覆盖 Decode
- **THEN** 系统接受两个阶段使用不同 A/F 设备池和 execution domains

#### Scenario: Selected phase has no link

- **WHEN** `af.phases` 包含某阶段但没有 link 覆盖该阶段
- **THEN** 系统拒绝该配置
- **AND** 错误信息标明缺少 link 的阶段

#### Scenario: Selected phase has ambiguous links

- **WHEN** 两条 link 同时覆盖同一个 AF 阶段
- **THEN** 系统拒绝该配置
- **AND** 错误信息说明第一版每阶段只允许一条 link

#### Scenario: Link endpoints do not provide required capabilities

- **WHEN** link 的 A 或 F 端点未声明相应 `af_attention` 或 `af_ffn` capability，或未绑定相符 execution domain
- **THEN** 系统拒绝该配置
- **AND** 错误信息包含 link、端点 group 和 domain 名称

## ADDED Requirements

### Requirement: Execution domains separate topology from physical placement

系统 MUST 将 Prefill、Decode A 和 Decode F 建模为独立 execution domains，并分别保存 logical instances、domain-local rank topology 与 rank-to-device mapping。每个 domain MUST 独立校验其 TP、PP、DP 以及适用的 EP 关系和 mapping 完整性；系统 MUST NOT 要求所有 P/A/F physical workers 数之和等于 `TP * PP * DP`，也 MUST NOT 将一套 global rank 顺序隐式复用于全部 domains。每个 `(logical_instance, domain_rank)` identity MUST 映射到唯一 physical worker，同一 domain 内一个 physical worker MUST NOT 被多个 logical rank identities 复用，且该 domain worker group 的每个 physical worker MUST 恰好被映射一次。EP SHALL 描述 F domain 内 expert placement 或 expert communication group，不得被无条件当作额外 physical worker 乘数。

#### Scenario: P, A, and F use different topologies

- **WHEN** Prefill、Decode A 和 Decode F domains 分别声明不同的 logical instance 数、TP/PP/DP/EP 配置和合法 device mapping
- **THEN** 系统逐 domain 接受并校验这些配置
- **AND** 任一 domain 的 rank 编号都只在该 domain 内解释

#### Scenario: One physical device backs multiple logical ranks

- **WHEN** 一个 mapping 将同一 execution domain 内的多个 logical rank identities 映射到同一 physical device 或 compute slot
- **THEN** 系统在模拟启动前拒绝该配置
- **AND** 错误信息列出冲突的 domain、logical instance、domain rank 与 physical worker

#### Scenario: Domain mapping is incomplete

- **WHEN** 一个显式多 rank domain 缺少 rank、device 或并行组映射，导致拓扑不能无歧义解析
- **THEN** 系统在模拟前拒绝配置并标明 domain 与缺失 mapping
- **AND** 系统不得按 worker group 顺序猜测 global ranks

#### Scenario: Legacy single-device configuration remains valid

- **WHEN** AF 被关闭或合法单设备 AF 配置未显式声明新 domain topology
- **THEN** 系统生成无歧义的单实例 domain view
- **AND** 既有单卡执行行为保持不变
