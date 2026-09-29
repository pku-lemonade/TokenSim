## MODIFIED Requirements

### Requirement: AF communication separates submission, transport, and completion

启用 `af.communication_timing` 时，每个 endpoint leg 与 collective MUST 按 host submission、fabric transport 和 completion/synchronize 三段执行。Submission MUST 竞争源端 host resource并在提交完成后释放；transport MUST 仅竞争规范化 fabric resource；completion service非零时 MUST 竞争目标端host resource。原 logical DAG node completion MUST 发生在transport与completion完成后，下游不得只等待submission。Dependency wait只用于解释等待，MUST NOT重复计入service time。

#### Scenario: Host submission overlaps transport

- **WHEN** 同一host连续提交两条消息且第一条已进入fabric
- **THEN** host可提交第二条而第一条继续传输
- **AND** trace分别记录host_submit与transfer的queue/service区间

#### Scenario: Completion cost does not occupy fabric

- **WHEN** transport完成且配置非零host completion cost
- **THEN** completion在目标host资源上执行
- **AND** fabric在transport结束时释放，可服务下一条消息

#### Scenario: Separated timing rejects ambiguous measurements

- **WHEN** 分段模式的transfer provider返回`legacy_unspecified`或`eager_interval` timing scope
- **THEN** executor在使用该estimate前明确失败
- **AND** 不把可能已经包含host间隙的数值作为transport service再次叠加固定成本

#### Scenario: Legacy timing remains compatible

- **WHEN** communication timing disabled
- **THEN** executor保持单段provider service行为
- **AND** 不生成host_submit或completion service事件

### Requirement: AF transfer latency follows physical endpoint topology

每个跨域 endpoint leg MUST 使用其实际source/target physical endpoints解析topology path、lowest common level、link、可用ports与oversubscription，并按该路径估算transport service。P2P oversubscription大于1时 MUST 不降低延迟。只有配置显式声明flat-link兼容模式时，系统 MAY 绕过placement使用单一link class。

#### Scenario: Cross-rack transfer is not priced as same-node

- **WHEN** 两条payload相同的legs分别位于同节点和跨机架endpoints
- **THEN** provider依据各自topology path选择link与层级成本
- **AND** 跨机架结果不因忽略placement退化为同节点结果

#### Scenario: Oversubscription cannot improve P2P latency

- **WHEN** 其他条件相同且topology oversubscription从1增加到大于1
- **THEN** P2P transport latency保持不变或增加
- **AND** 不得除以oversubscription使传输更快

### Requirement: Communication capacities represent independent lanes

AF communication resource的`capacity=N` MUST 表示N条独立服务lane，每条lane使用声明的link service model。相同resource key若由多个links/domains声明，network、duplex与capacity MUST 一致，否则系统在启动前失败。系统 MUST NOT 将capacity增加解释为多个flow各自获得完整共享总带宽，除非配置确实代表独立物理lanes。

#### Scenario: Conflicting fabric aliases fail

- **WHEN** 两个link为相同resource group/lane声明不同network、duplex或capacity
- **THEN** 系统在构建resources前报告冲突
- **AND** 不通过first-wins或`setdefault`静默选择一个定义

#### Scenario: Capacity one serializes service

- **WHEN** 两条ready transfers竞争同一capacity-one lane
- **THEN** service intervals不重叠
- **AND** 后获得lane的transfer记录queue duration

### Requirement: Whole-model AF execution uses one dependency and resource system

Transformer boundary operations与逐层AF DAG MUST 使用同一个core executor、runtime plan、provider、physical compute/fabric resources和failure handling。系统 MUST NOT 在`scripts`中的executor subclass旁路通用communication timing、trace或resource semantics。

#### Scenario: Boundary collective uses normalized resources

- **WHEN** embedding或logits boundary需要TP collective
- **THEN** collective使用domain plan声明的participants、topology和resource lane
- **AND** 与层内collective按相同capacity竞争

#### Scenario: Layer failure cancels the whole model step

- **WHEN** 任一boundary或layer node查询失败或返回非法duration
- **THEN** 当前model step停止并报告operation、layer、rank和query上下文
- **AND** 不输出部分成功的V3 token结果
