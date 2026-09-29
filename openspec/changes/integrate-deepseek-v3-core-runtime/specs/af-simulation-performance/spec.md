## Purpose

在不改变离散事件结果、排队和资源竞争语义的前提下，消除 AF 长序列仿真中的无用 trace 构造、重复 DAG planning、重复延迟估值和过细事件开销。

## ADDED Requirements

### Requirement: Disabled trace incurs no event-construction work

当 trace disabled 时，系统 MUST 在构造 `AFTraceEvent`、序列化 local work、复制 tensor slices/routing edges 或生成 trace-only metadata 前短路。Trace 状态 SHALL 仅影响观察数据，不得改变 DAG、provider query、resource request、service duration、queue duration 或请求完成结果。

#### Scenario: Trace-off skips expensive conversion

- **WHEN** AF 仿真以 trace disabled 运行
- **THEN** executor 不调用 trace-only 的 `local_work.as_dict()` 或等价深层转换
- **AND** recorder 不接收先构造后丢弃的事件

#### Scenario: Trace modes are simulation-equivalent

- **WHEN** 相同 workload 分别以 trace enabled 和 disabled 运行
- **THEN** 两次请求完成顺序、模拟时间、makespan、provider query 数、通信字节和资源服务区间一致
- **AND** 唯一允许差异是 trace/profiling 输出与墙钟开销

### Requirement: Trace iteration lookup uses complete position indexes

Trace enabled 时，recorder SHALL 保留唯一的完整历史事件列表，并在追加事件时维护 iteration 到全局事件位置的索引。`events_for(iteration_id)` MUST 返回本轮全部事件且保持原记录顺序，MUST NOT 为筛选本轮事件扫描其他 iteration。索引 MUST 指向当前存储位置，以便事件原位置替换后校验和导出读取最新内容。历史事件 MUST NOT 因本轮优化而被截断或丢弃。

#### Scenario: Interleaved iterations remain isolated and complete

- **WHEN** 多个 iteration 的事件交错追加，并分别执行校验和摘要
- **THEN** 每轮读取自身的全部事件且顺序与从完整列表筛选得到的参考结果一致
- **AND** 不访问其他轮次的事件来筛选，也不丢失原有依赖、payload 或资源校验

#### Scenario: Replacement is visible through the index

- **WHEN** 模型边界逻辑在原位置替换事件以补充 predecessor，且 iteration identity 保持不变
- **THEN** 随后的索引读取、校验和导出均返回替换后的事件
- **AND** 不保留或返回旧事件对象作为索引副本

#### Scenario: Lookup work grows with relevant events

- **WHEN** 固定每轮事件量的 8、16、32-step 用例每轮执行固定次数的检索、校验和摘要
- **THEN** 事件检索的累计访问次数随总事件数线性增长
- **AND** 该断言以访问计数验证，不以不稳定的绝对 wall-time 阈值代替

### Requirement: Runtime trace summaries avoid duplicate event serialization

每轮摘要 SHALL 通过一次遍历累计资源 busy time，保持 resource capacity、多资源事件和零 makespan 的既有统计口径，MUST NOT 为每个资源重新扫描整轮事件。Engine 常驻摘要 SHALL 仅保留 iteration identity 和聚合统计，MUST NOT 在轮次完成时深层序列化并长期保存完整事件字典副本；完整事件继续由 recorder 保存。

#### Scenario: Resource aggregation preserves accounting

- **WHEN** 本轮包含多资源事件、共享资源以及不同 resource capacities
- **THEN** 单遍累计得到与参考实现一致的 utilization，每个事件对同一资源只计入一次
- **AND** makespan 为零和资源输出排序保持既有行为

#### Scenario: Completing iterations does not materialize event dictionaries

- **WHEN** 多轮执行完成且尚未请求完整结果导出
- **THEN** engine 摘要不调用完整事件的 `as_dict` 或等价深层序列化
- **AND** recorder 的原始完整历史仍可通过索引读取，engine 不另存完整事件字典集合

### Requirement: Deferred trace export preserves the public result contract

结果导出边界 SHALL 按 iteration 索引取得当前完整事件并按既有 schema 序列化。对外 JSON 的字段、事件顺序、provenance、依赖、时间、payload 和聚合统计 MUST 与参考实现一致。导出 MUST 无副作用，不消耗历史或将完整序列化副本长期缓存回 engine。系统 MUST 如实区分运行期存储改善与一次性完整导出的内存峰值，不将本优化描述为有界内存或流式导出。

#### Scenario: Repeated exports preserve results and recorder state

- **WHEN** 对未发生状态变化的已完成仿真连续导出两次完整结果
- **THEN** 两次结果与参考 JSON 一致，recorder 的事件及索引不变
- **AND** 随后的事件追加不会修改此前已经返回的导出结果

#### Scenario: Disabled or empty trace remains compatible

- **WHEN** trace disabled 或查询尚无事件的 iteration
- **THEN** 保持既有空事件输出约定，不报错或读取其他 iteration
- **AND** trace disabled 时不累计事件索引、不创建 trace 事件或深层序列化数据，模拟结果保持等价

### Requirement: Static execution structure is compiled once and bound per iteration

系统 SHALL 将 AF planning 分为不可变 DAG template 编译与逐 iteration/block 动态 binding。Template MUST 可复用不变的节点类型、依赖、rank/worker placement、tensor ownership、collective plan、resource identity 和静态 shape；binding MUST 注入当轮 identity、token/context、routing、dynamic workload 和 payload。SimPy process/event/timeout/resource request MUST 每轮独立创建，不得跨轮缓存。

#### Scenario: Repeated uniform decode reuses templates

- **WHEN** 连续 decode steps 具有相同 phase、layer type、topology、dtype 和 routing structure
- **THEN** planner 复用已校验的 Dense/MoE template
- **AND** 不重新创建和校验等价静态 DAG 对象

#### Scenario: Dynamic routing cannot hit an incompatible template

- **WHEN** routing skew 改变 Expert endpoints、ownership branches 或通信结构
- **THEN** cache key 或 binding validation 区分该结构
- **AND** 系统不得仅因 batch size 相同复用错误 DAG

### Requirement: Service estimates are cached independently of scheduling

系统 SHALL 对规范化且语义等价的 compute/transfer query复用不可变 service estimate。缓存命中 MUST 保留 source、dataset、match type、timing scope、payload 和 component details。无论是否命中缓存，每次执行 MUST 重新申请当前 SimPy resources并产生当轮 queue/overlap/completion状态。

#### Scenario: Same query avoids repeated lookup

- **WHEN** 多轮产生完全相同的规范化 operator query
- **THEN** provider 可复用先前 estimate而不重新执行表匹配、插值或 analytical composition
- **AND** 每轮仍独立进入 worker或fabric队列

#### Scenario: Context-sensitive query does not collide

- **WHEN** MLA context length、batch、dtype、device、local shape或fallback policy不同
- **THEN** estimate cache key区分这些影响service time或provenance的字段
- **AND** 不返回另一 shape或policy的结果

### Requirement: Static cache fingerprints are scoped to active configuration

系统 SHALL 在初始化或首次使用时为当前 runtime 实际使用的不变配置生成静态缓存指纹，并在后续 DAG/service 缓存查询中复用。系统 MUST NOT 枚举所有可能 batch、context、routing 或 shape 组合，也不得为预计算指纹提前执行未请求的 service 估值。动态 key MUST 保留所有影响结构、估值或 provenance 的维度。

#### Scenario: Repeated queries reuse static fingerprints

- **WHEN** 同一配置生命周期内连续执行多层和多个 decode steps
- **THEN** 缓存查询复用已生成的静态配置指纹，不逐次递归转换整份配置
- **AND** context、routing workload、ownership 或 endpoints 变化仍经动态 key 或 binding validation 正确区分

#### Scenario: Configuration updates cannot reuse stale entries

- **WHEN** 已生成指纹后请求修改配置、lookup policy 或 dataset
- **THEN** 支持更新的入口切换配置版本并使相关指纹和缓存失效；不支持更新的入口显式拒绝修改
- **AND** runtime 使用的不可变配置快照不受外部原地修改影响

### Requirement: Cache hits avoid eager full query construction

内置 provider 的缓存快路径 SHALL 通过静态 query 骨架和动态规范化 key 查询已有 service 缓存，命中时避免仅为查缓存而构造完整 query 和临时 shape/dtype 容器。未命中、缓存关闭及兼容 provider 路径 MUST 保持完整 query 语义。系统 MAY 复用已编译不可变模板的静态校验结果，但 MUST 保留动态输入校验以及手工或未缓存节点的完整校验。

#### Scenario: Hit skips query construction but retains scheduling

- **WHEN** 内置 provider 收到与缓存条目语义相同的服务请求
- **THEN** executor/provider 不调用仅供估值的完整 query 构造器，返回包含完整 provenance 的缓存 estimate
- **AND** 本次资源申请、排队和 timeout 独立执行，trace 或诊断按需使用本轮上下文

#### Scenario: Miss and invalid input remain observable

- **WHEN** 紧凑 key 未命中、缓存关闭或请求包含非法动态输入
- **THEN** 合法请求通过完整 query 路径估值，非法请求按原有契约显式失败
- **AND** 缓存快路径不得绕过 ownership、workload 或通信 timing-scope 校验

### Requirement: Cache fast-path gains use an already-cached baseline

本轮静态指纹与按需 query 优化 MUST 以本轮修改前已开启 DAG/service 缓存的实现为性能对照，并保留缓存关闭路径的正确性差分。性能计时 MUST 与完整 service-record 观察分开运行；报告 SHALL 包含重复测量的冷启动与热缓存成本、wall/CPU、命中率及完整 query 构造次数，不以既有缓存收益代替本轮收益。

#### Scenario: Short and long decode expose amortization costs

- **WHEN** 在相同配置和种子下比较 61 层 V3 的短、长 decode workload
- **THEN** 分别重复计时并报告波动，计时运行不启用完整 service-record observer
- **AND** 单独通过差分验证模拟结果及 provenance 等价，无收益或回退也如实记录

### Requirement: Event fusion preserves scheduling boundaries

系统 MAY 将同一 physical compute resource 上严格串行、无外部消费者、无通信、无分支且原语义不可抢占的 components 合并为一个 service event，其 duration MUST 等于各 component duration之和并保留 breakdown。系统 MUST NOT 跨 communication、resource、fan-out、join 或可抢占 ExecutionNode 边界融合。

#### Scenario: Internal compute components are fused safely

- **WHEN** 一个 provider composition 内的多个 kernel components 本来作为一次不可抢占 compute service占用同一worker
- **THEN** executor使用一个resource request和timeout执行总duration
- **AND** profiling仍可查看全部component duration与来源

#### Scenario: Communication boundary prevents fusion

- **WHEN** 两项compute之间存在collective、A-to-F/F-to-A transfer或另一分支的依赖点
- **THEN** 系统保留独立SimPy事件与resource release/acquire语义
- **AND** 不因融合消除合法overlap或改变插队机会

### Requirement: Performance optimizations have differential oracles

每项快路径优化 MUST 与未缓存、未融合的参考路径进行确定性差分验证，至少比较 request TTFT/TBT/completion、makespan、service/queue intervals、operation counts、communication bytes、resource utilization和operator provenance。墙钟基准 SHALL 分开报告 planning、lookup、SimPy和trace成本。

#### Scenario: V3 long decode remains exact

- **WHEN** 61层V3 workload分别通过参考路径与优化路径运行
- **THEN** 所有模拟结果在声明的浮点容差内一致
- **AND** 报告模板命中率、estimate命中率、事件数和墙钟时间

#### Scenario: Multi-token skipping is not inferred from equal service time

- **WHEN** 多个decode steps具有相同单步service estimate但请求到达、batch成员、queue、KV或routing状态可能变化
- **THEN** 本变更仍逐轮推进必要SimPy状态
- **AND** 不使用单步时间乘token数替代动态仿真
