## Context

现有 V3 路径已经复用 `HardwareContext -> ModelSpec -> compile_af_runtime_plan -> provider -> AFExecutor` 的大部分核心设施，但仍在脚本中派生 `V3Executor`，自行补充 embedding、final norm、LM head、token selection 和 boundary collective。另一方面，AF planner 在每个 decode step 的每一层重新创建完整 DAG；即使 trace recorder disabled，executor 仍先构造昂贵的 trace 对象。通信分段配置、host resources、trace event 类型、文档和测试已经存在，但 executor 未执行对应三阶段流程。

## Goals / Non-Goals

**Goals:**

- V3 经由标准 TokenSim serving 入口运行完整模型生命周期。
- 通用核心表达 MLA、Dense/MoE 混合层与模型边界算子，不保留 V3 专用执行语义。
- 修复 AF 通信计时、拓扑和 oversubscription 正确性。
- 在保持离散事件语义的前提下降低构图、trace、lookup 和事件调度开销。
- 用慢路径/快路径差分测试证明结果等价。

**Non-Goals:**

- 不加载或执行真实 DeepSeek checkpoint。
- 不实现 MTP、FP8 数值精度或真实 noaux_tc 路由。
- 不用 `单步延迟 * token 数` 代替动态 scheduler/resource 仿真。
- 不在本变更中实现稳定周期检测或多 token 时间跳跃。
- 不把 analytical operator data 宣称为硬件实测。

## Decisions

### 1. 通用 whole-model 执行生命周期

在核心库中引入通用模型执行层，顺序为 embedding、可选 boundary collective、逐层 DAG、final norm、LM head、可选 logits collective、token selection。所有 boundary operation 使用与 transformer node 相同的 provider、runtime plan、physical resources、通信三阶段和 provenance。`LLMEngine` 根据 `ModelSpec` 调用该执行层；脚本只能提供 CLI 默认值和实验矩阵。

### 2. Planner 分为 template compile 与 dynamic bind

`compile_template` 只处理 phase、layer type、domain topology、placement、tensor ownership、collective plan、resource identity 和不变 shape，产生不可变、可校验的整数索引 DAG。`bind` 注入 iteration/block/request identity、token count、context lengths、routing assignments、动态 payload 和本轮 local workload。SimPy process、event、timeout 和 resource request 永不跨轮缓存。

模板缓存键包含 phase、layer type、A/F domain 与 replica topology、TP/DP/EP、router location、effective dtype contract、batch/shape class 和 routing topology signature。若动态 routing 改变 endpoint branches、ownership 或 payload structure，必须重新 binding 或编译匹配模板，不得错误命中仅按 batch size 建立的缓存。

### 3. 服务估值缓存与调度分离

Compute/transfer provider 在规范化 query key 上缓存不可变 `AFServiceEstimate`。缓存只避免 operator-table lookup、插值、analytical composition 和重复 component 构造；executor 每次仍申请当前 SimPy resource 并执行 timeout，从而保留 queue、overlap 和 completion order。缓存值必须保留 source id、dataset version、match type、timing scope、payload 和 component breakdown。

### 4. 安全的事件融合边界

只允许融合位于同一 physical worker、严格串行、无通信、无分支、无外部消费者且中间不释放资源的 compute components。融合 service duration 是组成项之和，component breakdown 继续保存在 estimate details。不得跨越 collective、A/F transfer、不同 worker、router fan-out、expert branch、weighted combine join 或任何可被其他 block 插入的显式调度边界。

### 5. Trace 采用构造前短路

当 trace disabled 时，executor 在构造 `AFTraceEvent`、`local_work.as_dict()`、tensor slice/routing metadata 和 profiling breakdown 之前返回轻量路径。trace enabled/disabled 使用相同 DAG、service estimate、resources 和 timeout；差异仅是观察数据，不得改变模拟状态。

### 6. 通信拆为 submit、transport、completion

每个 endpoint leg/collective 的原 DAG node id 表示完成事件。启用分段模式后内部生成：源端 host submit、fabric transport、目标端 host completion/synchronize。submit 结束即释放 host；transport 只占 fabric；completion 为零时仍可产生零服务同步事件以保持依赖可解释。多 rank collective 的参与 host 并行 submit，全部提交完成后进入 transport。

分段模式要求 provider 返回 `timing_scope=transport_service`。`legacy_unspecified` 和 `eager_interval` 明确失败，避免把已包含 host 间隙的测量再次叠加固定成本。

### 7. Endpoint-aware topology 与资源语义

AF transfer query 使用每条 executable leg 的实际 source/target physical endpoint，通过 placement 求最低公共 topology level/path，再选择对应 link、ports、oversubscription 和 P2P 模型。P2P oversubscription 通过降低有效带宽或增加传输时间实现，绝不让 oversubscription > 1 降低时延。

`capacity=N` 表示 N 条独立且各自具有声明带宽的 lane。若未来需要多个流共享一个总带宽，应引入独立的 fluid/shared-bandwidth resource，不用增大 capacity 近似。

### 8. 已有缓存的静态指纹与按需 query 构造

缓存路径增量包含两项：预计算静态缓存键，以及缓存命中时减少对象构造。现有 DAG template 与 service estimate cache 已实现；这里优化其查询和绑定入口，不新增另一套缓存，不改变 SimPy 调度器。

静态指纹按当前 executor/planner/provider 实际使用的配置，在初始化或首次使用时生成。复用配置、link、layer 结构、effective dtype、lookup policy 和 dataset 身份中不变的部分；不枚举 batch/context/routing/shape 组合，也不提前计算未遇到的 service。动态 key 仍包含全部影响 DAG 结构、service time、payload 或 provenance 的运行时字段，尤其是 context、local workload、expert ownership、participants 与 endpoints；不能仅按 batch 或 token 数匹配。

在引入预计算前明确配置生命周期：优先由 runtime 持有不可变配置快照并将指纹限制在该实例内。若某入口支持配置、policy 或 dataset 更新，则更新入口必须切换版本并使相关指纹、模板和估值缓存失效；不允许原地修改后静默读取旧结果。不支持运行中更新的入口必须显式拒绝修改，不能只依赖调用方自觉。不得为检测变化在每次命中路径重新递归扫描整份配置。

Service 查询拆为预编译静态 query 骨架、动态规范化 key 和按需完整 query 构造。先用紧凑 key 查询已有缓存，仅在 miss、缓存关闭或兼容 provider 需要完整 query 时实例化 query 与 shape 等临时容器。保持现有完整 query 接口作为参考及兼容路径；不得通过缩减字段或绕过 timing-scope/非法输入校验换取命中。Trace 和错误诊断按需取得本轮上下文，缓存 estimate 仍完整保留 provenance、payload 和 component details。第三方 provider 的可观察行为保持不变，不强制其支持紧凑 key。

已编译不可变模板的静态校验可移至编译阶段；手工/未缓存节点仍走完整校验，动态 workload、routing、ownership 等约束仍须逐次验证。每次执行继续申请资源并推进 timeout，不缓存排队或完成状态。

先在当前缓存开启的普通执行路径做函数级 profiling，分别测量 key、query 构造与估值成本。比较本轮修改前后两条缓存开启路径，并保留缓存关闭的正确性 oracle。61 层短/长 decode 样本重复计时，区分冷启动和热缓存，报告 wall/CPU、命中率、query 构造次数与波动；计时运行不挂完整 service-record observer，差分验证单独运行。若无收益或出现回退，明确记录，不把历史缓存收益归因于本轮优化。

### 9. Trace 按 iteration 索引、单份存储与导出时序列化

Trace 开启时，`events_for(iteration_id)` 当前每轮过滤整个历史事件列表，校验和标准入口摘要会反复触发扫描。`summarize` 又深层转换事件，engine 长期保存其字典副本。此次 trace 增量统一处理检索和重复存储两个成本，不改变事件生成、SimPy 调度或校验范围。

保留 `trace.events` 作为唯一完整历史事件存储，新增 `iteration_event_indices: dict[int, list[int]]`。`record(event)` 追加事件时同步追加其全局位置；`events_for` 按这些位置返回本轮完整、有序的事件集合，不扫描其他 iteration。索引存位置而非事件对象，因为模型边界逻辑会在原位置替换事件以补充 predecessor；后续读取必须得到替换后的当前事件。所有生产代码的追加/替换入口需核对：追加统一维护索引，替换保持位置和 iteration identity。若已有 reset/clear 入口，必须同时重置索引；本增量不通过删除历史事件控制内存。

本轮校验通过该索引读取相同的全部事件，继续检查依赖、排队、payload 和模型边界。摘要以一次按原事件顺序的遍历累计各资源 busy time；同一事件对同一 resource 只贡献一次，保留 multi-resource、capacity 和零 makespan 的原计算口径。资源名排序只用于稳定输出，不对每个资源重新遍历事件。

`af_iteration_summaries` 仅保留 iteration ID、起止时间、makespan、block completion、barrier wait、resource utilization 等摘要，不长期保留完整事件字典。结果导出边界通过摘要 ID 从 recorder 取得当前事件，再按原 schema 序列化；统一检查 `get_af_stats`、结果文件及其他摘要消费方，避免内部结构调整导致外部字段丢失。重复导出必须无副作用，不能消耗历史、改变统计或将导出副本缓存回 engine；后续追加事件不应使早先导出的独立结果发生变化。

Trace disabled 时继续跳过事件构造和深层序列化，不积累事件索引，并保持原空事件输出约定。当前全部历史的存储仍随事件数增长，索引额外占用线性空间；一次性构建完整 JSON 时也仍有临时副本和导出内存峰值。本增量不实现流式导出、历史截断、DAG 缓存准入或指纹淘汰策略修改。

验证以未修改 trace 管理前的当前实现为参考，比较 trace 字段、顺序、资源统计、导出 JSON 和仿真结果，覆盖交错 iteration、原位置事件替换、空记录和 trace-off。用固定每轮事件量的 8/16/32-step 用例统计实际检索访问次数，证明在每轮固定次数校验/汇总下累计访问量随总事件数线性增长；单遍资源累计按事件与资源关联数计量，不将该结论扩大为整个 invariant validator 或 simulator 都是线性。分别记录执行期、导出期 wall/CPU 和内存，验证运行期间未物化和保留完整事件字典副本，不承诺未经测量的总加速比例。

## Risks / Trade-offs

### 遗留测试迁移（2026-09-29 用户确认纳入范围）

`test_mla.py`、`test_deepseek_full_decode.py` 与 `test_operator_batching.py`
迁入 `ModelSpec`、`OperatorResolver`、`OperatorLookup`、`work_for_table`、
核心 model executor 和当前 package schema。保留 latent KV、TP 权重、Dense/MoE
混合层、分组 workload、边界依赖、缺数据失败和 measured override 的行为断言。

不恢复 `ShapeDiscovery`、`RecordingLookup`、旧 `ModelOperatorManifest` 或旧脚本
factory 签名。旧 shape-scoped `CalibrationRule` 已无对应运行入口，本变更明确
使用 package 内 table/default affine calibration；用按表隔离与实测值不重复校准
的测试取代旧 rule 匹配断言，不宣称两者具有相同的 shape 选择能力。
非法 multiplier/offset 仍须显式拒绝。旧全量 shape 包生成测试由当前包的
save/load、exact lookup 和 missing-shape 回归承接。

完整等长 prefill 的 executor 契约独立于底层 analytical work 公式；底层公式可
表示有 cached context 的 query，不代表标准入口承诺 chunked prefill。

- [Risk] 模板缓存键遗漏 routing/ownership 维度会复用错误 DAG。→ 使用结构签名、不可变模板、bind 后校验和慢路径差分 oracle。
- [Risk] 算子融合改变跨 block 插队机会。→ 仅融合原先已属于一次不可抢占 compute service 的 provider components；不融合独立 ExecutionNodes，除非 spec 明确其不可抢占语义。
- [Risk] 三阶段通信使旧结果发生变化。→ 默认关闭保持 legacy；新 V3 标准配置显式开启，并要求 transport-only 数据口径。
- [Risk] topology-aware leg 定价增加小消息数量和事件数。→ 先保证正确，随后通过 endpoint-leg 模板和 estimate cache降低墙钟开销。
- [Risk] 通用 whole-model executor影响其他模型。→ 对非 MLA 和 AF-disabled 建立相同输入的回归基线，V3 脚本迁移分阶段完成。
- [Risk] Trace 索引缓存了被替换前的事件或混入并发 iteration。→ 存储全局位置，覆盖交错追加与模型边界替换的差分测试。
- [Risk] 延迟序列化改变摘要消费方和导出 schema。→ 内部摘要与导出 DTO 分离，逐字段比较现有 JSON，并验证重复导出和 trace-off 行为。

## Migration Plan

1. 建立慢路径基线与差分 oracle，先修复 trace 短路和 P2P oversubscription。
2. 接入三阶段通信与 timing-scope 校验，恢复既有专项测试。
3. 将模型边界 operation 迁入通用 runtime，并让 V3 通过标准入口完成小规模 decode。
4. 拆分 planner template/bind，加入 estimate cache；逐项证明 trace、queue、bytes 和 makespan 等价。
5. 在安全边界内融合 provider components并记录墙钟基准。
6. 将 V3 专用脚本缩减为薄入口，删除 `V3Executor`；保留兼容命令并给出迁移说明。
7. 在既有实现上测量缓存查询成本，落实配置生命周期，加入静态指纹和命中时按需 query 快路径，再分别执行正确性差分与无重型观测器的重复性能验证。
8. 保存当前 trace/export 基线，建立 iteration 位置索引与单遍摘要，将完整事件序列化移到导出边界；验证历史完整性、结果等价、检索复杂度和运行期内存后完成回归。

## Open Questions

- 已确认：V3 标准入口支持完整等长、未分块 prefill 与完整 decode；部分缓存、分块或非等长 prefill 显式拒绝。该范围由用户于 2026-09-29 确认。
- 已实现：跨域网络默认要求可解析的 physical endpoints；只有配置显式选择 `flat_link` 时允许平坦链路兼容模式。


### 核心路由修复（用户已确认）

Synthetic routing 按 seed、phase、layer、request 和 token identity 确定 expert，排除 AF block、iteration 和调用顺序。Uniform 使用循环均匀分配，hot/skew/burst 使用每 token 独立的确定性采样且 top-k 无重复；热点不足时从剩余 experts 补齐。旧统计型 histogram API 保持原口径，显式外部 assignments 保持不变。该修复会改变旧 synthetic 仿真结果，需重新验证优化开关下的结果等价性。
