## 1. 统一 dtype 与执行契约

- [x] 1.1 新增公共 canonical dtype registry，统一 dtype alias、role 校验和 byte width（含 FP8=1、BF16/FP16=2、FP32=4），移除 executor/runtime/analytical 中的私有映射；用新增 dtype unit tests 验证同一 dtype 在三条路径返回相同结果且未知 dtype 明确失败
- [x] 1.2 在填充 defaults 前记录 raw dtype field presence，再解析 `weight_dtype`、`activation_dtype`、`kv_cache_dtype`、`a_to_f_dtype`、`f_to_a_dtype`，实现 legacy-only `af_dtype` 映射及新旧混用拒绝，并让 operator manifest 只校验最终 effective roles 而不覆盖 runtime；用 `tests.test_af_config` 覆盖无字段默认值、五角色覆盖、legacy-only、弃用 metadata、manifest mismatch 与冲突错误
- [x] 1.3 扩展 AF compute/transfer query 类型，使其携带 layer type、original/assignment workload、layout/owner、五类 effective dtype、logical stage id 与 endpoint 信息；用 query construction tests 验证字段不会在 executor-provider 边界丢失

## 2. 修正基础 topology、EP 与 collective 配置

- [x] 2.1 将 A/F domain 的基础 world size 与 runtime mapping 固定为 `TP * DP`（AF `PP=1`），移除 F rank 数中的无条件 EP 乘数；用 `tests.test_af_config` 验证 `TP=2,DP=2` 在任意合法 EP 模式下都只要求四个 F workers
- [x] 2.2 实现 `EP=1` 与 `EP=TP*DP` 两种模式及稳定的 `ep_rank=dp_rank*TP+tp_rank` rank-reuse identity，拒绝中间值和隐式 `TP*EP*DP` mapping；用 `tests.test_moe_expert_parallelism` 和 `tests.test_af_parallel_planning` 验证 placement、Expert TP 与错误上下文
- [x] 2.3 让 `EP=1` 的 Routed Experts在所选 F DP replica的 TP group执行并按 DP复制，让 rank-reuse Routed Experts跨 F world放置且 Expert TP=1，同时让 Dense/Shared始终使用所选 F TP group；用 placement tests 验证每个 Expert、component与 physical rank的唯一解析
- [x] 2.4 在 runtime plan前严格校验 hidden、query width、Attention heads、FFN intermediate等 TP shard axes，并只保留已定义的 KV-head replication例外；用 shape table tests 验证合法分片、KV复制和 `8 heads / TP=3` 早失败且不产生 `ceil` workload
- [x] 2.5 新增独立 communication fabric registry，并扩展 collective config 校验，使 operation、after-stage、replica-local `tp_rank` participants、network、resource group 和显式 resource lane 唯一且可执行；在每个 DP instance 将 participants 映射为 actual domain ranks，用 `tests.test_af_config` 覆盖 participant 越界/重复、DP rank 重映射、缺 fabric/lane、operation 差异、独立 fabric 及显式共享 AF fabric lane
- [x] 2.6 迁移现有 `TP*EP*DP` sample configs与test fixtures到 `EP=1` 或 `EP=TP*DP`，补充两个命名清楚的合法样例；加载全部 AF JSON并断言无样例仍依赖额外EP-only workers
- [x] 2.7 实现彼此独立的全局 round-robin A owner 与 F service replica 选择器；用连续 blocks 验证两侧 DP 独立轮转、`EP=1` Routed work 随 F service replica执行、rank-reuse Routed work 可跨整个 F world 且不把全 world 锁成单一 service replica

## 3. 重建 runtime rank、内存与 ownership plan

- [x] 3.1 重构 F runtime state编译，使每个基础 TP/DP rank只有一个 physical state，rank-reuse EP只添加membership view，`EP=1`保留per-replica Expert TP group；用 `tests.test_af_runtime_state` 验证 `TP=2,DP=2,EP=1/4` 都恰好产生四个 states且容量不重复
- [x] 3.2 按逐层 Dense/MoE metadata编译 A Attention/Router 与 F Dense/Routed/Shared component residency，并以stable component identity去重；用 runtime component tests验证Dense层无Router、A不持有F FFN权重、Shared placement与同rank多membership只记一次
- [x] 3.3 修正 runtime memory公式：SwiGLU使用gate/up/down三矩阵，MHA/GQA/MQA使用真实Q/K/V/O dimensions，权重与KV分别使用对应dtype；用 `tests.test_af_runtime_state` 的手算oracle验证component bytes、KV bytes/block和OOM判断
- [x] 3.4 在 runtime plan 中加入稳定 A DP request owner、以 global hidden interval 表示的 A/F TP tensor ownership、`sharded`/`replicated`/`partial` source layout、canonical replicated producer、residual residency、F partial destination和必要 layout transition；用 ownership tests 验证 `A_TP/F_TP=2/4` 与 `4/2` 的区间映射、每份 partial 唯一返回一个 A tensor owner且非owner没有隐式 activation
- [x] 3.5 扩展预运行一致性检查和runtime summary，输出EP mode/group、effective dtype roles、component formula/source、ownership/layout与collective plan；用序列化round-trip tests验证summary可重建rank数、bytes和group membership，并验证冲突在event loop前失败

## 4. 用逐 token assignments 取代 histogram 执行

- [x] 4.1 新增不可变 `RoutingAssignment` 与逐 `(phase, decode_step, layer)` 的 `LayerRoutingSnapshot`，保留request/token identity、A DP/TP owner、Expert id和routing weight，并只从assignments派生histogram；用unit tests验证 `T*K` edges、weight/owner保留和确定性聚合
- [x] 4.2 修改 synthetic balanced router，让它对整个 layer block连续生成assignments而不是逐request重置；用四个Decode tokens、`top_k=2`、八个Experts的回归测试验证八个Experts各获得一次assignment
- [x] 4.3 接入外部routing数据校验：assignments与histogram同时存在时必须完全一致，只给histogram时只允许统计而不得驱动正式ownership combine；用missing-owner、count mismatch和跨step/layer cache污染tests验证早失败
- [x] 4.4 修改 AF block/build/submit流程，在执行前为每个MoE layer冻结独立routing snapshot，并让Dense layer不创建routing state；用多层block tests验证两个MoE layers不复用snapshot且Prefill/Decode caches隔离

## 5. 生成逐层 Dense/MoE DAG 与 endpoint legs

- [x] 5.1 扩展 `RankLocalWork`、`ExecutionNode` 与 `AFExecutionDAG`，表示original tokens、assignments、owners/layouts、Dense/Routed/Shared/Combine/Residual/Cast stages、logical stage和多个terminal nodes；用DAG validation tests验证重复id、缺predecessor和错误terminal集合被拒绝
- [x] 5.2 实现Dense layer builder：A Attention/collective、A-to-F legs、F Dense TP compute/collective、F-to-A owner legs、A residual/layout；用 `tests.test_af_parallel_planning` 验证Dense trace plan不含Router、Routed/Shared Expert或weighted combine
- [x] 5.3 实现MoE layer builder：A Router后按assignments生成Routed branches，按原始tokens生成一个Shared TP branch，并在A owners上生成weighted combine、shared sum与一次residual；用DAG workload tests验证Routed=`T*K`、Shared/Residual=`T`且分支可并行
- [x] 5.4 将 A-to-F 建模为一个 logical stage 下的 unique endpoint legs，按 source/target global interval 交集生成 payload；`sharded` 只发交集、`replicated` 只由 canonical producer 发、`partial` 保留所有数学贡献，每条 leg 只等待自己的 A producer；用 payload tests 验证同 slice/同 endpoint 只计一次、fan-out endpoint 分别计费且 F branch 无全 stage barrier
- [x] 5.5 将 F-to-A 建模为一个 logical stage 下的 owner-specific legs，按 source/target global interval 交集和目标 A TP layout 切分完整 hidden output，保留同一 shard 的不同 EP partials且每条 leg 只等待自己的 F producer；用 `A_TP/F_TP=2/4`、`4/2`、`top_k=2` tests验证 owner 精确收取所需 slices/partials
- [x] 5.6 在每个A owner收齐partials后生成本地weighted combine/residual，并在下一层需要不同layout时生成显式A collective；用dependency tests验证下一层不能在最后一个必要owner/collective完成前开始，也不能因A leader或iteration barrier提前开始
- [x] 5.7 当 transfer dtype 与 activation dtype 不同时生成发送 encode 和接收 decode compute nodes；encode/decode 去重键必须包含 stable tensor item id、slice interval、worker、stage和dtype pair，用 fan-out fixed-provider tests验证同一item一次source encode、每个target一次decode、每条physical leg一次transfer，并验证不同token或不同Top-K partial即使interval相同也不互相去重，dtype相同时不生成虚构cast
- [x] 5.8 让collective node采用配置的真实operation、resolved participants、fabric与resource lane而不是通用 `tp_collective` 或AF link fallback；用planning tests验证all-reduce/all-gather/reduce-scatter产生不同queries/resources，且full-duplex共享fabric时不再默认占用`a_to_f` lane

## 6. 执行、资源竞争与 trace

- [x] 6.1 修改 executor按layer metadata调用Dense/MoE planner，并等待DAG全部A-side terminal nodes后才完成layer/block；用 `tests.test_af_executor` 验证多层依赖、单次request推进与最晚block决定iteration makespan
- [x] 6.2 让每条 endpoint leg 和 collective 独立申请已解析的 `(resource_group, resource_lane)`，logical stage只聚合展示而不充当 barrier；用 `tests.test_af_parallel_execution` 验证 producer级重叠、full/half-duplex、独立collective fabric不与AF竞争，并以A-to-F、F-to-A、collective同时ready的oracle验证collective只与显式lane竞争、容量为一时正确serialization与queue time
- [x] 6.3 为所有新compute stages与communication operations构造typed provider queries，并保持provider只返回service、executor只产生queue；用stub provider tests验证每个logical operation恰好查询一次且negative/NaN/infinite/missing estimate明确失败
- [x] 6.4 扩展trace事件，记录layer type、assignment/original count、owner/layout、logical stage/leg endpoints、payload breakdown、effective dtype、participants、resource group/lane、ready/start/end、queue/service和provenance；用 `tests.test_af_trace` 验证trace round-trip可重建owner join、资源利用率和makespan
- [x] 6.5 增加event-loop前后的执行不变量检查，覆盖assignment唯一消费、payload去重、Shared每token一次、F无residual、A combine输出仍为原始token数和terminal完整性；用故意破坏plan的tests验证在静默低估时延前失败

## 7. 完成 operator-data 与离线 analytical 语义

- [x] 7.1 升级operator-data schema/generator/package version，使GEMM/MoE keys同时保留weight/activation dtype，Attention额外保留KV dtype，communication保留方向dtype、operation和endpoint信息；用package schema tests验证新package round-trip以及旧/缺字段package给出迁移错误
- [x] 7.2 将compute provider拆成互斥的Attention、Router、Dense FFN、Routed Expert、Shared Expert、Weighted Combine、Residual、Encode/Decode和Layout compositions；用 `tests.test_operator_data_latency_provider` 验证breakdown覆盖完整且没有重复residual、communication或Top-K
- [x] 7.3 让Routed Expert按逐Expert assignment workload与当前Expert TP local shape查询，让Shared Expert按原始tokens和扩大的shared intermediate查询，让Combine保留输入edges与输出token数；用skew、`top_k=2`和多Shared Expert fixtures验证query keys与service sums
- [x] 7.4 让 transfer provider 逐 endpoint leg 估算 payload 和 service，让 collective query 携带每 rank logical input/output bytes、all-reduce/all-gather/reduce-scatter、participants、fabric与resource lane，并仅由 provider 应用一次算法 wire-volume 系数；用 provider tests 验证 logical bytes 不被 planner 预放大、wire bytes 不被重复放大、service query不含queue且不再要求独立 EP dispatch/combine records
- [x] 7.5 修正离线 analytical generator formulas并与runtime ledger共用exact model shape：SwiGLU三矩阵、真实GQA/MQA Q/K/V/O、每request Decode context和sliding window；用手算FLOPs/bytes和package-generation tests验证 `[1,9]` contexts逐请求聚合、window cap、公式来源进入版本化records以及MLA/MFA明确不进入该公式
- [x] 7.6 收紧runtime lookup规则，分别限制GEMM、Decode Attention、Routed/Shared Expert及elementwise插值轴，不允许跨operation、dtype或多维插值，也不允许exact/interpolation miss后调用analytical runtime fallback；用exact/interior/out-of-range/mixed-role/missing tests验证确定性匹配与完整错误

## 8. 集成、迁移与端到端验收

- [x] 8.1 在engine/benchmark入口接通逐层model metadata、runtime plan、五类dtype和新providers，并在结果中标记`decode_af_rank_local`与`prefill_aggregate`；用Decode-only集成测试验证Prefill走现有完整aggregate path且Decode进入新DAG
- [x] 8.2 更新示例配置、operator manifests/packages和用户文档，说明两种EP模式、F worker计数、dtype归属、collective fabric/lane及breaking migration；通过加载全部示例和运行文档中的最小Decode-only命令验证内容可执行
- [x] 8.3 重写deterministic DAG oracles，覆盖Dense+MoE多层、多block overlap、Shared Expert、owner return、rank reuse、dtype cast和collective contention；逐项从trace重建critical path并断言makespan而非对rank service直接求和
- [x] 8.4 构建generated operator-data端到端oracle，覆盖package exact/interpolated/missing query、离线公式record provenance、endpoint payload与资源queue；运行 `python -m unittest tests.test_operator_data_latency_provider -v` 并确认所有service events只来自package record或合法插值且可追溯
- [x] 8.5 为既有 AF-Prefill 路径增加 compatibility adapter，使其适配公共 rank、dtype、query 与 fabric 契约而不采用新的 Decode rank-local DAG；用迁移后的合法 AF-Prefill fixtures 验证继续走现有 AF block/planner，且 timing/request/KV progression 不因公共类型迁移而中断
- [x] 8.6 增加兼容回归，覆盖 Decode-only AF 下正常 Prefill、既有 AF-Prefill capability仍可选择、`af.enabled=false`、request/KV只推进一次以及 trace 开关不改变 timing；运行相关 AF executor/engine 测试并比较开启与关闭 trace 的 makespan
- [x] 8.7 运行 `python -m unittest tests.test_af_config tests.test_moe_expert_parallelism tests.test_af_runtime_state tests.test_af_parallel_planning tests.test_af_parallel_execution tests.test_af_trace tests.test_af_executor tests.test_operator_data_latency_provider -v`，修复全部定向回归并记录测试数与结果
- [x] 8.8 运行 `python -m unittest discover -s tests -v`、`python -m compileall TokenSim`、`openspec validate complete-decode-af-execution-semantics --strict` 和 `git diff --check`，确认全量测试、语法、规范与补丁格式全部通过后再进入verify/archive
