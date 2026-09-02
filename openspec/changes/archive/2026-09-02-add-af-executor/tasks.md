## 1. AF 数据契约与端到端夹具

- [x] 1.1 建立 `TokenSim/af/` 包并实现 immutable `AFBlock`、`AFIteration`、compute/transfer query 与 `AFServiceEstimate`，通过从真实 AF cluster JSON 构造 iteration 的测试验证字段、秒单位和稳定 ID 可被后续 trace 使用。
- [x] 1.2 实现 AF block builder 按 scheduler 顺序和 `af_block_size` 连续分组，通过 Prefill/Decode 端到端夹具验证 5 个 requests 形成 2/2/1 blocks、request 不重复、Prefill token 求和、Decode token 等于 request 数且不拆分单个 Prefill request。
- [x] 1.3 实现阶段识别与 `phase_to_af_link` 运行时绑定，通过 Decode-only、Prefill-only 和两阶段真实配置测试验证只为所选阶段构造 AF iteration，未选阶段仍进入既有完整阶段路径。
- [x] 1.4 定义可注入 compute/transfer provider 协议及仅用于测试的 deterministic providers，通过入口级 fixture 验证查询携带 phase、block、layer、shape、设备与方向上下文，返回值携带明确 `test-stub` source_id 且未新增 AIConfigurator/InferSim/Parquet 运行依赖。

## 2. Engine 级计算与通信资源

- [x] 2.1 由规范化 worker groups 建立稳定的物理 worker ID、group membership 和共享 compute slots，通过真实配置启动测试验证 A 所有者、F-only worker 和复合 capability worker 均映射到唯一物理资源。
- [x] 2.2 实现 A worker affinity，使 block 的每层 A compute 只申请提交 iteration 的 source A slot，并通过两个 A workers 并发提交的时间线验证结果返回原 A 且各自资源身份不串换。
- [x] 2.3 实现带稳定 worker token 的 F pool 获取与归还，通过多个同时 ready blocks 的端到端测试验证容量不超过 F worker 数、worker ID 可追踪且同时间选择规则确定。
- [x] 2.4 按 `af_communication_resources` 为每个 resource key 只创建一个 SimPy resource，通过 full-duplex 配置验证两个方向键独立、half-duplex 配置验证共享键唯一且容量与配置一致。
- [x] 2.5 让完整阶段计算与 AF 计算共享同一物理 worker slot，通过部分阶段 AF 的并发测试验证复合 capability 设备不会重复服务，并通过 AF disabled baseline 证明零竞争 acquire/release 不改变模拟时间。
- [x] 2.6 将资源注册表提升到 Engine 生命周期并由所有 worker/iterations 共享，通过两个并发 iterations 或共享 resource group 的测试验证资源不会按 iteration 重建而绕过竞争。

## 3. AF block 状态机与流水

- [x] 3.1 实现通用 stage runner 的 acquire、ready/start/end、timeout 和 `finally` release，通过繁忙/空闲资源的事件断言验证 `queue_duration=start-ready`、`service_duration=end-start` 且总阶段时间为两者之和。
- [x] 3.2 实现单 block 单层 `A -> A-to-F -> F -> F-to-A` 链，通过 2/1/3/1 ms provider 的真实 SimPy 时间线验证依赖顺序和 7 ms 完成时间。
- [x] 3.3 将四段链扩展为逐层循环，通过单 block 多层测试验证下一层 A 的 ready time 不早于上一层 F-to-A end，且总时长可由每层事件重建。
- [x] 3.4 为 iteration 中每个 block 启动独立 SimPy process 并用 AllOf 建 barrier，通过两个 blocks 的时间线验证可流水重叠、iteration end 取最后 block 完成而非 block duration 求和。
- [x] 3.5 记录早完成 block 的 barrier wait，并通过不等长 blocks 测试验证 barrier 前没有 request 推进、barrier wait 不计入 compute/transfer service 或 block 自身完成时间。
- [x] 3.6 对 provider unsupported、负数、NaN 和 Inf 实现带 iteration/block/layer/stage 上下文的失败，通过失败后再次提交 iteration 的端到端测试验证 sibling processes 被终止、worker/link 资源全部归还且 requests 未推进。
- [x] 3.7 扩展 SimPy event queue 耗尽保护以报告未完成 AF blocks 和等待资源，通过构造不可完成事件的集成测试验证错误可诊断而不被当作正常 iteration 完成。

## 4. Trace 与时间汇总

- [x] 4.1 实现 append-only AF event record 和 recorder，使用单层真实执行验证 compute/transfer 事件包含 iteration、block、layer、stage/direction、resource、ready/start/end、queue/service、request/token 数、worker、source_id，且 transfer 包含 payload bytes。
- [x] 4.2 实现 block/iteration summary，通过多 block 时间线验证 makespan、block completion 和 barrier wait 可从原始 records 重算且 iteration 时长不等于 blocks 时长简单求和。
- [x] 4.3 实现按 service intervals 计算的资源利用率，通过含排队与 barrier wait 的测试验证 queue/barrier 不进入 busy time，重叠容量的利用率口径明确且可重算。
- [x] 4.4 实现 trace disabled 的 no-op recorder，通过相同输入 trace on/off 对照验证资源选择顺序、所有 block 完成时间、iteration makespan 和 request 结果完全一致。

## 5. LLM Engine 与 benchmark 接入

- [x] 5.1 在 Engine 初始化中区分 request-owning A workers、F-only tokens 与完整阶段 workers，并构造共享 AF executor，通过两阶段真实 cluster 配置验证 Prefill/Decode 调度池和 A/F group 均来自显式 capability/link 而非硬件名称。
- [x] 5.2 在 worker STEP 边界接入 AF iteration，同时保留 AF disabled 和未选阶段的原 `dynamic_batch` 路径，通过 Prefill-only、Decode-only、两阶段和关闭 AF 的完整请求生命周期测试验证分支选择。
- [x] 5.3 保持 request 推进与 scheduler update 在 executor barrier 之后的既有单一入口，通过多 block 多层测试对每个 request 计数，验证一次 iteration 恰好 step 一次且 stage/block 完成不 step。
- [x] 5.4 保持 KV Cache 和阶段间 KV transfer 所有权，通过相同请求的 AF/控制路径对照验证 AF block 不增加 BlockManager allocate/free 次数，F worker 不持有 KV，Prefill/Decode A groups 不同时仍使用既有 KV transfer 而非 AF link。
- [x] 5.5 将 benchmark 的固定“AF executor 尚未实现”拒绝改为 executor/provider/model metadata 预检，通过注入完整 runtime 的入口测试验证合法 AF 可运行，并通过缺失任一依赖的入口测试验证在产生 AF 事件前明确失败且不回退 Roofline/LLMCompass 完整 iteration 时延。
- [x] 5.6 支持按 `trace_enabled` 收集并输出 AF trace/result，同时保持现有非 AF 输出兼容，通过真实 benchmark 参数路径测试验证开启 AF 能取得结构化结果、关闭 AF 不新增 AF 事件字段或改变既有结果。

## 6. AF 端到端事件 Oracle

- [x] 6.1 建立从真实 cluster JSON、模型元数据、scheduler requests、deterministic providers 运行到最终 result/trace 的 E2E harness，并验证测试不绕过 Engine/worker 主流程、不直接调用内部 stage helper。
- [x] 6.2 固定单 block/双 block oracle：A=2 ms、A-to-F=1 ms、F=3 ms、F-to-A=1 ms、各容量为 1，断言单 block 7 ms、双 block 完全串行上界 14 ms、实际流水 makespan 10 ms，并逐事件验证合法重叠。
- [x] 6.3 增加 A 慢/F 快与 A 快/F 慢场景，通过 queue 和 service 时间线验证瓶颈分别落在 A/F 资源，并报告 `T_C<T_A`、`T_C<T_F` 与 micro-batch 数下界仅作辅助检查、事件时间线作为最终判定。
- [x] 6.4 增加 full-duplex 与 half-duplex 对照，验证 full 模式相反方向可独立重叠、half 容量 1 时相反方向不重叠且等待计入 queue duration。
- [x] 6.5 增加同 resource group 多 links/blocks 与不同 resource groups 对照，验证相同 key 跨来源竞争、不同 key 独立，通信 total 始终等于 queue 加 service。
- [x] 6.6 增加 1 个与多个 F workers 对照，验证并发服务数不超过 worker 数、扩容后的 makespan 非增，并能由 F service/queue intervals 解释变化。
- [x] 6.7 增加 Prefill-only、Decode-only、两阶段和不同 A groups 场景，验证 phase 选择、token_count、阶段间 KV transfer、逐层 AF link 和最终 request 生命周期均符合规范。
- [x] 6.8 增加 provider 缺失/unsupported/非法值、trace on/off 和 AF disabled 对照，验证失败原子性、计时不受 trace 影响及所有现有路径兼容。

## 7. 文档、回归与 OpenSpec 验证

- [x] 7.1 更新 AF 文档和配置示例的运行状态，说明 executor 已支持的阶段、四段链、资源等待、barrier、trace 字段、stub 只验证机制以及缺少正式 provider 时的失败行为，并核对不宣称物理时延准确。
- [x] 7.2 明确记录 operator-data 后续边界：本 change 不生成/加载 Parquet、不实现实测匹配/插值/公式校准，后续 provider 通过 query/estimate 接口接入且 `source_id` 与 match/composition metadata 不参与调度，并用依赖清单检查没有新增相关运行包。
- [x] 7.3 运行完整 AF E2E 矩阵，输出每个场景的 makespan、关键事件区间、queue/service 和通过条件，确认不是仅依赖 smoke 或 isolated unit tests 判定完成。
- [x] 7.4 运行仓库完整测试套件并对 Prefill、Decode、KV transfer、并行、MoE、trace timestamp 和 latency baseline 做回归比较，确认 AF disabled 行为无变化且 Windows 上 Linux-only backend 的测试替代方式仍被明确记录。
- [x] 7.5 运行 formatter、compileall、`git diff --check` 和 `openspec validate add-af-executor --strict`，确认代码格式、导入、空白与 proposal/spec/design/tasks 一致性全部通过。
