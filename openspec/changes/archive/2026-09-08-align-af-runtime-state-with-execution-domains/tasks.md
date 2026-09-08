## 1. 锁定现有错误与 KV 状态边界

- [x] 1.1 增加失败测试，复现全局 `TP=1`、A domain `TP=2` 时只有 leader 按 TP=1 计算 KV/模型内存的现状，并验证测试在修复前确实失败。
- [x] 1.2 盘点 block manager 的 allocate、append、free、swap-in、swap-out、fork/ref-count 和 preemption 状态转移，为每类转移增加保持现有逻辑 block 语义的契约测试。

## 2. 编译唯一 AF Runtime Plan

- [x] 2.1 定义不可变的 rank binding、replica runtime 和 runtime-plan 数据结构，实现从 execution domains 一次编译 DP/TP/EP groups、leader 与 physical mappings，并用 A `TP=2,DP=2`、F `TP=2,DP=2,EP=2` 单元测试逐 rank 断言。
- [x] 2.2 实现 domain-rank binding 与 physical-device state 分离，按 physical identity 聚合跨阶段引用，并验证同一 physical worker 的共享组件不重复计数、不同组件正确累加。
- [x] 2.3 让 `LLMEngine` 在创建 AF workers 前编译 runtime plan，为每个 A replica 选定唯一 scheduler leader，并验证 `af.enabled=false` 时不创建该 plan。

## 3. 实现 Rank-local 模型与 KV 容量计算

- [x] 3.1 实现使用 A-domain TP 的 KV-head 分片/复制策略，输出 global head ids、replica factor 和 bytes per block，并用 `8 heads/TP2`、`1 head/TP2` 以及不合法整除组合测试验证。
- [x] 3.2 实现 AF 组件所有权与权重字节计算，覆盖 A Attention TP shard、A-leader Router、A embedding/lm-head、F routed expert placement 以及 F shared/dense FFN 复制，并用小型手算模型逐 rank 断言 component bytes。
- [x] 3.3 为每个 physical worker 建立去重的模型/KV 内存账本与 local GPU/CPU block 容量，并验证模型权重超过设备容量时在模拟前报出含 domain、rank、worker 和字节明细的错误。
- [x] 3.4 对多 rank AF 所需的 shape、dtype 或 component ownership 元数据实现显式失败，并验证不会静默回退到“整模型字节除以 rank 数”。

## 4. 实现 A Replica 组级 KV Cache

- [x] 4.1 实现 logical block 到每个 A TP rank local block 的 mapping 和 rank-local block pools，并验证 `TP=2` 的一次逻辑 allocate 恰好产生两个归属正确 rank 的 local shards。
- [x] 4.2 实现按稳定 rank 顺序的 reserve/commit/rollback 事务，并用“第一 rank 成功、第二 rank 无容量”测试验证所有 rank 的 block 数恢复到失败前状态。
- [x] 4.3 将 free、append/fork 与 reference-count 变化接入组级 mapping，并验证释放逻辑 block 后无孤立 shard、共享 block 在最后一个引用释放前仍存在。
- [x] 4.4 将 swap-in、swap-out 和 preemption 接入组级事务，并验证 GPU/CPU 两类 shards 在成功与中途失败后都保持一致。
- [x] 4.5 向 scheduler 暴露 `min(rank-local usable blocks)` 的 replica GPU/CPU 逻辑容量，并用 ranks 容量 100/80 的异构用例验证 scheduler 看到 80 而不是 100 或 180。

## 5. 接入 Scheduler、KV Connector 与 Planner

- [x] 5.1 使每个 A leader 的 scheduler/block manager 使用所属 replica 的逻辑 KV 视图，保留一个 replica 一份 request queue 和逻辑分配，并验证非 leader TP ranks 不会重复调度请求。
- [x] 5.2 让现有 KV connector 在逻辑 block 层面与 replica KV 视图交互，并验证一次逻辑目的端变化会创建/释放完整 TP shard 集且不额外增加 AF transfer 事件。
- [x] 5.3 将 `ParallelExecutionPlanner` 的 replica、group、physical endpoint 与 expert-residency 查询切换到同一 runtime plan，并验证现有 TP/DP/EP DAG 节点和 A→F→A 依赖不变。
- [x] 5.4 让 F compute ranks 关联实际 physical memory state 与 expert residency，并验证 expert compute 只能在已驻留该 expert 且模型可装载的 rank 上计划。

## 6. 增加语义校验与可观测输出

- [x] 6.1 实现启动前静态校验，覆盖 leader 唯一性、rank/physical binding、TP/EP group、KV-head 策略、component ownership 与 expert placement，并用故意损坏的 runtime plan 测试每类错误都在事件循环前失败。
- [x] 6.2 实现 logical/local block mapping、rank 容量和 request-owner 的动态 invariants，并用缺失 shard、孤立 shard、超容量和 owner 冲突用例验证带 replica/rank 上下文失败。
- [x] 6.3 在结果中输出带 schema/version 的 runtime summary，包含每 physical worker 的 component bytes、KV bytes/block、local capacity/usage 和每 A replica 的 logical capacity/usage，并通过序列化测试验证可逐 rank 对账。

## 7. 组合验收与回归

- [x] 7.1 增加全局 `TP=1`、A `TP=2` 的端到端 oracle，验证两个 A ranks 均有正确 KV/模型状态、scheduler 只运行一份且 planner DAG 仍使用两个 ranks。
- [x] 7.2 增加 A `TP=2,DP=2` 组合 oracle，验证两个 DP replicas 的容量与 shards 独立，request 后续 Decode 稳定复用原 owner，无 KV transfer 时不跨 replica 移动。
- [x] 7.3 更新 `TP=2,DP=2,EP=2` 端到端 oracle，同时断言 A KV shards、A/F component bytes、F expert residency、现有计算 DAG 和每层一次 A-to-F/F-to-A。
- [x] 7.4 增加 AF `TP=1` 退化路径与 `af.enabled=false` 回归，验证前者的 logical/local 容量一致，后者的现有全局 `ParallelConfig`、CacheConfig、request progression 和 timing 结果不变。
- [x] 7.5 更新 AF/KV/并行文档和示例输出，运行 AF、block manager、KV connector、parallel、MoE、operator-data 与完整测试集，然后执行 `openspec validate align-af-runtime-state-with-execution-domains --strict` 并记录验证结果。

## 8. 恢复 A 域 Router 与旧版通信顺序

- [x] 8.1 更新 Planner 失败测试，锁定 Router 必须位于所选 A leader、A→F payload 按 routed token count 计算，以及 A→F 之后存在 F 域 dispatch/combine nodes。
- [x] 8.2 修改 Planner DAG：Router 紧接 A TP collective 并在 A leader 上执行；A→F 等待 Router，之后保留 Dispatch→Expert→Combine，并保持 F TP collective 与 F→A 依赖正确。
- [x] 8.3 修改 AFRuntimePlan 组件所有权与校验，使 Router 权重从 F leader 迁回每个 A DP replica leader，并更新逐 rank 内存断言。
- [x] 8.4 更新 AF/operator-data 文档与组合 oracle，运行 Planner、Executor、runtime-state、operator-data 和完整测试集，再执行 OpenSpec 严格校验。

## 验证记录

- 2026-09-08：`python -m unittest discover -s tests -t .`，226 个测试全部通过。
- 2026-09-08：`python -m black --check TokenSim/af/runtime.py TokenSim/block/replica_kv.py TokenSim/llm/llm_engine.py TokenSim/af/planning.py TokenSim/af/executor.py tests/test_af_runtime_state.py`，格式检查通过。
- 2026-09-08：`openspec validate align-af-runtime-state-with-execution-domains --strict`，严格校验通过。
- 2026-09-08：Router/EP 边界调整后，Planner、Executor、runtime-state、operator-data 和 trace 组合测试共 82 个全部通过。
- 2026-09-08：`python -m unittest discover -s tests -t .`，228 个测试全部通过。
- 2026-09-08：本轮涉及的 6 个 Python 文件通过 `python -m black --check`。
- 2026-09-08：Router/EP 边界调整后再次执行 `openspec validate align-af-runtime-state-with-execution-domains --strict`，严格校验通过。
- 2026-09-08：恢复 A 域 Router 与 routed A→F payload 后，Planner 8 项、Executor 25 项、runtime-state 24 项、operator-data 21 项和 trace 4 项测试全部通过。
- 2026-09-08：恢复 A 域 Router 后运行 `python -m unittest discover -s tests -t .`，228 个测试全部通过；6 个相关 Python 文件通过 Black 检查；OpenSpec strict 校验通过。
- 2026-09-08：修正 AF 通信与 F 侧计算量语义后，Planner、Executor、MoE 和 operator-data focused 回归共 74 个测试全部通过。
- 2026-09-08：运行 `python -m unittest discover -s tests -t .`，233 个测试全部通过。
- 2026-09-08：使用 `data/operator_data/af_e2e_input.yaml` 成功生成并校验 `dataset=example-v1, schema=1` 的 operator-data package。
- 2026-09-08：本 change 涉及的 29 个 Python 文件通过 Black 检查，`git diff --check` 与 OpenSpec strict 校验通过。

## 9. 修正 AF 通信与 F 侧计算量语义

- [x] 9.1 增加失败测试，证明 routed histogram 在 Expert analytical 公式中被 Top-K 重复放大，并验证 F `TP=2` 未缩小 MoE/Dense rank-local compute。
- [x] 9.2 将 `moe.num_tokens` 固定为 routed assignments，按逐 Expert workload 查询并聚合 rank latency；使 MoE 和 Dense FFN 的 analytical/operator keys 使用 F TP 本地 shape。
- [x] 9.3 合并 A→F 与 EP dispatch、F→A 与 EP combine，使每层只搬运两次 routed payload，并验证 endpoints 覆盖 A leader 与 active F Expert ranks。
- [x] 9.4 对外部 histogram 增加 `sum == step_tokens * top_k` 校验，并按 Prefill/Decode 隔离 normalized cache。
- [x] 9.5 更新 AF、并行与 operator-data 文档和样例，增加 `top_k=2, TP=2, EP=2` 端到端 oracle，运行 focused/full tests、Black 与 OpenSpec strict 校验。
