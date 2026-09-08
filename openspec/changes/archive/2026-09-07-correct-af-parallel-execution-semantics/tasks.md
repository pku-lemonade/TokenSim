## 1. 锁定配置与拓扑语义

- [x] 1.1 为 AF `PP>1`、A 域 `EP>1`、缺失/冲突 rank 坐标和不完整 expert placement 增加失败测试，并验证这些测试在旧实现上能暴露错误接受行为
- [x] 1.2 为 `DomainRankMapping` 增加显式 `ep_rank`，实现 capability-aware 的 required rank count 与 `(dp_rank, ep_rank, tp_rank, pp_rank=0)` 坐标解析，并通过 A/F 单 rank及多 rank 坐标单元测试
- [x] 1.3 将每个 AF `LogicalInstanceConfig` 归一化为一个 DP replica，校验 instance 数、每 replica rank coverage、domain-rank 唯一性及 physical worker 一对一映射，并验证不再要求每个 instance 重复覆盖完整 DP world
- [x] 1.4 在所有 AF 运行入口前拒绝 `PP>1`，限制 A domain 的 EP 为 1，并验证错误包含 domain、实际并行值和支持边界
- [x] 1.5 为 F domain 构建每 DP replica 完整复制的 expert placement view，校验 expert coverage、EP rank 和 physical worker 引用，并通过 linear/round-robin placement 测试
- [x] 1.6 迁移仓库中的多 rank AF fixtures 与示例配置到新 replica/rank 坐标，验证配置测试通过且旧单 rank AF 与 `af.enabled=false` 配置仍可加载

## 2. 实现确定性 DP 选择与 A 侧所有权

- [x] 2.1 实现按 `(phase, domain)` 维护 cursor 的全局 round-robin replica selector，并用 2 replicas/4 blocks 测试验证稳定序列 `0,1,0,1`
- [x] 2.2 在 request 首次 AF admission 时持久记录 A DP owner，并验证后续 Decode iterations 复用同一 owner、不会无 KV transfer 地移动 request
- [x] 2.3 扩展 block 构建流程，使一个 block 只包含同一 A owner 的 requests，并在提交顺序中预先选择 F DP replica后生成不可变 `AFBlockPlacement`；验证运行时不再使用 `block_id % instance_count`
- [x] 2.4 增加并发 DP 测试，验证一个 replica 的长任务不会锁定其他 replica 的 compute ranks，且相同输入多次运行得到相同 placement

## 3. 构建逐 rank ParallelExecutionPlanner

- [x] 3.1 定义不可变的 local-work、`ExecutionNode` 和 `AFExecutionDAG` 数据对象及结构校验，验证 node id 唯一、predecessor 存在、图无环且一个 compute node 只绑定一个 physical worker
- [x] 3.2 实现 A 侧 TP sharding policy，为每个 TP rank生成本地 shape/workload 和语义所需 collective，并验证 `TP=1` 无 collective、`TP=2` 不重复完整 shape
- [x] 3.3 将 Router/Top-K 固定到选定 A replica 的 `tp_rank=0` leader，在 block/layer planning context 中保存不可变 routing histogram，并验证 Router 依赖 A TP 汇合且先于 A-to-F
- [x] 3.4 将 routing histogram 与每 replica expert placement 聚合为 per-expert/per-EP-rank workload，只为非零目标生成 expert 分支，并验证 10/2 不均衡负载不会被平均化且空 rank 无非零 compute
- [x] 3.5 为每个命中 expert shard展开 F 侧 TP rank compute 和必要 TP collective，使 EP combine 依赖全部必要分支，并验证其 ready time 由最慢分支决定
- [x] 3.6 实现 A-to-F、F-internal EP dispatch/combine 和 F-to-A 节点及统一 payload 计算，验证每层恰好一次双向跨域 transfer、域内通信不伪装为 AF transfer 且 Top-K 不被重复乘算
- [x] 3.7 用 planner 单元测试覆盖单 rank、TP-only、DP-only、EP-only 与 TP+DP+EP 图的 nodes、participants、resources、payload 和依赖拓扑

## 4. 收紧 Provider 查询边界

- [x] 4.1 扩展 `AFComputeQuery`，加入 operation、domain、DP/TP/EP 坐标、logical/physical identity、local shape、local token workload 与 expert shard 元数据，并更新 stub provider 后验证旧完整 workload 查询不再进入 AF 多 rank 路径
- [x] 4.2 更新 operator-data compute provider，使 A、Router/Top-K 和 Expert composition 使用各自局部查询且互不重叠，并验证 breakdown 恰好计时一次及缺失 local shape/workload 时明确失败
- [x] 4.3 扩展 typed communication query，明确区分 `af_a_to_f`、`af_f_to_a`、`tp_collective`、`ep_dispatch` 和 `ep_combine`，并验证 provider 保留 planner participants/payload、不会生成额外 transfer 或 queue time
- [x] 4.4 对 planner payload 与 provider 返回 payload 增加一致性校验，并用不匹配测试验证系统带 node/query context 失败而非静默采用其中一个值
- [x] 4.5 更新 operator-data 与固定时延 provider 测试，覆盖 rank-local exact/interpolated lookup、不同 expert workload、全部通信 operation 和 provenance 序列化

## 5. 切换 DAG Executor、资源与 Trace

- [x] 5.1 重构 `AFResourceRegistry`，在启动前解析 compute slots、AF boundary、A/F TP 与 F EP fabric，移除 FFN pool 任意选 worker和运行时动态创建通信资源 fallback，并验证缺失资源配置提前失败
- [x] 5.2 将 `AFExecutor` 切换为逐层 DAG runner：等待 predecessors、稳定排序申请资源、调用单 node provider、推进时间并释放资源；用 fan-out/fan-in 测试验证 rank 分支可并行且 join 等待最晚前驱
- [x] 5.3 保留并强化 iteration 失败传播、取消和资源释放逻辑，验证任一 node/provider 失败后无悬挂 process、无泄漏 resource 且错误包含 iteration/block/layer/node context
- [x] 5.4 扩展 `AFTraceEvent` 记录 node/predecessor、replica/group/rank、local work、participants 和全部时间/来源字段，并验证 trace 开关不改变调度结果
- [x] 5.5 实现 trace invariant checker，验证依赖时间、join ready time、资源容量、每层两个跨域节点以及 payload/provider 一致性，并用故意损坏的 trace 测试每类 invariant
- [x] 5.6 更新 makespan、barrier wait 与 utilization 汇总，使其从 DAG 最终节点和 service 区间重建，并通过多 rank 手算时间线测试验证不对 rank duration 求和

## 6. 端到端语义验收与回归

- [x] 6.1 增加 DP 端到端 oracle，验证 A owner 稳定、F 全局 round-robin、replica 独立并发和确定性 trace
- [x] 6.2 增加 TP 端到端 oracle，验证 per-rank local shapes、必要 collective、Router 等待 TP 汇合及 `TP=1` 兼容路径
- [x] 6.3 增加 EP 端到端 oracle，验证 A 侧 Router、F 内部 dispatch/combine、expert placement、非均匀 workload、空 rank 和 straggler 对 makespan 的影响
- [x] 6.4 增加 `TP=2, DP=2, PP=1, EP=2` 的组合 oracle，逐 node 断言运行流程、每层一次 A-to-F/F-to-A、资源竞争和可手算 makespan
- [x] 6.5 运行 AF config/executor、parallel、MoE、operator-data 相关测试，修复回归并验证 `af.enabled=false`、单 rank AF、request progression 和 KV accounting 保持不变

## 7. 文档、迁移与最终验证

- [x] 7.1 更新 AF 配置和执行文档，给出新 rank 坐标、DP replica、TP/EP fabric、`PP=1` 限制及旧多 rank配置迁移示例，并验证示例能够被实际配置解析器加载
- [x] 7.2 运行项目格式化、静态检查和完整测试套件，记录无法运行的环境依赖，并确保本 change 相关测试全部通过
- [x] 7.3 运行 `openspec validate correct-af-parallel-execution-semantics --strict` 并逐条对照 proposal/specs/design/tasks，确认无遗漏需求、重复 AF 通信或未验证任务
