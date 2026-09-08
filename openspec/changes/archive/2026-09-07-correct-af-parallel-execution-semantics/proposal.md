## Why

当前 TokenSim 虽然允许配置 TP、PP、DP 和 EP，也能建立 logical rank 与 worker 的映射，但这些配置没有完整转化为各 rank 的本地计算、数据依赖和通信事件：多个 DP rank 可能被锁成一个实例，TP/EP 缺少真实分片负载，PP 没有 stage 执行语义，AF 数据流还把 Router 放在 F 侧并重复模拟跨域通信。因此，现有结果会让“配置合法”看起来等同于“并行语义已运行”，但 trace 和 makespan 并不能代表该并行方案。

## What Changes

- 引入统一的并行执行规划层，根据阶段、并行配置、rank placement 和 MoE routing result 生成逐 rank 执行 DAG；Executor 只调度该 DAG，Provider 只估算本地算子或通信事件的 service time。
- 让 TP 真正产生每个 rank 的本地 shape、计算负载和必要 collective，并在 trace 中记录参与 ranks、局部 payload 与同步依赖。
- 让 DP rank 作为互不锁步的独立副本运行；第一版在可用副本间采用确定性的全局 round-robin 选择，并分别维护资源竞争和完成时间。
- **BREAKING**：AF 第一阶段不实现 Pipeline Parallel 执行；任何参与 AF 的 execution domain 配置 `PP > 1` 时 MUST 在仿真开始前被拒绝，而不是接受后按非 PP 路径运行。
- 将 Router/Top-K 固定在 A 域，并把单层 AF 数据流改为：选择 DP replica、A 侧 TP 计算、A 侧 Router/Top-K、一次 A-to-F、F 域内 EP dispatch、逐 expert rank 计算、F 域内 combine、一次 F-to-A；删除把 EP dispatch/combine 再建模为跨 A/F 通信的重复路径。
- 让 EP 根据 Top-K routing histogram 和 expert placement 生成目标 expert rank、per-rank token workload、dispatch/combine 参与者与 payload，并用最慢必要分支决定该层 expert 阶段完成时间，从而显式表现负载不均与 straggler。
- 统一计算、域内 collective、域间 transfer 的资源和 trace 语义，使每个事件都能追溯到 planner 节点、logical rank、physical worker、DP replica、TP/EP group、queue time 和 service time，并可从逐 rank 事件重建 makespan。
- 保持 `af.enabled=false` 与单 rank AF 配置的既有行为；对当前不能真实执行的并行组合明确失败，不再静默降级或仅做配置校验。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `af-separation-config`：并行参数必须对应可执行语义；AF 第一阶段拒绝 `PP > 1`，并明确 DP replica、TP group、EP group、expert placement 及 rank/worker 映射的校验契约。
- `af-execution`：从按 worker pool 的粗粒度阶段链改为 planner 生成的逐 rank DAG；规定 DP 选择、A 侧 Router/Top-K、TP collective、F 域内 EP dispatch/combine、straggler 汇合和单次双向 AF transfer 的执行顺序。
- `operator-data-latency-provider`：Provider 查询改为接收 planner 给出的 rank-local shape、token workload、参与 ranks 和 payload，只返回对应本地计算或单次通信的 service time，不决定拓扑、路由、排队或同步。

## Impact

- 主要影响 `TokenSim/af/planning.py`、`TokenSim/af/executor.py`、`TokenSim/af/resources.py`、`TokenSim/af/latency.py`、`TokenSim/operator_data/providers.py`、`TokenSim/moe/routing.py`、`TokenSim/moe/placement.py`、`TokenSim/latency/roofline.py` 以及并行与 AF 配置解析代码。
- AF planner、executor、provider 和 trace 之间的数据结构与调用契约会调整；显式配置 `PP > 1` 的 AF 场景将由当前的错误接受改为启动前失败。
- 测试需要新增 TP local-shape/collective、DP round-robin 与副本独立性、EP 不均衡与 straggler、Router 所属域、跨域 transfer 不重复、非法 PP 配置以及逐 rank trace/makespan 重建场景。
- 不新增或伪造 vLLM 的 AFD 实现；调度、MoE placement 和通信抽象借鉴 vLLM，AF 数据流以 Step-3 的 Attention-FFN 分离语义为准。
