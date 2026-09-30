# AgentX 会话路由与 DP 同步

本次接入参考版本的 consistent-hash 路由，以及 DP 同步的 GPU forward 和完成状态检查。仍使用 HBM-only 管线；CPU 后端、原生算子计时、MTP 随机进度与 prefill 限流继续分开实施。误差标准已按用户更新为相对误差 ≤25% 可接受、≤10% 优先；此标准不会放宽身份、计数和引用守恒检查。

## 路由

参考 `vllm-project/router` 的 `v0.1.14`，tag revision 为 `0a6f74ec88c87b5477dac8e24466ab760b9f1f35`。本地 Python 移植了 FurcHash、MurmurHash64A 的 64-bit wrapping、160 个虚拟节点及有序环查找；保留 Apache 2.0 许可证和来源说明。源码校验值、原日志校验值及覆盖八个 rank 的观测向量保存于 [validation.json](../data/reference/agentx-router/validation.json)，重算入口为 `scripts/validate_agentx_router.py`。

参考后端 URL 是 `http://localhost:8889`，DP 逻辑节点是 `http://localhost:8889@0` 至 `@7`。URL 字符串参与 hash，不能随意用不同名字替代。Benchmark 将 correlation ID 别名成 X-Session-ID，实际路由 key 为 `header:x-session-id:<ID>`。原生回放为每个会话提供稳定 ID，独立 subagent 使用自己的 ID，父 ID 不参与其路由；批量请求逐个选择目标，再按 worker 分组发送。

完整历史日志中的 B200 32,098 条、B300 47,269 条 key→worker 映射均精确匹配，零差异。这一核对包含日志里的预热和测量路由消息；与先前仅按 benchmark 导出关联的请求口径不同。它验证了算法及 URL 环；模拟器中的会话 UUID 仍由自身回放实例派生，因此正常模拟的逐会话 rank 不保证与历史 UUID 完全一致。固定实测 worker 诊断仍需单独接入已核实的回放身份映射。

## DP 同步

依据只读 vLLM `30f5c01d4ebb` 的 `DPEngineCoreProc.run_busy_loop`、`_has_global_unfinished_reqs`、`GPUWorker.execute_dummy_batch` 与 `ParallelConfig.sync_dp_state`。每步各 rank 同时完成本地 forward，步长取所有本地耗时的最大值；空闲 rank 执行 dummy pass，不能缺席 EP 协作。Dummy 使用源码的 uniform decode query 长度 1+K，但不分配真实请求 KV、不产生输出、不写请求指标。

当前源码的完成状态同步间隔默认 **16 步**，旧计划中的 32 步属于此前版本记录。完成状态在第 1 步和每 16 步检查；其余步即使所有真实请求刚结束，仍执行 dummy，直到完成共识后停止波次、清零共享 step counter。GPU 的逐步同步与这个 CPU 完成状态检查分开处理。CPU 控制 all-reduce 传输两个 int32，其额外时间留给框架开销校准，没有再次按 GPU collective 收费。

输入可在 kernel 期间到达，参与下一步；取消在当前同步 kernel 结束后释放物理引用。低优先级的零时间 `StepBoundary` 等待同一模拟时间的正常回调全部处理，再调度下一步，避免完成回调发出的下一请求被无故推迟一个 dummy step。全局 token 数与各 rank token 数传给 worker，供后续原生 MegaMoE 成本模块使用；现有通用耗时后端仍按本地形状估算，不能将其视为全局融合 EP 实测。

CLI 增加 `--router_policy`、`--router_backend_url`、`--dp_lockstep`、`--dp_sync_interval`。完整原生 AgentX 默认选择 consistent-hash 和 DP lockstep；其他负载保留现有默认行为。PP 同步、CPU 异步加载与保存将在相应后续阶段实现，不静默混入当前 DP 模式。

## 验证

12 个行为用例覆盖：不等长 rank 的最大值步长、空闲 rank dummy 成本、完成共识与波次重置、同时间完成回调发出新请求、kernel 中到达的请求、kernel 中取消、统计清零不改变步序，以及会话稳定性、独立子会话、批量路由、历史观测向量。`python -m unittest discover -s tests -v`：**355 项通过，82.472 秒**。

结果保存于 `results/agentx-dsv4-full/routing-dp/20260930-consistent-lockstep/{b200,b300}/`：使用全部 393 条 trace，固定负载种子 42、模拟种子 0，完成预热后发请求 3,600 秒，随后 30 秒宽限与最多 10 秒取消收尾。每分钟缓存窗口和最终结果记录每 rank 的真实／dummy 批次、计算及等待时长、波次数和 CPU 完成状态检查次数。模型与 HBM 预算沿用上一阶段，所有未匹配配置继续标注，不能将基线吞吐直接作为正式精度验收。

| 完整窗口结果 | B200 c196 | B300 c384 |
|---|---:|---:|
| 预热成功／发出 | 2,171/2,171 | 4,249/4,249 |
| 预热模拟秒 | 9,483.88 | 2,196.43 |
| 测量发出／成功／取消 | 949/681/268 | 5,131/4,678/453 |
| 测量阶段含收尾秒 | 3,630.95 | 3,630.02 |
| 成功请求指标分母秒 | 3,626.07 | 3,629.78 |
| GPU 命中率，成功输入 token 分母 | 21.913% | 94.104% |
| 输出 token/s，8 GPU 合计 | 91.14 | 736.78 |
| p90 TTFT 秒 | 788.58 | 6.36 |
| p90 TPOT 秒 | 1.222 | 0.748 |
| 总 GPU 步时中的同步等待比例 | 53.56% | 65.27% |

逐请求日志与阶段计数完全一致；成功输出长度等于请求原始长度；取消全部确认，未返回数为零；八个物理池最终均只有 null slot。会话在同一回放实例内始终映射到同一 rank，每个 rank 的真实批次数加 dummy 数等于全局完成步数。各 rank 的本地计算与等待之和等于 8×全局步时，误差仅为浮点累计舍入。完整核对保存在 [pipeline-validation.json](../data/reference/agentx-router/pipeline-validation.json)。

## 与上一阶段的差异及后续问题

上一阶段独立 rank 的输出吞吐为 B200 347.76、B300 2,413.81 token/s；本次分别为 91.14、736.78。增加正确的同步后吞吐下降，不能据此撤销同步或用额外系数补平。两项机制在本阶段一同接入，这些值是联合变化，不能直接分解成路由或同步的单独收益。

全局步长现在由最慢 rank 决定。B200 测量中 rank0 本地计算 1,094.18 秒、等待 2,536.77 秒；rank1 本地计算 2,210.29 秒、等待 1,420.67 秒，总和均为 3,630.95 秒。这验证了等待的来源。现有调度器仍可让长 prefill 取得整个 8,192/16,384-token 预算，并将该步延迟传播到所有 rank；下一阶段需要按源码接入 512-token threshold、DP-only interval/defer/capacity-bound 和正确的 decode 预算。

算子估算仍沿用通用结构、未完整适配原生模块；原生 MegaMoE 的全局 token/route、融合通信和 MTP 计算仍待后续对齐。框架时间校准应在这些行为与成本项完成后进行。本次结果仍是管线基线，吞吐与参考的差异未达到新接受范围，也没有把缺失 CPU 后端或不同 UUID 分布伪装成同配置对照。固定参考 worker 诊断和历史 rank3 负载归因继续保留为未完成任务。
