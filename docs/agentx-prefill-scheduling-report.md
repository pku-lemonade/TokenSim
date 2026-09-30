# AgentX 长 prefill 限流与 DP 调度

本次在已有路由和 DP 同步上接入当前 vLLM 的 prefill threshold、cadence、defer、capacity-bound，以及 FCFS running 顺序和 context 分配的抢占顺序。参考只读源码 `30f5c01d4ebb`，不运行真实 vLLM scheduler，不下载历史镜像。

## 源码行为

`scheduler.py:605–632` 先决定是否 defer，再计算有效 threshold。只有 running 与 waiting 的总数大于一时才使用 512-token cap；单个请求可以使用整个预算。`long_prefill_token_threshold_adaptive` 当前默认 False，显式启用后 cap 为 `max(configured_threshold, input_budget / eligible_count)`。因此本次没有把 512 理解为任何情况下的硬上限。

`core.py:_should_throttle_prefills` 在 DP 下使用共享 step counter，interval 8 的非对齐步传入 throttle。Scheduler 只有在 throttle、非 capacity-bound 且 running 包含 decode 时才 defer。运行中的 prefill chunk 被跳过，decode 继续；等待请求先查询缓存，剩余计算大于一个 token 时停止准入。无 decode 时继续 prefill，不因 cadence 空转。放行步骤结束后仍有 waiting 则记录 capacity-bound，下一步不 defer；无抢占的 admission 阶段才更新该标志。

当前源码的 FCFS running 列表保持插入顺序。恢复的旧请求追加到现有 running 之后，不能再按原 arrival 时间排序将它提升到队首。KV 分配不足时从尚未处理的 running 尾部抢占，最后才抢占当前请求；这条规则现在同样用于原生 prefill/recompute 的分配。完整输入准入检查和逐 chunk 物理分配沿用上一阶段，不能把输入一次性全部分配，也不能只检查本步 token。

原生完整 AgentX 默认 threshold 512、interval 8、adaptive False；其他负载默认 threshold 0、interval 1。日志新增 defer、capacity-bound、context/decode 调度 token、完整输入门槛和 max_num_seqs 门槛的计数；正式阶段开始只清空观测计数，保留影响行为的 capacity-bound 状态。

## 验证范围

逐步用例覆盖多个请求各取得 512、单请求使用整步、显式 adaptive、公平预算截断、已有 decode 时同时 defer running/waiting prefill、无 decode 时不 defer、capacity-bound 例外、max_num_seqs 门槛、DP-only 共享计数、恢复旧请求后的 running 顺序，以及无效参数。现有 chunk、缓存和取消测试继续验证抢占后的计算与引用生命周期。

完整一小时结果保存于 `results/agentx-dsv4-full/scheduling/20261001-prefill-cadence/{b200,b300}/`：全部 393 条 trace 为负载池，完成预热后发请求 3,600 秒，再执行 30 秒宽限和最多 10 秒取消；固定工作负载种子 42、模拟种子 0。模型、HBM 预算、consistent-hash 路由、DP 完成检查间隔 16 沿用上一阶段。

MTP 的 1+K query 预算必须与随机 2/3-token 进度、目标校验及 draft 成本一起接入；本次仍为一个输出 token 的管线基线，没有仅增加虚拟预算就宣称完成 MTP。CPU 异步加载的在途预留也继续由后续 offload 阶段实现。本次完成的只是上述 HBM 同步调度行为，完整计划仍保持未完成。

全仓回归：**364 项测试通过，83.373 秒**。两个完整窗口结束后的逐请求日志与阶段终态计数一致，成功输出长度未改变；所有取消得到确认，未返回数为零；最终八个池均只保留 null slot。各 rank 本地计算加等待等于 8×全局 GPU 步时，浮点误差范围内守恒。结构化证据见 [pipeline-validation.json](../data/reference/agentx-scheduling/pipeline-validation.json)。

| 本阶段结果 | B200 c196 | B300 c384 |
|---|---:|---:|
| 预热成功／发出 | 2,171/2,171 | 4,249/4,249 |
| 预热模拟秒 | 2,811.91 | 2,790.00 |
| 测量发出／成功／取消 | 1,501/1,164/337 | 10,006/9,621/385 |
| 测量阶段含收尾秒 | 3,630.06 | 3,630.34 |
| 成功请求指标分母秒 | 3,620.51 | 3,629.79 |
| 输出 token/s，8 GPU 合计 | 187.28 | 1,941.08 |
| GPU 命中率，成功输入 token 分母 | 55.838% | 94.840% |
| p90 TTFT 秒 | 575.62 | 28.75 |
| p90 TPOT 秒 | 0.752 | 0.200 |
| E2E mean 秒 | 571.86 | 154.01 |
| GPU 步时中的同步等待比例 | 53.77% | 19.01% |

相对上一阶段，输出吞吐从 B200 91.14、B300 736.78 提升至上述数值。B300 的同步等待比例从 65.27% 降至 19.01%，但 p90 TTFT 从 6.36 秒增至 28.75 秒；cadence 有效减少混合步对 decode 的干扰，同时延后部分 prefill。这个取舍来自源码规则，不通过缩短等待或修改 trace 输出消除。

B200 的容量门槛仍突出：八个 rank 在 6,665 个步骤中，完整输入门槛各阻止准入 1,220–5,303 次，capacity-bound 因而在这些 rank 经常关闭 defer。当前自算 HBM 预算较参考低 25.449%，CPU offload 尚未接入，不能将剩余排队全部归给 threshold。B300 的容量更宽裕，更多步骤按 cadence defer。后续继续核查预算与后端实际行为，不用参考池容量覆盖计算结果。

本次全一小时覆盖 B200 测量 209 条 trace、1,501 个唯一原始请求，B300 335 条 trace、9,834 个唯一原始请求；两者始终使用完整 393-trace 池。结果仍为通用算子与单 token 进度的管线基线，正式原生成本、MTP 及 CPU 分项未验收。框架耗时拟合应在这些行为完成后进行。
