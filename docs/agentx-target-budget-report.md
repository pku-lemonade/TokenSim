# MTP target query、KV 与 logits 范围修正

日期：2026-10-07；接续 `2f388dc`，仅修正 target 的预算与 head 行数。
完整 draft 模块、cached-prefill 单 token padding 和 mHC 重复残差保持
独立后续任务，不把本部分当成全部 MTP 已完成。

只读 vLLM `30f5c01d4ebb` 的 scheduler/async scheduler 保留完整 draft
placeholder，末步也不缩短投机 query batch；输出在 sampling 后按停止条件
处理。当前模拟器将 output 剩余数同时用来裁剪 query，导致少算最后一步
的 compute、token budget 和 KV。新实现末步仍用 1+K=4 个 query，51% /
49% 的 2/3 输出推进仍只截断实际输出，不改原始长度或接受流。

`_reserve_decode_slot` 已接收 query_tokens，但此前将 accepted_tokens
传给 KV manager。现在按 query + K lookahead 预留：computed=250、
accepted=2、query=4、K=3 时应覆盖到 257，而非 255，会跨越 256-token
full KV 页。新增真实共享池测试验证这个页与每个窗口组，并检查最终
释放守恒；静态容量不变。

vLLM V2 的 target logits 总行数是 request 数 + scheduled draft 数，
包含每个 partial prefill 的 bonus 行。DSV4 backend 现在对 decode
按每个 query、对所有 prefill 按一行计 head；generic 模型保留原行为。
记录并导出 `dsv4_target_logits` 的 decode/prefill/total rows 和 steps，
warmup 后清零，与 expanded decode 查询行数逐项核对。

源文件 SHA、旧实现的直接逻辑复现和数据边界证据见
`data/reference/agentx-mtp-kv/target-budget-followup.json`。全部 98,827
条原始请求 input + output + 7 最大为 996,586，未超过 1M 模型上限；
本部分的完整负载不需要数据裁剪。其他模型长度边界及 cached-prefill
padding 的行为继续按源码单独核对。

## 验证

目标测试：48 passed。包括最后一个输出仍需四 query token budget、
真实 KV 页边界、所有组覆盖和释放、两个 partial prefill + 四 verify
logits 行及统计 reset。完整测试与 B200/B300 各完整窗口验收后填写。
全仓已通过 **499 passed**、14 subtests passed、32 个已有 warning，
135.75 秒；日志 `/tmp/agentx-target-budget-tests.log`。两个硬件点已在
独立进程会话启动完整窗口，进程描述保存于
`data/reference/agentx-target-budget/process.json`，运行时源文件保持冻结。
结果根目录计划为 `native-target-budget/20261007-target-scope/`，配对
基线为 MTP tile 复用阶段的 B200 v2 / B300 retry1，393-trace 池与 seed0。

后续只读缓存预算核查保存在 `memory-scope-followup.json`。固定
DeepGEMM 源码的 symmetric buffer 含 input、ring、shared 和按 top-k
分开的 combine 存储；信号结构通过 host C++ ABI 核对为 21,140,864 B。
Pro EP8 的 132/148 SM 场景在 token/ring 对齐后均给出约
2.0924 / 3.6790 GiB（B200/B300 配置）的 buffer footprint；SM 场景
尚不作为实测硬件事实，独立 output tensor 和其他 live allocations 也
必须另外核对。当前通信 envelope 与 selected-expert 激活不能直接当作
这个具体 kernel 的分项存储，后续需要避免重复和漏计。

上游 AIC 的 3.5 GiB `other_mem` 注释是为覆盖内存计算误差增加的保守
余量，backend 将其用于 CUDA/cuBLAS 等；它不是一个单独测得的 vLLM
workspace。本阶段未改 reserve 或容量，后续也不依据历史池大小删除
余量或倒填容量。

B200 已完整结束且严格验收通过（`b200-validation.json`）：成功
7,918 / 取消 89，输出 1,845.484 token/s，较 tile 基线 +2.30%，
相对参考 −48.73%；p90 TTFT 59.792 s、p90 ITL 0.130114 s。
GPU/CPU 首次命中为 30.3632% / 64.6821%，抢占 126（基线 83）。
decode / prefill / total logits 行数为 11,733,268 / 90,813 / 11,824,081；
decode 行数精确等于 4 × request 数，indexer tile 数精确为 2 × request 数，
153,600 个计数 step 与 operator step 一致。源码、配置、静态容量、
逐请求来源、输出、取消、预留/引用释放及 component/DP 守恒均通过。
B300 完整窗口仍在测量中，整体最终精度尚未验收通过。

## 两组完整验收

B300 已完成完整预热和一小时测量，退出码 0；两组严格验收全部通过，
证据为 `data/reference/agentx-target-budget/pipeline-validation.json`。
使用全部 393 trace 为负载池，发请求窗口 3,600 秒，保留 grace 与取消；
不要求全部原始请求执行完。四个运行时源文件改变，配置、数据、硬件、
模型和静态容量保持不变；输出/来源、计数、引用、预留、取消和时间
守恒全部通过。最终 logits 计数与 decoder 查询逐项一致。

| 指标（8 GPU 合计） | B200 | B300 |
|---|---:|---:|
| profiling 成功 / 取消 | 7,918 / 89 | 16,333 / 301 |
| 输出吞吐 (token/s) | 1,845.484 | 3,865.835 |
| 相对 tile 基线变化 | +2.30% | −0.50% |
| 相对参考吞吐误差 | −48.73% | −39.22% |
| p90 TTFT (s) | 59.792 | 28.342 |
| p90 TTFT 相对误差 | +268.94% | +49.83% |
| p90 ITL (s) | 0.130114 | 0.100647 |
| p90 ITL 相对误差 | +158.96% | +103.39% |
| GPU 首次命中率 | 30.3632% | 94.8201% |
| GPU 相对误差 / 百分点差 | −55.49% / −37.8598 | +5.47% / +4.9201 |
| CPU 首次命中率 | 64.6821% | 0.6685% |
| CPU 相对误差 / 百分点差 | +172.62% / +40.9561 | −59.21% / −0.9705 |
| 抢占次数（tile 基线） | 126 (83) | 0 (0) |

decode / prefill / total logits 行数为 B300
24,834,052 / 183,807 / 25,017,859；两个硬件点 decode 行数都为
4 × request 数，indexer tiles 为 2 × request 数。B300 157,984 个
计数 step 与 operator step 一致，profile reset 未混入 warmup。

本阶段保证 forward/query 的预算与输出推进分开，不能用吞吐变化替代
逻辑正确性。更完整的 KV 预留使 B200 抢占增加，闭环也改变请求集合；
B300 没有资源压力，增加正确的 head 费用后 throughput 小幅下降。
**整体 25% 精度门槛仍未通过。** 完整 draft、cached-prefill padding、
mHC residual 重复计费、混合批投影与不均匀 MoE 计时仍待分阶段修正，
最终三 seed 矩阵必须使用修复后的完整实现重新验证。

累计 component GPU 秒：

| 分项 | B200 | B300 |
|---|---:|---:|
| attention | 5,611.087 | 10,213.965 |
| MoE | 8,699.932 | 7,999.787 |
| GEMM | 140.698 | 157.481 |
| elementwise | 774.434 | 1,140.600 |
| lm_head | 91.489 | 98.773 |
| comm / overhead | 0 / 0 | 0 / 0 |

复验：`python scripts/validate_agentx_target_budget.py
results/agentx-dsv4-full/native-target-budget/20261007-target-scope
--output data/reference/agentx-target-budget/pipeline-validation.json`。
