# 整批 decode attention 模块计时

日期：2026-10-07；本次仅修复 native decode 的长度分组计时。

原实现将每个 `(KV length, query width)` 分组视为独立 batch，累加完整
CSA/HCA 模块耗时。模块表包括输入和输出投影、compressor/indexer 及
sparse attention，因此不同长度组会重复支付投影权重读取和 kernel launch。
B200/B300 原 seed0 主点分别有 1,899,364/3,659,412 次 attention 查表，
对应 47,304/42,504 个 worker step，即平均每步约 20/43 个长度组，每组
分别估算 CSA 和 HCA。查表次数不代表 serving 实际调用次数；这里统计的
是每组请求触发的完整模块估算次数。

只读 vLLM `30f5c01d4ebb` 的
`vllm/models/deepseek_v4/attention.py:471` 中，`forward` 对整批
`hidden_states` 执行一次 `_run_parallel_input_projections`，随后执行
attention 和一次 `_o_proj`。AIC `f254959` 的
`collector/vllm/collect_dsv4_attn.py:451` 对形状为
`(batch_size, hidden_size)` 的 generation 模块整体测量。
KV 长度差异没有把这些共享投影变成独立模块调用。

新实现令 `R = Σ(count × query_width)`，对每个 KV 长度 `L` 查询完整
`R` 行的 uniform batch，按该长度占 query 行数的比例混合：

```
decode_us = Σ_L [rows(L) / R × module_us(batch=R, context=L)]
```

同长度 batch 精确恢复原有模块查询；不同长度的估算位于同规模 uniform
batch 耗时的最小值和最大值之间，共享模块成本只计一次。这是 ragged
batch 的显式近似，保留逐长度查表来源。没有用参考吞吐拟合比例，也没有
移除算子；两个 attention sensitivity scale 仍为 1.0。

本次保持 context 分组、MTP query 行展开和其他算子计时不变。MTP 多 query
是否共享 KV 读取、混合 context/decode 投影去重和完整 draft 模块成本，
都还需要单独核查；本修复不能证明这些要求已满足。

新增回归用例覆盖不同长度不重复 launch、同长度与实测查询一致、混合
成本边界、组件统计仅记录最终估算和 profile reset。native attention
累计耗时与 decode batch 计数现在能够穿过 result schema，保存在输出 JSON。

初次完整窗口暴露了表格网格边界：B200 的 68 行、737,408 context 查询
在 batch 插值中选到 batch=128、context 只测到 6,144 的邻点，触发 64x
上限；B300 同样遇到该问题。修复后仅在 generation 的邻点超界时使用
同 dtype、模型、TP 和层类型、能支持该 context 上限的 batch 切片，仍由
原查表器执行插值/外推及其 64x 限制。原始表没有删除或改写，越过所有
切片上限的查询仍报错；使用该路径的计数单独导出。
失败日志保存在 `20261007-full-batch/`，不计为验收成功。

完整验证结果记录在
`results/agentx-dsv4-full/native-decode-batch/20261007-full-batch-v2/`，
使用完整 393-trace 池、预热、3,600 秒模拟发请求窗口、30 秒 grace 和取消
收尾；不要求全部原始请求被执行。

## 完整窗口验收

全仓 `pytest -q`：482 passed、14 subtests passed、32 个已有 warning，
129.23 秒；日志 `/tmp/agentx-decode-batch-20261007-v2-tests.log`。
两个主点使用 seed0，源码和数据 SHA-256 均由严格 validator 逐项核对；
逐请求输入来源、完整输出、取消确认、预留释放、GPU null slot、CPU/DMA
pin 与 DP local compute/component 时间守恒全部通过。验收文件：
`data/reference/agentx-decode-batch/pipeline-validation.json`。

配对基线为 `native-checkpoints-final/seed0/`，包含相同原生后端和 MTP；
不与缺失 attention 成本的早期高吞吐窗口混作配对比较。吞吐均为 8 GPU 合计。

| 指标 | B200 | B300 |
|---|---:|---:|
| 成功请求 | 4,982 | 9,009 |
| 取消请求 | 210 | 380 |
| 输出吞吐 (token/s) | 1,053.931 | 1,862.202 |
| 相对配对基线变化 | +105.51% | +80.36% |
| 相对参考误差 | −70.72% | −70.72% |
| p90 TTFT (s) | 122.393 | 61.052 |
| 基线 p90 TTFT (s) | 237.518 | 102.389 |
| p90 ITL (s) | 0.217135 | 0.230528 |
| 基线 p90 ITL (s) | 0.403305 | 0.423130 |
| GPU 输入命中率 | 28.0897% | 95.0695% |
| CPU 输入命中率 | 66.3365% | 0.4407% |
| 抢占次数（基线） | 28 (4) | 0 (0) |

profiling 分项累计 GPU 秒（包括 DP dummy batch，不是墙钟秒）：

| 分项 | B200 | B300 |
|---|---:|---:|
| attention | 10,714.622 | 16,840.701 |
| MoE | 5,647.326 | 4,065.559 |
| GEMM | 82.603 | 74.375 |
| elementwise | 478.379 | 576.503 |
| lm_head | 52.469 | 44.497 |
| comm / overhead | 0 / 0 | 0 / 0 |

B200 27 次成功请求重复准入，重新加载 GPU/CPU token 为
814,336 / 3,418,880，重算 100,949 token；两组均无最终资源泄漏。
闭环执行速度改变了实际请求集合，抢占次数增加不能单独证明准入回归。
B200 仍有接近满池的 rank 和 full-input gate，逐 rank 证据保留在验收文件。

decode 批数为 89,112 / 72,576，长度组数为 1,738,765 / 3,015,985，
query 行数为 7,071,729 / 12,328,479；generation 支持切片救援查询为
286,188 / 990,536（包含 cache hit）。统计显示 ragged 网格是实际窗口的
常见路径，不能以仅测试表内统一长度代表完整覆盖。

## 尚未解决的误差

本阶段只修复整批计费，**没有通过整体 25% 精度门槛**。attention 仍占
累计分项约 63.1% / 78.0%。当前 `_analytical_work` 用 compressed-work
比例外推整个模块，连上下文无关的投影和权重读取也随 KV 长度增长。
例如 B200 batch=64 的 CSA 在 context=16K / 128K / 1M 时分别为
180.300 / 1,165.166 / 9,044.094 us；HCA 为
125.800 / 564.387 / 4,073.081 us。下一独立阶段按 SOL 分开固定投影、
稀疏 attention 和 indexer 的计算/读取量，保留实测边界和外推限制，
用留出实测点检查趋势；MTP KV 复用与完整 draft 成本再独立处理。

重新审计还发现旧 AIC 对照的量化和混合批次形状不匹配，以及 B300
历史 conversation-only worker 映射存在多个回放实例对应多个 rank。
这两项已重新打开计划任务，证据见 `scope-audit.json` 与
`routing-identity-followup.json`，不影响本阶段 consistent-hash 主点。
