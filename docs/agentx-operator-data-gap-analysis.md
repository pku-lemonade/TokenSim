# AgentX Operator-Data 差距分析与修复方案

## 1. 文档目的

本文记录 TokenSim 当前 AgentX 模拟结果与博客、InferenceX 参考结果之间的差距，并区分：

1. 配置或实现错误；
2. operator-data 没有覆盖当前查询形状的问题；
3. 模型和运行时本身的近似；
4. AgentX 闭环回放方法造成的不可直接比较。

本文只描述问题和修复方案。本次没有在 `main` 分支实施修复，当前分析文件位于 `agentx-support` 分支。

## 2. 背景

AgentX 是面向代码助手流量的推理测试方法，输入不是固定长度的一次性请求，而是 393 条 Claude Code 会话 trace。每条 trace 可能包含：

- 多轮对话；
- 很长的上下文；
- 主 agent 和 subagent 并发；
- 记录下来的前缀 hash，用于模拟 KV cache 重用；
- HBM 和 DRAM 两级 KV cache；
- 闭环请求间隔，即前一个请求完成后才会继续后面的请求。

本项目的模拟器不生成真实文本，而是使用 trace 中的输入长度、输出长度、hash 链和时间关系，模拟请求进入 scheduler、执行模型算子、进行并行通信和读写 KV cache。

远端 `main` 分支已经把原来的整体 roofline 模型替换成新的硬件目录和 operator-data 体系，包括：

- `data/devices/`：GPU 设备参数；
- `data/models/`：模型结构参数；
- `data/topologies/`：节点、NVLink、网络拓扑；
- `data/operator_data/`：GEMM、attention、MoE、DeepEP、collective 的实测数据；
- `TokenSim/latency/operator_table.py`：按实际请求形状查询 operator data。

本次重跑使用的是完整 393 条 trace、1800 秒 profile、10 秒系统空闲上限，并使用 `operator_table` backend。

## 3. 本次运行结果

完整结果目录：

```text
results/agentx-operator-data-full-v2/
```

完整对比文件：

- `results/agentx-operator-data-full-v2/comparison.json`
- `results/agentx-operator-data-full-v2/comparison.csv`

393 条 trace 中包含 98,827 个请求定义，但 AgentX 是闭环测试，所以 1800 秒内每个配置实际完成的请求数不同：

| 配置 | 完成请求数 |
|---|---:|
| B200 HBM | 206 |
| B200 DRAM | 999 |
| B300 HBM | 1 |
| B300 DRAM | 992 |

B300 HBM 只完成 1 个请求，因此不能把它当作稳定吞吐点，也不能用它计算有意义的 offload 收益。

### 3.1 与 InferenceX DRAM 点比较

| 指标 | B200 模拟 | B200 参考 | 差距 | B300 模拟 | B300 参考 | 差距 |
|---|---:|---:|---:|---:|---:|---:|
| 输出 tok/s/GPU | 5.82 | 449.91 | -98.71% | 4.28 | 795.03 | -99.46% |
| 输入 tok/s/GPU | 909.56 | 58,837.61 | -98.45% | 953.11 | 101,728.98 | -99.06% |
| p90 TTFT | 90.81 s | 16.21 s | +460.32% | 261.67 s | 18.92 s | +1283.35% |
| p90 TPOT/ITL | 593 ms | 50.25 ms | +1081.03% | 915 ms | 49.49 ms | +1747.90% |
| 平均 E2E | 211.09 s | 43.36 s | +386.78% | 349.14 s | 50.87 s | +586.27% |

### 3.2 与博客缓存数据比较

| 配置 | 模拟 HBM | 博客 HBM | 模拟 DRAM | 博客 DRAM |
|---|---:|---:|---:|---:|
| B200 DRAM | 63.34% | 73% | 32.53% | 20% |
| B300 DRAM | 49.97% | 91% | 44.91% | 1.36% |

博客中的参考值来自 [SemiAnalysis AgentX 文章](https://newsletter.semianalysis.com/p/agentx-inferencexv3-does-cuda-moat)。

## 4. 当前必须解决的问题

### 4.1 B300 设备被解析成了 GB300

AgentX B300 cluster 文件使用了 `B300`。但设备目录中：

- `gb300.yaml` 将 `B300` 列为 alias；
- `b300_sxm.yaml` 是真正的 B300 SXM 设备；
- 本次结果的 `latency_backends` 显示 `device_id = gb300`。

因此当前比较实际上是：

```text
InferenceX B300
vs
TokenSim GB300
```

这会同时改变 GPU 参数、operator data、collective data 和拓扑语义。B300 结果在修复设备选择前不能用于绝对性能比较。

修复要求：AgentX B300 配置必须显式使用 `B300-SXM` 或 `b300_sxm`，并增加测试确认结果中的设备 ID 为 `b300_sxm`。

### 4.2 EP8 被模拟成了 EP64

InferenceX 参考点的配置是 TP8、EP8、DP attention。对于 64 张 GPU，目标结构应是 8 个独立副本：

```text
DP0: 8 GPU，TP8/EP8
DP1: 8 GPU，TP8/EP8
...
DP7: 8 GPU，TP8/EP8
```

当前实现把 EP rank 数计算为：

```text
tensor_parallel_size * data_parallel_size = 8 * 8 = 64
```

并且 [TokenSim/parallel.py](../TokenSim/parallel.py) 的 EP group 没有按 `dp_rank` 分组。因此所有 64 张 GPU 被放进同一个 EP group。

这不是当前 AgentX 参考点的部署方式。

修复要求：增加显式 EP 配置，而不是在代码中固定写死 8：

```json
{
  "expert_parallel_scope": "per_dp",
  "expert_parallel_size": 8
}
```

`per_dp` 表示每个 DP 副本内部独立做 EP；`global` 表示所有 DP 副本共同做 wide EP。其他实验可以使用 `global`，但 AgentX B200/B300 参考点应使用 `per_dp`。

### 4.3 EP 通信因此走了错误的链路

由于 EP group 被扩大到 64 张 GPU，EP all-to-all 跨越 8 个节点，模拟器使用了跨节点 100Gb 网络。

正确的 AgentX EP8 通信应在一个 8-GPU 节点内部完成，使用 NVLink。当前结果中通信部分异常大：

- B200 DRAM 通信累计约 46,046 秒；
- B300 DRAM 通信累计约 79,017 秒；
- B300 HBM 在 1800 秒内只完成 1 个请求。

这说明当前最主要的问题是通信分组和拓扑，而不是单个 GEMM 算子慢了几倍。

### 4.4 operator data 没有被按正确形状查询

当前 operator data 中的 DeepEP 数据主要是：

```text
B200 SXM vLLM: EP8、单节点
B300 SXM vLLM: EP8、单节点
GB300 vLLM:    EP4、单节点
```

当前模拟器由于设备和 EP 分组错误，实际查询的是 EP64、8 节点。operator data 没有这类测量行，模拟器只能退回到公式估算。

因此，“换了 operator data 后性能突然变差”不能直接解释为 operator data 错了。更准确的解释是：新的 operator-table backend 暴露了错误的设备和通信查询形状。

### 4.5 KV cache 分布与参考结果不一致

当前 B300 DRAM 模拟得到 HBM 49.97%、DRAM 44.91%，而博客是 HBM 91%、DRAM 1.36%。

这说明以下行为还没有对齐：

- 会话是否固定路由到同一个 DP 副本；
- 主 agent 和 subagent 是否共享正确的 cache group；
- HBM 淘汰水位线；
- DRAM 写入和读取策略；
- prefix block 的对齐和生命周期；
- cache 命中率的统计分母。

### 4.6 HBM 和 DRAM 不是同一批请求

AgentX 是闭环测试。一个配置响应较慢，就会导致它在 1800 秒内完成更少的请求，后续看到的 trace 也会不同。

本次 B300 HBM 只完成 1 个请求，而 B300 DRAM 完成 992 个请求。因此不能把两者直接相减来计算 offload 收益。

修复后需要加入最低样本量检查、多 seed 运行和低样本警告。

### 4.7 DeepSeek-V4 仍是 proxy

当前 `DeepSeek-V4-Proxy` 只是近似模型，虽然已经接入新的 model catalog，并支持 compressed KV，但仍然没有真实 DeepSeek V4 的：

- kernel 形状；
- expert 路由分布；
- MTP/speculative decoding；
- chunked prefill；
- CUDA Graph；
- fused MoE 实现；
- 真实 scheduler 参数。

这会影响最终精度，但优先级低于设备选择和 EP 拓扑错误。

## 5. 修复建议

### 第一阶段：修正确定性错误

1. 将 AgentX B300 cluster 的硬件名称改为 `B300-SXM` 或 `b300_sxm`。
2. 增加 `expert_parallel_scope` 和 `expert_parallel_size`。
3. AgentX B200/B300 使用 `per_dp`、EP8。
4. `ExpertPlacement` 按 DP 副本独立分配 expert。
5. `ParallelCommunicator.ep_group()` 只选择同一 DP 副本的 rank。
6. `OperatorTableLatencyBackend` 在 per-DP EP 模式下不要把 token 数乘以 DP 数量。
7. 确认 EP group size 为 8、节点数为 1、链路为 NVLink。

### 第二阶段：修正 operator data 使用方式

1. 用 `table_only` 模式列出实际缺失的测量形状。
2. 确认 B200/B300 使用正确的 vLLM device package。
3. 补齐 B200/B300 的 EP8 DeepEP 数据。
4. 补齐 8-GPU NVLink collective 数据。
5. 补齐长上下文 attention 和 elementwise 数据。
6. 关键路径没有测量数据时直接报错或产生明确的缺失报告，不能静默用公式替代。

### 第三阶段：修正 cache 和回放方法

1. 校准 DP sticky routing。
2. 对齐 HBM block eviction 和 DRAM write-through 语义。
3. 分别记录主 agent、subagent 和新会话的 cache 生命周期。
4. 增加最低完成请求数检查。
5. 使用多个随机种子并报告结果范围。
6. 在 HBM/DRAM A/B 测试中尽量使用相同的请求序列或固定 schedule。

### 第四阶段：模型校准

使用真实或单机 profiling 数据校准：

- prefill token 数；
- decode batch；
- context length；
- TP/EP size；
- B200/B300 硬件；
- compressed-KV attention；
- MoE expert routing；
- communication latency。

## 6. 验收标准

在重新跑完整数据集前，先通过以下检查：

1. B300 结果中的 `device_id == b300_sxm`。
2. AgentX EP group size 为 8，不是 64。
3. EP group 只跨一个节点。
4. B200/B300 DeepEP 查询使用 `ep_size=8,nodes=1`。
5. B300 HBM 短测试至少完成足够多请求，不能只有 1 个。
6. B300 HBM/DRAM 的 cache 命中率接近博客参考值。
7. 关键 operator 查询缺失项已明确列出。
8. 最后再跑完整 393 条 trace、1800 秒四配置。

## 7. 当前结论

当前巨大误差不能归因于 operator data 本身。operator data 只是在新的 latency backend 中把错误暴露出来了。

最优先的问题是：

```text
B300 被解析成 GB300
EP8 被模拟成 EP64
EP 通信被放到了跨节点 Ethernet
错误查询形状触发了大量公式估算
```

先修复这四项，再判断 operator data 的绝对延迟是否需要重新校准。
