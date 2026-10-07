# 包含生成进度的缓存身份与恢复边界

接续 `72a0e14`。本部分对齐 native hybrid GPU/CPU 缓存的完整上下文
恢复。调度的 one-real-token padding、attention/graph metadata 与
model-length edge 单独推进，未并入本阶段。

只读 vLLM `30f5c01d4ebb` 的 `append_output_token_ids` 会更新完整 block
hash 链，`get_computed_blocks` 以 `request.num_tokens - 1` 限制命中。
此前模拟器将 keys 与 hit cap 都限于原始 prompt，恢复时会丢掉可用
的已生成上下文，并在 block-aligned prompt 上继续排除最后输入块。

新实现保留已知 prompt 的 shareable PrefixCacheKey；未知部分（partial
prompt/generated block）使用独立的 `RequestLocalPrefixKey` 类型。
键含 engine 内唯一 request ID、block 大小/index、最后已知 prompt
parent，以及 model/cache salt/reuse group 命名空间。它只能恢复同一
请求的状态，不能让 synthetic output 获得未经证明的跨请求命中。
native store 由 engine 持有，不跨独立模拟共享私有身份。

keys 随 accepted context 增长，GPU 和 external suffix 查询均按完整
context 的 n−1 处理。只发布已确认计算的完整块；proposed/lookahead
位置不会因分配而获得可复用身份。retention policy、全部 window/
EAGLE 状态检查、peek 丢弃、异步 write readiness、DMA pin 和容量
约束继续生效。命中上限扩大不等于所有新增位置都可恢复。

完整上下文的 GPU/CPU hit 与输入 token 统计分开：

- `gpu_cached_context_tokens` / external 记录当前恢复的完整前缀；
- 原有 cached/effective prefill 与首个 PrefillSourceStats 仍只记录
  原始输入，GPU + CPU + computed input 必须等于 prompt length；
- generated reload 使用独立的 GPU/CPU 累计字段，进入请求日志和
  `agentx_metrics`，不会改变原命中率分母；
- SimpleCPU 与 native Mooncake 复用同一异步 import 生命周期，
  Mooncake 底层 query 计数按完整 context，与 native source 一致。

## 逻辑验证

全仓 **517 passed**、16 subtests passed、32 个已有 warning，141.76 秒。
测试日志归档于 `data/reference/agentx-context-recovery/full-tests.txt`。

生成进度测试使用真实混合组：同一请求从 prompt 32,256、generated
1,025 的上下文恢复 32,768-token 保留检查点（含 512 generated token）；
另一请求不能复用私有 suffix；缺失 EAGLE peek 会退回较早边界。
独立 full-group 用例验证 context n−1 留一个真实 token，不把它当作
真实混合组的一 token 命中证据。SimpleCPU/Mooncake 的异步 CPU
恢复都保留原 input denominator，最终输出长度、引用和 pin 释放正确。

首轮新增 integration fixture 手动产生缓存后遗漏 worker 的
`wait_for_save` hook，external store 正确返回 no-ready-hit。fixture 已
补真实 save/DMA 完成流程；原失败日志保存为
`initial-fixture-failure.txt`，没有放宽命中或 readiness 条件。

## 完整窗口

B200 c196 / B300 c384 的 seed0 主点均完整结束并通过严格验收。
各使用全部 393 trace 池、完整预热、3,600 秒窗口与 grace/取消，
不要求全部原始请求完成。运行期间代码冻结，容量预算和数据输入
均通过对基线的身份核对。

- 结果：`results/agentx-dsv4-full/native-context-recovery/20261008-private-context/`；
- 源码/测试/PID/结果证据：`data/reference/agentx-context-recovery/`。

```bash
python scripts/validate_agentx_context_recovery.py \
  results/agentx-dsv4-full/native-context-recovery/20261008-private-context \
  --output data/reference/agentx-context-recovery/pipeline-validation.json
```

GPU/CPU 容量仍由原 hardware/model/config 预算计算，不使用参考容量
覆盖。原始数据、算子表、MTP acceptance 与 timing 参数保持一致。
本阶段不代表整体 25% 精度或最终三 seed 全矩阵通过。

## 成对完整结果（2026-10-08）

吞吐为 8 GPU 合计，baseline 为 `72a0e14`。

| 指标 | B200 c196 | B300 c384 |
|---|---:|---:|
| 输出 token/s | 1,781.095 | 3,805.127 |
| 相对上一阶段变化 % | -2.819 | -0.002 |
| 参考输出吞吐误差 % | -50.515 | -40.173 |
| p90 TTFT 秒 | 64.611 | 28.982 |
| p90 ITL 秒 | 0.128143 | 0.103331 |
| E2E 均值秒 | 97.716 | 85.681 |
| GPU 命中率 % | 31.1908 | 94.7920 |
| CPU 命中率 % | 63.7573 | 0.7799 |
| 抢占 | 78 | 0 |
| 成功请求 | 7,720 | 16,019 |
| 到期取消 | 121 | 266 |

两组全部取消返回，缓存引用、DMA pin、外部 pin 与在途预留释放，
DP/各组件时间守恒；MTP 三次 pass、输出头、mHC 和 MoE 的计数/
时间账本保持正确。runtime 差异严格限定为八个登记文件，静态容量、
完整输入及全部 timing 参数与基线一致。实际 benchmark wall time 为
B200 1,603.915 秒 / B300 3,226.484 秒。

| 成功请求恢复计数 | B200 | B300 |
|---|---:|---:|
| 重复准入 | 74 | 0 |
| 重新加载的输入 GPU token | 2,109,952 | 0 |
| 重新加载的输入 CPU token | 10,354,688 | 0 |
| 重新加载的 generated GPU token | 0 | 0 |
| 重新加载的 generated CPU token | 0 | 0 |
| 重算 token | 169,443 | 0 |

两个完整主点均未观测到 generated retention checkpoint reload。
专项回归证明真实混合组在符合状态/retention/readiness 条件时能恢复，
不能把该回归结果冒充本轮吞吐收益。B200 抢占 63 → 78，B300 保持
0；闭环请求群体和 GPU/CPU cache 发布身份/LRU 也发生变化，观察
本身不单独证明性能变化的因果。

| 最终 external resident pages | baseline | 本阶段 |
|---|---:|---:|
| B200 | 707,323 | 735,457 |
| B300 | 1,436,981 | 1,494,385 |

两个版本的各 rank CPU eviction 均为 0，具体使用字节与分 rank 数据
保存在 `paired-comparison.json`。

| 参考相对误差 % | B200 | B300 |
|---|---:|---:|
| 输入吞吐 | -50.038 | -35.962 |
| 输出吞吐 | -50.515 | -40.173 |
| p90 TTFT | +298.676 | +53.215 |
| p90 ITL | +155.032 | +108.810 |
| E2E 均值 | +125.334 | +68.416 |
| GPU 命中率 | -54.281 | +5.442 |
| CPU 命中率 | +168.723 | -52.415 |

GPU 命中率绝对差为 -37.0322 / +4.8920 个百分点。

CPU 命中率绝对差为 +40.0313 / -0.8591 个百分点。

模拟命中率仍以成功请求首次准入的原始 input token 为分母，历史
telemetry 的分母差异继续单列。整体 25% 精度未通过，最终三 seed
矩阵与全部原定范围继续推进。

## 下一部分

`padding-metadata-followup.json` 单列了 request progress、短 query
attention metadata、selected logits 和 graph eligibility。源码的
is_prefilling 取决于 original prompt 的已计算进度；它与生成上下文
重算不是同一状态。下一部分处理 waiting/loaded 的单真实 token
placeholder padding 与相应预算、KV/head 行数，继续保持实际输出为
一 token；max-model-length cap 也单独核对。后续混合投影、非均匀
DP MoE、buffer 生命周期、历史映射、匹配 AIC 与完整最终验收均
保留在原研究计划中。
