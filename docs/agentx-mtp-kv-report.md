# MTP verify 的 indexer KV tile 读取复用

日期：2026-10-07；接续 SOL 阶段 `a2051c3`。本部分只修正 target verify
中的 indexer KV 读取，输出推进的 51% / 49% 接受长度、draft 成本和
context/decode 混合批投影仍保持各自独立任务。

历史 B200/B300 server.log 均记录 `use_flattening=False`、next_n=4 和
FP4 indexer cache。当前只读 vLLM `30f5c01d4ebb` 的 SM100 builder 走
varlen；其 request indices 在固定的 DeepGEMM `e1f418c2a4f` kernel 中
重新按 request 分组，不能仅凭 reshape/flatten 名称判断 KV 复用量。
`sm100_mqa_logits.hpp` 明确 `BLOCK_Q = 128 / num_heads`，Pro 的
indexer head 为 64，因此每个 Q-block 最多两个 query。scheduler 的
`RequestInfo.num_q_blocks` 为 ceil(num_q_tokens/BLOCK_Q)，逐 Q-block
遍历 KV split；q=4 需要两个读取 block。源文件 SHA 与 URL 记录在
`data/reference/agentx-mtp-kv/scope-audit.json`。

令整步 query 行数为 `R`，某 context 的 query 行数与 KV-reading Q-block 数为 `r` / `t`，
其中 `t = Σ(request_count × ceil(query_width / BLOCK_Q))`。
仍先查询完整 `R` 行的 q=1 模块，使投影和 query 计算的预算保持整批；
只将 SOL 中 indexer KV 读取乘 `t/r`：

```
source_us = max(query_math_us, source_memory_us)
verify_us = max(query_math_us,
                source_memory_us - (1 - t/r) * source_indexer_bytes / HBM_bw)
estimated_module_us = measured_full_query_module_us * verify_us / source_us
```

q=4 的 indexer KV tile 读取估算为四个独立 q=1 请求的 1/2；projection、
compressor、indexer query/logits 计算保持全部 query 行。FlashInfer 稀疏
attention 的索引可能逐 query 不同，本部分仍逐 query 计其读取，不将
整模块直接乘四或将所有 KV 都除四。HCA 无 indexer，因此本部分不改变
HCA 成本。计算成为瓶颈时，KV 复用不再降低模块的 SOL 估算。

保留实测优先、64x 边界、ragged 支持切片和现有 small-grid clamp；复用
公式用 lookup 实际采用的 query geometry，不额外改变保守的 query 行数。
source KV/indexer 仍为 584 / 132 B；没有加入 source-to-reference layout
或 FlashMLA-to-FlashInfer 系数。roofline 比例是显式估算，不是 q=4 配对实测。跨 Q-block 的 L2/HBM 复用没有实测流量证据，本部分
不额外抵扣，报告的是保守 streaming SOL 估算。
同长度 q=1 批次仍精确恢复原模块值。

记录 request/query/Q-block 数、source geometry 的 indexer bytes 前后值和估算
节省的时间。字节字段是 SOL 估算，不能当作硬件实测 HBM counter。
profile reset 清空这些统计；最终 native attention 时间只累计修正后的
费用，保持 component/DP 守恒。
match counters 分类的是 q=1 source geometry 查表；q>1 的复用修正属于
SOL 估算，不能将 source exact 标签解释为 q=4 模块实测。

SOL 的长度无关 FLOP/byte 常数同时按 geometry 缓存，减少完整窗口中
数百万查询的报告字典构建。完整 `work()` 仍提供逐分项审计；新增差分
测试验证 fast path 与原 FLOP/byte 分解在 CSA/HCA/SWA、TP1/4/8、
prefill/decode 和长 context 下等价。

## 验证

单元回归覆盖 q=1、q=4、混合 query 宽度、source clamp、math-bound 复用
无增益、HCA 中性及统计清零。初次全仓 495 项通过，但该版每请求读一次的近似不符合 Q-block 源码，
其完整运行已主动中止，保留为未验收记录。修正后全仓 **496 passed**、14 subtests passed、32 个已有 warning，
131.51 秒；日志 `/tmp/agentx-mtp-kv-v2-tests.log`。两组完整窗口已在测试
通过后启动，结果保存在
`results/agentx-dsv4-full/native-mtp-kv/20261007-query-tiles-v2/`，配对基线为
`native-attention-sol/20261007-physical-growth/`；使用全部 393 trace 池与 seed0。
本阶段不能代表完整 draft 成本或最终三 seed 精度验证已完成。

初次每请求一次的旧形状诊断 `shape-audit.json` 与旧窗口
`20261007-request-reuse/` 保留，不用作验收。按 Q-block 修正的形状
诊断为 `shape-query-tiles.json`；后续填写对应的完整结果。

16 请求 / 64 query / 32 Q-block 的修正后形状诊断：

| context | B200 verify / expanded-q1 | B300 verify / expanded-q1 |
|---|---:|---:|
| 16K | 0.9769 | 0.9770 |
| 128K | 0.8646 | 0.8653 |
| 1M | 0.6554 | 0.6565 |

后续只读核查保存为 `target-budget-followup.json`：末步 target query
被输出剩余数截短、KV 预留路径未使用传入的 query_tokens，以及 target
head 只计一次 decode 请求并遗漏多数中间 prefill 的 bonus logit 行。
这些不属于本次 tile 复用修改，下一独立阶段修正并验证。全 98,827 个
原始请求的 input + output + 7 均小于模型上限，最大为 996,586 token；
不需要裁剪数据来满足 query/lookahead 预算。

## 当前完整窗口进度

B200 v2 已完整结束并通过严格验收：成功 7,734、取消 116；输出
1,803.965 token/s，较 SOL 基线 +2.51%，相对参考 −49.88%。p90 TTFT
63.428 s、p90 ITL 0.129575 s；GPU / CPU 首次命中为
28.4334% / 66.6288%，抢占 83（基线 56）。静态预算、源码/数据 SHA、
逐请求输出和输入来源、取消/引用释放、component/DP 时间守恒全部通过。
SOL source-geometry indexer bytes 前后比约 0.5；累计估算节省
419.882 GPU 秒，不能当成硬件实测或跨闭环窗口的直接时间差。

B300 v2 的 supervisor 返回 143，实际 benchmark PID 随后核实已消失，
最后窗口仍是 profiling（simulation_time=5,400），无完整 result JSON
和 run_status；终止原因未确定，不能用于验收。B200 结果保留。B300
在独立进程会话中从完整预热重新运行，目录为
`native-mtp-kv/20261007-query-tiles-b300-retry1/`，进程描述保存为
`data/reference/agentx-mtp-kv/b300-retry-process.json`。当前实现阶段等
B300 同配置完整窗口结束后再汇总和提交。

## 两组完整验收结果

B300 retry1 已完成预热和完整一小时窗口，退出码 0。严格跨目录验收
通过，结果为 `data/reference/agentx-mtp-kv/pipeline-validation.json`；
逐点源码/数据 SHA、配置与静态容量、输出与输入来源、取消确认、最终
null slot / CPU/DMA pin / 在途预留，以及 native attention / component /
DP local compute 守恒全部通过。只改变四个运行时源文件。

| 指标（8 GPU 合计） | B200 | B300 |
|---|---:|---:|
| profiling 成功 / 取消 | 7,734 / 116 | 16,345 / 284 |
| 输出吞吐 (token/s) | 1,803.965 | 3,885.424 |
| 较 SOL 基线变化 | +2.51% | +6.91% |
| 相对参考吞吐误差 | −49.88% | −38.91% |
| p90 TTFT (s) | 63.428 | 28.024 |
| p90 ITL (s) | 0.129575 | 0.101005 |
| GPU 输入命中率 | 28.4334% | 94.8039% |
| GPU 相对误差 / 百分点差 | −58.32% / −39.7896 | +5.45% / +4.9039 |
| CPU 输入命中率 | 66.6288% | 0.7289% |
| CPU 相对误差 / 百分点差 | +180.83% / +42.9028 | −55.53% / −0.9101 |
| 抢占次数（基线） | 83 (56) | 0 (0) |

profiling 分项累计 GPU 秒（含 dummy）：

| 分项 | B200 | B300 |
|---|---:|---:|
| attention | 5,537.613 | 10,231.623 |
| MoE | 8,717.093 | 7,998.570 |
| GEMM | 139.900 | 157.656 |
| elementwise | 767.621 | 1,141.100 |
| lm_head | 90.622 | 97.376 |
| comm / overhead | 0 / 0 | 0 / 0 |

两组 indexer tile 字节比为 0.500098 / 0.500104，混合末步 q=3 使其
略高于 0.5；估算节省累计 419.882 / 975.708 GPU 秒。request/query/
Q-block 计数分别为 B200 2,903,851 / 11,596,032 / 5,801,207，
B300 6,188,289 / 24,712,097 / 12,362,873。字段是 source-geometry SOL
估算，不是实际 HBM 流量，也不直接等于不同闭环窗口的总时间差。

**整体仍未通过 25% 门槛。** B200 五个 rank 在 full-input gate 频繁
触发并达到池上限；成功请求重复准入 83 次，重载 CPU/GPU token 为
11,889,920 / 1,436,928，重算 179,133。更快闭环改变实际请求集合，
不能以抢占数量单独断言准入回归。完整 draft 模块、target query/KV/
logits 范围、mHC 重复残差和混合批投影仍待独立修正；最终三 seed
矩阵不复用本阶段的 seed0 作为完成声明。

复验命令：

```bash
python scripts/validate_agentx_mtp_kv.py \
  results/agentx-dsv4-full/native-mtp-kv/20261007-query-tiles-v2 \
  --b300-root results/agentx-dsv4-full/native-mtp-kv/20261007-query-tiles-b300-retry1 \
  --output data/reference/agentx-mtp-kv/pipeline-validation.json
```

另一个后续形状诊断保存为 `moe-shape-followup.json`：在合成的
`[4096, 64, …, 64]` EAGER EP8 输入中，实际全组 4,544 行，当前
max-rank uniform 查询代表全组 32,768 行；均匀等效输入为 568 行/rank。
对应两个 uniform 表的 routed 时间相差约 4.8 倍。这不是实际 serving
性能比，也不证明可以直接改用均值；需核对源端 dispatch、全局 expert
负载、不同 block geometry 和 slow-rank 组成。本阶段 MoE 计时保持原实现。

mHC scope 后续核查在 `mhc-scope-followup.json`：collector pre 使用
`norm_weight=None`，其 norm 仍需单独估算；post 已包含 weighted residual
重建，native 分支又叠加 generic residual_add 是重复工作。后续单独移除
这部分，保留 unfused table 与 fused serving 的明确差异。本阶段 mHC
计时未改变，也未使用 collector 注释中 H20 的融合时延比修正两 SKU。
