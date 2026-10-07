# 主干输出头、最终归一化与 mHC 时间汇总

接续 `cf5bde3`。本部分补齐主干 hc_head 的成本和最后 RMSNorm 的行数，
同时汇总并导出实际 main/draft mHC 统计。缓存、调度、MTP 接受进度、
表数据和容量预算沿用上一阶段。

只读 vLLM `30f5c01d4ebb` 的 target model forward 先在所有 T 行上
执行 hc_head，把 `(T, hc_mult, H)` 的 BF16 residual 归约为 `(T, H)`，
再做 final norm；最后才选择 lm_head 的输入行。此前缺少主干
hc_head，并把 final norm 错计为 selected logits 行。新实现分别计量
full-forward rows 与 selected logits rows；SP gather 后的输出头仍用
完整 T，不除以 TP。符合 FULL_DECODE_ONLY 条件时计入 DP 同步的
capture padding，context rank 则继续使用各自的 eager T。

MTP decoder forward 返回 flat pre-hc_head residual，其 hc_head 和 shared
norm 延后到 compute_logits，因此保持 selected rows。其 FULL capture
包含整个 forward + compute_logits + sample routine，已有 captured request
padding 的 head 行数继续保留。独立 HCHead helper 复用上阶段 draft
的物理公式，不修改其同形状费用：

- FP32 参数 bytes：`4 * (hc_mult² * H + hc_mult + 1)`；
- 每行 BF16 input/output bytes：`H * (2 * hc_mult + 2)`；
- 每行 FP32 FLOP 估算：`H * (2 * hc_mult² + 8 * hc_mult)`；
- 时间：`max(math, memory / existing memory_efficiency) + kernel_launch_us`。

没有 matched hc_head 模块实测，仍是明确的 D 级解析估算。最终/shared
RMSNorm 与 selected-row BF16 lm_head 单独计费。MoE、MTP 与 target
output 使用同一 uniform decode capture 行数规则，保证三条路径一致；
不改变 borrowed table 的插值、外推或系数。

新增 `dsv4_hc_head_counts/seconds` 保留 target/draft 调用、行数与费用。
`dsv4_target_logits` 同时记录实际 forward、output forward、hc_head 和
final norm 行数。主干 mHC 的每层形状估算按 `layers_local` 复制，draft
每次估算已经是一层；导出的 `native_mhc_main_seconds` / draft / total
采用这份真实时间账本。`native_mhc_counts` 分 main/draft，仍表示查表
次数，不把它冒充 fused kernel 的实际调用数。

全仓 **512 passed**、14 subtests passed、32 个已有 warning，137.30 秒。
回归覆盖短 prefill 的全 forward norm 与 selected logits、TP4/DP2 的
capture padding、远端 context 的 eager 行数、实际 result JSON
序列化、主干层数权重及预热后重置。

B200 c196 / B300 c384 两个 seed0 主点均完整结束并通过严格验收。
各使用全部 393 trace 池、完整预热、3,600 秒窗口和 grace/取消；
运行期间源码保持冻结。
结果、PID/命令、源码和完整测试校验值位于：

- `results/agentx-dsv4-full/native-output-head/20261008-full-forward/`；
- `data/reference/agentx-output-head/`。

验收入口：

```bash
python scripts/validate_agentx_output_head.py \
  results/agentx-dsv4-full/native-output-head/20261008-full-forward \
  --output data/reference/agentx-output-head/pipeline-validation.json
```

本部分已完成成对误差与组件分析，单独提交。本部分不代表整体
25% 精度或最终三 seed 全矩阵验收通过。

## 成对完整结果（2026-10-08）

吞吐为 8 GPU 合计，baseline 为 `cf5bde3`。

| 指标 | B200 c196 | B300 c384 |
|---|---:|---:|
| 输出 token/s | 1,832.757 | 3,805.194 |
| 相对上一阶段变化 % | -2.914 | 0.512 |
| 相对参考输出吞吐误差 % | -49.080 | -40.172 |
| p90 TTFT 秒 | 61.511 | 29.047 |
| p90 ITL 秒 | 0.129332 | 0.103109 |
| E2E 均值秒 | 97.487 | 84.813 |
| GPU 命中率 % | 31.6175 | 94.9031 |
| CPU 命中率 % | 63.4443 | 0.6359 |
| 抢占 | 63 | 0 |
| 成功请求 | 7,822 | 16,085 |
| 到期取消 | 96 | 298 |

两组 request/input/output 分解、取消返回、引用、DMA pin、在途预留
和 DP/组件时间守恒均通过。静态运行配置、缓存预算及数据校验值
保持一致，runtime 差异严格限定为六个已登记文件。

| 输出头/归一化范围 | B200 行数 | B300 行数 |
|---|---:|---:|
| target 实际 forward | 55,963,538 | 113,332,204 |
| target 输出头与 final norm（含 capture padding） | 57,053,326 | 117,211,160 |
| target selected logits | 12,039,731 | 24,525,936 |
| draft selected head/logits | 11,337,007 | 22,026,986 |

| 独立账本（GPU 秒） | B200 | B300 |
|---|---:|---:|
| target hc_head | 1.138002 | 1.830229 |
| draft hc_head | 1.554248 | 1.680011 |
| main mHC（61 层计权） | 566.812218 | 822.981880 |
| draft mHC（三次一层 pass） | 19.742126 | 24.161325 |
| native mHC 合计 | 586.554344 | 847.143205 |

mHC 合计分别低于 global elementwise 695.423529 / 1,000.772321 GPU
秒；其余 norm/rope/input 等费用保留。head、selected logits、三个
draft pass 和 native MoE 账本也全部守恒，仍无重复 residual_add。

相对参考的诊断误差：

| 指标误差 % | B200 | B300 |
|---|---:|---:|
| 输入吞吐 | -49.262 | -35.836 |
| 输出吞吐 | -49.080 | -40.172 |
| p90 TTFT | +279.546 | +53.561 |
| p90 ITL | +157.399 | +108.361 |
| E2E 均值 | +124.806 | +66.710 |
| GPU 命中率 | -53.656 | +5.565 |
| CPU 命中率 | +167.404 | -61.201 |

GPU 命中率绝对差为 -36.6055 / +5.0031 个百分点。

CPU 命中率绝对差为 +39.7183 / -1.0031 个百分点。

命中率的模拟分母仍是成功请求首次准入的输入 token，历史 telemetry
分母差异继续单列。整体 25% 精度未通过，不能用查表范围或逻辑
验收代替精度验收。

## 固定形状费用差分

`paired-shape-diagnostic.json` 在每个 SKU 上比较 uniform decode、
unequal decode 与 mixed prefill/decode 三个固定 request/snapshot。baseline
实例绑定 git `cf5bde3` 的原始 `_compose` AST，其余输入、当前表与
draft/MoE 算法相同，两个实例的请求和 pass 状态独立。

| 形状 | B200 增量 μs / 增幅 | B300 增量 μs / 增幅 |
|---|---:|---:|
| uniform decode | 3.819 / 0.01042% | 3.814 / 0.01052% |
| unequal decode | 4.075 / 0.01160% | 4.068 / 0.01154% |
| mixed prefill/decode | 12.495 / 0.01486% | 12.434 / 0.01556% |

六例仅 lm_head 分项变化，其他分项差均低于 1e-12 秒。该差分只
核对工作范围与同形状费用，不是精度 benchmark，也不生成吞吐
修正系数。完整闭环的 request order、请求群体和缓存竞争另有反馈；
当前单 seed 的 2.914% 吞吐变化需由最终三 seed 结果进一步评估。

## 后续修正依据

`cached-context-followup.json` 核对了 source hit cap 使用包含已生成
输出的 `request.num_tokens - 1`，当前 native manager 的 keys 与上限
仍只含 prompt。后续先修正 resumed context 的边界/身份，再处理
one-real-token padding 以及 attention/graph metadata。允许的命中上限
还须经过全部 window/EAGLE、retention 与 tier-availability 检查；不给
未经证明的 generated output 跨请求复用。混合投影、非均匀 DP MoE、
buffer 生命周期及完整三 seed 矩阵仍按原研究计划继续，不缩小范围。
