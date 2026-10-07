# Query padding、元数据角色与长度边界

接续 `b48c165`。本部分对齐 waiting/loaded 的单真实 token padding，
分开 request progress、attention metadata、logits 与 graph eligibility，
并补齐 native max-model-length query/output/KV 边界。

只读 vLLM `30f5c01d4ebb` 的 waiting 分支在 K>0、real suffix=1、
当前没有 running prefill、已有 running work 或 positive prefix、完整
query 预算和 model-length 空间满足时，调度 1+K 行并填 K 个 invalid
draft。synthetic sampler 也拒绝 invalid draft，因此进度和输出仍是
一 token。新实现记录真实 `scheduled_tokens`、query 与 placeholder
数量，按 query+K 预留 KV，完整预算不足时保持 waiting；loaded 页
继续保留，不为了准入反复释放/重载。

源码的 `prefill_scheduled` 只从本轮之前 running 的请求累计，新
waiting admission 不改变该条件。source scheduler 的 `is_prefill_chunk`
是上一轮 complete-context chunk pending 状态；runner 的 is_prefilling
是 original prompt 的计算进度。两者已分别用 pending 标记与 prompt
进度建模。graph 允许 positive-prefix 的 one-real-token extension，
以及 original prompt 已恢复的生成上下文；实际 capture 仍须 query
宽度统一且匹配 FULL_DECODE_ONLY 条件。

SM100 indexer 支持 varlen query。短 query（≤1+K）可进入 decode
attention metadata，继续保留 Q-block 内 KV 复用估算；这不改变真实
context build 的一 token 输出。target logits 独立按 proposals 记录：
普通 context 为一 bonus row，invalid placeholder padding 仍需 1+K
个 logits。主干和 MTP 图形状复用同一 metadata 角色。

native query 保证 computed+query+1≤max_model_len；query+lookahead 的
物理 end 也被限制在 model length 内。输出达到 model/context 上限
时 length-stop，保留原 `decode_len` 请求值，触发 completion 与释放。
prompt 无法留出输出位置则在入队前拒绝。全部正式 trace 的原请求
input+output+7≤996,586，低于 1,048,576，不裁剪数据集或正式输出。

## 逻辑验证

全仓 **527 passed**、16 subtests passed、32 个已有 warning，142.14 秒。
测试日志归档于 `data/reference/agentx-padding-metadata/full-tests.txt`。

新增回归覆盖四 query / 一 token 进度与输出、完整预算不足时未取得
GPU prefix、loaded 页保留、running-prefill 与 new-waiting 条件差异、
max length 下取消 padding、query/output 截断与 completion/释放、
short context attention 和 logits 分离、generated replay 的 graph 角色。
测试 helper 以 module import 复用，避免重复收集原 fixture 的测试。

## 完整窗口

B200 c196 与 B300 c384 的 seed0 完整窗口均成功结束并通过严格验收。
全部 393 trace 为负载池，完整预热后模拟 3,600 秒，按 30 秒 grace 和取消收尾；
不要求全部 98,827 条原始请求完成。运行期间冻结 runtime 源码。

- 结果：`results/agentx-dsv4-full/native-padding-metadata/20261008-query-roles/`；
- 证据：`data/reference/agentx-padding-metadata/`。

```bash
python scripts/validate_agentx_padding_metadata.py \
  results/agentx-dsv4-full/native-padding-metadata/20261008-query-roles \
  --output data/reference/agentx-padding-metadata/pipeline-validation.json
```

两组已核对进度/query/placeholder/logits 账本、原始输出长度、缓存释放、
时间守恒、吞吐与各误差。容量由原预算计算，
表数据、AL2.49、随机种子、负载池及 timing 参数沿用 baseline。
整体 25% 精度和最终三 seed 全矩阵仍待原研究计划的全部工作完成。

## 成对结果与误差

吞吐为 8 GPU 合计；ITL 采用 AIPerf 对每请求的 TPOT 口径。
所有以下指标与 `b48c165` baseline 完全一致，变化均为 0%。

| 指标 | B200 | 参考相对误差 | B300 | 参考相对误差 |
|---|---:|---:|---:|---:|
| 输入吞吐 token/s | 235,169.608 | -50.038% | 521,159.014 | -35.962% |
| 输出吞吐 token/s | 1,781.095 | -50.515% | 3,805.127 | -40.173% |
| p90 TTFT 秒 | 64.611415 | +298.676% | 28.981745 | +53.215% |
| p90 ITL 秒 | 0.128143 | +155.032% | 0.103331 | +108.810% |
| 平均 E2E 秒 | 97.715935 | +125.334% | 85.680760 | +68.416% |
| GPU 命中率 % | 31.1908 | -54.281% | 94.7920 | +5.442% |
| CPU 命中率 % | 63.7573 | +168.723% | 0.7799 | -52.415% |

GPU 命中率相对参考分别差 −37.032 / +4.892 个百分点；CPU 分别差
+40.031 / −0.859 个百分点。两种误差口径分别记录，验收仍用相对误差。

| 完整窗口账本 | B200 | B300 |
|---|---:|---:|
| 预热成功 | 2,171 | 4,249 |
| 测量成功 | 7,720 | 16,019 |
| 到期取消 | 121 | 266 |
| 抢占 | 78 | 0 |
| context real/query token | 44,120,258 | 88,267,712 |
| scheduler selected logits | 11,712,214 | 24,411,622 |
| target logits（含 idle dummy） | 11,712,654 | 24,411,630 |
| target forward rows | 55,743,742 | 112,500,104 |
| target output/final norm rows | 56,686,634 | 116,588,032 |

placeholder query、padded admission、short-context decode metadata rows、
context graph eligibility 和正式 length cap，两组均为 0。全部成功请求
保持原输出长度；取消返回、引用、DMA pin、在途预留和 DP 时间守恒通过。
源码/配置/容量/输入身份通过独立核对，完整验收 JSON 保存结果校验值。

| 累计组件时间（秒，含 target/draft） | B200 | B300 |
|---|---:|---:|
| gemm | 554.512020 | 629.667310 |
| attention | 5,551.360779 | 9,960.590016 |
| moe | 8,709.345953 | 8,097.875755 |
| elementwise | 684.584685 | 993.003210 |
| lm_head | 184.018516 | 195.698857 |
| comm | 0.000000 | 0.000000 |
| overhead | 0.000000 | 0.000000 |

上述时间全部与 baseline 数值一致，累计量是所有 rank 的组件时间，
不是 wall time。native attention + draft attention、routed/shared MoE、
main/draft mHC 与各组件汇总守恒。

本 seed 没有触发新增 padding/short-context/长度边界行为，不能以
该窗口代替专项 source-derived 回归，也不能宣称当前性能收益。
本阶段扩大实现对配置和边界的忠实范围；整体精度仍未达标。

下一部分独立处理 mixed attention 的共享投影：保留 sparse core 的
decode/prefill 分支，避免每个 context 长度组与 decode 反复计入整套
输入/输出投影。只读调查和逻辑 fixture 见
`data/reference/agentx-padding-metadata/mixed-projection-followup.json`，
其中的合成成本不是实测或校准倍数。
