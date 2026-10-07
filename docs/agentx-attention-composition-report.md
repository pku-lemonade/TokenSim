# 混合 attention 的共享投影与分支成本

接续 `dd7da91`。本部分只修正 context 长度组与 decode 混合批的
attention 共享工作归属，保持调度、缓存容量、trace、MoE/MTP 和
算子原始数据不变。整体 25% 精度仍未验收。

## 源码范围与实现

只读 vLLM `30f5c01d4ebb` 的 `DeepseekV4Attention.forward` 先对完整
hidden tensor 执行输入投影，然后准备 query、compressor/indexer，
最终对整个 attention output 执行一次输出投影。compressor-input
GEMM、indexer query 与 weight projection 也使用完整 tensor。
FlashInfer sparse core 保留 decode 和 prefill 两个独立调用。

旧实现对每个 context `(query length, prefix)` 组查询完整模块，再加
一个完整 decode 模块；这反复包含共享投影、权重读取与输入准备。
此前的 ragged decode 修复只解决 decode 内部的长度组。

现有借表测量完整 FlashMLA 模块，无法直接取得各组件实测时间。
本部分采用已存在的 DSV4 FLOP/byte SOL 分解：

- 共享：主输入/输出投影、compressor-input GEMM、index-query/weight
  投影、权重读取及 activation preparation；
- 分支：sparse attention、context-dependent index scoring/KV 读取和
  compressor pointwise 工作。context 组和 decode 分支分别保留。

对于每个源形状，分别求共享 bound `C` 和分支 bound `R`，再令
`eta = measured_module / (C + R)`；两个分项之和严格回到该实测
完整模块。混合批共享效率按实际 query 行数加权，使用整批行数计算
一次共享工作，加上原分支估算。decode 的 context 分项继续采用
之前的整批 query-row 权重和 request-local Q-block KV 复用。

保留各分支的 source-grid clamp；共享行数不低于整批真实行数及最大
source floor。源表匹配、插值、外推范围和量化/layout 参数均不变。
纯 decode 和单 context 组保持原完整模块查询，不改变同形状锚点。
没有吞吐拟合系数，也没有新增 aux-stream 或 kernel overlap 抵扣。

结果导出 `native_dsv4_attention_composition`：共享/分支时间、原独立
模块总时间、两者之差、真实/source-floor 行数和共享层调用次数。
统计在预热后清零，worker 汇总与结果 schema 均保留这些分项。
独立模块时间与共享后时间的差允许为负，不以“必须变快”限制估算。

## 验证与限制

回归验证共享 intercept 只计一次、分支 residual 保留、context prefix
坐标、真实 query 行数、source floor、Q-block 读取、导出和重置。
两个 SKU 的 TP1/4/8 离线锚点对比均完全保留同形状估算，证据为
`data/reference/agentx-attention-composition/shape-diagnostic.json`。
离线混合形状的变化是成本诊断，不能用作完整回放的精度证据。

模块表无法识别单个 launch、ragged fusion 和 aux-stream overlap；
`C/R` 分配仍是物理模型估算。不同 context 长度组保留各自分支
residual，尽管源码将 prefill 集合送入一个 varlen kernel，不为此
追加未实测的 launch 收益。既有 SOL 缺失的状态写入细节、FlashMLA
借表与 FlashInfer serving 的 layout/后端差异仍需后续独立验证。

## 完整窗口

全仓 **537 passed**、29 subtests passed、32 个已有 warning，156.72 秒。
日志保存于 `data/reference/agentx-attention-composition/full-tests.txt`。
已冻结 runtime 源码，启动 B200 c196 与 B300 c384 的 seed0 完整检查点：全部 393 trace 作为池、完整预热、
3,600 秒 issue 窗口、30 秒 grace 与取消释放。两组已完整结束并通过严格验收，
取消、在途预留、引用和全部组件时间守恒通过。

```bash
python scripts/validate_agentx_attention_composition.py \
  results/agentx-dsv4-full/native-attention-composition/20261008-shared-projections \
  --output data/reference/agentx-attention-composition/pipeline-validation.json
```

实验原参数直接来自上一阶段，B200/B300 分别保留自己的原始数据。
不更改 workload、acceptance length、GPU/CPU capacity 或校准 scale。

## 完整成对结果

全部 393 trace 池、完整预热、3,600 秒 issue 窗口、30 秒 grace
与取消清理通过。运行时与结果校验值、原始输入/完整输出、配置、
容量、DP 和组件守恒均通过，完整证据为 `pipeline-validation.json`。
只有五个 runtime 文件发生变化，缓存/路由/调度/模型和原表保持原值。

吞吐为 8 GPU 合计，ITL 使用 AIPerf 的每请求 TPOT 口径。

| 指标 | B200 | 对 baseline 变化 | B300 | 对 baseline 变化 |
|---|---:|---:|---:|---:|
| 输入吞吐 token/s | 238,123.588 | +1.256% | 527,256.623 | +1.170% |
| 输出吞吐 token/s | 1,816.955 | +2.013% | 3,857.297 | +1.371% |
| p90 TTFT 秒 | 56.418604 | -12.680% | 28.978698 | -0.011% |
| p90 ITL 秒 | 0.124732 | -2.662% | 0.100912 | -2.341% |
| 平均 E2E 秒 | 95.554230 | -2.212% | 82.958472 | -3.177% |
| GPU 命中率 % | 28.8014 | -7.661% | 94.5597 | -0.245% |
| CPU 命中率 % | 66.1666 | +3.779% | 0.9282 | +19.020% |

命中率表的变化为相对值；GPU 命中率的绝对变化为 −2.3894 /
−0.2323 个百分点，CPU 为 +2.4094 / +0.1483 个百分点。

| 参考相对误差 % | B200 | B300 |
|---|---:|---:|
| 输入吞吐 token/s | -49.411 | -35.213 |
| 输出吞吐 token/s | -49.519 | -39.353 |
| p90 TTFT 秒 | +248.123 | +53.199 |
| p90 ITL 秒 | +148.244 | +103.923 |
| 平均 E2E 秒 | +120.350 | +63.065 |
| GPU 命中率 % | -57.783 | +5.183 |
| CPU 命中率 % | +178.878 | -43.365 |

整体 25% 精度未通过；B300 GPU 命中率进入 10% 优先目标，其他
指标的差异继续保留，单 seed 主点不替代最终三 seed 全矩阵。

| 完整窗口与成本账本 | B200 | B300 |
|---|---:|---:|
| 预热成功 | 2,171 | 4,249 |
| 测量成功 | 7,819 | 16,365 |
| 取消 | 140 | 295 |
| 抢占 | 132 | 0 |
| 组合批 | 35,895 | 19,680 |
| 混合阶段批 | 35,783 | 19,603 |
| 多个 context 组的批 | 21,172 | 19,611 |
| 组合批真实 query rows | 47,546,540 | 93,882,632 |
| 共享投影累计秒 | 1,923.991612 | 3,514.544095 |
| 分支累计秒 | 1,963.256161 | 3,630.430579 |
| 相同 query 独立模块累计秒 | 4,091.417432 | 7,398.774606 |
| 相同 query 共享成本差额秒 | 204.169659 | 253.799932 |

共享成本差额是**当前窗口形状**如果分别查询完整模块的账本差，
不是 baseline 闭环时间预测，也不是实测 kernel profiling。源表
source-floor 行数可能高于真实 query，下一节单列已定位的原因。
正式 length cap 和成功请求 generated reload 两组均为 0。

| 累计组件秒（target + draft） | B200 baseline → 当前 | B300 baseline → 当前 |
|---|---:|---:|
| gemm | 554.512 → 558.928 | 629.667 → 644.524 |
| attention | 5,551.361 → 5,425.216 | 9,960.590 → 9,967.826 |
| moe | 8,709.346 → 8,817.692 | 8,097.876 → 8,183.781 |
| elementwise | 684.585 → 691.125 | 993.003 → 1,019.141 |
| lm_head | 184.019 → 185.354 | 195.699 → 199.908 |
| comm | 0.000 → 0.000 | 0.000 → 0.000 |
| overhead | 0.000 → 0.000 | 0.000 → 0.000 |

B200 抢占从 78 增至 132，GPU 命中率下降。六个 rank 曾触及满池；
rank2/5 的 full-input gate 从 6,191/3,437 变为 8,711/5,382。
rank3/7 成功数下降，rank1/5/6 上升；成对 per-rank 请求量、输入
和命中记录保存在验收 JSON。缓存源码、容量与调度配置完全相同，
闭环时间变化改变工作集与抽样完成位置，不能把这些差异隐藏在
总体吞吐提升中，也不能以逻辑守恒通过宣称缓存精度通过。

B300 无抢占，CPU 命中率虽改善，相对参考仍低 43.365%。native
attention + draft attention、routed/shared MoE、main/draft mHC、
输出头行数和 DP 总时间分别守恒；最终加载/写入、DMA pin、引用、
在途预留和未返回取消均归零。统一使用原 AL2.49、scale1.0 与
step_overhead0，没有按 SKU 拟合参数。

## Query 与 prefix 耦合的查表诊断

`source-floor-diagnostic.json` 的固定 prefix 样本未复现 B200 的明显
行数扩张；扩充真实源表的对角线 prefix 后找到了原因。部分 context
样本使用 `prefix = 65536 - query`，因此同一 prefix 下唯一测量的
query 不是其他 query 的最小合法值。

例如 TP1、batch1、query512、prefix49152：当前预处理把 query 改为
16384；B200 CSA/HCA 成本为 13,326.3 / 8,073.5 微秒。直接用原始
query512 在已有同几何二维表中查找，可以插值为 576.793 / 338.916
微秒，约相差 23 倍。query256/prefix61440 也被放大为 query4096。
实测 query512/prefix65024 锚点保持精确，两 SKU 都复现同类问题。

证据为 `diagonal-context-followup.json`。这是预查表的坐标处理错误，
不是实测模块性能突然跳变，也不是新的整体吞吐系数。当前完整窗口
继续保留该既有行为，确保本部分只处理共享归属；下一部分优先独立
修正错误的 query 截断，验证真正的源表下限、邻接 prefix、精确点
及两个完整主点。非均匀 MoE、graph projection 与 buffer 生命周期
继续作为后续独立任务，不因发现此问题而省略。
