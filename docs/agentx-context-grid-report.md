# Context query 与 prefix 耦合查表修正

接续 `55ba777`。本部分只修正 native attention 的 context query
预查表下限，保留真实请求与预算形状，使用既有二维表插值。
没有调整原始数据、capacity、路由、MTP acceptance 或 timing scale。

## 问题与范围

源表的部分长上下文点为 `prefix = 65536 - query`。例如
prefix49152 只测了 query16384，不代表 query512 在这个 prefix
下不可计算。原实现按“固定其他轴时的最小测量 query”截断输入，
把合法 query512 改为16384；常规同几何二维查表却支持原 query，
可以在该 query 的其他 prefix 点之间插值。

在两 SKU 的 TP1 源表，query512/prefix49152 的整层 CSA/HCA 成本
被放大约23倍。query256/prefix61440 也被放大为4096。原 query
在相邻 prefix49151/49153 上没有相同截断，制造了非物理跳变。
证据已保存在上一部分的 `diagonal-context-followup.json`，该比值
用于定位坐标错误，不是实际端到端吞吐系数或新的校准参数。

新实现按同 geometry/batch 的全部 prefix 获取真正的 query floor，
不再使用对角线 prefix 下唯一的 query 作为其他 query 的下限。
保留 zero-prefix 特殊下限、generation batch 下限、原插值/64x
外推政策与模块范围。query/prefix 的数值仍按 collector 的定义；
本修改不合并两阶段 sparse kernel，也不改变请求进度或 KV 预算。

## 验证

专项 **45 passed / 47 subtests passed**。新增回归覆盖两 SKU 的
TP1/2/4/8、query512/prefix49152 原形状、短 context 尾部、邻接
prefix 连续性、对角线上的精确实测点、真实 source floors、缓存
统计重置及源表保持原值。测试 helper 以 module import 复用。

全仓 **543 passed / 63 subtests passed**，32 个已有 warning，
166.79 秒。归档：`data/reference/agentx-context-grid/full-tests.txt`。
runtime 只有 `TokenSim/latency/deepseek_v4_attention.py` 改动。

## 完整窗口

运行期间冻结 runtime，B200 c196、B300 c384 的 seed0 主点均已
完整结束并通过严格验收。全部
393 trace 作为池、完整预热、3,600 秒 issue 窗口、30 秒 grace 和
取消释放。两组 exit0，严格验收核对完整输出、取消和最终释放。

- 结果：`results/agentx-dsv4-full/native-context-grid/20261008-preserve-query/`；
- 证据：`data/reference/agentx-context-grid/`。

```bash
python scripts/validate_agentx_context_grid.py \
  results/agentx-dsv4-full/native-context-grid/20261008-preserve-query \
  --output data/reference/agentx-context-grid/pipeline-validation.json
```

验收保留原数据输出与输入分解、源码身份、容量/配置、metadata 与
MTP/head/MoE/mHC 行数/时间、共享/分支与 DP 时间守恒、最终引用与
预留释放。成对分析继续报告吞吐、TTFT/ITL/E2E、GPU/CPU 命中、
抢占及 source clamp 差异。整体25%和最终三 seed 全矩阵仍待完成。

## 完整成对结果

吞吐为8 GPU合计；ITL 使用 AIPerf 每请求 TPOT 口径。

| 指标 | B200 | baseline变化 | B300 | baseline变化 |
|---|---:|---:|---:|---:|
| 输入吞吐 token/s | 316,008.485 | +32.708% | 668,895.910 | +26.863% |
| 输出吞吐 token/s | 2,451.044 | +34.898% | 5,054.759 | +31.044% |
| p90 TTFT 秒 | 41.831277 | -25.856% | 20.515456 | -29.205% |
| p90 ITL 秒 | 0.090656 | -27.320% | 0.072827 | -27.831% |
| 平均 E2E 秒 | 71.902900 | -24.752% | 62.904263 | -24.174% |
| GPU 命中率 % | 36.2642 | +25.911% | 94.5890 | +0.031% |
| CPU 命中率 % | 59.0805 | -10.710% | 0.8904 | -4.076% |

| 参考相对误差 % | B200 | B300 |
|---|---:|---:|
| 输入吞吐 token/s | -32.864 | -17.809 |
| 输出吞吐 token/s | -31.901 | -20.526 |
| p90 TTFT 秒 | +158.114 | +8.457 |
| p90 ITL 秒 | +80.425 | +47.169 |
| 平均 E2E 秒 | +65.809 | +23.646 |
| GPU 命中率 % | -46.845 | +5.216 |
| CPU 命中率 % | +149.012 | -45.674 |

GPU/CPU 命中分别报告；参考百分点差为：

| 参考百分点差 | B200 | B300 |
|---|---:|---:|
| gpu_hit_fraction | -31.9588 | +4.6890 |
| cpu_hit_fraction | +35.3545 | -0.7486 |

整体25%未通过。B300 的吞吐、TTFT、E2E 和GPU命中已进入25%
范围；ITL与CPU命中仍超界，B200仍有多项误差。单 seed主点不
替代最终三seed完整曲线、后端、消融和灵敏度实验。

| 窗口及形状账本 | B200 | B300 |
|---|---:|---:|
| 预热成功 | 2,171 | 4,249 |
| 测量成功 | 9,781 | 20,492 |
| 取消 | 72 | 212 |
| 抢占 | 59 | 0 |
| 旧 context clamp queries | 4,440 | 9,090 |
| 新 context clamp queries | 0 | 0 |
| 组合批 | 43,734 | 27,490 |
| 组合批真实query rows | 58,563,628 | 116,959,220 |
| logical/source-floor projection rows | 3,572,381,308 | 7,134,512,420 |

两个窗口的 source-floor projection rows 与 logical rows 完全相同。
原对角线 query放大消失；精确点、真正的下限、64x界限和原始
表保持原值。请求进度/query/placeholder/logits 账本、完整输出、
模型长度及缓存配置不变，正式 length cap两组均为0。

| 累计组件秒（target+draft） | B200 baseline→当前 | B300 baseline→当前 |
|---|---:|---:|
| gemm | 558.928 → 797.161 | 644.524 → 878.695 |
| attention | 5,425.216 → 5,908.080 | 9,967.826 → 10,844.527 |
| moe | 8,817.692 → 11,246.668 | 8,183.781 → 10,839.424 |
| elementwise | 691.125 → 931.343 | 1,019.141 → 1,330.708 |
| lm_head | 185.354 → 261.998 | 199.908 → 278.808 |
| comm | 0.000 → 0.000 | 0.000 → 0.000 |
| overhead | 0.000 → 0.000 | 0.000 → 0.000 |

累计时间随完成工作量变化，不作为相同形状成本的直接比较。
按每百万输出token归一化，仅作结果分析：

| attention 成本归一化 | B200 | B300 |
|---|---:|---:|
| baseline 秒/百万输出 | 822.569 | 712.081 |
| 当前 秒/百万输出 | 664.079 | 591.043 |
| 变化 % | -19.268 | -16.998 |

归一化保存在 `paired-work-normalization.json`，没有写入运行时
或用作校准倍数。B200同步等待占比由46.012%降至34.074%，抢占
132→59；GPU命中率28.801%→36.264%，容量与缓存源码未改。
B300同步等待占比降至16.768%，无抢占。闭环速度变化改变请求
完成位置与rank工作集，GPU/CPU精度仍分别验收，不合并或隐藏。

所有reference/配置/源码/原始trace身份、DP与组件时间、MTP/head
行数和routed/shared、main/draft归属均通过严格核对。最终引用、
DMA pin、加载/写入、在途预留及未返回取消均归零。

## 后续独立任务

下一部分首先区分 native MegaMoE 的完整捕获输入与有效路由行数：
当前源码默认 padding top-k=-1，SP输入与mask连续切分；idle/dummy
及MTP各pass需保留单独有效行数。shape证据见
`moe-padding-followup.json`；没有把有效行数平均值直接替换源表
最大rank成本。后续计时必须同时保留block几何、最慢发送/合并
成本、shared输入范围及已有uniform实测锚点。

捕获主干工作范围、内存生命周期、历史映射发生实例、匹配AIC、
SOL留出误差和最终完整验收继续保留在研究计划中。
