# AgentX 基准产物核对与对比工具

日期：2026-09-30。分支：`agentx-support`。

本轮完成研究计划的「基准与口径」子任务，提交后停止。实现参考提取、主检查点矩阵和按实验身份对比的工具；未改模拟器、完整模型或请求回放。以下完整数据核对不是 §5.1 的一小时模拟验收，不能作为模拟精度证据。后续先完成完整模型与并行配置，再按计划推进请求回放、GPU 分配与缓存复用、调度和算子耗时。

## 改动

- `util/agentx_reference.py`：参考请求统计、原始长度映射、端点去重、日志与 router 解析。
- `scripts/extract_agentx_reference.py`：读取全部原始 trace 和两组完整参考产物，保存校验、时间序列、来源文件与 SHA-256，不覆盖既有输出目录。
- `data/experiments/agentx.yaml`：录入已核实的 B200 c196、B300 c384 主点；从真实启动命令检查 TP/DP/EP、GPU 数、block、seq 和 token 预算。完整曲线点随后补齐，不从 TP 数猜 GPU 数。
- `scripts/compare_agentx_results.py`：按矩阵和结果 manifest 读取逐点配置，按模拟的实际 GPU 数重新计算输入／输出吞吐。模型、拓扑、后端、参考身份、窗口或 trace 池不一致时保留结果并说明不可比较，不计算误导性的参考误差。
- `tests/test_agentx_reference.py`：覆盖统计边界、错误与取消、原始身份映射、端点身份、日志可选字段、矩阵对比与缺失值。
- `data/reference/agentx/`：提交紧凑证据摘要、输入校验值、已核对的历史源码和 Apache-2.0 许可证。较大的逐 rank 时间序列与全部 router 映射保留在本地结果目录。

另有独立提交 `88c0382`：既有通信外推回归用例显式选择它要验证的 analytical 策略，并纠正默认策略说明。现有默认策略为 empirical；未改变模拟耗时算法。修改前完整测试为 225 项、1 项失败，修正后 225 项通过。

## 完整参考数据核对

读取全部 **393 条 trace、98,827 条原始请求**，原始文件只读。两组参考均为预热后 3,600 秒发请求、30 秒宽限、最多 10 秒取消收尾；未裁剪 JSONL、server metrics 或日志。

| 项目 | B200 c196 / 440239 | B300 c384 / 439522 |
|---|---:|---:|
| 全部 JSONL 记录 | 16,162 | 29,960 |
| profiling 成功记录 | 13,991 | 25,696 |
| profiling 错误记录 | 0 | 15 |
| profiling 发出请求（wire log） | 14,124 | 25,923 |
| profiling 到期取消（wire log） | 133 | 212 |
| 吞吐观测时长（秒） | 3,629.805930 | 3,629.409517 |
| 输出 tok/s/GPU，8 GPU | 449.907834 | 795.033011 |
| 输入 tok/s/GPU，含命中 | 58,837.609879 | 101,728.980226 |
| TTFT p90（秒） | 16.206517 | 18.915776 |
| ITL p90（秒） | 0.050246 | 0.049485 |
| E2E mean（秒） | 43.364842 | 50.874388 |
| 重算指标对 bmk 的最大相对误差 | 0.043910% | 0.040616% |

TTFT、ITL、E2E、ISL、OSL 的 mean/p50/p75/p90/p95/std 逐项对照 bmk，均小于 0.5%；同时保存 p99。最大误差来自 bmk 的五位小数舍入。ITL 从 `(latency - TTFT) / (OSL - 1)` 重算，OSL=1 不产生 ITL 样本。吞吐边界使用成功 profiling 请求的最早 start 与最晚 end，先以整数纳秒相减，未将分母硬编码成 3,600 或 3,640 秒。

两个主点各 45 项核对全部通过，完整提取实际耗时约 57.8 秒。完整输出在 `results/agentx-dsv4-full/reference/`；可提交摘要为 `data/reference/agentx/summary.json`，输入身份和来源为同目录 `manifest.json`。每个主点保存 `key_point.json`、`bmk_validation.json`、`request_accounting.json`、`output_length_audit.json`、`sampling.json`、`router.json`、`server_metrics.json`、`server_log.json`、原始 bmk 和两类时间序列。两个 key point 也已提交到 `data/reference/<experiment-id>/key_point.json`，对比时可用 `--reference-root data`，无需重新读取临时研究缓存。

## 数据和日志揭示的问题

### 历史 expected OSL 查错请求

历史 [InferenceX 聚合源码](https://github.com/SemiAnalysisAI/InferenceX/blob/843c6254/utils/agentic/aggregation/trace_metadata.py) 只收集主请求，剥去子代理／并行会话后缀，再用会话内 `turn_index` 查原始主列表。这种索引在拆分会话中指向另一条请求。

本轮分别重现两种查法：错误查法得到 bmk 的 1,261.542874 / 1,219.870755；正确查法用 `source_trace_id + source_outer_idx + source_inner_idx`，得到 933.787792 / 898.350054。正确 expected 与成功 profiling 请求 actual **逐条完全一致**，没有缺失映射，也没有非零 `osl_mismatch_diff_pct`。原始 OSL=0 的 6 / 9 条记录按 AIPerf 最小 max_tokens=1 处理。不能用历史错误 expected 的差额证明窗口截断、取消或幸存者偏差，更不能调整 trace OSL 抵消模拟器耗时误差。

### 取消记录不能只看 JSONL

两组 JSONL 都没有取消记录，但 wire log 分别记录 133 / 212 个取消请求，且 `sent = completed + cancelled` 守恒。B300 的 completed=25,711 包含 15 个错误，指标使用 25,696 个成功请求；预热记录 2,171 / 4,249 中分别有 5 / 7 个解析错误。这些预热错误与 wire log 的 errors=0 属于不同计数位置，分别保留，不互相覆盖。

### router 和引擎端口重复暴露相同 series

按显式 endpoint 别名、engine 和所有其他 labels 识别同一来源，优先 `localhost:8889/metrics` 的直连引擎数据；未配置别名的端点不自动合并。保留所有端点统计，不能固定除二。

| 独立引擎总量 | B200 | B300 |
|---|---:|---:|
| prompt tokens，直连引擎 | 1,734,067,913 | 2,988,512,269 |
| prompt tokens，router 转发 | 1,734,067,913 | 2,988,561,237 |
| generation tokens，两端点相同 | 13,377,634 | 23,835,368 |
| 成功 profiling JSONL usage prompt | 1,708,552,842 | 2,953,729,032 |
| 成功 profiling JSONL usage completion | 13,064,625 | 23,084,003 |

B300 两端点的 prompt 差 48,968；旧稿直接除二得到的是二者的平均值。端点异步抓取的边界不同，不能把平均值声称为某个引擎的真实总量。

服务器总量减成功 JSONL usage，prompt 差 25,515,071 / 34,783,237，generation 差 313,009 / 751,365。服务器还观察到错误／取消／未完成请求，采样边界也不同，因此不能要求其与成功请求之和相等。**剩余差额的逐请求归因尚无完整取消记录支持，保留未分解状态**，不将整笔差额强行归给取消请求。

GPU/CPU 命中率由去重后的 `prompt_tokens_by_source` 重算，分别为 0.6822266 / 0.2372618 和 0.8990018 / 0.0163892，与 bmk 一致。`prefix_cache_queries` 是查询计数，不能替代全部 prompt 的分母；external hit 仍保留相对于 GPU 未命中 token 的独立口径。

### trace 顺序、router 身份与 rank 压力

参考初始通道顺序与原始池顺序匹配 **196/196、384/384**；旧 TokenSim 使用 seed-shuffled pool，即便取相同 seed=42，也只匹配 0/196、3/384。参考树完成后从共享顺序 sampler 领取下一条，旧实现按 lane/play 算固定位置。这里只确认差异，回放修复留在后续阶段。

router 日志单次请求有两条路由消息，只统计 `CONSISTENT_HASH_DEBUG`，按 worker URL 的 `@rank` 识别 GPU，不用 worker 列表的 `index`。两组共 16,295 / 30,172 次路由、2,180 / 4,617 个会话；全部 JSONL 记录可映射，无会话跨 rank。子代理记录 9,240 / 17,566 条，没有一条复用 parent correlation ID。

B200 rank3 有 1,966 次路由，约为 rank 均值的 0.965 倍，却达到 100% KV 使用率、prefix hit 0.3219。仅按请求数量的 hash 倾斜解释不了它的压力；需要在后续缓存阶段检查会话长度、存活时间、保留状态和容量。B300 各 rank 没有这一饱和情况。本轮保留证据，不改路由或硬编码容量。

日志中的 `Deferred` 是可选字段，部分样本省略 external hit。字段分别解析，缺失命中值记为 null，不丢弃忙碌或无查询时的 running/waiting 样本。B200/B300 的 525 / 568 条 `Reqs Running` 日志行全部被解析，没有遗漏；另存 364 / 361 个 MTP 接受率样本及完整逐 rank gauge 切片。Rust frontend running/waiting 是聚合值，不能再乘 GPU 数。

日志等效 KV token 总量为 21,943,624 / 43,335,400，明确标记为核对值，不能输入或覆盖容量。B300 rank4 的等效值为 5,416,820，其他 rank 为 5,416,940；保留差异。权重为 125.88 / 126.08 GiB；可用 KV 内存只在 rank0 观察到 32.27 / 103.07 GiB，未伪造其他 rank 的占用数据。

## 运行与迁移

安装新增的流式 JSON 依赖 `ijson` 后，以完整原始数据生成一个新的输出目录：

```bash
python scripts/extract_agentx_reference.py /path/to/reference-artifacts \
  --dataset dataset/agentx/traces.jsonl \
  --matrix data/experiments/agentx.yaml \
  --output results/agentx-dsv4-full/reference-next
```

参考产物根目录包含 `b200/` 与 `b300/`，内部为 `agentic/`、`bmk/`。`--source-file`、`--license` 可附带已核对源码和许可证归档；正式参考摘要已提交，不依赖研究用 `/tmp` 路径永久存在。原始大文件通过来源和校验值定位，未提交或复制原始 trace。

对比工具从运行 manifest 的 `experiments` 数组读取结果文件位置与实际配置。例如：

```json
{
  "experiments": [{
    "id": "b200-c196-dep8-simplecpu",
    "reference_id": 440239,
    "result_file": "b200-c196-dep8-simplecpu/seed-0/agentx_c196.json",
    "seed": 0,
    "num_gpus": 8,
    "hardware": "b200",
    "concurrency": 196,
    "parallelism": {"tp": 1, "dp": 8, "ep": 8, "pp": 1},
    "cache_backend": "simple_cpu",
    "model": "DeepSeek-V4-Pro",
    "profiling_window_s": 3600,
    "trace_count": 393,
    "cache_rate_denominator": "all_prompt_tokens"
  }]
}
```

这展示 manifest 格式，**不代表该完整模型模拟已执行**。统一运行入口会在后续实现并生成这些真实元数据。执行对比：

```bash
python scripts/compare_agentx_results.py results/agentx-dsv4-full/your-batch \
  --reference-root results/agentx-dsv4-full \
  --result-manifest results/agentx-dsv4-full/your-batch/run_manifest.json
```

旧 `cases` manifest 不含实际 GPU 数与逐点身份，工具明确拒绝。确需分析旧数据时，应记录其真实 proxy、64 GPU、1,800 秒等配置；该数据与当前参考不兼容，不得改元数据冒充完整模型。关闭 offload 时 CPU 命中可为 0；缺少计数或分母时输出 null，同口径未确认时不计算命中率误差。旧结果未修改。

## 测试、时间分解与下一部分

完整执行 `python -W error::ResourceWarning -m unittest discover -s tests -v`：**251 项测试全部通过**，新增 26 项；并通过 Python 编译和 `git diff --check`。新增测试验证逻辑；完整参考提取验证全部保存的数据。没有运行 smoke 测试，也没有缩小负载池。

参考日志不包含完整算子时间分解，本轮没有模拟时间分解可报告，未填 0 或从吞吐倒推。**B200/B300 新的一小时完整模拟尚未运行**：当前模拟器仍缺少已确认的完整模型和回放流程，不能拿旧 proxy／短窗口替代验收。下一部分完成完整模型与并行配置，按计划记录权重与容量推导、算子覆盖，打通主检查点完整运行，再保留后续行为与耗时阶段的独立提交边界。

发现的是数据口径与现有实现的事实差异，没有新增需选择的近似方案。本部分完成后停下，不继续修改回放、缓存或算子。
