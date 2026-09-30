# AgentX 原始请求图与初始抽样核对

日期：2026-09-30；分支：`agentx-support`。本次修改完成内部解析、会话划分、跨流前驱和初始 snapshot 的结构验证。执行回放、缓存接入及一小时性能验收仍未完成。

## 实现

新增 `TokenSim/workload/agentx_native/`，直接读取原始 WEKA JSONL，建立最小请求图。每条原始请求保留 trace、outer/inner 索引、时间戳、API duration、输入／输出长度、原始 model 和完整 hash 序列；同一原始请求必须恰好属于一个会话。

会话划分采用固定版本 AIPerf `49634d1c` 的前缀链、时间可行性、compaction seam、跨 model 分流和辅助／并行 worker 分类规则。NVIDIA 的 chain 算法在 TokenSim 内适配，保留原版权头与 Apache-2.0 许可证；原始文件的内容校验值与修改说明保存在 `source_manifest.json`。使用原生紧凑记录和 unsigned 64-bit hash，支持大于 2^63 的合法 hash ID。没有 AIPerf 包导入、加载器进程、tokenizer、合成提示文本或离线转换输入。

子代理内部相对时间按源码转换到原 trace 时间轴；保留原始 inner index，不能用拆分后 turn index 查回原始长度。根会话、显式子代理、flat agent、auxiliary 和 worker group 的会话 ID 与参考产物一致。SPAWN/JOIN 按分组的 spawning turn、子会话和 join target 保存；terminal 子代理保留为后台分支，空子代理保留结构。无父前驱的异常 marker 会报错，防止静默丢请求；当前完整池没有这一情况。

跨流前驱保留每个独立流最近完成的请求，再删除可以通过其他 frontier 推导的前驱。时间算术按参考的毫秒顺序执行：精确相接有前驱，相同开始时间无前驱，缺失／无效 API duration 按零宽区间处理。查询索引按 end/start/key 排序，用二分选择最近完成项；随机边界差分测试与原始直接扫描规则一致。

新增 snapshot、共享顺序池和 primer 计划：工作负载种子 42，用 seed/trace/lane 的 SHA-256 派生 NumPy RNG，在整棵树请求开始时间跨度的 25%–75% 抽样。保留等待子会话的父会话、根结束后仍活跃的后台子会话和各流已完成前缀。primer 计划重放各活跃流起点前最后一条请求，按全局提前量分布，跨度由 10 秒系统 idle cap 限制。回收由同一个顺序池领取下一条，而非 lane/play 固定位置。缓存 scope 包含 benchmark/pass/lane/base trace，父子共享；预热与 profiling 不改变 continuing tree 的 scope。

## 全部数据与日志核对

| 项目 | 原始请求图计算 | 参考日志／产物 |
|---|---:|---:|
| trace 池 | 393 | 393 |
| 原始请求 | 98,827 | 98,827 |
| conversation | 9,843 | 9,843 |
| 跨流门控请求 | 61,136 | 61,136 |
| B200 逐请求会话／turn／来源完全匹配 | 16,162 / 16,162 | 全部记录 |
| B300 逐请求会话／turn／来源完全匹配 | 29,960 / 29,960 | 全部记录 |
| B200 初始 lane 完全匹配 | 196 / 196 | 全部 lane |
| B300 初始 lane 完全匹配 | 384 / 384 | 全部 lane |
| B200 必要 primer | 211 | 211 |
| B300 必要 primer | 409 | 409 |

前驱宽度 1–14 的数量依次为 **51,527 / 6,771 / 1,804 / 492 / 149 / 80 / 62 / 89 / 74 / 52 / 21 / 11 / 2 / 2**，逐项对齐两组日志的 `join-widths`。所有 lane 的 trace ID、抽样百分比、root next index、root turn 数、live／ready 会话数也逐项相同。两组 primer 发出范围均为 9.9079946 秒。统计与原始身份由解析生成，未硬编码参考总数、会话拆分或前驱列表。

完整审核约 63 秒，12 项检查全部通过。原始数据 SHA-256 与前两部分一致，数据只读。可提交证据在 `data/reference/agentx-request-graph/validation.json`；完整运行审核产物在 `results/agentx-dsv4-full/request-graph/20260930-native-parser/`。

## 分析中纠正的两项语义

**依赖 scope 与缓存 scope 不同。** AIPerf 为每个 captured subagent 单独推导 interval frontier，该子代理的内部 overflow 流共享此 scope；父子之间由 SPAWN/JOIN 连接。若对整棵树无差别增加 interval 前驱，本次完整数据会得到 89,686 个门控请求，额外串行化真实并行。按源码分开后得到正确的 61,136。缓存 hash 仍属于整个原始 trace，不能因分开依赖作用域而切断父子共享前缀。

**抽样时间边界是请求开始时间。** 参考 `_trace_time_bounds` 取整棵树所有请求的最早／最晚 timestamp，未用末次 API end 或 subagent duration 扩展抽样范围。开始时间跨度用于 t*，实际记录的活动结束单独保留；两者混用会改变 warmup 游标和 primer 数。修正后所有初始通道与 primer 数对齐。

## 验证与后续

新增 30 项测试，覆盖前缀 seam、并行／跨 model、preamble、unsigned hash、原始索引、子代理时间与 JOIN、空／后台子代理、前驱边界和 transitive frontier、scope、snapshot、共享抽样池、primer 及 cache scope。完整执行：

```bash
python -W error::ResourceWarning -m unittest discover -s tests -v
python scripts/validate_agentx_graph.py \
  --dataset dataset/agentx/traces.jsonl \
  --reference-artifacts /path/to/complete/reference-artifacts \
  --output results/agentx-dsv4-full/request-graph/your-batch
```

完整测试 **299 项通过**，Python 编译与 diff 检查通过。单元测试与完整原始数据审核验证解析语义，没有运行 smoke，也未将其计作一小时性能验收。

当前旧 benchmark 回放仍使用旧接口；本次的新图和 snapshot 会接入下一步执行回放。还需实现每通道 10 个请求的压力预热与交接、跨流等待与树／系统 idle cap、逐请求状态、宽限和取消；随后与混合缓存、统一入口一起运行两个主点完整一小时。请求回放任务 7、8 的执行集成与一小时验证尚未完成，整体复选框继续保持未完成。
