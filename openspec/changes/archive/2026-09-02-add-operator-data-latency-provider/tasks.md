## 1. 数据包 schema 与运行依赖

- [x] 1.1 定义 hardware、generation metadata、shape manifest、source catalog、五类 compute perf/analysis 表及独立 communication 表的版本化 schema，并用合法最小 fixture 验证所有必填字段和单位可被解析。
- [x] 1.2 增加 `pyarrow` 直接读取 Parquet 的显式依赖与延迟导入入口预检，并验证 AF-disabled 与 stub provider 路径在不导入 reader 时仍可运行、正式 provider 缺依赖时给出可操作错误。
- [x] 1.3 实现数据包级 loader 与全量预检，覆盖未知版本、缺列、类型错误、非有限/负 `latency_us`、悬空 source、单位错误和 duplicate exact key，并验证错误包含文件、字段或完整重复键。

## 2. 离线数据生成与验证

- [x] 2.1 实现 benchmark、公开资料和公式输入到统一 observation/source record 的 adapter，保留 device/backend、方法、输入哈希和 A/B/C/D 证据等级，并验证公式或公开峰值不会被标记为目标设备 measured。
- [x] 2.2 按 operator-data 文档实现 GEMM、Prefill/Decode Attention、MoE、elementwise 的 FLOPs、内存搬运与硬件参数公式及类别 calibration 接口，并以已知小 shape 手算 oracle 验证公式分项和单位。
- [x] 2.3 实现“合格实测优先、公式补齐 manifest 缺口”的确定性候选选择和 runtime/analysis 分表输出，并验证改变输入记录顺序不会改变所选 perf 行、metadata 或数据集哈希。
- [x] 2.4 实现 calibration/holdout 的稳定 manifest 隔离与按算子类别的 coverage、mean/max relative error、P50/P90/P95、bias、bottleneck consistency、worst-shape 报告，并验证 holdout 未参与拟合且无 ground truth 时报告只声明模型自洽。
- [x] 2.5 由小型可审计源样本生成一个完整 Parquet 测试数据包，并验证重新生成得到等价表内容、来源引用和 generation metadata。

## 3. Typed lookup 与受控插值

- [x] 3.1 为 GEMM、context/generation Attention、MoE、elementwise 和 communication 定义 typed key、规范化与唯一索引，并验证 dtype、phase、device/backend 和所有 shape 维度均参与正确匹配。
- [x] 3.2 实现 exact-first 查询及 `match_type=exact` provenance，验证 exact 与可插值邻点并存时始终选择 exact 且返回原始 `latency_us` 和 `source_id`。
- [x] 3.3 实现 GEMM 沿 `m`、Prefill Attention 沿 `input_seq_len`、Decode Attention 沿 `context_len`、MoE 沿 `num_tokens` 的确定性内部单维插值，并验证邻点、权重、来源和结果与表行顺序无关。
- [x] 3.4 实现 elementwise 第一版限制以及越界、缺邻点、跨离散键、跨两维和所有外推的 typed missing error，并验证完整规范化 key 被报告且不会调用 Roofline、LLMCompass 或默认 fallback。

## 4. A/F compute provider

- [x] 4.1 扩充 AF compute query/model metadata 与强类型 provider result，使 dataset version、match type、source 和可序列化 breakdown 可表达，同时验证既有 stub provider 通过明确 `test-stub` identity 保持兼容。
- [x] 4.2 实现 model/operator manifest 和 A composition（RMSNorm、QKV、RoPE、Attention core、O projection、Residual Add、必要 reorder），并用生成数据包验证 breakdown 总和、微秒到秒恰好转换一次以及 QKV/O projection 不被 core 重复包含。
- [x] 4.3 实现 F composition（RMSNorm、Router/Gating、grouped/fused expert、必要 elementwise、Residual Add），并验证 Router、dispatch/combine、SwiGLU 等 fused/unfused 逻辑 operation 恰好计时一次且通信不进入 compute sum。
- [x] 4.4 实现 composition 完整性与 `includes` 冲突校验，并验证缺少任一必要 component 或逻辑 operation 重复覆盖时整次查询失败、不返回部分总和。

## 5. 通信 service provider

- [x] 5.1 实现 payload bytes 的计算/校验和 direction、source/target device、network 的 typed lookup，并验证 A-to-F、F-to-A、AllReduce、MoE dispatch/combine 不会错误共用键。
- [x] 5.2 实现 `fixed_latency + payload_bytes / effective_bandwidth` 的确定性 service time、来源与单位换算，并用手算链路 oracle 验证结果；若使用实测模型则验证其 dataset/version 和适用键被保留。
- [x] 5.3 验证 transfer provider 不创建或申请 SimPy 资源：两个相同查询返回相同 service time，而容量为一的共享链路只在 executor trace 中为后一次产生 queue time。

## 6. Executor 装配与 trace

- [x] 6.1 增加从 AF 配置显式选择并装配 operator-data compute/transfer providers 的入口，验证缺数据包、版本不兼容和 query missing 在首个不支持点明确失败，且不影响 AF-disabled 与旧 backend。
- [x] 6.2 将 dataset、match type、source、component/interpolation details 传递到 AF trace 序列化，并验证 details 可重建 compute service、transfer payload 和来源，同时只读取 service duration 参与调度。
- [x] 6.3 对同一数据包和 workload 运行 trace on/off 对照，验证 worker/resource 选择、事件顺序、block completion 和 makespan 完全一致。

## 7. 完整端到端验收

- [x] 7.1 从源样本实际运行离线 generator、加载生成的 Parquet 包、经 scheduler/AF executor 执行 Prefill-only、Decode-only 和两阶段多 block workload，验证最终 result 与 trace 能重建 A/F compute、双向通信、queue、overlap、barrier 和 makespan。
- [x] 7.2 在端到端路径分别触发 exact、interpolated、missing/out-of-range、duplicate/invalid schema、单位转换、A/F 无重复计时、direction/payload 和 trace provenance 场景，验证错误与成功结果均符合 delta specs。
- [x] 7.3 运行既有单 block 7 ms、双 block 10 ms 等 stub oracle、通信竞争 oracle、AF-disabled 测试和仓库完整 Python 回归，验证 provider 引入未改变现有时序与非 AF 行为，并记录 Windows 原生扩展占位限制。
- [x] 7.4 使用独立 kernel/link holdout 或可追溯真实系统 trace 运行精度报告，分别对照 compute service、transfer service、queue/wait、TTFT、TPOT 和吞吐；若目标硬件 ground truth 尚未交付，则将该项结果明确标为“机制与模型自洽通过、物理准确性待实测”，不得替代为模拟器自身输出。
- [x] 7.5 更新 AF separation/operator-data 使用文档，给出数据包生成、来源等级、正式 provider 配置、失败诊断和精度声明边界，并通过文档命令从干净 fixture 完整复现一次端到端运行。
