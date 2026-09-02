## 1. AF 阶段与基础配置模型

- [x] 1.1 定义 `prefill`/`decode` AF phase、`prefill`/`decode_full`/`af_attention`/`af_ffn` capability 常量，并通过单元测试验证允许值、重复值和未知值。
- [x] 1.2 实现默认关闭的 `AFConfig`（两阶段默认、request-count `af_block_size`、空 links、trace 开关），并验证缺省 AF 节与显式 `enabled=false` 均不改变现有配置语义。
- [x] 1.3 为 `phases` 非空集合和 `af_block_size` 严格正整数实现校验，并通过参数化测试覆盖两阶段、Prefill-only、Decode-only、空列表、重复阶段、未知阶段、0、负数和布尔值。

## 2. Worker capability 与旧配置兼容

- [x] 2.1 扩展 worker group/worker 配置以携带稳定 `name` 和 capabilities，并验证一个 group 可组合完整阶段与 AF capabilities，同时保留硬件、数量和默认网络。
- [x] 2.2 按当前代码实现 `prefill -> prefill`、`decode -> decode_full`、`hybrid -> prefill + decode_full` 的 legacy 归一化和确定性名称生成，并运行现有 cluster/worker/KV transfer 测试确认旧配置行为不变。
- [x] 2.3 检测未知/重复 capability、重复 group 名、非正 worker 数量、缺失默认网络以及 role/capabilities 冲突，并验证每类失败均抛出包含字段或 group 上下文的 `ConfigurationError`。

## 3. 显式 AF link 与阶段拓扑

- [x] 3.1 实现 `AFLinkConfig` 的名称、阶段、A/F group、network、resource group、duplex 和模式相关容量字段，并通过单元测试验证 full/half 合法输入及缺字段、非法枚举、非正/布尔容量。
- [x] 3.2 构建确定性的 `phase -> link` 映射，要求每个所选阶段恰好一条 link，并通过测试覆盖单 link 两阶段、两 link 分阶段、缺失 link 和同阶段多 link。
- [x] 3.3 校验 link 端点存在、A/F group 不同且分别声明 `af_attention`/`af_ffn`，并验证错误信息包含 phase、link 和端点 group 上下文。
- [x] 3.4 校验未选择阶段仍有完整执行 capability：Decode-only 需要 `prefill`，Prefill-only 需要 `decode_full`，并通过正反例测试验证。
- [x] 3.5 校验 link 的 network 引用存在于集群 networks，并验证硬件名称不会被用于推断 A/F 角色。

## 4. 通信竞争资源归一化

- [x] 4.1 为 full-duplex resource group 生成独立 A→F/F→A 资源键，为 half-duplex resource group 生成共享资源键，并通过测试验证相同 group 共享、不同 group 隔离。
- [x] 4.2 聚合同一 resource group 的 links 并要求 network、duplex 和有效容量一致，通过测试覆盖合法共享以及网络、双工和容量冲突。
- [x] 4.3 在规范化通信描述中保留 queue/service 分离所需的资源键、方向和容量元数据，并通过对象断言验证 `capacity` 表示槽位而非带宽倍数。

## 5. KV 边界与阶段性运行保护

- [x] 5.1 保持 `benchmark.py --block_size` 的 KV Cache 兼容语义，在入口使用明确的 `kv_block_size` 局部命名，并通过调用断言证明 `af_block_size` 不进入 Request/BlockManager 的 KV block 路径。
- [x] 5.2 确保配置文件加载和 Python 直接构造使用同一归一化/校验入口，并通过等价输入测试比较规范化结果。
- [x] 5.3 在 AF executor 尚未实现时，让运行入口对合法 `af.enabled=true` 在 SimPy 启动前抛出明确 `ConfigurationError`，并通过入口测试证明不会静默调用 Roofline/LLMCompass 或完整 iteration backend。

## 6. 示例、文档与回归验证

- [x] 6.1 添加不含虚构性能数值的两阶段 AF 配置样例，包含独立 Prefill/A/F groups、显式 link、network 和 resource group，并验证可由 `ClusterConfig.from_file()` 解析。
- [x] 6.2 添加 Decode-only AF 与 half-duplex 共享资源样例，验证未选 Prefill 使用完整能力且双向映射到同一资源键。
- [x] 6.3 更新配置说明，记录阶段选择、capability 词汇、真实 legacy role 映射、AF/KV block 区别、resource group 竞争语义和阶段性运行保护，并核对示例字段与规格一致。
- [x] 6.4 记录后续 operator-data 契约：离线层读取 `shape_manifests/<experiment-id>.yaml`，按 GEMM、Prefill Attention、Decode Attention、MoE、逐元素算子生成一一对应的 `*_perf.parquet`/`*_analysis.parquet` 和 `generation_meta.yaml`；TokenSim 运行时不重新计算 roofline；Groq 第一版公开峰值参数必须逐项携带来源且不得冒充算子实测。
- [x] 6.5 记录结构化查询与组合边界：精确匹配优先，仅允许文档规定的一维内插且禁止外推；返回 `latency_us`、`match_type`、`source_id` 和完整缺失主键；A 聚合 QKV/Attention/输出投影/逐元素，F 聚合 Router/MoE/逐元素，dispatch/combine/AllReduce/A↔F 保持独立通信事件。
- [x] 6.6 核对当前 change 未新增 AIConfigurator/InferSim 运行依赖、Groq 数据文件、离线生成器或查询 provider，并运行新增配置测试和现有完整测试套件，确认 AF 默认关闭时 Prefill、Decode、KV transfer、并行、MoE 和 latency baseline 回归全部通过。

## 7. 下游 AF 验证契约

- [x] 7.1 交付后续 executor change 可直接执行的 AF 端到端验证矩阵，覆盖单 block 单层、单 block 多层、双 block 流水、A 慢/F 快、A 快/F 慢、全/半双工、多个 F worker、Prefill、Decode 和非 AF 回归，并逐项定义依赖顺序、重叠、竞争、queue/service、iteration barrier、请求恰好推进一次和 trace 完整性的预期断言；同时报告 `T_C<T_A`、`T_C<T_F` 和 micro-batch 数下界，但以事件时间线作为最终判定。
- [x] 7.2 在验证矩阵中固定双 block oracle：A compute 2 ms、A→F 1 ms、F compute 3 ms、F→A 1 ms，容量为 1 时单 block 为 7 ms、完全串行为 14 ms、正确流水 makespan 为 10 ms；注明该测试从真实配置加载运行到最终 trace/result，但使用 stub providers，只验证执行机制而非物理精度。
- [x] 7.3 定义计算数据与校验契约，要求 A（官方规格或算子/层级实测）、B（公开端到端基准）、C（公开结果校准 roofline）、D（项目假设）四级 `source_id` 元数据，calibration/holdout 隔离，并按 GEMM、Prefill/Decode Attention、MoE 和逐元素算子报告 coverage、平均/最大相对误差、P50/P90/P95、bias、瓶颈分类和 worst-error shapes。
- [x] 7.4 定义通信 benchmark 数据契约，覆盖 payload、方向、双工、并发、本地/跨设备/跨节点、resource group 和内部 dispatch/combine；规定 service time 对照实测、queue time 对照事件 oracle、total 等于 queue 加 service。
- [x] 7.5 定义公开与真实系统 trace 对齐契约，要求记录模型版本、精度、batch/并发、输入输出长度、芯片数、主机通信和指标定义；匹配条件下比较 A/F kernel、transfer、queue/wait、TTFT、TPOT 与吞吐，BERT 论文结果作模型级校验，信息不完整的 Mixtral 吞吐只作弱范围检查。
- [x] 7.6 在配置说明和验证契约中将查询方式 `match_type=exact|interpolated` 与证据来源 `source_id` 分开；C/D 数据不得声明物理准确，并记录 AIConfigurator/InferSim 只是设计参考，Roofline 仅用于离线生成而非 AF 运行时 fallback。
- [x] 7.7 核对本 change 仅交付上述验证矩阵、数据和报告契约；实际 AF executor E2E harness、kernel/link 采集、provider 校准和真实系统对照留给对应后续 changes。
- [x] 7.8 定义敏感性分析契约，分别扰动算子延迟、有效带宽和固定延迟，报告 AF 结论的稳定区间、结论发生变化的边界、已知误差来源和仍缺失的数据。
